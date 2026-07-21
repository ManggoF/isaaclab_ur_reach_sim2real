#!/usr/bin/env python3
"""Interactively estimate the payload mounted after a force/torque sensor."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.utils.tool_com_estimator import (  # noqa: E402
    ToolComFit,
    fit_tool_com,
    quality_warnings,
    quaternion_xyzw_to_matrix,
    transform_point,
)


DEFAULT_WRENCH_TOPIC = "/force_torque_sensor_broadcaster/wrench"
DEFAULT_OUTPUT = PROJECT_ROOT / "calibration" / "tool_com_result.json"


@dataclass(frozen=True)
class WrenchRecord:
    receive_time_s: float
    frame_id: str
    wrench: np.ndarray


@dataclass(frozen=True)
class PoseMeasurement:
    gravity_m_s2: np.ndarray
    wrench_mean: np.ndarray
    wrench_std: np.ndarray
    raw_sample_count: int
    used_sample_count: int
    gravity_motion_deg: float


def clean_frame_id(frame_id: str) -> str:
    return frame_id.strip().lstrip("/")


def vector_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1.0e-12:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def robust_wrench_average(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Reject isolated spikes with a median absolute deviation filter."""

    values = np.asarray(samples, dtype=np.float64)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    scale_floor = np.array([0.01, 0.01, 0.01, 0.001, 0.001, 0.001], dtype=np.float64)
    robust_scale = np.maximum(1.4826 * mad, scale_floor)
    inlier_mask = np.all(np.abs(values - median) <= 4.5 * robust_scale, axis=1)
    if int(inlier_mask.sum()) < max(10, values.shape[0] // 2):
        inlier_mask = np.ones(values.shape[0], dtype=bool)
    inliers = values[inlier_mask]
    sample_std = np.std(inliers, axis=0, ddof=1) if inliers.shape[0] > 1 else np.zeros(6)
    return np.mean(inliers, axis=0), sample_std, int(inliers.shape[0])


class ToolComMeasurementNode(Node):
    def __init__(self, wrench_topic: str) -> None:
        super().__init__("tool_com_measurement")
        self.tf_buffer = Buffer(cache_time=Duration(seconds=20.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._lock = threading.Lock()
        self._records: deque[WrenchRecord] = deque(maxlen=100_000)
        self.create_subscription(
            WrenchStamped,
            wrench_topic,
            self._wrench_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"Listening for raw wrench data on {wrench_topic}")

    def _wrench_callback(self, msg: WrenchStamped) -> None:
        wrench = np.array(
            [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(wrench).all():
            return
        record = WrenchRecord(
            receive_time_s=time.monotonic(),
            frame_id=clean_frame_id(msg.header.frame_id),
            wrench=wrench,
        )
        with self._lock:
            self._records.append(record)

    def latest_record(self) -> WrenchRecord | None:
        with self._lock:
            if not self._records:
                return None
            return self._records[-1]

    def records_between(
        self,
        start_time_s: float,
        end_time_s: float,
        message_frame: str | None,
    ) -> np.ndarray:
        with self._lock:
            records = [
                record.wrench.copy()
                for record in self._records
                if start_time_s <= record.receive_time_s <= end_time_s
                and (message_frame is None or record.frame_id == message_frame)
            ]
        if not records:
            return np.empty((0, 6), dtype=np.float64)
        return np.asarray(records, dtype=np.float64)

    def lookup_transform_arrays(
        self,
        target_frame: str,
        source_frame: str,
        timeout_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        transform = self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time(),
            timeout=Duration(seconds=timeout_s),
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            np.array([translation.x, translation.y, translation.z], dtype=np.float64),
            np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float64),
        )

    def gravity_in_frame(
        self,
        base_frame: str,
        wrench_frame: str,
        gravity_m_s2: float,
        timeout_s: float,
    ) -> np.ndarray:
        _, quaternion_base_wrench = self.lookup_transform_arrays(
            base_frame, wrench_frame, timeout_s
        )
        rotation_base_wrench = quaternion_xyzw_to_matrix(quaternion_base_wrench)
        gravity_base = np.array([0.0, 0.0, -gravity_m_s2], dtype=np.float64)
        return rotation_base_wrench.T @ gravity_base


def resolve_wrench_frame(node: ToolComMeasurementNode, requested_frame: str) -> tuple[str, str | None]:
    latest = node.latest_record()
    if latest is None:
        raise RuntimeError("No force/torque message has been received.")
    if requested_frame:
        return clean_frame_id(requested_frame), None
    if not latest.frame_id:
        raise RuntimeError(
            "The WrenchStamped header.frame_id is empty. Pass --wrench-frame explicitly."
        )
    return latest.frame_id, latest.frame_id


def wait_until_ready(
    node: ToolComMeasurementNode,
    args: argparse.Namespace,
) -> tuple[str, str | None]:
    deadline = time.monotonic() + args.startup_timeout
    last_error = "waiting for wrench data"
    while time.monotonic() < deadline and rclpy.ok():
        latest = node.latest_record()
        if latest is None or time.monotonic() - latest.receive_time_s > args.data_timeout:
            time.sleep(0.1)
            continue
        try:
            wrench_frame, message_frame = resolve_wrench_frame(node, args.wrench_frame)
            node.gravity_in_frame(
                args.base_frame, wrench_frame, args.gravity, args.tf_timeout
            )
            return wrench_frame, message_frame
        except Exception as exc:  # TF exception types differ across ROS2 patch releases.
            last_error = str(exc)
            time.sleep(0.2)
    raise RuntimeError(
        f"Input did not become ready within {args.startup_timeout:.1f}s: {last_error}. "
        "Check the wrench topic and TF frame names."
    )


def capture_pose(
    node: ToolComMeasurementNode,
    args: argparse.Namespace,
    wrench_frame: str,
    message_frame: str | None,
) -> PoseMeasurement:
    time.sleep(args.settle_seconds)
    gravity_before = node.gravity_in_frame(
        args.base_frame, wrench_frame, args.gravity, args.tf_timeout
    )
    start_time_s = time.monotonic()
    time.sleep(args.sample_seconds)
    end_time_s = time.monotonic()
    gravity_after = node.gravity_in_frame(
        args.base_frame, wrench_frame, args.gravity, args.tf_timeout
    )

    values = node.records_between(start_time_s, end_time_s, message_frame)
    if values.shape[0] < args.min_window_samples:
        raise RuntimeError(
            f"Only {values.shape[0]} wrench samples arrived; need at least "
            f"{args.min_window_samples}."
        )
    wrench_mean, wrench_std, used_count = robust_wrench_average(values)
    gravity_motion_deg = vector_angle_deg(gravity_before, gravity_after)
    gravity_average = gravity_before + gravity_after
    gravity_norm = float(np.linalg.norm(gravity_average))
    if gravity_norm < 1.0e-9:
        gravity_average = gravity_before
    else:
        gravity_average *= args.gravity / gravity_norm
    return PoseMeasurement(
        gravity_m_s2=gravity_average,
        wrench_mean=wrench_mean,
        wrench_std=wrench_std,
        raw_sample_count=int(values.shape[0]),
        used_sample_count=used_count,
        gravity_motion_deg=gravity_motion_deg,
    )


def reject_reason(
    measurement: PoseMeasurement,
    accepted: list[PoseMeasurement],
    args: argparse.Namespace,
) -> str | None:
    if measurement.gravity_motion_deg > args.max_gravity_motion_deg:
        return (
            f"tool moved during sampling ({measurement.gravity_motion_deg:.2f} deg gravity change; "
            f"limit {args.max_gravity_motion_deg:.2f} deg)"
        )
    max_force_std = float(np.max(measurement.wrench_std[:3]))
    if max_force_std > args.max_force_std:
        return f"force is not stable (max std {max_force_std:.4f} N)"
    max_torque_std = float(np.max(measurement.wrench_std[3:]))
    if max_torque_std > args.max_torque_std:
        return f"torque is not stable (max std {max_torque_std:.5f} Nm)"
    if accepted:
        nearest_angle = min(
            vector_angle_deg(measurement.gravity_m_s2, item.gravity_m_s2)
            for item in accepted
        )
        if nearest_angle < args.min_pose_separation_deg:
            return (
                f"orientation is too similar to an accepted pose ({nearest_angle:.1f} deg; "
                f"need {args.min_pose_separation_deg:.1f} deg)"
            )
    return None


def collect_measurements(
    node: ToolComMeasurementNode,
    args: argparse.Namespace,
    wrench_frame: str,
    message_frame: str | None,
) -> tuple[list[PoseMeasurement], ToolComFit]:
    accepted: list[PoseMeasurement] = []
    fit_error = ""
    print(
        "\nMove the robot only with the teach pendant/freedrive. Keep the spoon empty and "
        "make sure the tool touches nothing."
    )
    print(
        "Use clearly different tilts so gravity points toward different sensor axes. "
        "Do not re-zero the sensor between poses."
    )
    print("At each static pose press Enter to sample; enter 'r' to remove the last pose or 'q' to finish.\n")

    while len(accepted) < args.max_poses:
        if len(accepted) >= args.pose_count:
            try:
                fit = fit_tool_com(
                    np.asarray([item.gravity_m_s2 for item in accepted]),
                    np.asarray([item.wrench_mean for item in accepted]),
                )
                return accepted, fit
            except ValueError as exc:
                fit_error = str(exc)
                print(f"Pose coverage is still insufficient: {fit_error}")

        prompt = f"Pose {len(accepted) + 1}/{args.pose_count} ready [Enter/r/q]: "
        command = input(prompt).strip().lower()
        if command == "q":
            if len(accepted) < 4:
                print("At least 4 accepted poses are required.")
                continue
            try:
                fit = fit_tool_com(
                    np.asarray([item.gravity_m_s2 for item in accepted]),
                    np.asarray([item.wrench_mean for item in accepted]),
                )
                return accepted, fit
            except ValueError as exc:
                print(f"Cannot fit these poses yet: {exc}")
                continue
        if command == "r":
            if accepted:
                accepted.pop()
                print("Removed the last accepted pose.")
            continue
        if command:
            print("Unknown command. Press Enter, or use 'r'/'q'.")
            continue

        print(
            f"  settling {args.settle_seconds:.1f}s, then sampling "
            f"{args.sample_seconds:.1f}s ..."
        )
        try:
            measurement = capture_pose(
                node, args, wrench_frame=wrench_frame, message_frame=message_frame
            )
        except Exception as exc:
            print(f"  rejected: {exc}")
            continue
        reason = reject_reason(measurement, accepted, args)
        if reason:
            print(f"  rejected: {reason}")
            continue
        accepted.append(measurement)
        force = measurement.wrench_mean[:3]
        torque = measurement.wrench_mean[3:]
        print(
            f"  accepted: F={np.round(force, 4)} N, T={np.round(torque, 5)} Nm, "
            f"samples={measurement.used_sample_count}/{measurement.raw_sample_count}"
        )

    if fit_error:
        raise RuntimeError(
            f"Reached --max-poses={args.max_poses} but the fit is still invalid: {fit_error}"
        )
    fit = fit_tool_com(
        np.asarray([item.gravity_m_s2 for item in accepted]),
        np.asarray([item.wrench_mean for item in accepted]),
    )
    return accepted, fit


def save_samples_csv(
    path: Path,
    measurements: list[PoseMeasurement],
    wrench_frame: str,
) -> None:
    header = [
        "pose_index",
        "wrench_frame",
        "gravity_x_m_s2",
        "gravity_y_m_s2",
        "gravity_z_m_s2",
        "force_x_n",
        "force_y_n",
        "force_z_n",
        "torque_x_nm",
        "torque_y_nm",
        "torque_z_nm",
        "force_std_x_n",
        "force_std_y_n",
        "force_std_z_n",
        "torque_std_x_nm",
        "torque_std_y_nm",
        "torque_std_z_nm",
        "raw_sample_count",
        "used_sample_count",
        "gravity_motion_deg",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for index, measurement in enumerate(measurements, start=1):
            writer.writerow(
                [index, wrench_frame]
                + measurement.gravity_m_s2.astype(float).tolist()
                + measurement.wrench_mean.astype(float).tolist()
                + measurement.wrench_std.astype(float).tolist()
                + [
                    measurement.raw_sample_count,
                    measurement.used_sample_count,
                    measurement.gravity_motion_deg,
                ]
            )


def save_result(
    node: ToolComMeasurementNode,
    args: argparse.Namespace,
    measurements: list[PoseMeasurement],
    fit: ToolComFit,
    wrench_frame: str,
) -> tuple[Path, Path, np.ndarray | None, list[str]]:
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path = output_path.with_name(f"{output_path.stem}_samples.csv")

    warnings = quality_warnings(
        fit,
        force_rmse_limit_n=args.max_fit_force_rmse,
        torque_rmse_limit_nm=args.max_fit_torque_rmse,
    )
    com_sim_frame: np.ndarray | None = None
    sim_transform: dict[str, object] | None = None
    if args.sim_frame:
        sim_frame = clean_frame_id(args.sim_frame)
        try:
            translation, quaternion = node.lookup_transform_arrays(
                sim_frame, wrench_frame, args.tf_timeout
            )
            com_sim_frame = transform_point(fit.com_offset_m, translation, quaternion)
            sim_transform = {
                "target_frame": sim_frame,
                "source_frame": wrench_frame,
                "translation_m": translation.astype(float).tolist(),
                "quaternion_xyzw": quaternion.astype(float).tolist(),
            }
        except Exception as exc:
            warnings.append(
                f"could not transform center of mass into {sim_frame}: {exc}"
            )

    fit_data = fit.to_dict()
    result = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool_name": args.tool_name,
        "method": "multi_pose_static_gravity_wrench_least_squares",
        "measurement": {
            "wrench_topic": args.wrench_topic,
            "wrench_frame": wrench_frame,
            "base_frame": clean_frame_id(args.base_frame),
            "gravity_m_s2": args.gravity,
            "pose_count": len(measurements),
            "sample_seconds_per_pose": args.sample_seconds,
        },
        "payload": {
            "mass_kg": fit.mass_kg,
            "center_of_mass": {
                "frame": wrench_frame,
                "xyz_m": fit.com_offset_m.astype(float).tolist(),
            },
        },
        "sensor_model": {
            "signed_mass_kg": fit.signed_mass_kg,
            "force_bias_n": fit.force_bias_n.astype(float).tolist(),
            "torque_bias_nm": fit.torque_bias_nm.astype(float).tolist(),
        },
        "fit_quality": {
            **fit_data["fit_error"],
            **fit_data["conditioning"],
            "status": "warning" if warnings else "good",
            "warnings": warnings,
        },
        "simulation_equivalent_payload": None,
        "frame_transform": sim_transform,
        "samples_csv": str(samples_path),
        "usage_note": (
            "This is the combined payload after the force sensor. Assign it to a separate fixed "
            "tool rigid body, or combine it with an existing link's mass properties; do not overwrite "
            "the wrist link mass directly. Static measurements do not identify the inertia tensor."
        ),
    }
    if com_sim_frame is not None:
        result["simulation_equivalent_payload"] = {
            "frame": clean_frame_id(args.sim_frame),
            "mass_kg": fit.mass_kg,
            "center_of_mass_xyz_m": com_sim_frame.astype(float).tolist(),
        }

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary_path.replace(output_path)
    save_samples_csv(samples_path, measurements, wrench_frame)
    return output_path, samples_path, com_sim_frame, warnings


def print_result(
    fit: ToolComFit,
    wrench_frame: str,
    sim_frame: str,
    com_sim_frame: np.ndarray | None,
    output_path: Path,
    samples_path: Path,
    warnings: list[str],
) -> None:
    print("\nMeasurement result")
    print(f"  mass: {fit.mass_kg:.6f} kg")
    print(
        f"  COM in {wrench_frame}: "
        f"[{fit.com_offset_m[0]:.6f}, {fit.com_offset_m[1]:.6f}, {fit.com_offset_m[2]:.6f}] m"
    )
    if com_sim_frame is not None:
        print(
            f"  COM in {clean_frame_id(sim_frame)}: "
            f"[{com_sim_frame[0]:.6f}, {com_sim_frame[1]:.6f}, {com_sim_frame[2]:.6f}] m"
        )
    print(
        f"  fit error: force={fit.force_rmse_n:.4f} N, "
        f"torque={fit.torque_rmse_nm:.5f} Nm, gravity span={fit.gravity_span_deg:.1f} deg"
    )
    if fit.signed_mass_kg < 0.0:
        print("  sensor convention: reaction-wrench sign detected; physical mass above is positive")
    if warnings:
        print("  quality warnings:")
        for warning in warnings:
            print(f"    - {warning}")
    print(f"  result JSON: {output_path}")
    print(f"  pose samples: {samples_path}")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the combined mass and center of mass of the gripper/spoon mounted "
            "after a force/torque sensor. The script never commands robot motion."
        )
    )
    parser.add_argument("--wrench-topic", default=DEFAULT_WRENCH_TOPIC)
    parser.add_argument(
        "--wrench-frame",
        default="",
        help="Frame in which wrench components are expressed; empty uses WrenchStamped.header.frame_id.",
    )
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument(
        "--sim-frame",
        default="wrist_3_link",
        help="Also express the fitted COM in this simulation/link frame; empty disables it.",
    )
    parser.add_argument("--tool-name", default="gripper_and_spoon")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--pose-count", type=int, default=8)
    parser.add_argument("--max-poses", type=int, default=14)
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    parser.add_argument("--min-window-samples", type=int, default=30)
    parser.add_argument("--gravity", type=float, default=9.80665)
    parser.add_argument("--startup-timeout", type=float, default=15.0)
    parser.add_argument("--data-timeout", type=float, default=1.0)
    parser.add_argument("--tf-timeout", type=float, default=1.0)
    parser.add_argument("--min-pose-separation-deg", type=float, default=15.0)
    parser.add_argument("--max-gravity-motion-deg", type=float, default=0.5)
    parser.add_argument("--max-force-std", type=float, default=0.20)
    parser.add_argument("--max-torque-std", type=float, default=0.015)
    parser.add_argument("--max-fit-force-rmse", type=float, default=0.20)
    parser.add_argument("--max-fit-torque-rmse", type=float, default=0.03)

    argv = sys.argv[1:]
    if "--ros-args" in argv:
        ros_index = argv.index("--ros-args")
        app_args = argv[:ros_index]
        ros_args = argv[ros_index:]
    else:
        app_args = argv
        ros_args = []
    args = parser.parse_args(app_args)
    if args.pose_count < 4:
        parser.error("--pose-count must be at least 4")
    if args.max_poses < args.pose_count:
        parser.error("--max-poses must be greater than or equal to --pose-count")
    if args.gravity <= 0.0 or args.sample_seconds <= 0.0 or args.settle_seconds < 0.0:
        parser.error("gravity/sample timing values must be positive")
    return args, ros_args


def main() -> int:
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = ToolComMeasurementNode(args.wrench_topic)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, name="tool-com-ros-spin", daemon=True)
    spin_thread.start()

    try:
        wrench_frame, message_frame = wait_until_ready(node, args)
        print(
            f"Ready: wrench={args.wrench_topic}, wrench_frame={wrench_frame}, "
            f"base_frame={clean_frame_id(args.base_frame)}"
        )
        measurements, fit = collect_measurements(
            node, args, wrench_frame=wrench_frame, message_frame=message_frame
        )
        output_path, samples_path, com_sim_frame, warnings = save_result(
            node, args, measurements, fit, wrench_frame
        )
        print_result(
            fit,
            wrench_frame,
            args.sim_frame,
            com_sim_frame,
            output_path,
            samples_path,
            warnings,
        )
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\nMeasurement cancelled; no result was written.")
        return 130
    except Exception as exc:
        print(f"\nMeasurement failed: {exc}", file=sys.stderr)
        return 1
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

