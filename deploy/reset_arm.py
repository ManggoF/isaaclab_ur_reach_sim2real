import argparse
import math
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class ArmResetNode(Node):
    """Send the UR5e to the policy training default joint pose."""

    STATE_TOPIC = "/scaled_joint_trajectory_controller/controller_state"
    CMD_TOPIC = "/scaled_joint_trajectory_controller/joint_trajectory"
    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    JOINT_NAME_TO_IDX = {name: idx for idx, name in enumerate(JOINT_NAMES)}
    DEFAULT_JOINT_POS = np.deg2rad(np.array([-45.0, -80.0, 110.0, -90.0, 0.0, 60.0], dtype=np.float64))

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("arm_reset_node")
        self.args = args
        self.current_joint_pos: np.ndarray | None = None
        self.sent = False
        self.finished = False
        self.exit_requested = False
        self.publish_count = 0
        self.shutdown_timer = None
        self.target_joint_pos: np.ndarray | None = None
        self.trajectory_duration_s: float | None = None
        self.execute_start_time_ns: int | None = None
        self.reached_since_ns: int | None = None

        self.create_subscription(JointTrajectoryControllerState, self.STATE_TOPIC, self.joint_state_callback, 10)
        self.pub = self.create_publisher(JointTrajectory, self.CMD_TOPIC, 10)
        self.timer = self.create_timer(0.1, self.step)

        mode = "EXECUTE，会发布归位轨迹" if args.execute else "preview，只打印不发布；加 --execute 才会运动"
        self.get_logger().info(f"arm reset 节点已启动: {mode}")
        self.get_logger().info(f"等待 {self.STATE_TOPIC} ...")

    def joint_state_callback(self, msg: JointTrajectoryControllerState) -> None:
        pos = np.full(6, np.nan, dtype=np.float64)
        for msg_idx, joint_name in enumerate(msg.joint_names):
            if joint_name not in self.JOINT_NAME_TO_IDX:
                continue
            dst_idx = self.JOINT_NAME_TO_IDX[joint_name]
            pos[dst_idx] = float(msg.feedback.positions[msg_idx])

        if np.any(np.isnan(pos)):
            missing = [name for name, idx in self.JOINT_NAME_TO_IDX.items() if np.isnan(pos[idx])]
            self.get_logger().warn(f"controller_state 缺少关节: {missing}", throttle_duration_sec=1.0)
            return

        self.current_joint_pos = pos

    def step(self) -> None:
        if self.sent:
            self.monitor_execution()
            return
        if self.current_joint_pos is None:
            return
        if self.args.execute and self.pub.get_subscription_count() == 0:
            self.get_logger().warn(f"等待 {self.CMD_TOPIC} 出现订阅者...", throttle_duration_sec=1.0)
            return

        target = self.DEFAULT_JOINT_POS.copy()
        delta = target - self.current_joint_pos
        max_delta = float(np.max(np.abs(delta)))
        duration_s = max(self.args.min_duration, max_delta / max(self.args.max_joint_velocity, 1.0e-6))

        self.get_logger().info(
            "\n[arm reset]\n"
            f"current(deg)={np.round(np.rad2deg(self.current_joint_pos), 2)}\n"
            f"target(deg)={np.round(np.rad2deg(target), 2)}\n"
            f"delta(deg)={np.round(np.rad2deg(delta), 2)}\n"
            f"duration={duration_s:.2f}s, max_joint_velocity={self.args.max_joint_velocity:.3f}rad/s"
        )

        if not self.args.execute:
            self.get_logger().info("preview 完成：未发布轨迹。确认无误后加 --execute。")
            self.sent = True
            self.shutdown_later()
            return

        traj = JointTrajectory()
        traj.joint_names = list(self.JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [float(target[self.JOINT_NAME_TO_IDX[name]]) for name in self.JOINT_NAMES]
        sec = int(duration_s)
        nanosec = int((duration_s - sec) * 1.0e9)
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)
        traj.points.append(point)

        self.sent = True
        self.target_joint_pos = target
        self.trajectory_duration_s = duration_s
        self.execute_start_time_ns = self.get_clock().now().nanoseconds
        self.reset_traj = traj
        self.publish_timer = self.create_timer(0.2, self.publish_reset_trajectory)

    def publish_reset_trajectory(self) -> None:
        self.pub.publish(self.reset_traj)
        self.publish_count += 1
        if self.publish_count == 1:
            self.get_logger().info(
                f"已发布归位轨迹。订阅者数量={self.pub.get_subscription_count()}。运动期间请保持急停可用并观察机械臂。"
            )
        if self.publish_count >= self.args.publish_repeats:
            self.publish_timer.cancel()

    def monitor_execution(self) -> None:
        if self.finished or not self.args.execute:
            return
        if self.current_joint_pos is None or self.target_joint_pos is None:
            return

        now_ns = self.get_clock().now().nanoseconds
        max_error = float(np.max(np.abs(self.target_joint_pos - self.current_joint_pos)))
        tolerance = math.radians(self.args.goal_tolerance_deg)

        if max_error <= tolerance:
            if self.reached_since_ns is None:
                self.reached_since_ns = now_ns
                self.get_logger().info(
                    f"已到达归位目标附近: max_error={math.degrees(max_error):.2f}deg，等待稳定..."
                )
            elif (now_ns - self.reached_since_ns) * 1.0e-9 >= self.args.settle_time:
                self.finished = True
                self.get_logger().info(
                    f"归位完成，自动退出。final_error={math.degrees(max_error):.2f}deg"
                )
                self.request_exit()
            return

        self.reached_since_ns = None
        if self.execute_start_time_ns is None or self.trajectory_duration_s is None:
            return
        elapsed_s = (now_ns - self.execute_start_time_ns) * 1.0e-9
        timeout_s = self.trajectory_duration_s + self.args.execution_timeout_margin
        if elapsed_s > timeout_s:
            self.finished = True
            self.get_logger().warn(
                "归位执行超时，自动退出。"
                f"elapsed={elapsed_s:.1f}s, timeout={timeout_s:.1f}s, max_error={math.degrees(max_error):.2f}deg。"
                "如果机械臂没有动，检查 External Control 和 speed_scaling。"
            )
            self.request_exit()

    def shutdown_later(self) -> None:
        self.shutdown_timer = self.create_timer(self.args.keepalive_after_publish, self.request_exit)

    def request_exit(self) -> None:
        self.exit_requested = True


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Reset UR5e to the policy default joint pose.")
    parser.add_argument("--execute", action="store_true", help="Actually publish the reset trajectory. Omit for preview only.")
    parser.add_argument("--max-joint-velocity", type=float, default=0.25, help="Velocity used to choose trajectory duration in rad/s.")
    parser.add_argument("--min-duration", type=float, default=4.0, help="Minimum trajectory duration in seconds.")
    parser.add_argument("--publish-repeats", type=int, default=5, help="Number of times to publish the same reset trajectory.")
    parser.add_argument("--keepalive-after-publish", type=float, default=3.0, help="Seconds to keep this node alive after publishing.")
    parser.add_argument("--goal-tolerance-deg", type=float, default=0.1, help="Exit after all joints are within this many degrees of the target.")
    parser.add_argument("--settle-time", type=float, default=0.5, help="Seconds the robot must remain within tolerance before exiting.")
    parser.add_argument("--execution-timeout-margin", type=float, default=8.0, help="Extra seconds after planned duration before timing out.")
    return parser.parse_known_args()


def main() -> None:
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = ArmResetNode(args)
    try:
        while rclpy.ok() and not node.exit_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.001)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
