"""Estimate a rigid tool payload from static force/torque measurements.

The input wrench is assumed to be expressed at one fixed sensor origin and in
the sensor frame.  The sensor bias is fitted together with the payload.  This
is useful for the UR driver because a zeroing operation is not required at
every pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def skew(vector: np.ndarray) -> np.ndarray:
    """Return the matrix such that ``skew(v) @ x == v cross x``."""

    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a ROS xyzw quaternion to a 3x3 rotation matrix."""

    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm < 1.0e-12:
        raise ValueError("Quaternion norm is zero.")
    x, y, z, w = np.asarray([x, y, z, w], dtype=np.float64) / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_point(
    point_in_source: np.ndarray,
    translation_target_source: np.ndarray,
    quaternion_target_source_xyzw: np.ndarray,
) -> np.ndarray:
    """Transform a point from ``source`` into ``target`` using a TF pose."""

    rotation = quaternion_xyzw_to_matrix(quaternion_target_source_xyzw)
    return rotation @ np.asarray(point_in_source, dtype=np.float64) + np.asarray(
        translation_target_source, dtype=np.float64
    )


@dataclass(frozen=True)
class ToolComFit:
    """Result of the static payload fit.

    ``com_offset_m`` is measured from the wrench origin in the wrench frame.
    ``signed_mass_kg`` retains the sign implied by the sensor wrench
    convention.  ``mass_kg`` is the physical positive mass.  Keeping the
    signed value makes the result independent of whether the sensor reports
    the reaction wrench or the opposite wrench.
    """

    signed_mass_kg: float
    mass_kg: float
    com_offset_m: np.ndarray
    force_bias_n: np.ndarray
    torque_bias_nm: np.ndarray
    force_rmse_n: float
    force_component_rmse_n: float
    torque_rmse_nm: float
    torque_component_rmse_nm: float
    force_max_error_n: float
    torque_max_error_nm: float
    force_rank: int
    torque_rank: int
    force_condition_number: float
    torque_condition_number: float
    gravity_span_deg: float
    gravity_centered_singular_values: np.ndarray
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible scalar and list values."""

        return {
            "signed_mass_kg": float(self.signed_mass_kg),
            "mass_kg": float(self.mass_kg),
            "com_offset_m": np.asarray(self.com_offset_m, dtype=float).tolist(),
            "force_bias_n": np.asarray(self.force_bias_n, dtype=float).tolist(),
            "torque_bias_nm": np.asarray(self.torque_bias_nm, dtype=float).tolist(),
            "fit_error": {
                "force_rmse_n": float(self.force_rmse_n),
                "force_component_rmse_n": float(self.force_component_rmse_n),
                "force_max_error_n": float(self.force_max_error_n),
                "torque_rmse_nm": float(self.torque_rmse_nm),
                "torque_component_rmse_nm": float(self.torque_component_rmse_nm),
                "torque_max_error_nm": float(self.torque_max_error_nm),
            },
            "conditioning": {
                "force_rank": int(self.force_rank),
                "torque_rank": int(self.torque_rank),
                "force_condition_number": float(self.force_condition_number),
                "torque_condition_number": float(self.torque_condition_number),
                "gravity_span_deg": float(self.gravity_span_deg),
                "gravity_centered_singular_values": np.asarray(
                    self.gravity_centered_singular_values, dtype=float
                ).tolist(),
                "sample_count": int(self.sample_count),
            },
        }


def _condition_number(matrix: np.ndarray) -> float:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[-1] <= 1.0e-12:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def _gravity_span_deg(gravity_vectors: np.ndarray) -> float:
    unit = gravity_vectors / np.linalg.norm(gravity_vectors, axis=1, keepdims=True)
    cosine = np.clip(unit @ unit.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)).max())


def fit_tool_com(gravity_vectors: np.ndarray, wrenches: np.ndarray) -> ToolComFit:
    """Fit payload mass and center of mass from static pose averages.

    Args:
        gravity_vectors: ``(N, 3)`` gravity acceleration in the wrench frame,
            in m/s^2.  Each row is normally obtained from TF as
            ``R_base_sensor.T @ [0, 0, -g]``.
        wrenches: ``(N, 6)`` force/torque averages in N and Nm, in the same
            frame and at the same origin.

    The fitted equations are::

        f_i = b_f + alpha * g_i
        t_i = b_t - skew(g_i) * (alpha * r)

    where ``alpha`` can be negative for a reaction-wrench sensor convention,
    and ``r`` is the physical vector from the sensor origin to the payload
    center of mass.
    """

    gravity = np.asarray(gravity_vectors, dtype=np.float64)
    wrench = np.asarray(wrenches, dtype=np.float64)
    if gravity.ndim != 2 or gravity.shape[1] != 3:
        raise ValueError("gravity_vectors must have shape (N, 3).")
    if wrench.ndim != 2 or wrench.shape[1] != 6:
        raise ValueError("wrenches must have shape (N, 6).")
    if gravity.shape[0] != wrench.shape[0]:
        raise ValueError("gravity_vectors and wrenches must have the same number of rows.")
    if gravity.shape[0] < 4:
        raise ValueError("At least 4 static poses are required.")
    if not np.isfinite(gravity).all() or not np.isfinite(wrench).all():
        raise ValueError("Gravity vectors and wrenches must contain only finite values.")

    gravity_norm = np.linalg.norm(gravity, axis=1)
    if np.any(gravity_norm < 1.0e-6):
        raise ValueError("A gravity vector has near-zero magnitude.")
    expected_gravity = float(np.median(gravity_norm))
    if np.max(np.abs(gravity_norm - expected_gravity)) > max(0.05, 0.02 * expected_gravity):
        raise ValueError("Gravity vector magnitudes are inconsistent between poses.")

    sample_count = gravity.shape[0]
    force_design = np.zeros((3 * sample_count, 4), dtype=np.float64)
    torque_design = np.zeros((3 * sample_count, 6), dtype=np.float64)
    for index, gravity_i in enumerate(gravity):
        rows = slice(3 * index, 3 * index + 3)
        force_design[rows, 0] = gravity_i
        force_design[rows, 1:4] = np.eye(3)
        torque_design[rows, 0:3] = -skew(gravity_i)
        torque_design[rows, 3:6] = np.eye(3)

    force_rank = int(np.linalg.matrix_rank(force_design))
    torque_rank = int(np.linalg.matrix_rank(torque_design))
    if force_rank < 4:
        raise ValueError(
            "Gravity directions do not identify the mass. Tilt the tool to more distinct directions."
        )
    if torque_rank < 6:
        raise ValueError(
            "Gravity directions do not identify the 3-D center of mass. Use at least three non-coplanar tilts."
        )

    force_solution, _, _, _ = np.linalg.lstsq(
        force_design, wrench[:, :3].reshape(-1), rcond=None
    )
    torque_solution, _, _, _ = np.linalg.lstsq(
        torque_design, wrench[:, 3:].reshape(-1), rcond=None
    )

    signed_mass = float(force_solution[0])
    if abs(signed_mass) < 1.0e-6:
        raise ValueError("The fitted mass is nearly zero; check the wrench frame and raw sensor data.")
    q = torque_solution[:3]
    com_offset = q / signed_mass

    force_bias = force_solution[1:4]
    torque_bias = torque_solution[3:6]
    force_prediction = (force_design @ force_solution).reshape(sample_count, 3)
    torque_prediction = (torque_design @ torque_solution).reshape(sample_count, 3)
    force_error = force_prediction - wrench[:, :3]
    torque_error = torque_prediction - wrench[:, 3:]
    force_norm_error = np.linalg.norm(force_error, axis=1)
    torque_norm_error = np.linalg.norm(torque_error, axis=1)
    centered_gravity = gravity - gravity.mean(axis=0, keepdims=True)
    centered_singular_values = np.linalg.svd(centered_gravity, compute_uv=False)

    return ToolComFit(
        signed_mass_kg=signed_mass,
        mass_kg=abs(signed_mass),
        com_offset_m=com_offset,
        force_bias_n=force_bias,
        torque_bias_nm=torque_bias,
        force_rmse_n=float(np.sqrt(np.mean(force_norm_error**2))),
        force_component_rmse_n=float(np.sqrt(np.mean(force_error**2))),
        torque_rmse_nm=float(np.sqrt(np.mean(torque_norm_error**2))),
        torque_component_rmse_nm=float(np.sqrt(np.mean(torque_error**2))),
        force_max_error_n=float(force_norm_error.max()),
        torque_max_error_nm=float(torque_norm_error.max()),
        force_rank=force_rank,
        torque_rank=torque_rank,
        force_condition_number=_condition_number(force_design),
        torque_condition_number=_condition_number(torque_design),
        gravity_span_deg=_gravity_span_deg(gravity),
        gravity_centered_singular_values=centered_singular_values,
        sample_count=sample_count,
    )


def quality_warnings(
    fit: ToolComFit,
    force_rmse_limit_n: float = 0.20,
    torque_rmse_limit_nm: float = 0.03,
    condition_limit: float = 250.0,
) -> list[str]:
    """Return non-fatal warnings useful for deciding whether to repeat a fit."""

    warnings: list[str] = []
    if fit.force_rmse_n > force_rmse_limit_n:
        warnings.append(
            f"force residual {fit.force_rmse_n:.4f} N exceeds {force_rmse_limit_n:.4f} N"
        )
    if fit.torque_rmse_nm > torque_rmse_limit_nm:
        warnings.append(
            f"torque residual {fit.torque_rmse_nm:.4f} Nm exceeds {torque_rmse_limit_nm:.4f} Nm"
        )
    if max(fit.force_condition_number, fit.torque_condition_number) > condition_limit:
        warnings.append(
            "pose geometry is poorly conditioned; add larger, non-coplanar tilts"
        )
    if np.linalg.norm(fit.com_offset_m) > 1.0:
        warnings.append("the fitted center of mass is more than 1 m from the wrench origin")
    return warnings
