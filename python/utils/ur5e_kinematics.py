import math

import numpy as np
from scipy.spatial.transform import Rotation as R


class UR5eDLSIK:
    """Small damped-least-squares IK helper matching IsaacLab's relative pose action."""

    def __init__(self, damping: float = 0.05, finite_difference_eps: float = 1.0e-4) -> None:
        self.damping = damping
        self.eps = finite_difference_eps
        self.lower_limits = np.full(6, -2.0 * math.pi, dtype=np.float64)
        self.upper_limits = np.full(6, 2.0 * math.pi, dtype=np.float64)

        # UR5e modified DH dimensions in meters; used only to convert small TCP deltas into joint deltas.
        self.a = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0], dtype=np.float64)
        self.d = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996], dtype=np.float64)
        self.alpha = np.array([math.pi / 2.0, 0.0, 0.0, math.pi / 2.0, -math.pi / 2.0, 0.0], dtype=np.float64)

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, R]:
        transform = np.eye(4, dtype=np.float64)
        for theta, a, d, alpha in zip(q, self.a, self.d, self.alpha):
            transform = transform @ self._dh(theta, a, d, alpha)
        return transform[:3, 3].copy(), R.from_matrix(transform[:3, :3])

    def joint_target_from_delta(
        self,
        q: np.ndarray,
        delta_pose: np.ndarray,
        max_joint_step: float,
    ) -> np.ndarray:
        """Convert policy delta pose into a safe one-step joint target."""
        q = np.asarray(q, dtype=np.float64)
        delta_pose = np.asarray(delta_pose, dtype=np.float64)

        pos, rot = self.fk(q)
        # IsaacLab apply_delta_pose: position is world/base displacement, rotation is left-multiplied angle-axis.
        target_pos = pos + delta_pose[:3]
        target_rot = R.from_rotvec(delta_pose[3:6]) * rot

        pose_error = np.concatenate([target_pos - pos, (target_rot * rot.inv()).as_rotvec()])
        jacobian = self._numerical_jacobian(q, pos, rot)
        lambda_matrix = (self.damping ** 2) * np.eye(6, dtype=np.float64)
        dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + lambda_matrix, pose_error)
        # 与仿真一致: 超过单步关节上限时整组同比缩放，避免逐关节截断改变 IK 运动方向和末端姿态。
        max_abs_dq = float(np.max(np.abs(dq)))
        if max_joint_step > 0.0 and max_abs_dq > max_joint_step:
            dq *= max_joint_step / max_abs_dq
        return np.clip(q + dq, self.lower_limits, self.upper_limits)

    def _numerical_jacobian(self, q: np.ndarray, pos: np.ndarray, rot: R) -> np.ndarray:
        jacobian = np.zeros((6, 6), dtype=np.float64)
        for joint_id in range(6):
            q_perturbed = q.copy()
            q_perturbed[joint_id] += self.eps
            pos_p, rot_p = self.fk(q_perturbed)
            jacobian[:3, joint_id] = (pos_p - pos) / self.eps
            jacobian[3:, joint_id] = (rot_p * rot.inv()).as_rotvec() / self.eps
        return jacobian

    @staticmethod
    def _dh(theta: float, a: float, d: float, alpha: float) -> np.ndarray:
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        return np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def quat_wxyz_to_rotation(quat_wxyz: np.ndarray) -> R:
    return R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def quat_apply_wxyz(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return quat_wxyz_to_rotation(quat_wxyz).apply(vector)