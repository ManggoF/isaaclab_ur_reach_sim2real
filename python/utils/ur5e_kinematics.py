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

    def joint_target_from_spoon_error(
        self, q: np.ndarray, spoon_error: np.ndarray, rotation_error: np.ndarray,
        spoon_offset_world: np.ndarray, max_joint_step: float,
        translation_step_limit: float, rotation_step_limit: float,
        position_deadband: float = 0.002, angle_deadband: float = math.radians(0.5),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prioritize spoon position, then align in its Jacobian null space.

        All vectors use the DH base frame. The offset is the measured wrist-to-
        spoon vector in that frame, so predictions are anchored to live TF.
        Return a bounded joint target and its predicted wrist pose delta.
        """
        q = np.asarray(q, dtype=np.float64)
        error = np.asarray(spoon_error, dtype=np.float64)
        rot_error = np.asarray(rotation_error, dtype=np.float64)
        offset = np.asarray(spoon_offset_world, dtype=np.float64)
        if (q.shape != (6,) or any(v.shape != (3,) for v in (error, rot_error, offset))
                or not all(np.all(np.isfinite(v)) for v in (q, error, rot_error, offset))):
            raise ValueError('位置优先 IK 输入无效')
        distance, angle = float(np.linalg.norm(error)), float(np.linalg.norm(rot_error))
        if distance <= position_deadband and angle <= angle_deadband:
            return q.copy(), np.zeros(6)
        dp = (error * min(1., translation_step_limit / max(distance, 1e-12))
              if distance > position_deadband else np.zeros(3))
        dr = (rot_error * min(1., rotation_step_limit / max(angle, 1e-12))
              if angle > angle_deadband else np.zeros(3))
        pos, rot = self.fk(q)
        jacobian = self._numerical_jacobian(q, pos, rot)
        x, y, z = offset
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        spoon_jacobian = jacobian[:3] - skew @ jacobian[3:]
        damping_sq = max(self.damping, 1e-5) ** 2
        dq_position = spoon_jacobian.T @ np.linalg.solve(
            spoon_jacobian @ spoon_jacobian.T + damping_sq * np.eye(3), dp,
        )
        # Use an actual null-space projector: a damped projector would let
        # orientation correction undo the spoon's translation near singularities.
        null = np.eye(6) - np.linalg.pinv(spoon_jacobian, rcond=1e-5) @ spoon_jacobian
        orient_jacobian = jacobian[3:] @ null
        dq_orientation = null @ orient_jacobian.T @ np.linalg.solve(
            orient_jacobian @ orient_jacobian.T + damping_sq * np.eye(3),
            dr - jacobian[3:] @ dq_position,
        )
        # Backtrack against nonlinear FK, not just the local Jacobian prediction.
        # Inside the position deadband, permit small position error while leveling.
        for orientation_scale in (1., 0.5, 0.):
            dq = dq_position + orientation_scale * dq_orientation
            largest = float(np.max(np.abs(dq)))
            scale = min(1., max_joint_step / max(largest, 1e-12))
            for i in range(6):
                if dq[i] > 1e-12:
                    scale = min(scale, max(0., (self.upper_limits[i] - q[i]) / dq[i]))
                elif dq[i] < -1e-12:
                    scale = min(scale, max(0., (self.lower_limits[i] - q[i]) / dq[i]))
            for _ in range(12):
                candidate = q + scale * dq
                next_pos, next_rot = self.fk(candidate)
                relative_rot = next_rot * rot.inv()
                wrist_delta = next_pos - pos
                spoon_delta = wrist_delta + relative_rot.apply(offset) - offset
                remaining = float(np.linalg.norm(error - spoon_delta))
                next_angle = float((R.from_rotvec(rot_error) * relative_rot.inv()).magnitude())
                position_ok = (remaining < distance - 1e-9 if distance > position_deadband
                               else remaining <= position_deadband + 1e-9 and next_angle < angle - 1e-9)
                if (position_ok and np.linalg.norm(spoon_delta) <= translation_step_limit + 1e-9
                        and relative_rot.magnitude() <= rotation_step_limit + 1e-9
                        and np.all(candidate >= self.lower_limits) and np.all(candidate <= self.upper_limits)):
                    return candidate, np.r_[wrist_delta, relative_rot.as_rotvec()]
                scale *= 0.5
        # No acceptable progress: hold instead of alternating backward/forward.
        return q.copy(), np.zeros(6)

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
