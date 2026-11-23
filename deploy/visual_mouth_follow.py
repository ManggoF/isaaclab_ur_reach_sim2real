"""Visual handoff for the deployment node, using its existing ROS controller."""

from dataclasses import dataclass
import math

import numpy as np
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from scipy.spatial.transform import Rotation as R

from python.utils.ur5e_kinematics import quat_apply_wxyz


@dataclass
class VisualFollowConfig:
    standoff: float = 0.06
    insert_depth: float = 0.015
    target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    horizontal: bool = True
    locked_feed: bool = True
    track_speed: float = 0.03
    feed_speed: float = 0.03
    angular_speed: float = 0.15
    command_interval_s: float = 0.10
    open_mm: float = 10.0
    close_mm: float = 8.0
    open_hold_s: float = 0.3
    timeout_s: float = 0.4
    fixed_tolerance: float = 0.015
    align_tolerance: float = 0.008
    height_tolerance: float = 0.003
    feed_tolerance: float = 0.001
    max_pose_jump: float = 0.05
    max_axis_jump_rad: float = math.radians(30.0)
    max_travel: float = 0.25
    feed_motion_tolerance: float = 0.02
    feed_axis_tolerance_rad: float = math.radians(10.0)
    filter_alpha: float = 0.35
    position_deadband: float = 0.002
    angle_deadband_deg: float = 0.5
    feed_axis_tolerance_deg: float = 1.0
    feed_lateral_tolerance: float = 0.002
    feed_align_hold_s: float = 0.3

    @classmethod
    def from_args(cls, args, settings):
        mapping = {
            "standoff": "follow_standoff", "horizontal": "follow_horizontal",
            "track_speed": "follow_speed", "feed_speed": "follow_feed_speed",
            "open_mm": "mouth_open_mm", "close_mm": "mouth_close_mm",
            "open_hold_s": "mouth_open_hold", "timeout_s": "follow_timeout",
            "fixed_tolerance": "fixed_reach_tolerance",
        }
        depth = settings.VISUAL_INSERT_DEPTH_M
        try:
            offset = np.asarray(settings.VISUAL_TARGET_OFFSET_M, dtype=np.float64)
            position_deadband = float(settings.VISUAL_ONLY_POSITION_DEADBAND_M)
            angle_deadband = float(settings.VISUAL_ONLY_ANGLE_DEADBAND_DEG)
            feed_axis_tolerance = float(settings.VISUAL_FEED_AXIS_TOL_DEG)
            feed_lateral_tolerance = float(settings.VISUAL_FEED_LATERAL_TOL_M)
            feed_align_hold = float(settings.VISUAL_FEED_ALIGN_HOLD_S)
            valid_depth = (math.isfinite(depth)
                           and settings.VISUAL_INSERT_DEPTH_MIN_M <= depth <= settings.VISUAL_INSERT_DEPTH_MAX_M)
        except (TypeError, ValueError):
            raise ValueError('run_task.py 视觉配置必须为数值。') from None
        if not valid_depth:
            raise ValueError(
                f'run_task.py: VISUAL_INSERT_DEPTH_M 必须在 {settings.VISUAL_INSERT_DEPTH_MIN_M:.3f}'
                f'～{settings.VISUAL_INSERT_DEPTH_MAX_M:.3f}m 内；'
                '正值在视觉嘴平面内，0 在平面上，负值在平面前；总前送约为等待距离加此值。'
            )
        if offset.shape != (3,) or not np.all(np.isfinite(offset)) or np.any(np.abs(offset) > 0.20):
            raise ValueError('run_task.py: VISUAL_TARGET_OFFSET_M 必须是三个 [-0.20, 0.20]m 内的有限数。')
        if not math.isfinite(position_deadband) or not 0. < position_deadband <= 0.003:
            raise ValueError('run_task.py: VISUAL_ONLY_POSITION_DEADBAND_M 必须在 (0, 0.003]m 内。')
        if not math.isfinite(angle_deadband) or not 0. < angle_deadband <= 1.:
            raise ValueError('run_task.py: VISUAL_ONLY_ANGLE_DEADBAND_DEG 必须在 (0, 1]° 内。')
        if not math.isfinite(feed_axis_tolerance) or not 0. < feed_axis_tolerance <= 5.:
            raise ValueError('run_task.py: VISUAL_FEED_AXIS_TOL_DEG 必须在 (0, 5]° 内。')
        if not math.isfinite(feed_lateral_tolerance) or not 0. < feed_lateral_tolerance <= cls.align_tolerance:
            raise ValueError('run_task.py: VISUAL_FEED_LATERAL_TOL_M 必须在 (0, 0.008]m 内。')
        if not math.isfinite(feed_align_hold) or not 0. < feed_align_hold <= 2.:
            raise ValueError('run_task.py: VISUAL_FEED_ALIGN_HOLD_S 必须在 (0, 2]s 内。')
        return cls(insert_depth=float(depth), target_offset=tuple(map(float, offset)),
                   locked_feed=bool(settings.VISUAL_FEED_LOCKED_MODE),
                   position_deadband=position_deadband, angle_deadband_deg=angle_deadband,
                   feed_axis_tolerance_deg=feed_axis_tolerance, feed_lateral_tolerance=feed_lateral_tolerance,
                   feed_align_hold_s=feed_align_hold,
                   **{key: getattr(args, value) for key, value in mapping.items()})


class VisualMouthFollowMixin:
    visual_only_mode = False
    visual_follow_enabled = False
    visual_follow_started = False

    def init_visual_follow(self, args):
        self.visual_follow_enabled = bool(self.visual_only_mode or args.visual_follow_after_fixed)
        self.visual_follow_config = VisualFollowConfig.from_args(args, self)
        self.visual_pose_topic = args.follow_pose_topic
        self.visual_follow_started = False
        self.visual_mouth_pos = None
        self.visual_mouth_axis = None
        self.visual_raw_pos = None
        self.visual_raw_axis = None
        self.visual_pose_stamp_ns = None
        self.visual_pose_rejected = False
        self.visual_pose_received_count = 0
        self.visual_pose_accepted_count = 0
        self.visual_last_input_stamp_ns = None
        self.visual_pose_reject_reason = ''
        self.visual_opening_stamp_ns = None
        self.visual_gap_mm = None
        self.visual_paired_stamp_ns = None
        self.visual_paired_gap_mm = None
        self.visual_open_since_ns = None
        self.visual_last_open_sample_ns = None
        self.visual_open_samples = 0
        self.visual_open = False
        self.reset_visual_alignment()
        self.visual_feed_stable_steps = 0
        self.visual_feed_start_ns = None
        self.visual_feed_trajectory_planned = False
        self.visual_feed_trajectory_sent = False
        self.visual_feed_duration_s = 0.0
        self.visual_feed_timeout_s = 15.0
        self.visual_feed_interrupted_reason = ''
        self.visual_feed_mouth = None
        self.visual_feed_locked = False
        self.visual_feed_insert_depth = None
        self.visual_feed_start_spoon = None
        self.visual_feed_observed_mouth = None
        self.visual_feed_observed_axis = None
        self.visual_feed_height = None
        self.visual_feed_axis = None
        self.visual_feed_quat = None
        self.visual_anchor = None
        self.visual_holding = False
        self.visual_last_command_ns = None
        if self.visual_follow_enabled:
            self.create_subscription(PoseStamped, args.follow_pose_topic, self.visual_pose_callback, 10)
            self.create_subscription(Vector3Stamped, args.follow_opening_topic, self.visual_opening_callback, 10)

    @staticmethod
    def visual_stamp_ns(msg):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def visual_stamp_fresh(self, stamp):
        if stamp is None or stamp <= 0:
            return False
        age = (self.get_clock().now().nanoseconds - stamp) * 1e-9
        return -0.05 <= age <= self.visual_follow_config.timeout_s

    def visual_control_axis(self, detected_axis):
        axis = np.asarray(detected_axis, dtype=np.float64).copy()
        if np.linalg.norm(axis[:2]) < 0.1:
            raise ValueError('嘴轴接近竖直，无法确定水平喂入方向/本人左右方向')
        if self.visual_follow_config.horizontal:
            axis[2] = 0.0
        return self.normalize_vector(axis)

    def level_spoon_rotation(self, axis):
        """Point the spoon horizontally, with its bowl normal facing up."""
        forward = self.visual_control_axis(axis).astype(np.float64)
        up = np.array([0., 0., 1.])
        world_basis = np.column_stack([np.cross(up, forward), up, forward])
        local_forward = self.SPOON_FORWARD_AXIS_LOCAL.astype(np.float64)
        local_up = self.SPOON_UP_AXIS_LOCAL.astype(np.float64)
        local_basis = np.column_stack([np.cross(local_up, local_forward), local_up, local_forward])
        return R.from_matrix(world_basis @ local_basis.T)

    def visual_target_offset_world(self):
        """Offset the spoon target in entry / person's left / world up axes."""
        forward = self.visual_mouth_axis.astype(np.float64)
        up = np.array([0., 0., 1.])
        person_left = np.cross(forward, up)
        if np.linalg.norm(person_left) < 0.1:
            raise ValueError('嘴轴接近竖直，无法确定本人左右补偿方向')
        person_left /= np.linalg.norm(person_left)
        return np.column_stack([forward, person_left, up]) @ self.visual_follow_config.target_offset

    def corrected_visual_mouth(self):
        # Compensation affects only this node's visual control, never /mouth_pose.
        return self.visual_mouth_pos + self.visual_target_offset_world()

    def visual_feed_target(self):
        depth = self.visual_feed_insert_depth
        if depth is None:
            depth = self.visual_follow_config.insert_depth
        target = self.visual_feed_mouth + self.visual_feed_axis * depth
        if self.visual_feed_start_spoon is not None:
            # Finish centering BEFORE feeding. Preserve the small accepted lateral
            # offset instead of drawing an oblique line back to the mouth center.
            start = self.visual_feed_start_spoon
            start_depth = float((start - self.visual_feed_mouth) @ self.visual_feed_axis)
            target = start + self.visual_feed_axis * (depth - start_depth)
        if self.visual_follow_config.horizontal:
            target[2] = self.visual_feed_height
        return target

    def visual_pose_callback(self, msg):
        stamp = self.visual_stamp_ns(msg)
        self.visual_pose_received_count += 1
        self.visual_last_input_stamp_ns = stamp
        if not self.visual_stamp_fresh(stamp):
            age = (self.get_clock().now().nanoseconds - stamp) * 1e-9
            self.visual_pose_reject_reason = f'消息时间戳无效或过期 age={age:.3f}s（上限 {self.visual_follow_config.timeout_s:.2f}s）'
            self.get_logger().warn('视觉跟随消息被拒绝: ' + self.visual_pose_reject_reason, throttle_duration_sec=1.0)
            return
        if self.visual_pose_stamp_ns is not None and stamp <= self.visual_pose_stamp_ns:
            return
        try:
            pose = msg if msg.header.frame_id == self.target_frame else self.tf_buffer.transform(msg, self.target_frame)
            pos = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
            quat = np.array([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w])
            if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)) or np.linalg.norm(quat) < 1e-6:
                self.visual_pose_rejected = True
                self.visual_pose_reject_reason = '位置或四元数无效'
                return
            axis = -R.from_quat(quat).apply([0., 0., 1.])
            control_axis = self.visual_control_axis(axis)
        except Exception as exc:
            self.visual_pose_rejected = True
            self.visual_pose_reject_reason = f'位姿转换失败: {exc}'
            self.get_logger().warn(f"视觉跟随位姿转换失败: {exc}", throttle_duration_sec=1.0)
            return
        cfg = self.visual_follow_config
        if self.visual_raw_pos is not None:
            jump = float(np.linalg.norm(pos - self.visual_raw_pos))
            angle = math.acos(float(np.clip(np.dot(axis, self.visual_raw_axis), -1., 1.)))
            if jump > cfg.max_pose_jump or angle > cfg.max_axis_jump_rad:
                self.visual_pose_rejected = True
                self.visual_pose_reject_reason = f'目标跳变: distance={jump:.3f}m, angle={math.degrees(angle):.1f}deg'
                self.get_logger().warn("视觉跟随目标跳变，本帧拒绝更新: " + self.visual_pose_reject_reason, throttle_duration_sec=1.0)
                return
        self.visual_raw_pos, self.visual_raw_axis = pos.copy(), axis.copy()
        self.visual_pose_rejected = False
        self.visual_pose_reject_reason = ''
        self.visual_pose_accepted_count += 1
        if self.visual_mouth_pos is None:
            self.visual_mouth_pos, self.visual_mouth_axis = pos.copy(), control_axis.copy()
        else:
            self.visual_mouth_pos += cfg.filter_alpha * (pos - self.visual_mouth_pos)
            self.visual_mouth_axis = self.normalize_vector(
                (1. - cfg.filter_alpha) * self.visual_mouth_axis + cfg.filter_alpha * control_axis
            )
        self.visual_pose_stamp_ns = stamp
        self.pair_visual_sample()
        self.get_logger().info(
            f'已接收融合嘴部目标 {self.visual_pose_topic}: age={(self.get_clock().now().nanoseconds - stamp) * 1e-9:.3f}s, '
            f'pos={np.round(pos, 4)}', throttle_duration_sec=1.0,
        )

    def visual_wait_status(self):
        if self.visual_pose_received_count == 0:
            return f'尚未收到任何 {self.visual_pose_topic} 消息；检查融合节点'
        if self.visual_pose_reject_reason:
            return self.visual_pose_reject_reason
        if self.visual_pose_stamp_ns is None:
            return f'已收到 {self.visual_pose_received_count} 条消息，但没有有效位姿'
        age = (self.get_clock().now().nanoseconds - self.visual_pose_stamp_ns) * 1e-9
        return f'最后有效位姿 age={age:.3f}s（上限 {self.visual_follow_config.timeout_s:.2f}s）'

    def visual_opening_callback(self, msg):
        stamp = self.visual_stamp_ns(msg)
        if not self.visual_stamp_fresh(stamp) or (self.visual_opening_stamp_ns is not None and stamp <= self.visual_opening_stamp_ns):
            return
        gap_mm = float(np.linalg.norm([msg.vector.x, msg.vector.y, msg.vector.z]) * 1000.)
        if not math.isfinite(gap_mm) or gap_mm > 80.:
            return
        self.visual_gap_mm = gap_mm
        self.visual_opening_stamp_ns = stamp
        self.pair_visual_sample()

    def pair_visual_sample(self):
        if self.visual_pose_stamp_ns is not None and self.visual_pose_stamp_ns == self.visual_opening_stamp_ns:
            self.visual_paired_stamp_ns = self.visual_pose_stamp_ns
            self.visual_paired_gap_mm = self.visual_gap_mm

    def reset_visual_open_debounce(self):
        self.visual_open = False
        self.visual_open_since_ns = None
        self.visual_last_open_sample_ns = None
        self.visual_open_samples = 0
        self.reset_visual_alignment()

    def reset_visual_alignment(self):
        self.visual_aligned_steps = 0
        self.visual_align_since_ns = None
        self.visual_align_first_stamp_ns = None
        self.visual_align_last_stamp_ns = None
        self.visual_align_samples = 0

    def update_visual_alignment(self, aligned):
        """Require sustained measured alignment, confirmed by distinct vision frames."""
        if not aligned:
            self.reset_visual_alignment()
            return False
        now = self.get_clock().now().nanoseconds
        stamp = self.visual_pose_stamp_ns
        if self.visual_align_since_ns is None:
            self.visual_align_since_ns = now
            self.visual_align_first_stamp_ns = stamp
        self.visual_aligned_steps += 1
        if stamp != self.visual_align_last_stamp_ns:
            self.visual_align_last_stamp_ns = stamp
            self.visual_align_samples += 1
        hold_ns = round(self.visual_follow_config.feed_align_hold_s * 1e9)
        return (self.visual_aligned_steps >= 5 and self.visual_align_samples >= 3
                and now - self.visual_align_since_ns >= hold_ns
                and stamp - self.visual_align_first_stamp_ns >= hold_ns)

    def visual_axis_alignment(self, spoon, rotation, mouth, axis):
        """Measure the actual spoon axis and centerline independently of IK goals."""
        spoon_axis = rotation.apply(self.SPOON_FORWARD_AXIS_LOCAL)
        angle = math.acos(float(np.clip(np.dot(spoon_axis, axis), -1., 1.)))
        offset = spoon - mouth
        lateral = offset - float(offset @ axis) * axis
        if self.visual_follow_config.horizontal:
            lateral[2] = 0.  # Height has its own gate and is locked during feeding.
        return angle, float(np.linalg.norm(lateral))

    def return_to_visual_alignment(self, reason):
        self.fsm_state = 'VISUAL_TRACK'
        self.visual_feed_locked = False
        self.visual_feed_trajectory_planned = False
        self.visual_last_command_ns = None
        self.hold_visual_position('喂入尚未下发，重新进行最终对齐: ' + reason)

    def visual_opening_ready(self):
        # Both outputs must originate from the SAME camera acquisition, not a held pose.
        return (
            self.visual_stamp_fresh(self.visual_pose_stamp_ns)
            and self.visual_stamp_fresh(self.visual_paired_stamp_ns)
        )

    def update_visual_open_debounce(self):
        if not self.visual_opening_ready():
            self.reset_visual_open_debounce()
            return False
        stamp = self.visual_paired_stamp_ns
        gap = self.visual_paired_gap_mm
        cfg = self.visual_follow_config
        if stamp != self.visual_last_open_sample_ns:
            if (self.visual_last_open_sample_ns is not None
                    and (stamp - self.visual_last_open_sample_ns) * 1e-9 > cfg.timeout_s):
                self.reset_visual_open_debounce()
            if gap <= cfg.close_mm:
                self.reset_visual_open_debounce()
            elif gap >= cfg.open_mm or self.visual_open:
                if self.visual_open_since_ns is None:
                    self.visual_open_since_ns = stamp
                self.visual_open = True
                self.visual_open_samples += 1
            else:
                self.reset_visual_open_debounce()
            self.visual_last_open_sample_ns = stamp
        return (
            self.visual_open and self.visual_open_samples >= 3
            and (stamp - self.visual_open_since_ns) * 1e-9 >= cfg.open_hold_s
        )

    def hold_visual_position(self, reason):
        self.get_logger().warn(reason, throttle_duration_sec=1.0)
        if self.fsm_state in {"VISUAL_FEED", "SUCCESS_HOLD"} and self.visual_feed_trajectory_sent:
            # A feedback outage cancels feeding. Never resume a trajectory
            # that was replaced by a hold when data becomes available again.
            self.visual_feed_interrupted_reason = reason
        self.reset_visual_open_debounce()
        self.visual_feed_stable_steps = 0
        fresh_joint = (
            self.last_joint_state_time_ns is not None
            and (self.get_clock().now().nanoseconds - self.last_joint_state_time_ns) * 1e-9 <= self.args.joint_state_timeout
        )
        if (not self.visual_holding and not self.args.dry_run and fresh_joint
                and self.current_joint_pos is not None and np.all(np.isfinite(self.current_joint_pos))):
            self.publish_joint_target(self.current_joint_pos.copy())
        self.visual_holding = True

    def start_visual_follow(self, spoon_center, ee_quat_wxyz):
        self.visual_follow_started = True
        self.target_received = True
        self.visual_anchor = None if self.visual_only_mode else np.asarray(spoon_center).copy()
        self.fsm_state = "VISUAL_TRACK"
        self.feed_stable_steps = 0
        self.last_fsm_metrics = {}
        self.previous_arm_action.fill(0.)
        self.previous_safety_action.fill(0.)
        if self.visual_position_priority_enabled():
            wrist_index = self.JOINT_NAME_TO_IDX['wrist_2_joint']
            self.ik.lower_limits[wrist_index] = -2. * math.pi
            self.ik.upper_limits[wrist_index] = 2. * math.pi
            self.get_logger().info(
                '纯视觉位置优先：第5关节不锁定0°或限制±15°，沿用±2π关节边界；'
                '先靠近嘴前等待位置，再逐步对准水平姿态，预测退远的动作不下发。'
            )
        self.hold_visual_position(
            "直接进入纯视觉跟随；等待新鲜嘴部目标。" if self.visual_only_mode
            else "已到达固定点，切换视觉跟随；等待新鲜嘴部目标。"
        )

    def visual_position_priority_enabled(self):
        return bool(self.visual_only_mode and not self.args.visual_follow_after_fixed
                    and self.VISUAL_ONLY_POSITION_PRIORITY)

    def set_visual_safety_target(self, pos, axis):
        self.mouth_pos = np.asarray(pos, dtype=np.float32).copy()
        self.fixed_mouth_axis = np.asarray(axis, dtype=np.float32).copy()
        self.mouth_quat_wxyz = self.mouth_quat_from_policy_axis(axis)

    def abort_visual_feed(self, spoon_center, quat, reason):
        advance = ''
        if self.visual_feed_start_spoon is not None:
            actual = float((spoon_center - self.visual_feed_start_spoon) @ self.visual_feed_axis)
            planned = float((self.visual_feed_target() - self.visual_feed_start_spoon) @ self.visual_feed_axis)
            advance = f'；实际前送={actual * 1000.:.1f}/{planned * 1000.:.1f}mm'
        self.get_logger().warn(f"视觉喂入中止: {reason}{advance}，沿锁定方向撤回 10 cm 再归位。")
        self.set_visual_safety_target(self.visual_feed_mouth, self.visual_feed_axis)
        self.start_success_withdraw(spoon_center, quat, safety_triggered=True)
        if not self.episode_stopped:
            self.update_success_withdraw()

    def visual_feed_speed_scale(self, spoon):
        """Schedule near-lip slowdown before dispatch, without short commands.

        Use the learned filter's maximum possible slow gain at each path point;
        its live gains cannot request a slower speed than this envelope. The
        geometric path already converges to the mouth center. Live force,
        retreat and deviation checks still override the entire trajectory.
        """
        offset = spoon - self.visual_feed_mouth
        depth = float(offset @ self.visual_feed_axis)
        radial = float(np.linalg.norm(offset - depth * self.visual_feed_axis))
        if self.learned_safety_enabled:
            lip_risk = (
                float(np.clip(1.0 - abs(radial - self.SAFETY_POLICY_MOUTH_HOLE_RADIUS) / self.RULE_LIP_MARGIN, 0., 1.))
                * float(np.clip(1.0 - abs(depth) / self.SAFETY_POLICY_MOUTH_CHANNEL_FRONT, 0., 1.))
            )
            gain = lip_risk if lip_risk > self.RULE_TRIGGER_SCORE else 0.0
        elif self.rule_safety_enabled:
            near = self.inverse_ramp(float(np.linalg.norm(offset)), self.RULE_SLOW_START_DIST, self.RULE_SLOW_FULL_DIST)
            lip = (
                self.inverse_ramp(abs(radial - self.RULE_MOUTH_CHANNEL_RADIUS), self.RULE_LIP_MARGIN, 0.)
                * self.inverse_ramp(abs(depth), self.RULE_MOUTH_CHANNEL_FRONT, 0.)
            )
            gain = max(near, lip)
        else:
            gain = 0.0
        return 1.0 - float(np.clip(gain, 0., 0.95))

    def update_visual_feed_trajectory(self, ee_pos, quat, spoon, target):
        """Dispatch one complete feed and monitor actual arrival, never resend."""
        cfg = self.visual_follow_config
        if not self.visual_feed_trajectory_planned:
            actual_rot = R.from_quat(quat[[1, 2, 3, 0]])
            axis_error, lateral = self.visual_axis_alignment(
                spoon, actual_rot, self.visual_feed_mouth, self.visual_feed_axis,
            )
            pose_error = (R.from_quat(self.visual_feed_quat[[1, 2, 3, 0]]) * actual_rot.inv()).magnitude()
            if (axis_error > math.radians(cfg.feed_axis_tolerance_deg)
                    or lateral > cfg.feed_lateral_tolerance
                    or pose_error > math.radians(1. if cfg.horizontal else cfg.feed_axis_tolerance_deg)
                    or np.linalg.norm(spoon - self.visual_feed_start_spoon) > cfg.feed_tolerance
                    or np.max(np.abs(self.current_joint_vel)) >= 0.03):
                self.return_to_visual_alignment(
                    f'轴向误差={math.degrees(axis_error):.2f}deg，横向偏差={lateral * 1000.:.1f}mm，检查实际位姿/稳定性'
                )
                return
            # Anchor the axial line to the confirmed pose at dispatch, including
            # sub-millimeter settling since the preceding alignment callback.
            self.visual_feed_start_spoon = spoon.copy()
            self.visual_feed_height = float(spoon[2])
            self.visual_feed_quat = quat.copy()
            target = self.visual_feed_target()
        offset = spoon - self.visual_feed_mouth
        depth = float(offset @ self.visual_feed_axis)
        radial = float(np.linalg.norm(offset - depth * self.visual_feed_axis))
        rotation_error = (R.from_quat(self.visual_feed_quat[[1, 2, 3, 0]])
                          * R.from_quat(quat[[1, 2, 3, 0]]).inv()).magnitude()
        if cfg.horizontal and abs(spoon[2] - self.visual_feed_height) > cfg.height_tolerance:
            self.abort_visual_feed(spoon, quat, '实际喂入高度偏离锁定高度超过 3mm')
            return
        if (radial > cfg.align_tolerance + cfg.feed_tolerance
                or rotation_error > math.radians(5.) or depth > self.visual_feed_insert_depth + cfg.feed_tolerance):
            self.abort_visual_feed(spoon, quat, "实际喂入偏离锁定通道/姿态或深度超限")
            return
        error = float(np.linalg.norm(target - spoon))
        settled = np.max(np.abs(self.current_joint_vel)) < 0.03
        advanced = float((spoon - self.visual_feed_start_spoon) @ self.visual_feed_axis)
        planned = float((target - self.visual_feed_start_spoon) @ self.visual_feed_axis)
        self.get_logger().info(
            f'连续喂入: depth={depth * 1000.:.1f}/{self.visual_feed_insert_depth * 1000.:.1f}mm '
            f'（相对嘴平面）, advance={advanced * 1000.:.1f}/{planned * 1000.:.1f}mm, '
            f'target_offset={np.round(np.asarray(cfg.target_offset) * 1000., 1)}mm, '
            f'vision_depth={float((spoon - self.visual_feed_observed_mouth) @ self.visual_feed_axis) * 1000.:.1f}mm, '
            f'remaining={error * 1000.:.1f}mm, radial={radial * 1000.:.1f}mm, '
            f'height={spoon[2]:.4f}m, '
            f'trajectory_sent={self.visual_feed_trajectory_sent}, speed_scaling_raw={self.speed_scaling}',
            throttle_duration_sec=0.5,
        )
        if self.visual_feed_trajectory_sent and error <= cfg.feed_tolerance and settled:
            self.visual_feed_stable_steps += 1
            if self.visual_feed_stable_steps >= 5:
                self.get_logger().warn(f'视觉喂入实际到位: depth={depth * 1000.:.1f}mm，开始保持。')
                self.start_success_hold()
            return
        self.visual_feed_stable_steps = 0
        if self.visual_feed_trajectory_planned:
            return
        try:
            trajectory = self.plan_spoon_linear_trajectory(
                ee_pos, quat, target, self.visual_feed_quat, cfg.feed_speed,
                max_lateral_error=cfg.feed_tolerance,
                max_orientation_error=math.radians(2.),
                speed_scale_at_spoon=self.visual_feed_speed_scale,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.abort_visual_feed(spoon, quat, f'连续喂入轨迹规划失败: {exc}')
            return
        self.visual_feed_trajectory_planned = True
        end_time = trajectory.points[-1].time_from_start
        self.visual_feed_duration_s = end_time.sec + end_time.nanosec * 1e-9
        scaling = float(self.speed_scaling) if self.speed_scaling is not None else 1.0
        if scaling > 1.0:
            scaling /= 100.0
        expected_s = self.visual_feed_duration_s / max(scaling, 0.01)
        hold_s = self.SUCCESS_HOLD_STEPS_AFTER_REACHED * self.control_period
        self.visual_feed_timeout_s = max(15.0, expected_s * 2.0 + hold_s + 3.0)
        self.visual_feed_start_ns = self.get_clock().now().nanoseconds
        # Robot feedback is always required; locked feeding already captured
        # valid vision at alignment and does not depend on camera visibility.
        if self.data_is_stale() or (not self.args.dry_run and not self.ros_control_is_ready()):
            self.abort_visual_feed(spoon, quat, '规划后机器人反馈/控制器失效')
            return
        if not self.visual_feed_locked and (
                self.visual_pose_rejected or not self.visual_stamp_fresh(self.visual_pose_stamp_ns)
                or not self.visual_opening_ready()):
            self.abort_visual_feed(spoon, quat, '规划后视觉数据失效')
            return
        if not self.args.dry_run:
            self.pub.publish(trajectory)
            self.visual_feed_trajectory_sent = True
        self.get_logger().warn(
            f'{"预览" if self.args.dry_run else "已一次下发"}完整连续喂入轨迹: '
            f'travel={np.linalg.norm(target - spoon) * 1000.:.1f}mm, depth={self.visual_feed_insert_depth * 1000.:.1f}mm, '
            f'locked_feed={self.visual_feed_locked}, '
            f'duration={self.visual_feed_duration_s:.2f}s（UR速度缩放前，含近嘴减速）, '
            f'expected={expected_s:.2f}s, timeout={self.visual_feed_timeout_s:.2f}s；途中不重发。'
            f' axis_base={np.round(self.visual_feed_axis, 4)}, '
            f'start_spoon_base={np.round(self.visual_feed_start_spoon, 4)}, target_spoon_base={np.round(target, 4)}'
        )

    def step_visual_follow(self):
        if self.data_is_stale() or (not self.args.dry_run and not self.ros_control_is_ready()):
            self.hold_visual_position("视觉阶段等待机器人反馈/控制器恢复。")
            return
        pose = self.lookup_ee_pose(require_fresh=True)
        if pose is None:
            self.hold_visual_position("视觉阶段等待有效末端 TF。")
            return
        ee_pos, quat = pose
        spoon = ee_pos + quat_apply_wxyz(quat, self.SPOON_OFFSET)
        cfg = self.visual_follow_config
        now = self.get_clock().now().nanoseconds
        feeding = self.fsm_state in {"VISUAL_FEED", "SUCCESS_HOLD"}
        self.policy_step += 1
        timed_out = (
            self.episode_start_time_ns is not None
            and (now - self.episode_start_time_ns) * 1e-9 >= self.args.episode_length_s
        ) or self.policy_step >= self.max_episode_steps
        if timed_out:
            if feeding:
                self.abort_visual_feed(spoon, quat, "episode 超时")
            else:
                self.stop_episode("TIMEOUT", "视觉跟随等待超时", spoon)
            return
        if not (feeding and self.visual_feed_locked) and (
                self.visual_pose_rejected or not self.visual_stamp_fresh(self.visual_pose_stamp_ns)):
            if feeding:
                self.abort_visual_feed(spoon, quat, "视觉目标丢失、跳变或过期")
            else:
                self.hold_visual_position("等待新鲜视觉嘴部位姿: " + self.visual_wait_status())
            return
        if feeding:
            if self.visual_feed_interrupted_reason:
                self.abort_visual_feed(spoon, quat, "机器人反馈曾中断: " + self.visual_feed_interrupted_reason)
                return
            if not self.visual_feed_locked:
                movement = float(np.linalg.norm(self.visual_raw_pos - self.visual_feed_observed_mouth))
                angle = math.acos(float(np.clip(np.dot(self.visual_raw_axis, self.visual_feed_observed_axis), -1., 1.)))
                if not self.visual_opening_ready():
                    self.abort_visual_feed(spoon, quat, '同帧开口数据失效/过期')
                    return
                if self.visual_gap_mm <= cfg.close_mm:
                    self.abort_visual_feed(spoon, quat, f'嘴闭合: opening={self.visual_gap_mm:.1f}mm <= {cfg.close_mm:.1f}mm')
                    return
                if movement > cfg.feed_motion_tolerance or angle > cfg.feed_axis_tolerance_rad:
                    delta = self.visual_raw_pos - self.visual_feed_observed_mouth
                    self.abort_visual_feed(spoon, quat,
                        f'头部移动超限: shift={movement * 1000.:.1f}mm, angle={math.degrees(angle):.1f}deg, '
                        f'delta_base={np.round(delta * 1000., 1)}mm；喂入轨迹尚未完成不能视为深度不足')
                    return
            if (now - self.visual_feed_start_ns) * 1e-9 > self.visual_feed_timeout_s:
                self.abort_visual_feed(spoon, quat, "喂入超时")
                return
            # Wrench remains part of diagnostics and recording. During visual
            # feeding/holding, magnitude thresholds do not request withdrawal.
            target = self.visual_feed_target()
            goal_rot = R.from_quat(self.visual_feed_quat[[1, 2, 3, 0]])
        else:
            if self.visual_anchor is None:
                mouth = self.corrected_visual_mouth()
                if (not np.all(np.isfinite(mouth)) or np.any(mouth < self.WORKSPACE_MIN)
                        or np.any(mouth > self.WORKSPACE_MAX)):
                    self.hold_visual_position("首次视觉嘴部目标超出工作空间，等待有效目标。")
                    return
                self.visual_anchor = mouth.copy()
                self.get_logger().info(f"纯视觉跟随中心已锁定: mouth_base={np.round(self.visual_anchor, 4)}，跟随范围半径25 cm。")
            if max(np.linalg.norm(self.visual_raw_pos - self.visual_anchor),
                   np.linalg.norm(self.corrected_visual_mouth() - self.visual_anchor)) > cfg.max_travel:
                self.hold_visual_position("视觉目标超出跟随中心附近 25 cm 的范围。")
                return
            self.set_visual_safety_target(self.corrected_visual_mouth(), self.visual_mouth_axis)
            target = self.mouth_pos - self.compute_mouth_axis() * cfg.standoff
            current_rot = R.from_quat(quat[[1, 2, 3, 0]])
            desired_axis = self.compute_mouth_axis()
            if cfg.horizontal:
                goal_rot = self.level_spoon_rotation(desired_axis)
            else:
                current_axis = current_rot.apply(self.SPOON_FORWARD_AXIS_LOCAL)
                cross = np.cross(current_axis, desired_axis)
                dot = float(np.clip(np.dot(current_axis, desired_axis), -1., 1.))
                if np.linalg.norm(cross) < 1e-8 and dot < 0.:
                    self.hold_visual_position("勺子与嘴轴反向，暂停跟随并检查姿态。")
                    return
                correction = cross / max(float(np.linalg.norm(cross)), 1e-8) * math.acos(dot)
                goal_rot = R.from_rotvec(correction) * current_rot
        if not np.all(np.isfinite(target)) or np.any(target < self.WORKSPACE_MIN) or np.any(target > self.WORKSPACE_MAX):
            if feeding:
                self.abort_visual_feed(spoon, quat, "喂入目标越界")
            else:
                self.hold_visual_position("视觉跟随目标超出工作空间。")
            return
        self.visual_holding = False
        self.last_fsm_metrics = self.compute_fsm_metrics(spoon, quat)
        self.compute_wrist_wrench_observation()
        error = float(np.linalg.norm(target - spoon))
        current_rot = R.from_quat(quat[[1, 2, 3, 0]])
        rotation_error = (goal_rot * current_rot.inv()).as_rotvec()
        settled = np.max(np.abs(self.current_joint_vel)) < 0.03
        if not feeding:
            stable_open = self.update_visual_open_debounce()
            angle_tolerance = math.radians(1. if cfg.horizontal else 5.)
            height_aligned = not cfg.horizontal or abs(target[2] - spoon[2]) <= cfg.height_tolerance
            axis_error, lateral_error = self.visual_axis_alignment(
                spoon, current_rot, self.mouth_pos, self.visual_mouth_axis,
            )
            # The smoothed tracking axis can lag a turn of the head. A recent
            # accepted /mouth_pose must independently agree with the actual spoon.
            latest_axis = self.visual_control_axis(self.visual_raw_axis)
            latest_axis_error, _ = self.visual_axis_alignment(spoon, current_rot, self.mouth_pos, latest_axis)
            axis_tolerance = math.radians(cfg.feed_axis_tolerance_deg)
            aligned = (error <= cfg.align_tolerance and height_aligned and settled
                       and np.linalg.norm(rotation_error) <= angle_tolerance
                       and max(axis_error, latest_axis_error) <= axis_tolerance
                       and lateral_error <= cfg.feed_lateral_tolerance)
            alignment_confirmed = self.update_visual_alignment(aligned and stable_open)
            final_alignment = stable_open and error <= cfg.align_tolerance
            if final_alignment:
                self.get_logger().info(
                    f'喂入前最终对齐: axis_error={math.degrees(axis_error):.2f}deg, '
                    f'latest_axis_error={math.degrees(latest_axis_error):.2f}deg '
                    f'(limit={cfg.feed_axis_tolerance_deg:.2f}deg), '
                    f'lateral={lateral_error * 1000.:.1f}/{cfg.feed_lateral_tolerance * 1000.:.1f}mm, '
                    f'settled={settled}, aligned={aligned}, frames={self.visual_align_samples}, '
                    f'spoon_axis={np.round(current_rot.apply(self.SPOON_FORWARD_AXIS_LOCAL), 4)}, '
                    f'mouth_axis={np.round(self.visual_mouth_axis, 4)}', throttle_duration_sec=0.5,
                )
            if alignment_confirmed:
                self.fsm_state = "VISUAL_FEED"
                self.visual_feed_mouth = self.corrected_visual_mouth().copy()
                self.visual_feed_start_spoon = spoon.copy()
                self.visual_feed_observed_mouth = self.visual_raw_pos.copy()
                self.visual_feed_observed_axis = self.visual_raw_axis.copy()
                self.visual_feed_axis = self.visual_mouth_axis.copy()
                self.visual_feed_locked = bool(cfg.locked_feed)
                self.visual_feed_insert_depth = float(cfg.insert_depth)
                self.visual_feed_height = float(spoon[2])
                # Alignment is already complete; preserve the measured pose.
                # Do not combine the last wrist correction with insertion.
                self.visual_feed_quat = quat.copy()
                self.visual_feed_start_ns = now
                self.visual_feed_stable_steps = 0
                self.visual_feed_trajectory_planned = False
                self.visual_feed_trajectory_sent = False
                self.visual_feed_interrupted_reason = ''
                self.set_visual_safety_target(self.visual_feed_mouth, self.visual_feed_axis)
                self.get_logger().warn(
                    f"持续张嘴并已完成最终对齐，axis_error={math.degrees(axis_error):.2f}deg, "
                    f"lateral={lateral_error * 1000.:.1f}mm, hold={cfg.feed_align_hold_s:.2f}s，horizontal={cfg.horizontal}, "
                    f"喂入锁定高度={self.visual_feed_height:.4f}m，"
                    f"末端目标补偿[前,本人左,上]={np.round(np.asarray(cfg.target_offset) * 1000., 1)}mm，"
                    f"勺心目标相对补偿后嘴平面={cfg.insert_depth * 1000.:+.1f}mm（正内/负外）；"
                    f"locked_feed={self.visual_feed_locked}；"
                    "将保持当前姿态，沿锁定嘴轴一次下发连续喂入轨迹。"
                )
                if self.visual_feed_locked:
                    self.get_logger().info(
                        "本次喂入目标已锁定；后续视觉开口/位姿仅接收，不改变目标或触发中止，"
                        "到位并保持后沿锁定方向撤回归位；力觉继续记录，力/力矩超限不触发撤回，"
                        "机器人反馈和轨迹保护继续生效。"
                    )
                return
        elif not (error <= cfg.feed_tolerance and settled):
            self.visual_feed_stable_steps = 0

        # Use the controller horizon, rather than shrinking each new command to one timer tick.
        horizon = max(self.control_period, self.args.min_trajectory_duration)
        rotation_step = self.clip_vector_norm(rotation_error, cfg.angular_speed * horizon)
        next_rot = R.from_rotvec(rotation_step) * current_rot
        spoon_step = self.clip_vector_norm(target - spoon, (cfg.feed_speed if feeding else cfg.track_speed) * horizon)
        next_ee = spoon + spoon_step - next_rot.apply(self.SPOON_OFFSET)
        delta = np.concatenate([next_ee - ee_pos, rotation_step]).astype(np.float32)
        if self.learned_safety_enabled:
            delta, state = self.apply_learned_safety_filter(ee_pos, quat, spoon, delta)
        else:
            delta, state = self.apply_rule_based_safety_filter(ee_pos, quat, spoon, delta)
        if feeding and state["retreat_gain"] > 0.:
            self.abort_visual_feed(spoon, quat, "原安全层触发撤回")
            return
        if self.maybe_start_safety_strengthen(spoon, quat, state):
            return
        if feeding:
            if self.fsm_state == "SUCCESS_HOLD":
                self.update_success_hold(spoon, quat)
                return
            self.record_wrench_sample(spoon, np.zeros(6), np.zeros(6), np.zeros(6), delta, state, delta)
            self.update_visual_feed_trajectory(ee_pos, quat, spoon, target)
            return
        if self.visual_position_priority_enabled():
            error_ik = self.ros_delta_to_ik_delta(np.r_[target - spoon, rotation_error])
            offset_ik = self.ros_delta_to_ik_delta(np.r_[spoon - ee_pos, np.zeros(3)])[:3]
            position_deadband = cfg.position_deadband
            angle_deadband = math.radians(cfg.angle_deadband_deg)
            if final_alignment:
                position_deadband = min(position_deadband, cfg.feed_lateral_tolerance * 0.5)
                angle_deadband = min(angle_deadband, math.radians(cfg.feed_axis_tolerance_deg) * 0.5)
            try:
                joint_target, predicted_delta = self.ik.joint_target_from_spoon_error(
                    self.current_joint_pos, error_ik[:3], error_ik[3:], offset_ik,
                    self.args.max_joint_step, cfg.track_speed * horizon, cfg.angular_speed * horizon,
                    position_deadband, angle_deadband,
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                self.hold_visual_position(f'纯视觉位置优先 IK 无有效解: {exc}')
                return
            delta = self.ros_delta_to_ik_delta(predicted_delta)
            if (not np.any(predicted_delta) and (error > position_deadband
                    or np.linalg.norm(rotation_error) > angle_deadband)):
                self.get_logger().warn('纯视觉 IK 暂无满足前进和限位条件的解，保持当前位置。', throttle_duration_sec=1.0)
        else:
            joint_target = None
        safe, reason = self.delta_safety_status(ee_pos, quat, delta)
        if not safe:
            self.hold_visual_position(f"视觉动作被工作空间限制阻止: {reason}")
            return
        if joint_target is None:
            joint_target = self.ik.joint_target_from_delta(
                self.current_joint_pos, self.ros_delta_to_ik_delta(delta), self.args.max_joint_step
            )
        if not np.all(np.isfinite(joint_target)):
            self.stop_episode("FAILURE", "视觉跟随 IK 输出无效", spoon)
            return
        self.get_logger().info(
            f"视觉阶段={self.fsm_state}, target_error={error:.4f}m, "
            f"angle_error={math.degrees(np.linalg.norm(rotation_error)):.1f}deg, "
            f"pitch={self.last_fsm_metrics['handle_tilt_deg']:.1f}deg, "
            f"bowl_tilt={self.last_fsm_metrics['bowl_tilt_deg']:.1f}deg, "
            f"opening={self.visual_gap_mm}, dry_run={self.args.dry_run}", throttle_duration_sec=0.5,
        )
        self.record_wrench_sample(spoon, np.zeros(6), np.zeros(6), np.zeros(6), delta, state, delta)
        # Leave time for controller dispatch; replacing its trajectory on every
        # 30 Hz tick can continually restart execution on delayed controllers.
        if (not self.args.dry_run and (self.visual_last_command_ns is None
                or (now - self.visual_last_command_ns) * 1e-9 >= cfg.command_interval_s)):
            self.publish_joint_target(joint_target)
            self.visual_last_command_ns = now
