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
    track_speed: float = 0.03
    feed_speed: float = 0.015
    angular_speed: float = 0.15
    command_interval_s: float = 0.10
    open_mm: float = 10.0
    close_mm: float = 8.0
    open_hold_s: float = 0.3
    timeout_s: float = 0.4
    fixed_tolerance: float = 0.015
    align_tolerance: float = 0.008
    feed_tolerance: float = 0.003
    max_pose_jump: float = 0.05
    max_axis_jump_rad: float = math.radians(30.0)
    max_travel: float = 0.25
    feed_motion_tolerance: float = 0.02
    feed_axis_tolerance_rad: float = math.radians(10.0)
    filter_alpha: float = 0.35

    @classmethod
    def from_args(cls, args):
        mapping = {
            "standoff": "follow_standoff", "insert_depth": "follow_insert_depth",
            "track_speed": "follow_speed", "feed_speed": "follow_feed_speed",
            "open_mm": "mouth_open_mm", "close_mm": "mouth_close_mm",
            "open_hold_s": "mouth_open_hold", "timeout_s": "follow_timeout",
            "fixed_tolerance": "fixed_reach_tolerance",
        }
        return cls(**{key: getattr(args, value) for key, value in mapping.items()})


class VisualMouthFollowMixin:
    visual_follow_enabled = False
    visual_follow_started = False

    def init_visual_follow(self, args):
        self.visual_follow_enabled = bool(args.visual_follow_after_fixed)
        self.visual_follow_config = VisualFollowConfig.from_args(args)
        self.visual_follow_started = False
        self.visual_mouth_pos = None
        self.visual_mouth_axis = None
        self.visual_raw_pos = None
        self.visual_raw_axis = None
        self.visual_pose_stamp_ns = None
        self.visual_pose_rejected = False
        self.visual_opening_stamp_ns = None
        self.visual_gap_mm = None
        self.visual_paired_stamp_ns = None
        self.visual_paired_gap_mm = None
        self.visual_open_since_ns = None
        self.visual_last_open_sample_ns = None
        self.visual_open_samples = 0
        self.visual_open = False
        self.visual_aligned_steps = 0
        self.visual_feed_stable_steps = 0
        self.visual_feed_start_ns = None
        self.visual_feed_mouth = None
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

    def visual_pose_callback(self, msg):
        stamp = self.visual_stamp_ns(msg)
        if not self.visual_stamp_fresh(stamp) or (self.visual_pose_stamp_ns is not None and stamp <= self.visual_pose_stamp_ns):
            return
        try:
            pose = msg if msg.header.frame_id == self.target_frame else self.tf_buffer.transform(msg, self.target_frame)
            pos = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
            quat = np.array([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w])
            if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)) or np.linalg.norm(quat) < 1e-6:
                self.visual_pose_rejected = True
                return
            axis = -R.from_quat(quat).apply([0., 0., 1.])
        except Exception as exc:
            self.visual_pose_rejected = True
            self.get_logger().warn(f"视觉跟随位姿转换失败: {exc}", throttle_duration_sec=1.0)
            return
        cfg = self.visual_follow_config
        if self.visual_raw_pos is not None:
            jump = float(np.linalg.norm(pos - self.visual_raw_pos))
            angle = math.acos(float(np.clip(np.dot(axis, self.visual_raw_axis), -1., 1.)))
            if jump > cfg.max_pose_jump or angle > cfg.max_axis_jump_rad:
                self.visual_pose_rejected = True
                self.get_logger().warn("视觉跟随目标跳变，本帧拒绝更新。", throttle_duration_sec=1.0)
                return
        self.visual_raw_pos, self.visual_raw_axis = pos.copy(), axis.copy()
        self.visual_pose_rejected = False
        if self.visual_mouth_pos is None:
            self.visual_mouth_pos, self.visual_mouth_axis = pos.copy(), axis.copy()
        else:
            self.visual_mouth_pos += cfg.filter_alpha * (pos - self.visual_mouth_pos)
            self.visual_mouth_axis = self.normalize_vector(
                (1. - cfg.filter_alpha) * self.visual_mouth_axis + cfg.filter_alpha * axis
            )
        self.visual_pose_stamp_ns = stamp
        self.pair_visual_sample()

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
        self.visual_aligned_steps = 0

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
        self.visual_anchor = np.asarray(spoon_center).copy()
        self.fsm_state = "VISUAL_TRACK"
        self.feed_stable_steps = 0
        self.last_fsm_metrics = {}
        self.previous_arm_action.fill(0.)
        self.previous_safety_action.fill(0.)
        self.hold_visual_position("已到达固定点，切换视觉跟随；等待新鲜嘴部目标。")

    def set_visual_safety_target(self, pos, axis):
        self.mouth_pos = np.asarray(pos, dtype=np.float32).copy()
        self.fixed_mouth_axis = np.asarray(axis, dtype=np.float32).copy()
        self.mouth_quat_wxyz = self.mouth_quat_from_policy_axis(axis)

    def abort_visual_feed(self, spoon_center, quat, reason):
        self.get_logger().warn(f"视觉喂入中止: {reason}，沿锁定方向撤回 10 cm 再归位。")
        self.set_visual_safety_target(self.visual_feed_mouth, self.visual_feed_axis)
        self.start_success_withdraw(spoon_center, quat, safety_triggered=True)
        if not self.episode_stopped:
            self.update_success_withdraw()

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
        if self.visual_pose_rejected or not self.visual_stamp_fresh(self.visual_pose_stamp_ns):
            if feeding:
                self.abort_visual_feed(spoon, quat, "视觉目标丢失、跳变或过期")
            else:
                self.hold_visual_position("等待新鲜视觉嘴部位姿，不使用融合节点保持的历史目标。")
            return
        if feeding:
            moved = np.linalg.norm(self.visual_raw_pos - self.visual_feed_mouth) > cfg.feed_motion_tolerance
            angle = math.acos(float(np.clip(np.dot(self.visual_raw_axis, self.visual_feed_axis), -1., 1.)))
            if (not self.visual_opening_ready() or self.visual_gap_mm <= cfg.close_mm
                    or moved or angle > cfg.feed_axis_tolerance_rad):
                self.abort_visual_feed(spoon, quat, "嘴闭合、开口数据失效或头部移动超限")
                return
            if (now - self.visual_feed_start_ns) * 1e-9 > 15.:
                self.abort_visual_feed(spoon, quat, "喂入超时")
                return
            raw_wrench = self.get_sim_aligned_safety_wrench_raw()
            axial = abs(float(np.dot(quat_apply_wxyz(quat, raw_wrench[:3]), self.visual_feed_axis)))
            if (np.linalg.norm(raw_wrench[:3]) > self.RULE_FORCE_THRESHOLD_N
                    or np.linalg.norm(raw_wrench[3:]) > self.RULE_TORQUE_THRESHOLD_NM
                    or axial > self.RULE_AXIAL_FORCE_THRESHOLD_N):
                self.abort_visual_feed(spoon, quat, "喂入期间力/力矩超限")
                return
            target = self.visual_feed_mouth + self.visual_feed_axis * cfg.insert_depth
            goal_rot = R.from_quat(self.visual_feed_quat[[1, 2, 3, 0]])
        else:
            if np.linalg.norm(self.visual_raw_pos - self.visual_anchor) > cfg.max_travel:
                self.hold_visual_position("视觉目标超出固定点附近 25 cm 的跟随范围。")
                return
            self.set_visual_safety_target(self.visual_mouth_pos, self.visual_mouth_axis)
            target = self.mouth_pos - self.compute_mouth_axis() * cfg.standoff
            current_rot = R.from_quat(quat[[1, 2, 3, 0]])
            current_axis = current_rot.apply(self.SPOON_FORWARD_AXIS_LOCAL)
            desired_axis = self.compute_mouth_axis()
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
            aligned = error <= cfg.align_tolerance and np.linalg.norm(rotation_error) <= math.radians(5.) and settled
            self.visual_aligned_steps = self.visual_aligned_steps + 1 if aligned else 0
            if stable_open and self.visual_aligned_steps >= 5:
                self.fsm_state = "VISUAL_FEED"
                self.visual_feed_mouth = self.visual_mouth_pos.copy()
                self.visual_feed_axis = self.visual_mouth_axis.copy()
                self.visual_feed_quat = quat.copy()
                self.visual_feed_start_ns = now
                self.visual_feed_stable_steps = 0
                self.set_visual_safety_target(self.visual_feed_mouth, self.visual_feed_axis)
                self.get_logger().warn(f"持续张嘴并已对准，锁定嘴轴，喂入 {cfg.insert_depth:.3f}m。")
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
            if error <= cfg.feed_tolerance and settled:
                self.visual_feed_stable_steps += 1
                if self.visual_feed_stable_steps >= 5:
                    self.start_success_hold()
                return
        safe, reason = self.delta_safety_status(ee_pos, quat, delta)
        if not safe:
            if feeding:
                self.abort_visual_feed(spoon, quat, reason)
            else:
                self.hold_visual_position(f"视觉动作被工作空间限制阻止: {reason}")
            return
        joint_target = self.ik.joint_target_from_delta(
            self.current_joint_pos, self.ros_delta_to_ik_delta(delta), self.args.max_joint_step
        )
        if not np.all(np.isfinite(joint_target)):
            self.stop_episode("FAILURE", "视觉跟随 IK 输出无效", spoon)
            return
        self.get_logger().info(
            f"视觉阶段={self.fsm_state}, target_error={error:.4f}m, "
            f"opening={self.visual_gap_mm}, dry_run={self.args.dry_run}", throttle_duration_sec=0.5,
        )
        self.record_wrench_sample(spoon, np.zeros(6), np.zeros(6), np.zeros(6), delta, state, delta)
        # Leave time for controller dispatch; replacing its trajectory on every
        # 30 Hz tick can continually restart execution on delayed controllers.
        if (not self.args.dry_run and (self.visual_last_command_ns is None
                or (now - self.visual_last_command_ns) * 1e-9 >= cfg.command_interval_s)):
            self.publish_joint_target(joint_target)
            self.visual_last_command_ns = now
