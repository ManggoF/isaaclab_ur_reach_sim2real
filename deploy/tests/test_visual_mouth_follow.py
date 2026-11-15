"""Exercise visual handoff and feeding with FK feedback; no robot connection."""

from copy import deepcopy
from collections import OrderedDict
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped, Vector3Stamped
from rclpy.duration import Duration
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer

from deploy.face_pose_fusion import PoseFusionNode
from deploy.run_task import ArmPolicyDeployNode, parse_args
from deploy.visual_mouth_follow import VisualFollowConfig
from deploy.tests.test_success_withdraw import OfflineDeployNode, TimedTrajectoryDeployNode


class HybridNode(OfflineDeployNode):
    # Nominal geometry in most tests; production settings have their own test.
    VISUAL_INSERT_DEPTH_M = 0.015
    VISUAL_TARGET_OFFSET_M = (0., 0., 0.)

    def __init__(self, dry_run=False, locked_feed=None):
        OfflineDeployNode.__init__(self, dry_run=dry_run)
        self.target_frame = 'base_link'
        self.args.episode_length_s = 240.
        self.max_episode_steps = 7200
        self.wrench_sign = -1.
        self.args.min_spoon_z = 0.06
        self.args.max_mouth_spoon_distance = 3.
        self.args.visual_follow_after_fixed = False  # No ROS subscriptions in this fake.
        self.args.follow_pose_topic = '/mouth_pose'
        self.args.follow_opening_topic = '/mouth_opening'
        self.args.follow_standoff = 0.06
        self.args.follow_horizontal = True
        self.args.follow_speed = 0.03
        self.args.follow_feed_speed = 0.03
        self.args.mouth_open_mm = 10.
        self.args.mouth_close_mm = 8.
        self.args.mouth_open_hold = 0.3
        self.args.follow_timeout = 0.4
        self.args.fixed_reach_tolerance = 0.015
        self.init_visual_follow(self.args)
        if locked_feed is not None:
            self.visual_follow_config.locked_feed = locked_feed
        self.visual_follow_enabled = True
        self.fixed_mouth_axis = self.mouth_axis.copy()
        self.mouth_quat_wxyz = self.mouth_quat_from_policy_axis(self.mouth_axis)
        self.get_sim_aligned_safety_wrench_raw = Mock(return_value=np.zeros(6))
        self.compute_wrist_wrench_observation = Mock(return_value=np.zeros(6))
        self.record_wrench_sample = Mock()
        self.last_spoon, _ = self.spoon_pose()
        self.live_mouth = self.last_spoon + self.mouth_axis * 0.06

    def compute_mouth_axis(self, invert=True):
        return ArmPolicyDeployNode.compute_mouth_axis(self, invert)

    def send_sample(self, gap_mm=15., position=None, axis=None, opening=True, stamp=None):
        stamp = self.now_ns if stamp is None else stamp
        msg = PoseStamped()
        msg.header.frame_id = 'base_link'
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
        p = self.live_mouth if position is None else np.asarray(position)
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, p)
        q = self.mouth_quat_from_policy_axis(self.mouth_axis if axis is None else axis)
        msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z = map(float, q)
        lip = Vector3Stamped()
        lip.header.stamp = msg.header.stamp
        lip.header.frame_id = 'wrist_cam_optical_frame'
        lip.vector.y = gap_mm / 1000.
        self.visual_pose_callback(msg)
        if opening:
            self.visual_opening_callback(lip)
        return msg, lip

    def begin_tracking(self):
        spoon, quat = self.spoon_pose()
        self.start_visual_follow(spoon, quat)

    def reach_feeding(self):
        self.begin_tracking()
        for _ in range(30):
            self.send_sample()
            self.tick()
            if self.fsm_state == 'VISUAL_FEED':
                return
        raise AssertionError(f'Never entered feed: {self.fsm_state}, {self.logger.mock_calls[-3:]}')


class VisualMouthFollowTest(unittest.TestCase):
    def test_level_spoon_has_horizontal_forward_and_upward_bowl(self):
        node = HybridNode()
        for yaw in (-2.0, -0.5, 0.0, 1.5, 3.0):
            for pitch in (-0.4, 0.25):
                tilted = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
                goal = node.level_spoon_rotation(tilted)
                np.testing.assert_allclose(goal.apply(node.SPOON_FORWARD_AXIS_LOCAL), [np.cos(yaw), np.sin(yaw), 0.], atol=1e-6)
                np.testing.assert_allclose(goal.apply(node.SPOON_UP_AXIS_LOCAL), [0., 0., 1.], atol=1e-6)

    def test_horizontal_tracking_can_change_height_without_following_head_pitch(self):
        node = HybridNode()
        node.begin_tracking()
        start, _ = node.spoon_pose()
        tilted_axis = node.mouth_axis * np.cos(0.3) + np.array([0., 0., -np.sin(0.3)])
        for _ in range(180):
            node.send_sample(gap_mm=5., position=node.live_mouth + [0., 0., 0.015], axis=tilted_axis)
            node.tick()
        end, quat = node.spoon_pose()
        self.assertGreater(end[2] - start[2], 0.012)
        self.assertEqual(node.visual_mouth_axis[2], 0.)
        self.assertLess(node.visual_raw_axis[2], -0.2, 'Keep true head pose for movement monitoring')
        rot = R.from_quat(quat[[1, 2, 3, 0]])
        self.assertLess(abs(rot.apply(node.SPOON_FORWARD_AXIS_LOCAL)[2]), 0.001)
        self.assertGreater(rot.apply(node.SPOON_UP_AXIS_LOCAL)[2], 0.999)

    def test_depth_change_only_extends_feed_endpoint_at_locked_height(self):
        starts, targets, waiting = [], [], []
        for depth in (0.030, 0.040):
            node = HybridNode()
            node.visual_follow_config.insert_depth = depth
            node.reach_feeding()
            axis = node.visual_feed_axis
            start = node.visual_feed_start_spoon.copy()
            target = node.visual_feed_target()
            starts.append(start)
            targets.append(target)
            waiting.append(node.corrected_visual_mouth() - axis * node.visual_follow_config.standoff)
            self.assertAlmostEqual(target[2], start[2])
            self.assertAlmostEqual(float((target - start) @ axis), 0.060 + depth, delta=0.001)
        np.testing.assert_allclose(starts[0], starts[1])
        np.testing.assert_allclose(waiting[0], waiting[1])
        np.testing.assert_allclose(targets[1] - targets[0], axis * 0.010, atol=1e-7)

    def test_3d_target_compensation_has_person_view_signs_and_separate_depth(self):
        node = HybridNode()
        pose, _ = node.send_sample()
        observed = node.visual_raw_pos.copy()
        for yaw in (0., np.pi / 2., np.pi, -0.7):
            node.visual_mouth_axis = np.array([np.cos(yaw), np.sin(yaw), 0.])
            person_left = np.array([np.sin(yaw), -np.cos(yaw), 0.])
            for compensation in ((0.12, 0.04, 0.01), (-0.02, -0.03, -0.02)):
                node.visual_follow_config.target_offset = compensation
                corrected = node.corrected_visual_mouth()
                expected = (compensation[0] * node.visual_mouth_axis + compensation[1] * person_left
                            + np.array([0., 0., compensation[2]]))
                np.testing.assert_allclose(corrected - observed, expected, atol=1e-7)
                node.visual_feed_mouth = corrected.copy()
                node.visual_feed_axis = node.visual_mouth_axis.copy()
                node.visual_feed_height = float(corrected[2])
                goal = node.visual_feed_target()
                np.testing.assert_allclose(goal - corrected, node.visual_feed_axis * 0.015, atol=1e-7)
        np.testing.assert_allclose(node.visual_raw_pos, observed)
        self.assertEqual(pose.pose.position.z, observed[2])

    def test_horizontal_feed_and_withdraw_keep_height_with_3d_compensation(self):
        class TimedHybridNode(HybridNode, TimedTrajectoryDeployNode):
            publish_joint_target = TimedTrajectoryDeployNode.publish_joint_target
            receive_trajectory = TimedTrajectoryDeployNode.receive_trajectory
            tick = TimedTrajectoryDeployNode.tick

        node = TimedHybridNode()
        node.follow_commands = False
        node.scaling, node.speed_scaling = 0.54, 54.0
        node.interpolation = 'cubic'
        node.trajectory_target = node.trajectory_start = None
        node.trajectory_elapsed = node.trajectory_duration = node.dispatch_delay = 0.
        node.learned_safety_enabled = True
        node.wrist_wrench_policy_obs = np.zeros(6)
        node.safety_policy.act.return_value = np.zeros(3)
        node.visual_follow_config.target_offset = (0.06, 0.02, 0.01)
        # Synthetic 3D detection bias is deliberately known in this test.
        # Offset must fix all three components before alignment and feeding.
        offset = node.mouth_axis * 0.06 + np.cross(node.mouth_axis, [0., 0., 1.]) * 0.02 + [0., 0., 0.01]
        node.live_mouth -= offset
        tilted_axis = node.mouth_axis * np.cos(0.25) + np.array([0., 0., -np.sin(0.25)])
        node.begin_tracking()
        for _ in range(30):
            node.send_sample(axis=tilted_axis)
            node.tick()
            if node.fsm_state == 'VISUAL_FEED':
                break
        self.assertEqual(node.fsm_state, 'VISUAL_FEED')
        height = node.visual_feed_height
        self.assertEqual(node.visual_feed_axis[2], 0.)
        max_depth = -1.
        for _ in range(900):
            node.send_sample(axis=tilted_axis)
            node.tick()
            spoon, quat = node.spoon_pose()
            if node.fsm_state != 'RESET':
                self.assertAlmostEqual(spoon[2], height, delta=0.001)
                rot = R.from_quat(quat[[1, 2, 3, 0]])
                self.assertLess(abs(rot.apply(node.SPOON_FORWARD_AXIS_LOCAL)[2]), 0.002)
                self.assertGreater(rot.apply(node.SPOON_UP_AXIS_LOCAL)[2], 0.999)
            max_depth = max(max_depth, float((spoon - node.visual_feed_mouth) @ node.visual_feed_axis))
            if node.reset_started:
                break
        self.assertTrue(node.reset_started, node.logger.mock_calls[-5:])
        self.assertEqual(node.episode_stop_reason, 'SUCCESS')
        self.assertAlmostEqual(max_depth, 0.015, delta=0.001)
        self.assertEqual(sum(len(msg.points) > 1 for msg in node.trajectories), 2)

    def test_horizontal_mode_can_be_disabled(self):
        node = HybridNode()
        node.visual_follow_config.horizontal = False
        tilted = node.mouth_axis * np.cos(0.25) + np.array([0., 0., -np.sin(0.25)])
        node.send_sample(axis=tilted)
        np.testing.assert_allclose(node.visual_mouth_axis, tilted, atol=1e-6)

    def test_vertical_face_axis_is_rejected_in_horizontal_mode(self):
        node = HybridNode()
        node.send_sample(axis=[0., 0., 1.])
        self.assertTrue(node.visual_pose_rejected)
        self.assertIn('水平喂入方向', node.visual_pose_reject_reason)

    def test_fixed_reached_switches_to_tracking_instead_of_success_hold(self):
        node = HybridNode()
        spoon, quat = node.spoon_pose()
        node.mouth_pos = spoon.copy()
        for _ in range(node.FEED_STABLE_STEPS_REQUIRED):
            node.update_episode_fsm(spoon, quat)
        self.assertEqual(node.fsm_state, 'VISUAL_TRACK')
        self.assertTrue(node.visual_follow_started)
        self.assertEqual(node.success_hold_steps, 0)
        self.assertFalse(node.reset_started)

    def test_closed_mouth_tracks_moving_target_without_arm_actor(self):
        node = HybridNode()
        node.begin_tracking()
        start, _ = node.spoon_pose()
        for _ in range(120):
            node.send_sample(gap_mm=5., position=node.live_mouth + np.array([0., 0., 0.015]))
            node.tick()
        end, _ = node.spoon_pose()
        self.assertEqual(node.fsm_state, 'VISUAL_TRACK')
        self.assertGreater(end[2] - start[2], 0.010)
        self.assertLess(abs(np.dot(end - node.mouth_pos, node.compute_mouth_axis()) + 0.06), 0.003)
        node.policy.act.assert_not_called()
        self.assertFalse(node.reset_started)

    def test_stale_and_jumping_pose_hold_without_refreshing_target(self):
        node = HybridNode()
        node.begin_tracking()
        node.send_sample(gap_mm=5.)
        stamp = node.visual_pose_stamp_ns
        node.now_ns += 500_000_000
        before = len(node.commands)
        node.tick()
        self.assertEqual(node.visual_pose_stamp_ns, stamp)
        self.assertLessEqual(len(node.commands) - before, 1)
        node.send_sample(position=node.live_mouth + [0., 0., 0.10])
        self.assertEqual(node.visual_pose_stamp_ns, stamp)
        self.assertEqual(node.fsm_state, 'VISUAL_TRACK')

    def test_one_open_frame_or_unpaired_opening_cannot_start_feed(self):
        node = HybridNode()
        node.begin_tracking()
        node.send_sample()
        for _ in range(10):
            node.tick()
        self.assertEqual(node.fsm_state, 'VISUAL_TRACK')
        self.assertEqual(node.visual_open_samples, 1)
        for _ in range(20):
            node.send_sample(opening=False)
            node.tick()
        self.assertEqual(node.fsm_state, 'VISUAL_TRACK')

    def test_open_mouth_must_also_be_aligned_before_feed(self):
        node = HybridNode()
        node.begin_tracking()
        node.follow_commands = False
        for _ in range(30):
            node.send_sample(position=node.live_mouth + [0., 0., 0.025])
            node.tick()
        self.assertTrue(node.visual_open)
        self.assertEqual(node.fsm_state, 'VISUAL_TRACK')

    def test_closed_or_lost_visual_or_head_motion_aborts_into_withdrawal(self):
        for failure in ('closed', 'lost', 'moved', 'jump', 'force', 'timeout'):
            with self.subTest(failure=failure):
                node = HybridNode(locked_feed=False)
                node.reach_feeding()
                axis = node.visual_feed_axis.copy()
                if failure == 'closed':
                    node.send_sample(gap_mm=5.)
                elif failure == 'lost':
                    node.now_ns += 500_000_000
                elif failure == 'moved':
                    node.send_sample(position=node.live_mouth + [0., 0., 0.025])
                elif failure == 'jump':
                    node.send_sample(position=node.live_mouth + [0., 0., 0.10])
                elif failure == 'timeout':
                    node.visual_feed_start_ns -= 16_000_000_000
                    node.send_sample()
                else:
                    node.get_sim_aligned_safety_wrench_raw.return_value = np.array([20., 0., 0., 0., 0., 0.])
                    node.send_sample()
                node.tick()
                self.assertEqual(node.fsm_state, 'SAFETY_WITHDRAW')
                np.testing.assert_allclose(node.success_withdraw_direction, -axis, atol=1e-6)
                self.assertEqual(node.withdraw_distance_m, 0.10)
                self.assertEqual(len(node.trajectories), 1)

    def test_asynchronous_pose_and_opening_callbacks_do_not_false_abort(self):
        node = HybridNode()
        node.reach_feeding()
        node.now_ns += 30_000_000
        _, opening = node.send_sample(opening=False)
        node.tick()
        self.assertEqual(node.fsm_state, 'VISUAL_FEED')
        node.visual_opening_callback(opening)
        self.assertTrue(node.visual_opening_ready())

    def test_feed_then_hold_then_single_withdrawal_then_reset(self):
        node = HybridNode()
        node.reach_feeding()
        mouth = node.visual_feed_mouth.copy()
        axis = node.visual_feed_axis.copy()
        max_depth = -1.
        states = set()
        for _ in range(600):
            node.send_sample()
            node.tick()
            states.add(node.fsm_state)
            spoon, _ = node.spoon_pose()
            max_depth = max(max_depth, float(np.dot(spoon - mouth, axis)))
            if node.reset_started:
                break
        self.assertIn('SUCCESS_HOLD', states)
        self.assertIn('SUCCESS_WITHDRAW', states)
        self.assertTrue(node.reset_started, node.logger.mock_calls[-10:])
        self.assertLessEqual(max_depth, 0.0155)
        self.assertGreaterEqual(max_depth, 0.012)
        self.assertEqual(node.episode_stop_reason, 'SUCCESS')
        self.assertEqual(len(node.trajectories), 2, 'One complete feed and one complete withdrawal')
        node.policy.act.assert_not_called()

    def test_dry_run_tracks_and_previews_abort_without_publishing(self):
        node = HybridNode(dry_run=True, locked_feed=False)
        node.reach_feeding()
        node.send_sample(gap_mm=5.)
        node.tick()
        self.assertTrue(node.episode_stopped)
        self.assertFalse(node.reset_started)
        self.assertEqual(node.commands, [])
        self.assertEqual(node.trajectories, [])

    def test_lost_wrench_or_joint_feedback_does_not_continue_feed(self):
        for data in ('last_wrench_time_ns', 'last_joint_state_time_ns'):
            with self.subTest(data=data):
                node = HybridNode()
                node.reach_feeding()
                node.data_is_stale = lambda: ArmPolicyDeployNode.data_is_stale(node)
                setattr(node, data, node.now_ns - 1_000_000_000)
                before = node.current_joint_pos.copy()
                count = len(node.commands)
                node.step_callback()
                np.testing.assert_array_equal(node.current_joint_pos, before)
                self.assertLessEqual(len(node.commands) - count, 1)
                self.assertFalse(node.reset_started)

    def test_real_learned_safety_allows_open_center_crossing_but_keeps_force_gate(self):
        node = HybridNode()
        node.learned_safety_enabled = True
        node.wrist_wrench_policy_obs = np.zeros(6)
        node.safety_policy.act.return_value = np.zeros(3)
        node.reach_feeding()
        crossing = node.visual_feed_mouth - node.visual_feed_axis * 0.001
        _, quat = node.spoon_pose()
        state = node.compute_safety_policy_state(quat, crossing)
        self.assertFalse(state['linkage_contact_before_insert'])
        self.assertEqual(state['stage_entered'], 0.)
        node.wrist_wrench_policy_obs[0] = 1.6
        force_state = node.compute_safety_policy_state(quat, crossing)
        self.assertTrue(node.learned_safety_hazard_now(force_state))
        node.wrist_wrench_policy_obs.fill(0.)
        for _ in range(600):
            node.send_sample()
            node.tick()
            if node.reset_started:
                break
        self.assertTrue(node.reset_started)
        self.assertEqual(node.episode_stop_reason, 'SUCCESS')

    def test_close_during_success_hold_also_retracts(self):
        node = HybridNode(locked_feed=False)
        node.reach_feeding()
        for _ in range(300):
            node.send_sample()
            node.tick()
            if node.fsm_state == 'SUCCESS_HOLD':
                break
        self.assertEqual(node.fsm_state, 'SUCCESS_HOLD')
        node.send_sample(gap_mm=5.)
        node.tick()
        self.assertEqual(node.fsm_state, 'SAFETY_WITHDRAW')

    def test_scaled_delayed_controller_completes_feed_and_withdrawal(self):
        class TimedHybridNode(HybridNode, TimedTrajectoryDeployNode):
            publish_joint_target = TimedTrajectoryDeployNode.publish_joint_target
            receive_trajectory = TimedTrajectoryDeployNode.receive_trajectory
            tick = TimedTrajectoryDeployNode.tick

        node = TimedHybridNode()
        node.follow_commands = False
        node.scaling = node.speed_scaling = 0.55
        node.interpolation = 'cubic'
        node.trajectory_target = None
        node.trajectory_start = None
        node.trajectory_elapsed = 0.
        node.trajectory_duration = 0.
        node.dispatch_delay = 0.
        node.begin_tracking()
        for _ in range(1500):
            node.send_sample()
            node.tick()
            if node.reset_started or node.episode_stopped:
                break
        self.assertTrue(node.reset_started, node.logger.mock_calls[-10:])
        self.assertEqual(node.episode_stop_reason, 'SUCCESS')
        self.assertEqual(sum(len(msg.points) > 1 for msg in node.trajectories), 2)

    def test_continuous_feed_reaches_requested_depth_at_54_percent_scaling(self):
        class TimedHybridNode(HybridNode, TimedTrajectoryDeployNode):
            publish_joint_target = TimedTrajectoryDeployNode.publish_joint_target
            receive_trajectory = TimedTrajectoryDeployNode.receive_trajectory
            tick = TimedTrajectoryDeployNode.tick

        for requested_depth in (-0.020, -0.005, 0.0, 0.015, 0.025, 0.040, 0.080):
            with self.subTest(depth=requested_depth):
                node = TimedHybridNode()
                node.follow_commands = False
                node.scaling, node.speed_scaling = 0.54, 54.0
                node.interpolation = 'cubic'
                node.trajectory_target = node.trajectory_start = None
                node.trajectory_elapsed = node.trajectory_duration = node.dispatch_delay = 0.
                node.visual_follow_config.insert_depth = requested_depth
                node.learned_safety_enabled = True
                node.wrist_wrench_policy_obs = np.zeros(6)
                node.safety_policy.act.return_value = np.zeros(3)
                node.reach_feeding()
                started = node.now_ns
                before_feed = len(node.trajectories)
                path = []
                while node.fsm_state == 'VISUAL_FEED' and node.now_ns - started < 30_000_000_000:
                    node.send_sample()
                    node.tick()
                    spoon, _ = node.spoon_pose()
                    path.append(spoon)
                    if node.visual_feed_trajectory_sent and node.fsm_state == 'VISUAL_FEED':
                        self.assertEqual(len(node.trajectories), before_feed + 1, 'Monitoring must never replace the feed')
                self.assertEqual(node.fsm_state, 'SUCCESS_HOLD', node.logger.mock_calls[-5:])
                depth = float((path[-1] - node.visual_feed_mouth) @ node.visual_feed_axis)
                self.assertAlmostEqual(depth, requested_depth, delta=0.001)
                self.assertAlmostEqual(path[-1][2], node.visual_feed_height, delta=0.001)
                progress = (np.asarray(path) - path[0]) @ node.visual_feed_axis
                self.assertTrue(np.all(np.diff(progress) >= -0.00005), 'Continuous feed must not reverse')
                radial = np.asarray(path) - node.visual_feed_mouth
                radial -= (radial @ node.visual_feed_axis)[:, None] * node.visual_feed_axis
                self.assertLess(np.max(np.linalg.norm(radial, axis=1)), 0.001)
                trajectory = node.trajectories[before_feed]
                np.testing.assert_allclose(trajectory.points[0].velocities, 0.)
                np.testing.assert_allclose(trajectory.points[-1].velocities, 0.)
                self.assertTrue(all(np.linalg.norm(p.velocities) > 0. for p in trajectory.points[1:-1]))
                for p in trajectory.points:
                    self.assertTrue(np.all(np.abs(p.velocities) <= node.JOINT_MAX_VELOCITIES_RAD_S + 1e-9))

    def test_loss_after_dispatch_interrupts_continuous_feed(self):
        for failure in ('closed', 'lost', 'force'):
            with self.subTest(failure=failure):
                node = HybridNode(locked_feed=False)
                node.safety_grec_force_one = True
                node.follow_commands = False
                node.reach_feeding()
                node.send_sample()
                node.tick()
                self.assertTrue(node.visual_feed_trajectory_sent)
                self.assertEqual(len(node.trajectories), 1)
                if failure == 'closed':
                    node.send_sample(gap_mm=5.)
                elif failure == 'lost':
                    node.now_ns += 500_000_000
                else:
                    node.get_sim_aligned_safety_wrench_raw.return_value = np.array([20., 0., 0., 0., 0., 0.])
                    node.send_sample()
                node.tick()
                self.assertEqual(node.fsm_state, 'SAFETY_WITHDRAW')
                self.assertEqual(len(node.trajectories), 2)
                self.assertTrue(all(row['grec'] == 0. for row in node.safety_gain_curve_rows))
                warnings = [str(call.args[0]) for call in node.logger.warn.call_args_list]
                self.assertTrue(any('视觉喂入中止:' in message and '实际前送=' in message for message in warnings))

    def test_locked_feed_ignores_visual_failures_before_and_after_dispatch(self):
        for dispatched in (False, True):
            for failure in ('closed', 'lost', 'opening_lost', 'moved', 'jump', 'axis'):
                with self.subTest(dispatched=dispatched, failure=failure):
                    node = HybridNode()
                    node.follow_commands = False
                    node.reach_feeding()
                    self.assertTrue(node.visual_feed_locked)
                    target = node.visual_feed_target().copy()
                    if dispatched:
                        node.tick()
                        self.assertTrue(node.visual_feed_trajectory_sent)
                    if failure == 'closed':
                        node.send_sample(gap_mm=1.4)
                    elif failure == 'lost':
                        node.now_ns += 500_000_000
                    elif failure == 'opening_lost':
                        node.visual_paired_stamp_ns = None
                    elif failure == 'moved':
                        node.send_sample(position=node.live_mouth + [0., 0., 0.025])
                    elif failure == 'jump':
                        node.send_sample(position=node.live_mouth + [0., 0., 0.10])
                    else:
                        node.send_sample(axis=-node.mouth_axis)
                    node.tick()
                    self.assertEqual(node.fsm_state, 'VISUAL_FEED')
                    self.assertTrue(node.visual_feed_trajectory_sent)
                    self.assertEqual(len(node.trajectories), 1)
                    np.testing.assert_array_equal(node.visual_feed_target(), target)
                    self.assertFalse(node.safety_strengthen_active)

    def test_locked_feed_depth_is_captured_at_alignment(self):
        node = HybridNode()
        node.visual_follow_config.insert_depth = 0.040
        node.reach_feeding()
        target = node.visual_feed_target().copy()
        node.visual_follow_config.insert_depth = 0.080
        node.send_sample(gap_mm=1.4, position=node.live_mouth + [0., 0., 0.025])
        self.assertEqual(node.visual_feed_insert_depth, 0.040)
        np.testing.assert_array_equal(node.visual_feed_target(), target)

    def test_locked_feed_finishes_hold_withdrawal_and_reset_under_occlusion(self):
        class TimedHybridNode(HybridNode, TimedTrajectoryDeployNode):
            publish_joint_target = TimedTrajectoryDeployNode.publish_joint_target
            receive_trajectory = TimedTrajectoryDeployNode.receive_trajectory
            tick = TimedTrajectoryDeployNode.tick

        node = TimedHybridNode()
        node.follow_commands = False
        node.scaling, node.speed_scaling = 0.54, 54.0
        node.interpolation = 'cubic'
        node.trajectory_target = node.trajectory_start = None
        node.trajectory_elapsed = node.trajectory_duration = node.dispatch_delay = 0.
        node.learned_safety_enabled = True
        node.wrist_wrench_policy_obs = np.zeros(6)
        node.safety_policy.act.return_value = np.zeros(3)
        node.visual_follow_config.insert_depth = 0.080
        node.reach_feeding()
        mouth = node.visual_feed_mouth.copy()
        axis = node.visual_feed_axis.copy()
        height = node.visual_feed_height
        stamp = node.visual_pose_stamp_ns
        states, max_depth = set(), -1.
        # No camera messages arrive again after the alignment decision.
        for _ in range(1500):
            node.tick()
            states.add(node.fsm_state)
            spoon, _ = node.spoon_pose()
            max_depth = max(max_depth, float((spoon - mouth) @ axis))
            self.assertAlmostEqual(spoon[2], height, delta=0.001)
            if node.reset_started:
                break
        self.assertFalse(node.visual_stamp_fresh(stamp))
        self.assertIn('SUCCESS_HOLD', states)
        self.assertIn('SUCCESS_WITHDRAW', states)
        self.assertAlmostEqual(max_depth, 0.080, delta=0.001)
        self.assertTrue(node.reset_started, node.logger.mock_calls[-10:])
        self.assertEqual(node.episode_stop_reason, 'SUCCESS')
        paths = [message for message in node.trajectories if len(message.points) > 1]
        self.assertEqual(len(paths), 2, 'One complete feed and one complete withdrawal')
        self.assertAlmostEqual(node.withdraw_distance_m, 0.10)
        self.assertTrue(all(row['grec'] == 0. for row in node.safety_gain_curve_rows))

    def test_locked_feed_keeps_force_abort_when_vision_is_stale(self):
        node = HybridNode()
        node.follow_commands = False
        node.reach_feeding()
        node.tick()
        self.assertTrue(node.visual_feed_trajectory_sent)
        node.now_ns += 500_000_000
        node.get_sim_aligned_safety_wrench_raw.return_value = np.array([20., 0., 0., 0., 0., 0.])
        node.tick()
        self.assertEqual(node.fsm_state, 'SAFETY_WITHDRAW')
        self.assertEqual(len(node.trajectories), 2)

    def test_negative_depth_stops_before_visual_plane_and_completes_reset(self):
        node = HybridNode()
        node.visual_follow_config.insert_depth = -0.005
        node.reach_feeding()
        mouth = node.visual_feed_mouth.copy()
        axis = node.visual_feed_axis.copy()
        start = node.visual_feed_start_spoon.copy()
        target = node.visual_feed_target()
        self.assertAlmostEqual(float((target - start) @ axis), 0.055, delta=0.001)
        self.assertAlmostEqual(target[2], start[2])
        states, max_depth = set(), -1.
        for _ in range(600):
            # Camera stays occluded; success is based on actual endpoint feedback,
            # not the fixed-point policy's positive-depth success condition.
            node.tick()
            states.add(node.fsm_state)
            spoon, _ = node.spoon_pose()
            max_depth = max(max_depth, float((spoon - mouth) @ axis))
            if node.fsm_state == 'SUCCESS_HOLD':
                self.assertFalse(node.last_fsm_metrics['feed_reached'])
            if node.reset_started:
                break
        self.assertAlmostEqual(max_depth, -0.005, delta=0.001)
        self.assertIn('SUCCESS_HOLD', states)
        self.assertIn('SUCCESS_WITHDRAW', states)
        self.assertTrue(node.reset_started, node.logger.mock_calls[-10:])
        self.assertEqual(node.episode_stop_reason, 'SUCCESS')
        self.assertEqual(node.withdraw_distance_m, 0.10)

    def test_signed_depth_configuration_accepts_plane_and_shallower_targets(self):
        for depth in (-0.020, -0.005, 0., 0.0049, 0.005):
            with self.subTest(depth=depth), \
                    patch.object(ArmPolicyDeployNode, 'VISUAL_INSERT_DEPTH_M', depth), \
                    patch('sys.argv', ['run_task.py', '--visual-follow-after-fixed']):
                args, _ = parse_args()
                cfg = VisualFollowConfig.from_args(args, ArmPolicyDeployNode)
                self.assertEqual(cfg.insert_depth, depth)

    def test_locked_dry_run_previews_feed_with_occluded_camera_without_motion(self):
        node = HybridNode(dry_run=True)
        node.reach_feeding()
        node.now_ns += 500_000_000
        node.tick()
        self.assertEqual(node.fsm_state, 'VISUAL_FEED')
        self.assertTrue(node.visual_feed_trajectory_planned)
        self.assertFalse(node.visual_feed_trajectory_sent)
        self.assertEqual(node.commands, [])
        self.assertEqual(node.trajectories, [])

    def test_original_safety_hazard_does_not_interrupt_continuous_visual_feed(self):
        for backend in ('learned', 'rule'):
            with self.subTest(backend=backend):
                node = HybridNode()
                node.follow_commands = False
                node.reach_feeding()
                node.learned_safety_enabled = backend == 'learned'
                node.rule_safety_enabled = backend == 'rule'
                node.safety_grec_force_one = True
                # Exercise actual filters: policy-observation risk for the learned
                # backend; out-of-channel approach geometry for the rule backend.
                node.wrist_wrench_policy_obs = np.array([1.6, 0., 0., 0., 0., 0.])
                node.safety_policy.act.return_value = np.ones(3)
                node.RULE_RETREAT_START_DIST = 0.10
                for _ in range(3):
                    node.send_sample()
                    node.tick()
                self.assertEqual(node.fsm_state, 'VISUAL_FEED')
                self.assertTrue(node.visual_feed_trajectory_sent)
                self.assertEqual(len(node.trajectories), 1)
                self.assertFalse(node.safety_strengthen_active)
                self.assertTrue(all(row['grec'] == 0. for row in node.safety_gain_curve_rows))

    def test_feedback_recovery_does_not_resume_a_cancelled_feed(self):
        node = HybridNode()
        node.follow_commands = False
        node.reach_feeding()
        node.send_sample()
        node.tick()
        self.assertTrue(node.visual_feed_trajectory_sent)
        node.data_is_stale.return_value = True
        node.tick()
        self.assertTrue(node.visual_feed_interrupted_reason)
        node.data_is_stale.return_value = False
        node.send_sample()
        node.tick()
        self.assertEqual(node.fsm_state, 'SAFETY_WITHDRAW')

    def test_stalled_feed_times_out_without_resending_the_forward_path(self):
        node = HybridNode()
        node.follow_commands = False
        node.reach_feeding()
        node.send_sample()
        node.tick()
        for _ in range(10):
            node.send_sample()
            node.tick()
        self.assertEqual(len(node.trajectories), 1)
        node.visual_feed_start_ns -= int((node.visual_feed_timeout_s + 1.) * 1e9)
        node.send_sample()
        node.tick()
        self.assertEqual(node.fsm_state, 'SAFETY_WITHDRAW')
        self.assertEqual(len(node.trajectories), 2)

    def test_feed_timeout_accounts_for_slow_ur_speed_scaling(self):
        node = HybridNode()
        node.follow_commands = False
        node.speed_scaling = 10.
        node.reach_feeding()
        node.send_sample()
        node.tick()
        self.assertGreater(node.visual_feed_timeout_s, node.visual_feed_duration_s / 0.1)

    def test_failed_feed_planning_never_dispatches_forward_motion(self):
        node = HybridNode()
        node.reach_feeding()
        planner = node.plan_spoon_linear_trajectory
        calls = []

        def fail_forward(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise ValueError('unreachable feed target')
            return planner(*args, **kwargs)

        node.plan_spoon_linear_trajectory = fail_forward
        node.send_sample()
        node.tick()
        self.assertEqual(node.fsm_state, 'SAFETY_WITHDRAW')
        self.assertFalse(node.visual_feed_trajectory_sent)
        self.assertEqual(len(node.trajectories), 1, 'Only withdrawal may be dispatched')

    def test_parameter_validation_rejects_incompatible_or_invalid_values(self):
        invalid = [
            ['--visual-follow-after-fixed', '--vision-mouth'],
            ['--mouth-open-mm', '5', '--mouth-close-mm', '8'],
            ['--follow-speed', 'nan'], ['--follow-insert-depth', '0.10'],
            ['--visual-follow-after-fixed', '--withdraw-distance', '0.08'],
            ['--follow-mouth-offset', '0.05'],
        ]
        for argv in invalid:
            with self.subTest(argv=argv), patch('sys.argv', ['run_task.py', *argv]), patch('sys.stderr'):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_follow_defaults_use_the_existing_fused_pose(self):
        with patch('sys.argv', ['run_task.py', '--visual-follow-after-fixed']):
            args, _ = parse_args()
        self.assertEqual(args.follow_pose_topic, '/mouth_pose')
        self.assertEqual(args.follow_opening_topic, '/mouth_opening')
        self.assertEqual(args.follow_feed_speed, 0.03)
        self.assertTrue(args.follow_horizontal)
        cfg = VisualFollowConfig.from_args(args, ArmPolicyDeployNode)
        self.assertEqual(cfg.target_offset, tuple(ArmPolicyDeployNode.VISUAL_TARGET_OFFSET_M))
        self.assertEqual(cfg.insert_depth, ArmPolicyDeployNode.VISUAL_INSERT_DEPTH_M)
        self.assertEqual(cfg.locked_feed, ArmPolicyDeployNode.VISUAL_FEED_LOCKED_MODE)
        with patch('sys.argv', ['run_task.py', '--visual-follow-after-fixed', '--no-follow-horizontal']), \
                patch.object(ArmPolicyDeployNode, 'VISUAL_TARGET_OFFSET_M', (0.12, -0.04, 0.01)), \
                patch.object(ArmPolicyDeployNode, 'VISUAL_INSERT_DEPTH_M', 0.025), \
                patch.object(ArmPolicyDeployNode, 'VISUAL_FEED_LOCKED_MODE', False):
            args, _ = parse_args()
            cfg = VisualFollowConfig.from_args(args, ArmPolicyDeployNode)
            self.assertEqual(cfg.target_offset, (0.12, -0.04, 0.01))
            self.assertEqual(cfg.insert_depth, 0.025)
            self.assertFalse(cfg.horizontal)
            self.assertFalse(cfg.locked_feed)

    def test_invalid_code_configuration_fails_before_robot_node_is_created(self):
        cases = [('VISUAL_INSERT_DEPTH_M', value) for value in (
            -0.0201, 0.0801, 0.1, float('nan'), float('inf'), float('-inf'), 'bad')]
        cases += [('VISUAL_TARGET_OFFSET_M', value) for value in (
            (0., 0.), (0., float('nan'), 0.), (0., 0., 0.201), (float('inf'), 0., 0.), 'bad')]
        for name, value in cases:
            with self.subTest(name=name, value=value), patch.object(ArmPolicyDeployNode, name, value), \
                    patch('sys.argv', ['run_task.py', '--visual-follow-after-fixed']), patch('sys.stderr'):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_waiting_point_does_not_fake_entered_stage_or_widen_channel(self):
        node = HybridNode()
        node.begin_tracking()
        node.set_visual_safety_target(node.live_mouth, node.mouth_axis)
        node.wrist_wrench_policy_obs = np.zeros(6)
        spoon, quat = node.spoon_pose()
        state = node.compute_safety_policy_state(quat, spoon)
        self.assertFalse(state['inside_mouth_channel'])
        self.assertFalse(node.learned_safety_hazard_now(state))
        self.assertEqual(node.compute_safety_stage_flags()['stage_entered'], 0.)
        off_axis = node.mouth_pos + node.compute_mouth_axis() * 0.01 + np.array([0., 0., 0.04])
        state = node.compute_safety_policy_state(quat, off_axis)
        self.assertFalse(state['inside_mouth_channel'])
        self.assertTrue(node.learned_safety_hazard_now(state))


class MouthPoseFusionTest(unittest.TestCase):
    def fake_fusion(self):
        node = object.__new__(PoseFusionNode)
        node.now_ns = 10_000_000_000
        node.get_clock = Mock(return_value=SimpleNamespace(now=lambda: Time(nanoseconds=node.now_ns)))
        node.get_logger = Mock(return_value=Mock())
        node.target_frame = 'base_link'
        node.mouth_openings = {camera: OrderedDict() for camera in ('wrist', 'fixed')}
        node.pending_poses = {camera: OrderedDict() for camera in ('wrist', 'fixed')}
        node.camera_pose_cache = {}
        node.CAMERA_MODE = 'auto'
        node.last_selected_key = None
        node.last_published_pose = node.last_published_axis = node.last_accepted_time = None
        node.publisher = Mock()
        node.opening_publisher = Mock()

        def transform(msg, *args, **kwargs):
            transformed = deepcopy(msg)
            transformed.header.frame_id = 'base_link'
            return transformed

        node.tf_buffer = SimpleNamespace(can_transform=Mock(return_value=True), transform=Mock(side_effect=transform))
        return node

    def camera_sample(self, node, camera, gap_mm=15., stamp=None, publish_opening=True):
        source = HybridNode()
        pose, opening = source.send_sample(gap_mm=gap_mm, stamp=node.now_ns if stamp is None else stamp)
        pose.header.frame_id = f'{camera}_cam_optical_frame'
        opening.header.frame_id = pose.header.frame_id
        if camera == 'fixed':
            node.pose2_callback(pose)
        else:
            node.pose1_callback(pose)
        if publish_opening:
            node.mouth_opening_callback(camera, opening)
        return pose, opening

    def test_fixed_camera_pose_and_opening_are_used_together(self):
        node = self.fake_fusion()
        self.camera_sample(node, 'wrist', gap_mm=25.)
        pose, opening = self.camera_sample(node, 'fixed', gap_mm=5.)
        node.now_ns += 30_000_000
        node.fusion_timer_callback()
        result = node.publisher.publish.call_args.args[0]
        self.assertEqual(result.header.frame_id, 'base_link')
        self.assertEqual(result.header.stamp, pose.header.stamp)
        node.opening_publisher.publish.assert_called_once_with(opening)
        self.assertEqual(node.last_selected_key[0], 'fixed')

    def test_wrist_camera_is_used_when_fixed_camera_disappears(self):
        node = self.fake_fusion()
        self.camera_sample(node, 'fixed', gap_mm=5.)
        node.fusion_timer_callback()
        node.now_ns += 600_000_000
        pose, opening = self.camera_sample(node, 'wrist', gap_mm=25.)
        node.fusion_timer_callback()
        self.assertEqual(node.last_selected_key[0], 'wrist')
        self.assertEqual(node.publisher.publish.call_args.args[0].header.stamp, pose.header.stamp)
        self.assertEqual(node.opening_publisher.publish.call_args.args[0], opening)

    def test_acquisition_tf_is_used_without_blocking_timer(self):
        node = self.fake_fusion()
        pose, _ = self.camera_sample(node, 'wrist')
        node.now_ns += 30_000_000
        node.fusion_timer_callback()
        query = node.tf_buffer.transform.call_args
        self.assertEqual(query.args[0].header.stamp, pose.header.stamp)
        self.assertEqual(query.kwargs['timeout'], Duration())
        self.assertEqual(node.publisher.publish.call_args.args[0].header.stamp, pose.header.stamp)

    def test_current_wrist_only_selection_survives_missing_inputs(self):
        node = self.fake_fusion()
        node.CAMERA_MODE = PoseFusionNode.CAMERA_MODE
        self.assertEqual(node.CAMERA_MODE, 'wrist')
        node.fusion_timer_callback()
        self.camera_sample(node, 'fixed')
        node.fusion_timer_callback()
        node.publisher.publish.assert_not_called()
        self.camera_sample(node, 'wrist')
        node.fusion_timer_callback()
        self.assertEqual(node.last_selected_key[0], 'wrist')
        first = deepcopy(node.publisher.publish.call_args.args[0])
        node.now_ns += 500_000_000
        node.fusion_timer_callback()
        self.assertEqual(node.publisher.publish.call_args.args[0], first)

    def test_pending_frame_is_not_starved_by_newer_images_without_tf(self):
        node = self.fake_fusion()
        old, _ = self.camera_sample(node, 'wrist')
        node.tf_buffer.can_transform.return_value = False
        node.fusion_timer_callback()
        node.publisher.publish.assert_not_called()
        node.now_ns += 30_000_000
        new, _ = self.camera_sample(node, 'wrist')
        node.tf_buffer.can_transform.side_effect = lambda target, source, when: when.nanoseconds == self.stamp(old)
        node.fusion_timer_callback()
        self.assertEqual(node.publisher.publish.call_args.args[0].header.stamp, old.header.stamp)
        self.assertIn(self.stamp(new), node.pending_poses['wrist'])
        node.tf_buffer.can_transform.side_effect = None
        node.tf_buffer.can_transform.return_value = True
        node.now_ns += 20_000_000
        node.fusion_timer_callback()
        self.assertEqual(node.publisher.publish.call_args.args[0].header.stamp, new.header.stamp)

    @staticmethod
    def stamp(msg):
        return PoseFusionNode.stamp_ns(msg)

    def test_moving_wrist_camera_does_not_move_stationary_world_mouth(self):
        node = self.fake_fusion()
        node.tf_buffer = Buffer()
        target = np.array([0.7, 0.5, 0.25])
        # Current wrist position differs from the image's wrist position.
        for stamp, x in ((10_000_000_000, 0.), (10_100_000_000, 0.01), (10_200_000_000, 0.02)):
            transform = TransformStamped()
            transform.header.frame_id = 'base_link'
            transform.child_frame_id = 'wrist_cam_optical_frame'
            transform.header.stamp = Time(nanoseconds=stamp).to_msg()
            transform.transform.translation.x = x
            rotation = R.from_rotvec([0., 0., x * 6.])
            q = rotation.as_quat()
            (transform.transform.rotation.x, transform.transform.rotation.y,
             transform.transform.rotation.z, transform.transform.rotation.w) = map(float, q)
            node.tf_buffer.set_transform(transform, 'offline_test')
        node.now_ns = 10_250_000_000
        for stamp, x in ((10_000_000_000, 0.), (10_100_000_000, 0.01)):
            raw, _ = self.camera_sample(node, 'wrist', stamp=stamp)
            rotation = R.from_rotvec([0., 0., x * 6.])
            observed = rotation.inv().apply(target - [x, 0., 0.])
            raw.pose.position.x, raw.pose.position.y, raw.pose.position.z = map(float, observed)
            converted = node.get_pose_in_base(raw)
            self.assertIsNotNone(converted)
            np.testing.assert_allclose(node.pose_position_array(converted), target, atol=1e-8)

    def test_pending_pose_history_is_bounded_and_expired_frames_are_not_retimed(self):
        node = self.fake_fusion()
        for i in range(100):
            self.camera_sample(node, 'wrist', stamp=node.now_ns + i)
        self.assertEqual(len(node.pending_poses['wrist']), node.POSE_HISTORY_SIZE)
        node.now_ns += 500_000_000
        node.fusion_timer_callback()
        node.publisher.publish.assert_not_called()
        self.assertFalse(node.pending_poses['wrist'])

    def test_held_pose_does_not_refresh_detection_or_repeat_opening(self):
        node = self.fake_fusion()
        pose, opening = self.camera_sample(node, 'fixed')
        node.fusion_timer_callback()
        first = deepcopy(node.publisher.publish.call_args.args[0])
        node.opening_publisher.publish.reset_mock()
        node.now_ns += 2_000_000_000
        node.fusion_timer_callback()
        held = node.publisher.publish.call_args.args[0]
        self.assertEqual(held, first)
        self.assertEqual(pose.header.stamp, first.header.stamp)
        node.opening_publisher.publish.assert_not_called()

        # Repeated /mouth_pose publication must not authorize feeding on a
        # historical open mouth when both detectors have lost the target.
        task = HybridNode()
        task.begin_tracking()
        task.visual_pose_callback(first)
        task.visual_opening_callback(opening)
        for _ in range(30):
            task.now_ns += 30_000_000
            task.visual_pose_callback(held)
            task.tick()
        self.assertEqual(task.fsm_state, 'VISUAL_TRACK')
        self.assertFalse(task.visual_opening_ready())

    def test_late_opening_is_forwarded_only_for_the_selected_frame(self):
        node = self.fake_fusion()
        pose, opening = self.camera_sample(node, 'fixed', publish_opening=False)
        node.mouth_opening_callback('wrist', opening)
        node.fusion_timer_callback()
        node.opening_publisher.publish.assert_not_called()
        mismatched = deepcopy(opening)
        mismatched.header.stamp.nanosec += 1
        node.mouth_opening_callback('fixed', mismatched)
        node.fusion_timer_callback()
        node.opening_publisher.publish.assert_not_called()
        node.mouth_opening_callback('fixed', opening)
        node.fusion_timer_callback()
        node.opening_publisher.publish.assert_called_once_with(opening)
        self.assertEqual(node.publisher.publish.call_args.args[0].header.stamp, pose.header.stamp)

    def test_repeated_image_does_not_count_as_new_detection(self):
        node = self.fake_fusion()
        self.camera_sample(node, 'fixed')
        node.fusion_timer_callback()
        accepted = node.last_accepted_time
        first = deepcopy(node.publisher.publish.call_args.args[0])
        node.now_ns += 100_000_000
        node.fusion_timer_callback()
        self.assertEqual(node.last_accepted_time, accepted)
        self.assertEqual(node.publisher.publish.call_args.args[0], first)

    def test_opening_history_is_bounded(self):
        node = self.fake_fusion()
        for i in range(100):
            self.camera_sample(node, 'fixed', stamp=node.now_ns + i)
        self.assertEqual(len(node.mouth_openings['fixed']), node.OPENING_HISTORY_SIZE)

    def test_pose_copy_does_not_share_the_detection_stamp(self):
        node = self.fake_fusion()
        pose, _ = self.camera_sample(node, 'fixed')
        copied = node.copy_pose(pose)
        copied.header.stamp.sec += 1
        self.assertNotEqual(copied.header.stamp, pose.header.stamp)


if __name__ == '__main__':
    unittest.main()
