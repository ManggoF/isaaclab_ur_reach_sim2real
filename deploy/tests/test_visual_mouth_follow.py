"""Exercise visual handoff and feeding with FK feedback; no robot connection."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from geometry_msgs.msg import PoseStamped, Vector3Stamped

from deploy.face_pose_fusion import PoseFusionNode
from deploy.run_task import ArmPolicyDeployNode, parse_args
from deploy.tests.test_success_withdraw import OfflineDeployNode, TimedTrajectoryDeployNode


class HybridNode(OfflineDeployNode):
    def __init__(self, dry_run=False):
        OfflineDeployNode.__init__(self, dry_run=dry_run)
        self.target_frame = 'base_link'
        self.args.episode_length_s = 240.
        self.max_episode_steps = 7200
        self.wrench_sign = -1.
        self.args.min_spoon_z = 0.06
        self.args.max_mouth_spoon_distance = 3.
        self.args.visual_follow_after_fixed = False  # No ROS subscriptions in this fake.
        self.args.follow_standoff = 0.06
        self.args.follow_insert_depth = 0.015
        self.args.follow_speed = 0.03
        self.args.follow_feed_speed = 0.015
        self.args.mouth_open_mm = 10.
        self.args.mouth_close_mm = 8.
        self.args.mouth_open_hold = 0.3
        self.args.follow_timeout = 0.4
        self.args.fixed_reach_tolerance = 0.015
        self.init_visual_follow(self.args)
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
                node = HybridNode()
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
        self.assertEqual(len(node.trajectories), 1)
        node.policy.act.assert_not_called()

    def test_dry_run_tracks_and_previews_abort_without_publishing(self):
        node = HybridNode(dry_run=True)
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
        node = HybridNode()
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
        self.assertEqual(sum(len(msg.points) > 1 for msg in node.trajectories), 1)

    def test_parameter_validation_rejects_incompatible_or_invalid_values(self):
        invalid = [
            ['--visual-follow-after-fixed', '--vision-mouth'],
            ['--mouth-open-mm', '5', '--mouth-close-mm', '8'],
            ['--follow-speed', 'nan'], ['--follow-insert-depth', '0.10'],
            ['--visual-follow-after-fixed', '--withdraw-distance', '0.08'],
        ]
        for argv in invalid:
            with self.subTest(argv=argv), patch('sys.argv', ['run_task.py', *argv]), patch('sys.stderr'):
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


class LiveFusionTest(unittest.TestCase):
    def fake_fusion(self):
        node = object.__new__(PoseFusionNode)
        node.tracking_camera = 'wrist'
        node.pose1_raw = None
        node.pose2_raw = None
        node.target_frame = 'base_link'
        node.tracking_openings = {'wrist': None, 'fixed': None}
        node.tracking_pose_pub = Mock()
        node.tracking_opening_pub = Mock()
        node.tf_buffer = SimpleNamespace(transform=Mock(side_effect=lambda msg, *a, **kw: deepcopy(msg)))
        return node

    def test_live_output_keeps_acquisition_stamp_and_matches_camera(self):
        source = HybridNode()
        pose, opening = source.send_sample()
        pose.header.frame_id = 'wrist_cam_optical_frame'
        node = self.fake_fusion()
        node.pose1_raw = pose
        node.tracking_openings['wrist'] = opening
        node.publish_live_tracking(SimpleNamespace(nanoseconds=source.now_ns + 30_000_000))
        result = node.tracking_pose_pub.publish.call_args.args[0]
        self.assertEqual(PoseFusionNode.stamp_ns(result), source.now_ns)
        node.tracking_opening_pub.publish.assert_called_once_with(opening)
        node.tracking_opening_pub.publish.reset_mock()
        node.tracking_openings['wrist'] = None
        node.tracking_openings['fixed'] = opening
        node.publish_live_tracking(SimpleNamespace(nanoseconds=source.now_ns + 30_000_000))
        node.tracking_opening_pub.publish.assert_not_called()

    def test_live_output_does_not_publish_stale_pose_or_mismatched_opening(self):
        source = HybridNode()
        pose, opening = source.send_sample()
        node = self.fake_fusion()
        node.pose1_raw = pose
        node.tracking_openings['wrist'] = opening
        node.publish_live_tracking(SimpleNamespace(nanoseconds=source.now_ns + 500_000_000))
        node.tracking_pose_pub.publish.assert_not_called()
        opening.header.stamp.nanosec += 1
        node.publish_live_tracking(SimpleNamespace(nanoseconds=source.now_ns + 30_000_000))
        node.tracking_pose_pub.publish.assert_called_once()
        node.tracking_opening_pub.publish.assert_not_called()


if __name__ == '__main__':
    unittest.main()
