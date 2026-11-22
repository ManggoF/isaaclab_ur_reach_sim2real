"""Pure visual startup and feeding without ROS connections or learned models."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from deploy.run_task import ArmPolicyDeployNode, parse_args
from deploy.tests.test_visual_mouth_follow import HybridNode


class VisualOnlyTest(unittest.TestCase):
    def direct_node(self, dry_run=False):
        node = HybridNode(dry_run=dry_run)
        node.visual_only_mode = True
        node.visual_follow_enabled = True
        node.policy = node.safety_policy = None
        node.rule_safety_enabled = node.learned_safety_enabled = False
        node.wrist_wrench_policy_obs = np.zeros(6)
        node.target_received = False
        node.mouth_pos = node.fixed_mouth_axis = node.mouth_quat_wxyz = None
        node.episode_start_time_ns = None
        return node

    def test_code_switch_skips_all_model_loading_and_enables_visual_subscriptions(self):
        node = HybridNode()
        node.VISUAL_ONLY_MODE = True
        # Also override an enabled rule layer: pure visual mode uses no safety actor/rules.
        node.RULE_SAFETY_ENABLED = True
        node.create_subscription = Mock()
        with patch('deploy.run_task.HarlArmPolicy', side_effect=AssertionError('Must not load a policy')):
            node.load_control_policies(SimpleNamespace())
            node.init_visual_follow(node.args)
        self.assertIsNone(node.policy)
        self.assertIsNone(node.safety_policy)
        self.assertFalse(node.learned_safety_enabled)
        self.assertFalse(node.rule_safety_enabled)
        self.assertTrue(node.visual_follow_enabled)
        self.assertFalse(node.args.visual_follow_after_fixed)
        topics = [call.args[1] for call in node.create_subscription.call_args_list]
        self.assertEqual(topics, ['/mouth_pose', '/mouth_opening'])

    def test_switch_off_restores_policy_loading(self):
        node = HybridNode()
        node.VISUAL_ONLY_MODE = False
        args = SimpleNamespace(model_path='arm.pt', device='cpu', policy_backend='torchscript')
        with patch('deploy.run_task.HarlArmPolicy') as policy_class:
            node.load_control_policies(args)
            self.assertEqual(policy_class.call_count, 2)
        self.assertFalse(node.visual_only_mode)
        self.assertTrue(node.learned_safety_enabled)
        self.assertIsNotNone(node.policy)
        self.assertIsNotNone(node.safety_policy)

    def test_starts_tracking_without_fixed_target_or_policy_then_waits_for_vision(self):
        node = self.direct_node()
        before = node.current_joint_pos.copy()
        node.tick()
        self.assertTrue(node.visual_follow_started)
        self.assertEqual(node.fsm_state, 'VISUAL_TRACK')
        self.assertIsNone(node.mouth_pos)
        self.assertIsNone(node.visual_anchor)
        self.assertIsNotNone(node.episode_start_time_ns)
        for _ in range(20):
            node.tick()
        np.testing.assert_array_equal(node.current_joint_pos, before)
        self.assertEqual(node.fsm_state, 'VISUAL_TRACK')
        self.assertEqual(len(node.commands), 1, 'Only initial hold, no motion to the fixed point')
        node.send_sample(gap_mm=5.)
        node.tick()
        np.testing.assert_allclose(node.visual_anchor, node.live_mouth)
        self.assertIsNone(node.policy)

    def test_robot_readiness_is_required_before_startup(self):
        for missing in ('feedback', 'controller', 'tf'):
            with self.subTest(missing=missing):
                node = self.direct_node()
                if missing == 'feedback':
                    node.data_is_stale.return_value = True
                elif missing == 'controller':
                    node.ros_control_is_ready.return_value = False
                else:
                    node.lookup_ee_pose = Mock(return_value=None)
                node.tick()
                self.assertFalse(node.visual_follow_started)
                self.assertEqual(node.commands, [])
                self.assertIsNone(node.episode_start_time_ns)

    def test_follows_from_more_than_25_cm_away_then_enforces_target_travel_limit(self):
        node = self.direct_node()
        spoon, _ = node.spoon_pose()
        node.live_mouth = spoon + node.mouth_axis * 0.30
        node.tick()
        node.send_sample(gap_mm=5.)
        node.tick()
        np.testing.assert_allclose(node.visual_anchor, node.live_mouth)
        for _ in range(540):
            node.send_sample(gap_mm=5.)
            node.tick()
        actual, _ = node.spoon_pose()
        waiting = node.live_mouth - node.mouth_axis * 0.06
        self.assertLess(np.linalg.norm(actual - waiting), np.linalg.norm(spoon - waiting) - 0.10)
        self.assertGreater(len(node.commands), 50, 'Distance from startup must not block visual commands')
        self.assertEqual(node.fsm_state, 'VISUAL_TRACK')
        node.visual_raw_pos = node.visual_anchor + [0., 0., 0.251]
        node.tick()
        self.assertTrue(node.visual_holding)
        self.assertEqual(node.fsm_state, 'VISUAL_TRACK')

    def test_complete_direct_feed_and_withdrawal_uses_no_actors(self):
        node = self.direct_node()
        node.visual_follow_config.insert_depth = -0.008
        node.tick()
        for _ in range(60):
            node.send_sample()
            node.tick()
            if node.fsm_state == 'VISUAL_FEED':
                break
        self.assertEqual(node.fsm_state, 'VISUAL_FEED')
        self.assertEqual(node.visual_feed_insert_depth, -0.008)
        for _ in range(600):
            # Reuse the locked trajectory when the camera is occluded.
            node.tick()
            if node.reset_started:
                break
        self.assertTrue(node.reset_started, node.logger.mock_calls[-10:])
        self.assertEqual(node.episode_stop_reason, 'SUCCESS')
        self.assertIsNone(node.policy)
        self.assertIsNone(node.safety_policy)
        self.assertEqual(len(node.trajectories), 2)
        self.assertGreater(len(node.safety_gain_curve_rows), 0)
        self.assertTrue(all(row['grec'] == 0. for row in node.safety_gain_curve_rows))

    def test_direct_dry_run_publishes_no_joint_motion(self):
        node = self.direct_node(dry_run=True)
        node.live_mouth += [0., 0., 0.02]
        node.tick()
        for _ in range(30):
            node.send_sample(gap_mm=5.)
            node.tick()
        self.assertEqual(node.fsm_state, 'VISUAL_TRACK')
        self.assertEqual(node.commands, [])
        self.assertEqual(node.trajectories, [])

    def test_direct_mode_enforces_withdrawal_limit_without_cli_follow_flag(self):
        with patch.object(ArmPolicyDeployNode, 'VISUAL_ONLY_MODE', True), \
                patch('sys.argv', ['run_task.py', '--withdraw-distance', '0.08']), patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                parse_args()


if __name__ == '__main__':
    unittest.main()
