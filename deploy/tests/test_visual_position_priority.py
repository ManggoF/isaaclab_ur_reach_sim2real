"""Position-priority control tests using FK feedback, never a live robot."""

import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation as R

from deploy.run_task import ArmPolicyDeployNode
from deploy.tests import test_visual_only as visual_only_tests
from deploy.tests.test_visual_mouth_follow import HybridNode
from deploy.tests.test_success_withdraw import TimedTrajectoryDeployNode
from deploy.visual_mouth_follow import VisualFollowConfig


class VisualPositionPriorityTest(unittest.TestCase):
    def singular_start(self, target=(0.825, 0.252, 0.345), axis=(0.7, 0.71414284, 0.)):
        node = visual_only_tests.VisualOnlyTest().direct_node()
        node.current_joint_pos = node.DEFAULT_JOINT_POS.astype(float).copy()
        node.mouth_axis = np.asarray(axis, dtype=float)
        node.mouth_axis /= np.linalg.norm(node.mouth_axis)
        node.live_mouth = np.asarray(target, dtype=float)
        node.visual_follow_config.insert_depth = -0.008
        return node

    def test_fifth_joint_is_unrestricted_by_old_fifteen_degree_bounds(self):
        node = self.singular_start()
        node.ik.lower_limits[4], node.ik.upper_limits[4] = np.deg2rad([-15., 15.])
        node.tick()
        self.assertTrue(node.visual_position_priority_enabled())
        self.assertAlmostEqual(node.ik.lower_limits[4], -2. * math.pi)
        self.assertAlmostEqual(node.ik.upper_limits[4], 2. * math.pi)

    def test_other_modes_and_code_switch_keep_the_original_controller(self):
        for pure, flag, enabled in ((False, False, True), (False, True, True),
                                    (True, True, True), (True, False, False)):
            with self.subTest(pure=pure, flag=flag, enabled=enabled):
                node = visual_only_tests.VisualOnlyTest().direct_node()
                node.visual_only_mode = pure
                node.args.visual_follow_after_fixed = flag
                node.VISUAL_ONLY_POSITION_PRIORITY = enabled
                with patch.object(node.ik, 'joint_target_from_spoon_error', side_effect=AssertionError('Wrong mode')) as new_ik, \
                        patch.object(node.ik, 'joint_target_from_delta', wraps=node.ik.joint_target_from_delta) as original_ik:
                    node.begin_tracking()
                    node.send_sample(gap_mm=5.)
                    node.tick()
                new_ik.assert_not_called()
                original_ik.assert_called_once()

    def test_fixed_targets_converge_from_wrist_singularity_without_backward_steps(self):
        cases = [((0.825, 0.252, 0.345), (0.7, 0.71414284, 0.)),
                 ((0.85, 0.55, 0.26), (0.25625, 0.96661, 0.)),
                 ((0.7, 0.4, 0.3), (0.6, 0.8, 0.))]
        for target, axis in cases:
            with self.subTest(target=target):
                node = self.singular_start(target, axis)
                node.tick()
                waiting = node.live_mouth - node.mouth_axis * node.visual_follow_config.standoff
                errors, wrists = [], []
                for _ in range(900):
                    previous = node.current_joint_pos.copy()
                    node.send_sample(gap_mm=5.)
                    node.tick()
                    spoon, quat = node.spoon_pose()
                    errors.append(np.linalg.norm(spoon - waiting))
                    wrists.append(node.current_joint_pos[4])
                    self.assertLessEqual(np.max(np.abs(node.current_joint_pos - previous)), node.args.max_joint_step + 1e-8)
                    self.assertTrue(np.all(np.abs(node.current_joint_pos) <= 2. * math.pi))
                self.assertLess(errors[-1], 0.0021)
                self.assertLessEqual(max(np.diff(errors)), 1e-7)
                self.assertGreater(max(np.abs(wrists)), math.radians(15.))
                rot = R.from_quat(quat[[1, 2, 3, 0]])
                angle = (node.level_spoon_rotation(node.mouth_axis) * rot.inv()).magnitude()
                self.assertLess(angle, math.radians(0.6))
                self.assertEqual(node.fsm_state, 'VISUAL_TRACK')

    def test_small_vision_noise_does_not_restart_joint_corrections(self):
        node = visual_only_tests.VisualOnlyTest().direct_node()
        node.begin_tracking()
        original = node.current_joint_pos.copy()
        for index in range(90):
            sign = -1. if index % 2 else 1.
            axis = R.from_rotvec([0., 0., sign * math.radians(0.2)]).apply(node.mouth_axis)
            node.send_sample(gap_mm=5., position=node.live_mouth + [sign * 0.0008, 0.0003, 0.], axis=axis)
            node.tick()
        np.testing.assert_array_equal(node.current_joint_pos, original)
        self.assertEqual(node.fsm_state, 'VISUAL_TRACK')

    def test_position_priority_still_requires_opening_and_alignment_then_completes_withdrawal(self):
        node = self.singular_start()
        node.tick()
        for _ in range(900):
            node.send_sample(gap_mm=5.)
            node.tick()
        self.assertEqual(node.fsm_state, 'VISUAL_TRACK')
        for _ in range(60):
            node.send_sample(gap_mm=15.)
            node.tick()
            if node.fsm_state == 'VISUAL_FEED':
                break
        self.assertEqual(node.fsm_state, 'VISUAL_FEED')
        self.assertAlmostEqual(node.visual_feed_insert_depth, -0.008)
        # Locked feeding and withdrawal still finish when the camera is occluded.
        for _ in range(600):
            node.tick()
            if node.reset_started or node.episode_stopped:
                break
        self.assertTrue(node.reset_started, node.logger.mock_calls[-8:])
        self.assertEqual(node.episode_stop_reason, 'SUCCESS')
        self.assertEqual(len(node.trajectories), 2)
        self.assertIsNone(node.policy)
        self.assertIsNone(node.safety_policy)

    def test_scaled_delayed_controller_keeps_approaching(self):
        class TimedNode(HybridNode, TimedTrajectoryDeployNode):
            publish_joint_target = TimedTrajectoryDeployNode.publish_joint_target
            receive_trajectory = TimedTrajectoryDeployNode.receive_trajectory
            tick = TimedTrajectoryDeployNode.tick

        node = TimedNode()
        node.follow_commands = False
        node.scaling = node.speed_scaling = 0.55
        node.interpolation = 'cubic'
        node.trajectory_target = node.trajectory_start = None
        node.trajectory_elapsed = node.trajectory_duration = node.dispatch_delay = 0.
        node.visual_only_mode = True
        node.policy = node.safety_policy = None
        node.wrist_wrench_policy_obs = np.zeros(6)
        node.current_joint_pos = node.DEFAULT_JOINT_POS.astype(float).copy()
        node.mouth_axis = np.array([0.7, 0.71414284, 0.])
        node.mouth_axis /= np.linalg.norm(node.mouth_axis)
        node.live_mouth = np.array([0.825, 0.252, 0.345])
        node.begin_tracking()
        errors = []
        for _ in range(1500):
            node.send_sample(gap_mm=5.)
            node.tick()
            spoon, _ = node.spoon_pose()
            errors.append(np.linalg.norm(spoon - (node.live_mouth - node.mouth_axis * 0.06)))
        self.assertLess(errors[-1], 0.003)
        self.assertLessEqual(max(np.diff(errors)), 1e-6)
        self.assertEqual(node.fsm_state, 'VISUAL_TRACK')

    def test_deadbands_cannot_prevent_existing_feed_alignment(self):
        node = visual_only_tests.VisualOnlyTest().direct_node()
        for key, value in (('VISUAL_ONLY_POSITION_DEADBAND_M', 0.),
                           ('VISUAL_ONLY_POSITION_DEADBAND_M', 0.0031),
                           ('VISUAL_ONLY_POSITION_DEADBAND_M', float('nan')),
                           ('VISUAL_ONLY_ANGLE_DEADBAND_DEG', 1.01),
                           ('VISUAL_ONLY_ANGLE_DEADBAND_DEG', float('inf'))):
            with self.subTest(key=key, value=value), patch.object(ArmPolicyDeployNode, key, value):
                with self.assertRaises(ValueError):
                    VisualFollowConfig.from_args(node.args, ArmPolicyDeployNode)


if __name__ == '__main__':
    unittest.main()
