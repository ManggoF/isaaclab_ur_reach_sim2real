"""Verify final measured alignment and axial insertion without any ROS connection."""

import math
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation as R

from deploy.run_task import ArmPolicyDeployNode
from deploy.tests import test_visual_only as visual_only_tests
from deploy.visual_mouth_follow import VisualFollowConfig


class VisualFeedAlignmentTest(unittest.TestCase):
    def stationary_node(self, yaw_deg=0., lateral_m=0.):
        node = visual_only_tests.VisualOnlyTest().direct_node()
        spoon, _ = node.spoon_pose()
        node.mouth_axis = R.from_rotvec([0., 0., math.radians(yaw_deg)]).apply(node.mouth_axis)
        side = np.cross([0., 0., 1.], node.mouth_axis)
        node.live_mouth = spoon + node.mouth_axis * 0.06 + side * lateral_m
        node.follow_commands = False
        node.begin_tracking()
        return node

    def test_open_and_centered_spoon_cannot_feed_with_wrong_axis(self):
        for yaw in (-20., 5., 20.):
            with self.subTest(yaw=yaw):
                node = self.stationary_node(yaw_deg=yaw)
                for _ in range(35):
                    node.send_sample()
                    node.tick()
                self.assertTrue(node.visual_open)
                self.assertEqual(node.fsm_state, 'VISUAL_TRACK')
                self.assertIsNone(node.visual_feed_axis)
                self.assertFalse(node.visual_feed_trajectory_sent)
                self.assertEqual(node.trajectories, [])

    def test_three_mm_lateral_offset_requires_centering_even_with_matching_axis(self):
        node = self.stationary_node(lateral_m=0.0032)
        for _ in range(35):
            node.send_sample()
            node.tick()
        spoon, quat = node.spoon_pose()
        angle, lateral = node.visual_axis_alignment(
            spoon, R.from_quat(quat[[1, 2, 3, 0]]), node.corrected_visual_mouth(), node.visual_mouth_axis,
        )
        self.assertLess(angle, math.radians(1.))
        self.assertGreater(lateral, 0.003)
        self.assertLess(lateral, node.visual_follow_config.align_tolerance)
        self.assertEqual(node.fsm_state, 'VISUAL_TRACK')
        self.assertEqual(node.trajectories, [])

    def test_latest_visual_axis_blocks_feed_when_smoothed_axis_lags_head_turn(self):
        node = self.stationary_node()
        node.send_sample()
        node.tick()
        node.visual_follow_config.filter_alpha = 0.  # Isolate lagging tracking orientation.
        turned = R.from_rotvec([0., 0., math.radians(6.)]).apply(node.mouth_axis)
        for _ in range(35):
            node.send_sample(axis=turned)
            node.tick()
        spoon, quat = node.spoon_pose()
        rot = R.from_quat(quat[[1, 2, 3, 0]])
        tracked_error, _ = node.visual_axis_alignment(spoon, rot, node.live_mouth, node.visual_mouth_axis)
        latest_error, _ = node.visual_axis_alignment(spoon, rot, node.live_mouth, node.visual_raw_axis)
        self.assertLess(tracked_error, math.radians(1.))
        self.assertGreater(latest_error, math.radians(5.))
        self.assertEqual(node.fsm_state, 'VISUAL_TRACK')
        self.assertEqual(node.trajectories, [])

    def test_repeated_timer_ticks_cannot_confirm_alignment_from_one_visual_frame(self):
        node = self.stationary_node()
        # Isolate alignment confirmation from the independent opening debounce.
        node.update_visual_open_debounce = Mock(return_value=True)
        node.send_sample()
        for _ in range(10):
            node.tick()
        self.assertGreaterEqual(node.visual_aligned_steps, 5)
        self.assertEqual(node.visual_align_samples, 1)
        self.assertEqual(node.fsm_state, 'VISUAL_TRACK')
        for _ in range(15):
            node.send_sample()
            node.tick()
            if node.fsm_state == 'VISUAL_FEED':
                break
        self.assertEqual(node.fsm_state, 'VISUAL_FEED')
        self.assertGreaterEqual(node.visual_align_samples, 3)
        self.assertGreaterEqual(
            (node.now_ns - node.visual_align_since_ns) * 1e-9,
            node.visual_follow_config.feed_align_hold_s,
        )

    def test_wrist_finishes_alignment_before_feed_and_remains_free(self):
        node = self.stationary_node(yaw_deg=20.)
        node.follow_commands = True
        states = []
        for _ in range(400):
            node.send_sample()
            node.tick()
            states.append(node.fsm_state)
            if node.fsm_state == 'VISUAL_FEED':
                break
        self.assertIn('VISUAL_TRACK', states)
        self.assertEqual(node.fsm_state, 'VISUAL_FEED', node.logger.mock_calls[-5:])
        spoon, quat = node.spoon_pose()
        rot = R.from_quat(quat[[1, 2, 3, 0]])
        angle, lateral = node.visual_axis_alignment(spoon, rot, node.visual_feed_mouth, node.visual_feed_axis)
        self.assertLessEqual(angle, math.radians(1.))
        self.assertLessEqual(lateral, 0.002)
        self.assertLess((node.level_spoon_rotation(node.visual_feed_axis) * rot.inv()).magnitude(), math.radians(1.))
        self.assertAlmostEqual(node.ik.lower_limits[4], -2. * math.pi)
        self.assertAlmostEqual(node.ik.upper_limits[4], 2. * math.pi)

    def test_small_accepted_lateral_error_does_not_make_insertion_oblique(self):
        node = visual_only_tests.VisualOnlyTest().direct_node()
        node.live_mouth += np.cross([0., 0., 1.], node.mouth_axis) * 0.0015
        node.follow_commands = False  # Retain a known, acceptable centering error.
        node.reach_feeding()
        start = node.visual_feed_start_spoon.copy()
        axis = node.visual_feed_axis.copy()
        target = node.visual_feed_target()
        travel = target - start
        self.assertLess(np.linalg.norm(travel - float(travel @ axis) * axis), 1e-7)
        self.assertAlmostEqual(float((target - node.visual_feed_mouth) @ axis), node.visual_feed_insert_depth)
        self.assertAlmostEqual(target[2], start[2])
        self.assertGreater(np.linalg.norm(np.cross(start - node.visual_feed_mouth, axis)), 0.0001)
        node.tick()  # Locked feed is valid even without another visible mouth frame.
        self.assertTrue(node.visual_feed_trajectory_sent)
        self.assertEqual(len(node.trajectories), 1)
        ros_frame = R.from_rotvec([0., 0., math.pi])
        locked_rot = R.from_quat(node.visual_feed_quat[[1, 2, 3, 0]])
        for point in node.trajectories[0].points:
            pos, rot = node.ik.fk(np.asarray(point.positions))
            actual_rot = ros_frame * rot
            spoon = ros_frame.apply(pos) + actual_rot.apply(node.SPOON_OFFSET)
            delta = spoon - start
            self.assertLess(np.linalg.norm(delta - float(delta @ axis) * axis), 0.0002)
            self.assertLess((locked_rot * actual_rot.inv()).magnitude(), math.radians(0.05))

    def test_alignment_is_rechecked_before_dispatch_and_drift_returns_to_tracking(self):
        for drift in ('angle', 'position', 'velocity'):
            with self.subTest(drift=drift):
                node = visual_only_tests.VisualOnlyTest().direct_node()
                node.reach_feeding()
                spoon, quat = node.spoon_pose()
                rot = R.from_quat(quat[[1, 2, 3, 0]])
                if drift == 'angle':
                    rot = R.from_rotvec([0., 0., math.radians(5.)]) * rot
                elif drift == 'position':
                    spoon += np.cross([0., 0., 1.], node.visual_feed_axis) * 0.004
                else:
                    node.current_joint_vel[0] = 0.04
                ee = spoon - rot.apply(node.SPOON_OFFSET)
                node.lookup_ee_pose = Mock(return_value=(ee, rot.as_quat()[[3, 0, 1, 2]]))
                node.tick()
                self.assertEqual(node.fsm_state, 'VISUAL_TRACK')
                self.assertFalse(node.visual_feed_trajectory_sent)
                self.assertFalse(node.safety_strengthen_active)
                self.assertFalse(node.reset_started)
                self.assertEqual(node.trajectories, [])

    def test_invalid_final_alignment_settings_are_rejected(self):
        node = self.stationary_node()
        for key, values in (
            ('VISUAL_FEED_AXIS_TOL_DEG', (0., 5.01, float('nan'))),
            ('VISUAL_FEED_LATERAL_TOL_M', (0., 0.0081, float('inf'))),
            ('VISUAL_FEED_ALIGN_HOLD_S', (0., 2.01, float('nan'))),
        ):
            for value in values:
                with self.subTest(key=key, value=value), patch.object(ArmPolicyDeployNode, key, value):
                    with self.assertRaises(ValueError):
                        VisualFollowConfig.from_args(node.args, ArmPolicyDeployNode)


if __name__ == '__main__':
    unittest.main()
