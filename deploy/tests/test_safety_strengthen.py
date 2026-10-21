"""Exercise the real safety filters and the latched retreat/reset without ROS I/O."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from deploy.run_task import ArmPolicyDeployNode
from deploy.tests.test_success_withdraw import OfflineDeployNode, TimedTrajectoryDeployNode


def configure_tracking(node, backend="learned", hazard=True, force_one=True):
    node.safety_strenthen = True
    node.safety_grec_force_one = force_one
    node.args.episode_length_s = 120.0
    node.args.min_spoon_z = 0.04
    node.args.max_mouth_spoon_distance = 0.0
    node.args.action_limit = 3.0
    node.args.action_scale_arm = 0.008
    node.args.debug_wrench = False
    node.args.debug_obs = False
    node.max_episode_steps = 3600
    # Start 3 cm before the mouth: no feeding success, and a reachable 10 cm retreat.
    spoon, _ = node.spoon_pose()
    node.mouth_pos = spoon + node.mouth_axis * 0.03
    node.mouth_quat_wxyz = node.mouth_quat_from_policy_axis(node.mouth_axis)
    node.learned_safety_enabled = backend == "learned"
    node.rule_safety_enabled = backend == "rule"
    node.wrench_sign = -1.0
    node.wrist_wrench_policy_obs = np.array([1.6 if hazard else 0.0, 0, 0, 0, 0, 0])
    node.get_sim_aligned_safety_wrench_raw = Mock(return_value=np.array([20.0 if hazard else 0.0, 0, 0, 0, 0, 0]))
    node.compute_arm_observation = Mock(return_value=np.zeros(node.ARM_POLICY_OBS_DIM))
    node.compute_wrist_wrench_observation = Mock(side_effect=lambda: node.wrist_wrench_policy_obs)
    node.record_wrench_sample = Mock()
    node.policy.act = Mock(return_value=np.zeros(6))
    node.safety_policy.act = Mock(return_value=np.zeros(3))


class SafetyStrengthenTest(unittest.TestCase):
    def test_both_safety_backends_immediately_send_one_ten_cm_trajectory(self):
        for backend in ("learned", "rule"):
            with self.subTest(backend=backend):
                node = OfflineDeployNode()
                configure_tracking(node, backend)
                node.args.withdraw_distance = 0.16  # Safety retreat stays at exactly 10 cm.
                start_spoon, _ = node.spoon_pose()
                node.tick()
                self.assertTrue(node.safety_strengthen_active)
                self.assertEqual(node.fsm_state, "SAFETY_WITHDRAW")
                self.assertEqual(node.withdrawal_reason, "SAFETY")
                self.assertEqual(node.withdraw_distance_m, 0.10)
                self.assertEqual(node.success_hold_steps, 0)
                self.assertFalse(node.reset_started)
                self.assertEqual(len(node.commands), 1, "Discard the policy's incremental command")
                self.assertEqual(len(node.trajectories), 1)
                self.assertGreater(len(node.trajectories[0].points), 2)
                spoon, _ = node.spoon_pose()
                self.assertLess(np.linalg.norm(spoon - start_spoon + node.mouth_axis * 0.10), 0.0001)
                self.assertTrue(all(row["grec"] == 1.0 for row in node.safety_gain_curve_rows))
                self.assertTrue(all(row["mode"] == "SAFETY_WITHDRAW" for row in node.safety_gain_curve_rows))

    def test_switch_off_keeps_existing_incremental_retreat(self):
        for backend in ("learned", "rule"):
            with self.subTest(backend=backend):
                node = OfflineDeployNode()
                configure_tracking(node, backend)
                node.safety_strenthen = False
                node.tick()
                self.assertGreater(node.rule_safety_last_state["retreat_gain"], 0.0)
                self.assertEqual(node.rule_safety_last_state["mode"], "RETREAT")
                self.assertFalse(node.safety_strengthen_active)
                self.assertEqual(node.fsm_state, "APPROACH")
                self.assertEqual(len(node.commands), 1)
                self.assertEqual(node.trajectories, [])
                self.assertFalse(node.reset_started)

    def test_raw_actor_retreat_output_does_not_trigger_without_effective_grec(self):
        node = OfflineDeployNode()
        configure_tracking(node, hazard=False)
        node.safety_policy.act.return_value = np.ones(3)
        node.tick()
        self.assertGreater(node.rule_safety_last_state["slow_gain"], 0.0)
        self.assertEqual(node.rule_safety_last_state["retreat_gain"], 0.0)
        self.assertFalse(node.safety_strengthen_active)
        self.assertEqual(node.fsm_state, "APPROACH")
        self.assertEqual(node.trajectories, [])

    def test_latched_withdrawal_and_reset_finish_after_risk_clears(self):
        for backend, force_one in (("learned", True), ("rule", True), ("learned", False), ("rule", False)):
            with self.subTest(backend=backend, force_one=force_one):
                node = TimedTrajectoryDeployNode(scaling=0.55, interpolation="cubic")
                node.current_joint_pos = np.array([0.19602, -0.93687, 1.43608, -0.58323, 0.76631, 0.06622])
                configure_tracking(node, backend, force_one=force_one)
                node.RESET_AFTER_SUCCESS = False  # Strengthened safety still completes its reset.
                node.tick()
                node.policy.act.reset_mock()
                node.safety_policy.act.reset_mock()
                node.wrist_wrench_policy_obs.fill(0.0)
                node.get_sim_aligned_safety_wrench_raw.return_value = np.zeros(6)
                node.safety_policy.act.return_value = -np.ones(3)
                for _ in range(1800):
                    node.tick()
                    if node.reset_finished:
                        break
                    if force_one:
                        self.assertEqual(node.rule_safety_last_state["grec"], 1.0)
                    else:
                        self.assertEqual(node.rule_safety_last_state["grec"], node.rule_safety_last_state["retreat_gain"])
                    node.save_joint_angle_curve.assert_not_called()
                    node.save_safety_gain_curve_outputs.assert_not_called()
                self.assertTrue(node.success_withdraw_complete)
                self.assertTrue(node.reset_finished)
                self.assertEqual(node.fsm_state, "DONE")
                self.assertEqual(node.episode_stop_reason, "SAFETY")
                self.assertFalse(node.safety_strengthen_active)
                self.assertEqual(len(node.trajectories), 2)
                rows = node.safety_gain_curve_rows
                self.assertTrue(any(row["mode"] == "SAFETY_RESET" for row in rows))
                if force_one:
                    self.assertTrue(all(row["grec"] == 1.0 for row in rows[:-1]))
                else:
                    self.assertTrue(any(row["grec"] == 0.0 for row in rows[:-1]))
                self.assertEqual(rows[-1]["grec"], 0.0)
                self.assertLess(np.max(np.abs(node.current_joint_pos - node.DEFAULT_JOINT_POS)), node.RESET_GOAL_TOL_RAD)
                node.policy.act.assert_not_called()
                if backend == "learned" and not force_one:
                    self.assertGreater(node.safety_policy.act.call_count, 1)
                else:
                    node.safety_policy.act.assert_not_called()
                node.save_safety_gain_curve_outputs.assert_called_once_with("SAFETY")
                node.save_joint_angle_curve.assert_called_once()

    def test_stalled_strengthened_withdrawal_cannot_reset(self):
        node = OfflineDeployNode(follow_commands=False)
        configure_tracking(node)
        node.tick()
        for _ in range(30):
            node.tick()
        self.assertEqual(len(node.trajectories), 1)
        self.assertFalse(node.reset_started)
        node.now_ns += round(node.SUCCESS_WITHDRAW_TIMEOUT_S * 1e9)
        node.tick()
        self.assertTrue(node.episode_stopped)
        self.assertEqual(node.episode_stop_reason, "FAILURE")
        self.assertFalse(node.success_withdraw_complete)
        self.assertFalse(node.reset_started)

    def test_dry_run_strengthen_records_trigger_without_motion(self):
        node = OfflineDeployNode(dry_run=True)
        configure_tracking(node)
        node.tick()
        self.assertTrue(node.episode_stopped)
        self.assertEqual(node.episode_stop_reason, "SAFETY")
        self.assertEqual(node.commands, [])
        self.assertEqual(node.trajectories, [])
        self.assertFalse(node.reset_started)

    def test_unsafe_strengthened_goal_stops_without_reset(self):
        node = OfflineDeployNode()
        node.safety_strenthen = True
        state = node.default_rule_safety_state()
        state["retreat_gain"] = 0.4
        node.publish_safety_state(state)
        node.maybe_start_safety_strengthen(np.array([0.5, -0.645, 0.5]), np.array([1.0, 0, 0, 0]), state)
        self.assertTrue(node.episode_stopped)
        self.assertEqual(node.episode_stop_reason, "FAILURE")
        self.assertFalse(node.reset_started)
        self.assertEqual(node.trajectories, [])

    def test_strengthened_withdrawal_uses_latched_axis_when_vision_is_stale(self):
        node = OfflineDeployNode(follow_commands=False)
        configure_tracking(node)
        node.tick()
        direction = node.success_withdraw_direction.copy()
        node.args.fixed_mouth_pose = False
        self.assertFalse(ArmPolicyDeployNode.data_is_stale(node))
        node.mouth_axis = np.array([1.0, 0.0, 0.0])
        node.tick()
        self.assertTrue(np.allclose(node.success_withdraw_direction, direction))
        self.assertEqual(len(node.trajectories), 1)

    def test_original_output_tracks_live_actor_gains_and_preserves_normal_limits(self):
        node = OfflineDeployNode(follow_commands=False)
        configure_tracking(node, force_one=False)
        node.tick()
        self.assertEqual(node.rule_safety_last_state["grec"], 0.5)
        node.safety_policy.act.return_value = np.array([0.0, 0.6, 0.0])
        node.tick()
        self.assertAlmostEqual(node.rule_safety_last_state["grec"], 0.8, places=6)
        previous_obs = node.safety_policy.act.call_args.args[0]
        self.assertTrue(np.allclose(previous_obs[-3:], np.zeros(3)))
        node.safety_policy.act.return_value = np.array([0.0, -1.0, 0.0])
        node.tick()
        self.assertEqual(node.rule_safety_last_state["grec"], node.SAFETY_POLICY_RETRACT_GAIN_MIN)
        previous_obs = node.safety_policy.act.call_args.args[0]
        self.assertTrue(np.allclose(previous_obs[-3:], [0.0, 0.6, 0.0]))
        self.assertEqual(node.rule_safety_last_state["retreat_gain"], node.SAFETY_POLICY_RETRACT_GAIN_MIN)
        self.assertEqual(len(node.trajectories), 1, "Diagnostic safety outputs cannot replace the locked trajectory")
        self.assertEqual(len(node.commands), 1)
        node.policy.act.assert_called_once()
        self.assertTrue(node.safety_strengthen_active)

    def test_force_one_changes_recording_without_overwriting_trigger_gain(self):
        node = OfflineDeployNode(follow_commands=False)
        configure_tracking(node, force_one=True)
        node.tick()
        self.assertEqual(node.safety_gain_curve_rows[0]["grec"], 1.0)
        self.assertEqual(len(node.trajectories), 1)
        # A record-only update with no retreat output must not fabricate a hazard.
        state = node.default_rule_safety_state()
        node.publish_safety_state(state)
        self.assertEqual(state["retreat_gain"], 0.0)
        self.assertEqual(state["grec"], 1.0)
        self.assertEqual(node.safety_gain_curve_rows[-1]["grec"], 1.0)

    def test_recording_switch_has_no_effect_when_strengthening_is_disabled(self):
        for force_one in (False, True):
            with self.subTest(force_one=force_one):
                node = OfflineDeployNode()
                configure_tracking(node, force_one=force_one)
                node.safety_strenthen = False
                node.tick()
                self.assertFalse(node.safety_strengthen_active)
                self.assertEqual(node.rule_safety_last_state["grec"], 0.5)
                self.assertEqual(node.rule_safety_last_state["retreat_gain"], 0.5)

    def test_original_output_does_not_fabricate_zero_at_reset_completion(self):
        node = OfflineDeployNode(follow_commands=False)
        node.safety_strenthen = True
        node.safety_grec_force_one = False
        node.safety_strengthen_active = True
        node.success_withdraw_complete = True
        node.episode_stopped = True
        node.episode_stop_reason = "SAFETY"
        node.rule_safety_last_state["retreat_gain"] = 0.6
        node.update_strengthened_safety_state = Mock()
        node.current_joint_pos = node.DEFAULT_JOINT_POS.copy()
        node.start_reset_after_success()
        node.reset_reached_since_ns = node.now_ns - round(node.RESET_SETTLE_TIME_S * 1e9)
        node.tick()
        self.assertTrue(node.reset_finished)
        self.assertEqual(node.safety_gain_curve_rows[-1]["grec"], 0.6)
        self.assertEqual(node.safety_gain_curve_rows[-1]["mode"], "DONE")

    def test_curve_export_uses_grec_column_and_selected_recording_values(self):
        node = OfflineDeployNode(follow_commands=False)
        configure_tracking(node, force_one=False)
        node.tick()
        node.safety_gain_curve_saved = False
        node.plot_safety_gain_curve_matplotlib = Mock(return_value=True)
        node.plot_safety_gain_curve_cv2 = Mock(return_value=False)
        with TemporaryDirectory() as directory:
            with patch("deploy.run_task.SAFETY_GAIN_VIS_DIR", Path(directory)):
                ArmPolicyDeployNode.save_safety_gain_curve_outputs(node, "SAFETY")
            csv_path = next(Path(directory).glob("*.csv"))
            with csv_path.open() as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames[2:5], ["gslow", "glat", "grec"])
                records = list(reader)
            self.assertTrue(all(float(row["grec"]) == 0.5 for row in records))


if __name__ == "__main__":
    unittest.main()
