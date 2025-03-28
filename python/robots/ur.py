import io
from typing import Optional

import numpy as np
import torch

from controllers.policy_controller import PolicyController

class URReachPolicy(PolicyController):
    def __init__(self):
        super().__init__()
        self.dof_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.load_policy(
            "/home/louis/Documents/MyProjects/my_repo/isaaclab_ur_reach_sim2real/sample/ur_reach/ur_reach_policy.pt",
            "/home/louis/Documents/MyProjects/my_repo/isaaclab_ur_reach_sim2real/sample/ur_reach/ur_reach_env.yaml",
        )

        self._action_scale = 0.5
        self._previous_action = np.zeros(6)
        self._policy_counter = 0
        self.target_command = np.array([0.5, 0.0, 0.2, 0.7071, 0.0, 0.7071, 0.0])

        self.has_joint_data = False
        self.current_joint_positions = np.zeros(6)
        self.current_joint_velocities = np.zeros(6)

    def update_joint_state(self, position, velocity):
        self.current_joint_positions = np.array(position[:self.num_joints], dtype=np.float32)
        self.current_joint_velocities = np.array(velocity[:self.num_joints], dtype=np.float32)
        self.has_joint_data = True

    def _compute_observation(self, command):
        if not self.has_joint_data:
            return None
        obs = np.zeros(25)
        obs[:6] = self.current_joint_positions - self.default_pos
        obs[6:12] = self.current_joint_velocities
        obs[12:19] = command
        obs[19:25] = self._previous_action
        return obs

    def forward(self, dt, command):
        if not self.has_joint_data:
            return None
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            if obs is None:
                return None
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

            #### Logging ####
            # print("\n=== Policy Step ===")
            # print(f"{'Command:':<20} {np.round(command, 4)}\n")

            # print("--- Observation ---")
            # print(f"{'Δ Joint Positions:':<20} {np.round(obs[:6], 4)}")
            # print(f"{'Joint Velocities:':<20} {np.round(obs[6:12], 4)}")
            # print(f"{'Command:':<20} {np.round(obs[12:19], 4)}")
            # print(f"{'Previous Action:':<20} {np.round(obs[19:25], 4)}\n")

            # print("--- Action ---")
            # print(f"{'Raw Action:':<20} {np.round(self.action, 4)}")
            # processed_action = self.default_pos + (self.action * self._action_scale)
            # print(f"{'Processed Action:':<20} {np.round(processed_action, 4)}")

        joint_positions = self.default_pos + (self.action * self._action_scale)

        self._policy_counter += 1
        return joint_positions