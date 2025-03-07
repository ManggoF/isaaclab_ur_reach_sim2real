import numpy as np

###############################################################################
# Class: UR
#   Handles transformations between real/Gazebo joint states and the model’s
#   expected ordering, zero offsets, etc.
###############################################################################
class UR:
    def __init__(self):
        """
        Store any offsets, reorder indices, sign flips, etc. for your UR joints.
        For example:
          - self.joint_order_model = [0, 1, 2, 3, 4, 5]
          - self.joint_order_real = [0, 1, 2, 3, 4, 5]
          - self.offsets = [0.0, 0.0, ...]
          - self.scales = [1.0, 1.0, ...]
        Adapt to your real scenario.
        """
        self.num_joints = 6

        # Example: identity ordering (no reordering),
        # but you can change these if your Gazebo vs. model indices differ.
        self.model_joint_indices = [0, 1, 2, 3, 4, 5]  
        self.gazebo_joint_indices = [0, 1, 2, 3, 4, 5]

        # Example offsets or scales if needed
        self.position_offsets = np.zeros(self.num_joints, dtype=np.float32)
        self.velocity_offsets = np.zeros(self.num_joints, dtype=np.float32)
        self.action_offsets   = np.zeros(self.num_joints, dtype=np.float32)
        self.action_scales    = np.ones(self.num_joints,  dtype=np.float32)  # e.g. if you need to scale

    def transform_joint_positions_to_model(self, real_positions: np.ndarray) -> np.ndarray:
        """
        Reorder and offset the real (Gazebo) joint positions into the
        order/zero reference used by the model.
        """
        # Reorder first
        reordered = real_positions[self.gazebo_joint_indices]
        # Apply offsets (example: model might have zero at a different angle)
        transformed = reordered - self.position_offsets
        return transformed

    def transform_joint_velocities_to_model(self, real_velocities: np.ndarray) -> np.ndarray:
        """
        Reorder and offset the real (Gazebo) joint velocities into the
        order used by the model.
        """
        reordered = real_velocities[self.gazebo_joint_indices]
        transformed = reordered - self.velocity_offsets
        return transformed

    def transform_action_to_gazebo(self, model_action: np.ndarray) -> np.ndarray:
        """
        Transform the model's predicted action (joint angles or increments)
        back into the real (Gazebo) ordering (including any offsets if needed).
        """
        # Possibly add offsets or scale
        scaled_action = model_action * self.action_scales + self.action_offsets

        # If the model's joint ordering is different, reorder back to the real joints:
        # We'll create an array of the same size, then fill it.
        real_action = np.zeros(self.num_joints, dtype=np.float32)
        for idx_model, idx_gazebo in enumerate(self.model_joint_indices):
            real_action[idx_gazebo] = scaled_action[idx_model]

        return real_action

    def build_observation(
        self, 
        real_positions: np.ndarray, 
        real_velocities: np.ndarray, 
        target_pose: np.ndarray,
        last_actions: np.ndarray
    ) -> np.ndarray:
        """
        Build the complete observation vector in the order the model expects:
        [ joint_pos(6), joint_vel(6), pose(7), last_actions(6) ] = 25
        Return shape (1, 25).
        """
        # 1) Transform joints into model space
        joint_positions_model = self.transform_joint_positions_to_model(real_positions)
        joint_velocities_model = self.transform_joint_velocities_to_model(real_velocities)

        # 2) For now, we assume pose is used as-is
        #    If needed to reorder or offset the pose (x,y,z,qx,qy,qz,qw), do it here
        pose_model = target_pose  # shape (7,)

        # 3) If last_actions also need reordering, offset, etc.
        #    For example, maybe last_actions are in the real order but the model expects them in a certain order
        #    We'll just pretend last_actions is in the model space for this example
        #    If not, transform them similarly to `transform_action_to_model`.
        last_actions_model = last_actions

        # 4) Concatenate
        obs = np.concatenate([
            joint_positions_model,
            joint_velocities_model,
            pose_model,
            last_actions_model
        ])  # shape (25,)

        # 5) Reshape to (1, 25) for the batch dimension
        return obs.reshape(1, -1).astype(np.float32)