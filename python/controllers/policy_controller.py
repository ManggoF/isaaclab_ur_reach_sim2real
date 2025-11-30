import io
import os
from typing import Optional
import numpy as np
import torch

from utils.config_loader import parse_env_config, get_physics_properties, get_robot_joint_properties

# 定义归一化统计文件路径 (你需要根据实际部署路径设置)
DEPLOY_STATS_DIR = "/home/xry/isaaclab_ur_reach_sim2real/sample/ur_reach/"

class PolicyController:
    """
    A controller that loads and executes a policy from a file.
    """

    def __init__(self) -> None:
        self.obs_mean: Optional[np.ndarray] = None
        self.obs_std: Optional[np.ndarray] = None
        pass

    def load_policy(self, policy_file_path, policy_env_path) -> None:
        """
        Loads policy from a file.
        """
        # --- 1. 加载策略模型 ---
        print("\n=== Policy Loading ===")
        print(f"{'Model path:':<18} {policy_file_path}")
        print(f"{'Environment path:':<18} {policy_env_path}")

        with open(policy_file_path, "rb") as f:
            file = io.BytesIO(f.read())
        self.policy = torch.jit.load(file)

        # --- 2. 加载环境配置 ---

        self.policy_env_params = parse_env_config(policy_env_path)

        self._decimation, self._dt, self.render_interval = get_physics_properties(self.policy_env_params)

        print("\n--- Physics properties ---")
        print(f"{'Decimation:':<18} {self._decimation}")
        print(f"{'Timestep (dt):':<18} {self._dt}")
        print(f"{'Render interval:':<18} {self.render_interval}")

        self._max_effort, self._max_vel, self._stiffness, self._damping, self.default_pos, self.default_vel = get_robot_joint_properties(
            self.policy_env_params, self.dof_names
        )
        self.num_joints = len(self.dof_names)

        print("\n--- Robot joint properties ---")
        print(f"{'Number of joints:':<18} {self.num_joints}")
        print(f"{'Max effort:':<18} {self._max_effort}")
        print(f"{'Max velocity:':<18} {self._max_vel}")
        print(f"{'Stifness:':<18} {self._stiffness}")
        print(f"{'Damping:':<18} {self._damping}")
        print(f"{'Default position:':<18} {self.default_pos}")
        print(f"{'Default velocity:':<18} {self.default_vel}")

        # --- 3. 加载归一化统计量 (新增) ---
        mean_path = os.path.join(DEPLOY_STATS_DIR, 'obs_mean.npy')
        std_path = os.path.join(DEPLOY_STATS_DIR, 'obs_std.npy')

        try:
            self.obs_mean = np.load(mean_path).astype(np.float32)
            self.obs_std = np.load(std_path).astype(np.float32)
            
            # 避免除以零，将过小的标准差设为1.0
            # self.obs_std[np.where(self.obs_std == 0)] = 1.0 
            
            print(f"\n✅ Normalization stats loaded. Dim: {self.obs_mean.shape[0]}")
        except Exception as e:
            print(f"\n❌ ERROR: Failed to load normalization stats from {DEPLOY_STATS_DIR}. {e}")
            raise RuntimeError("Missing or corrupt normalization files.")
        print("\n=== Policy Loaded ===\n")

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Computes the action from the observation using the loaded policy.

        Args:
            obs (np.ndarray): The observation.

        Returns:
            np.ndarray: The action.
        """
        if self.obs_mean is None or self.obs_std is None:
            raise RuntimeError("Normalization stats not loaded before computing action.")
            
        # --- 归一化 (新增) ---
        obs_norm = (obs - self.obs_mean) / (self.obs_std + 0.01)

        # !!! 立即打印归一化后的数值 !!!
        # 检查是否有任何数值超过 5.0 或 10.0
        print(f"--- Normalized Obs Max Value: {np.max(np.abs(obs_norm)):.2f} ---")
        print(f"Normalized Δq (D0-5): {np.round(obs_norm[:6], 2)}")
        print(f"Normalized q_dot (D6-11): {np.round(obs_norm[6:12], 2)}")
        print(f"Normalized Command (D12-18): {np.round(obs_norm[12:19], 2)}") # 重点观察这个！
        print(f"Normalized A_prev (D19-24): {np.round(obs_norm[19:25], 2)}")

        with torch.no_grad():
            obs = torch.from_numpy(obs_norm).view(1, -1).float()
            action = self.policy(obs).detach().view(-1).numpy()
        return action

    def _compute_observation(self) -> NotImplementedError:
        """
        Computes the observation. Not implemented.
        """

        raise NotImplementedError(
            "Compute observation need to be implemented, expects np.ndarray in the structure specified by env yaml"
        )

    def forward(self) -> NotImplementedError:
        """
        Forwards the controller. Not implemented.
        """
        raise NotImplementedError(
            "Forward needs to be implemented to compute and apply robot control from observations"
        )