import numpy as np
from controllers.policy_controller import PolicyController


class URReachPolicy(PolicyController):
    """Policy controller for UR Reach using a pre-trained policy model."""

    def __init__(self) -> None:
        """Initialize the URReachPolicy instance."""
        super().__init__()
        self.current_tcp_position = np.zeros(3) # 新增一个变量来存储TCP位置
        self.dof_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        # Load the pre-trained policy model and environment configuration
        # YOU NEED TO CHANGE THE PATH
        self.load_policy(
            "/home/xry/isaaclab_ur_reach_sim2real/sample/ur_reach/model_1499_deploy.pt",
            "/home/xry/isaaclab_ur_reach_sim2real/sample/ur_reach/ur_reach_env.yaml",
        )

        self._action_scale = 0.5
        self._previous_action = np.zeros(6)
        self._policy_counter = 0
        # self.target_command = np.array([0.5, 0.4, 0.3, 0.7071, 0.0, 0.0, 0.7071]) # x, y, z, qx, qy, qz, qw
        self.has_joint_data = False
        self.current_joint_positions = np.zeros(6)
        self.current_joint_velocities = np.zeros(6)

    def update_tcp_position(self, tcp_position: np.ndarray) -> None:
        """从外部更新TCP的位置"""
        self.current_tcp_position = tcp_position

    def update_joint_state(self, position, velocity) -> None:
        """
        Update the current joint state.

        Args:
            position: A list or array of joint positions.
            velocity: A list or array of joint velocities.
        """
        self.current_joint_positions = np.array(position[:self.num_joints], dtype=np.float32)
        self.current_joint_velocities = np.array(velocity[:self.num_joints], dtype=np.float32)
        self.has_joint_data = True

    def _compute_observation(self, command: np.ndarray) -> np.ndarray:
        """
        Compute the observation vector for the policy network.

        Args:
            command: The target command vector. 即人脸位姿识别计算出的命令

        Returns:
            An observation vector if joint data is available, otherwise None.
        """
        if not self.has_joint_data:
            return None
        obs = np.zeros(25)
        obs[:6] = self.current_joint_positions - self.default_pos   #调试看看和仿真里能不能对应上
        obs[6:12] = self.current_joint_velocities
        obs[12:19] = command
        # 使用我们刚刚更新的、真实的TCP位置！
        # ball_position = self.current_tcp_position
        # obs[19:22] = ball_position
        obs[19:25] = self._previous_action
        return obs

    def forward(self, dt: float, command: np.ndarray) -> np.ndarray:
        """
        Compute the next joint positions based on the policy.

        Args:
            dt: Time step for the forward pass.
            command: The target command vector. 即人脸位姿识别计算出的命令

        Returns:
            The computed joint positions if joint data is available, otherwise None.
        """
        if not self.has_joint_data:
            return None

        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            if obs is None:
                return None
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

            # Debug Logging (commented out)
            # print("\n=== Policy Step ===")
            # print(f"{'Command:':<20} {np.round(command, 4)}\n")
            # print("--- Observation ---")
            # print(f"{'Δ Joint Positions:':<20} {np.round(obs[:6], 4)}")
            # print(f"{'Joint Velocities:':<20} {np.round(obs[6:12], 4)}")
            # print(f"{'Command:':<20} {np.round(obs[12:19], 4)}")
            # print(f"{'Previous Action:':<20} {np.round(obs[19:25], 4)}\n")
            # print("--- Action ---")
            # print(f"{'Default_pos:':<20} {np.round(self.default_pos, 4)}")
            # print(f"{'Raw Action:':<20} {np.round(self.action, 4)}")
            # processed_action = self.default_pos + (self.action * self._action_scale)
            # print(f"{'Processed Action:':<20} {np.round(processed_action, 4)}")

            # 辅助变量：弧度转角度的因子
            r2d = 180 / np.pi

            print("\n=== Policy Step (单位: 度 °) ===")
            # Command 人脸位姿识别计算出的命令
            print(f"{'Command:':<20} {np.round(command, 4)}\n")

            print("--- Observation ---")
            print(f"{'Current_pos:':<20} {np.round(np.array(self.current_joint_positions) * r2d, 2)}")
            print(f"{'Default_pos:':<20} {np.round(np.array(self.default_pos) * r2d, 2)}")
            print(f"{'Δ Joint Positions:':<20} {np.round(obs[:6] * r2d, 2)}")
            # 速度单位变为：度/秒 (deg/s)
            print(f"{'Joint Velocities:':<20} {np.round(obs[6:12] * r2d, 2)}")
            print(f"{'Command (in obs):':<20} {np.round(obs[12:19], 4)}")
            # 上一步动作也转为角度
            print(f"{'Previous Action:':<20} {np.round(obs[19:25] * r2d, 2)}\n")

            print("--- Action ---")
            
            # 注意：Raw Action 通常是归一化数值（如 -1 到 1），强行转角度物理意义不大，
            # 但为了检查网络是否输出爆炸数值，这里也按你的要求转了。
            print(f"{'Raw Action:':<20} {np.round(self.action * r2d, 2)}")

            # 计算最终动作并转换
            processed_action = self.default_pos + (self.action * self._action_scale)
            print(f"{'Processed Action:':<20} {np.round(processed_action * r2d, 2)}")
        # 核心：动作反归一化/缩放和集成
        joint_positions = self.default_pos + (self.action * self._action_scale)
        self._policy_counter += 1
        return joint_positions  #关节角度有问题
