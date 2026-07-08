import argparse
import math
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rclpy.duration import Duration as RclpyDuration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_geometry_msgs  # noqa: F401  # 让 tf2_ros 能自动转换 PoseStamped。
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from controllers.harl_arm_policy import HarlArmPolicy
from utils.ur5e_kinematics import UR5eDLSIK, quat_apply_wxyz, quat_xyzw_to_wxyz


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 默认部署用户指定的 6.15 arm checkpoint；命令行仍可用 --model-path 临时覆盖。
DEFAULT_MODEL_PATH = PROJECT_ROOT / "results/isaaclab/Isaac-UR5-Mouth-Feed-Direct-v0/happo/test/6.15_ours_a_best/best_model/actor_agent_arm.pt"
DEFAULT_ALGO_CFG = PROJECT_ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/ur5_mouth_marl_env/agents/harl_happo_cfg.yaml"


class ArmPolicyDeployNode(Node):
    """ROS2 node that deploys only the trained HARL arm policy on UR5e."""

    STATE_TOPIC = "/scaled_joint_trajectory_controller/controller_state"
    CMD_TOPIC = "/scaled_joint_trajectory_controller/joint_trajectory"
    FACE_TOPIC = "/face_pose"

    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    JOINT_NAME_TO_IDX = {name: idx for idx, name in enumerate(JOINT_NAMES)}
    DEFAULT_JOINT_POS = np.deg2rad(np.array([-45.0, -80.0, 110.0, -90.0, 0.0, 50.0], dtype=np.float32))
    SPOON_OFFSET = np.array([0.0, -0.02, 0.1185], dtype=np.float32)
    WRENCH_SCALE = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1], dtype=np.float32)
    MOUTH_FEATURE_OFFSETS = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.025, 0.0, 0.0],
            [-0.025, 0.0, 0.0],
            [0.0, 0.018, 0.0],
            [0.0, -0.018, 0.0],
            [0.0, 0.0, 0.035],
        ],
        dtype=np.float32,
    )

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("arm_policy_deploy_node")
        self.args = args
        self.target_frame = args.target_frame
        self.ee_frame = args.ee_frame
        self.control_period = 1.0 / args.control_hz

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.policy = HarlArmPolicy(args.model_path, args.algo_cfg, device=args.device, backend=args.policy_backend)
        self.ik = UR5eDLSIK(damping=args.ik_damping)

        self.current_joint_pos: np.ndarray | None = None
        self.current_joint_vel: np.ndarray | None = None
        self.mouth_pos: np.ndarray | None = None
        self.mouth_quat_wxyz: np.ndarray | None = None
        self.previous_arm_action = np.zeros(6, dtype=np.float32)
        self.wrist_wrench_obs = np.zeros(6, dtype=np.float32)
        self.target_received = False
        self.warned_missing_tf = False
        self.last_face_time_ns: int | None = None
        self.last_joint_state_time_ns: int | None = None

        self.create_subscription(JointTrajectoryControllerState, self.STATE_TOPIC, self.joint_state_callback, 10)
        self.create_subscription(PoseStamped, self.FACE_TOPIC, self.face_pose_callback, 10)
        if args.wrench_topic:
            self.create_subscription(WrenchStamped, args.wrench_topic, self.wrench_callback, 10)

        self.pub = self.create_publisher(JointTrajectory, self.CMD_TOPIC, 10)
        self.timer = self.create_timer(self.control_period, self.step_callback)

        mode = "dry-run，只打印不发关节轨迹" if args.dry_run else "motion enabled，会发布关节轨迹"
        self.get_logger().info(f"arm policy 部署节点已启动: {mode}")
        self.get_logger().info(f"policy request: {Path(args.model_path).expanduser().resolve()}")
        self.get_logger().info(f"policy loaded: {self.policy.loaded_path}")
        self.get_logger().info(f"policy backend: {self.policy.backend}")
        self.get_logger().info(f"等待 {self.FACE_TOPIC}、{self.STATE_TOPIC} 和 TF {self.target_frame}->{self.ee_frame} ...")

    def joint_state_callback(self, msg: JointTrajectoryControllerState) -> None:
        pos = np.zeros(6, dtype=np.float32)
        vel = np.zeros(6, dtype=np.float32)
        for msg_idx, joint_name in enumerate(msg.joint_names):
            if joint_name not in self.JOINT_NAME_TO_IDX:
                continue
            dst_idx = self.JOINT_NAME_TO_IDX[joint_name]
            # UR driver 的 controller_state 在 Humble 下使用 feedback 作为真实反馈。
            pos[dst_idx] = float(msg.feedback.positions[msg_idx])
            vel[dst_idx] = float(msg.feedback.velocities[msg_idx])
        self.current_joint_pos = pos
        self.current_joint_vel = vel
        self.last_joint_state_time_ns = self.get_clock().now().nanoseconds

    def face_pose_callback(self, msg: PoseStamped) -> None:
        try:
            transformed = self.tf_buffer.transform(msg, self.target_frame, timeout=RclpyDuration(seconds=0.1))
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(f"无法将 {msg.header.frame_id} 下的人脸位姿转换到 {self.target_frame}: {exc}", throttle_duration_sec=1.0)
            return

        pos = transformed.pose.position
        ori = transformed.pose.orientation
        self.mouth_pos = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
        self.mouth_quat_wxyz = quat_xyzw_to_wxyz(np.array([ori.x, ori.y, ori.z, ori.w], dtype=np.float32))
        self.last_face_time_ns = self.get_clock().now().nanoseconds
        if not self.target_received:
            self.target_received = True
            self.get_logger().info("已接收到第一个人脸/嘴部目标，arm policy 开始参与控制。")

    def wrench_callback(self, msg: WrenchStamped) -> None:
        raw = np.array(
            [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z,
            ],
            dtype=np.float32,
        )
        # 仿真观测里使用归一化后的腕部力/力矩；没有传感器时默认保持 0。
        self.wrist_wrench_obs = np.clip(raw * self.WRENCH_SCALE, -5.0, 5.0)

    def step_callback(self) -> None:
        if not self.target_received or self.current_joint_pos is None or self.current_joint_vel is None:
            return
        if self.data_is_stale():
            return

        ee_pose = self.lookup_ee_pose()
        if ee_pose is None:
            return
        ee_pos, ee_quat_wxyz = ee_pose

        obs = self.compute_arm_observation(ee_pos, ee_quat_wxyz)
        raw_action = self.policy.act(obs)
        action = np.clip(raw_action, -self.args.action_limit, self.args.action_limit).astype(np.float32)
        arm_delta = action * self.args.action_scale_arm

        if not self.delta_is_safe(ee_pos, arm_delta):
            self.get_logger().warn(f"策略目标越界，跳过本步。arm_delta={np.round(arm_delta, 4)}", throttle_duration_sec=1.0)
            return

        joint_target = self.ik.joint_target_from_delta(
            self.current_joint_pos,
            arm_delta,
            max_joint_step=self.args.max_joint_step,
        ).astype(np.float32)
        self.previous_arm_action = action

        if self.args.dry_run:
            self.get_logger().info(
                f"dry-run action={np.round(action, 3)}, joint_target(deg)={np.round(np.rad2deg(joint_target), 1)}",
                throttle_duration_sec=0.5,
            )
            return

        self.publish_joint_target(joint_target)

    def lookup_ee_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            transform = self.tf_buffer.lookup_transform(self.target_frame, self.ee_frame, rclpy.time.Time(), timeout=RclpyDuration(seconds=0.05))
        except tf2_ros.TransformException as exc:
            if not self.warned_missing_tf:
                self.get_logger().warn(f"等待 TF {self.target_frame}->{self.ee_frame}: {exc}")
                self.warned_missing_tf = True
            return None

        trans = transform.transform.translation
        rot = transform.transform.rotation
        ee_pos = np.array([trans.x, trans.y, trans.z], dtype=np.float32)
        ee_quat_wxyz = quat_xyzw_to_wxyz(np.array([rot.x, rot.y, rot.z, rot.w], dtype=np.float32))
        return ee_pos, ee_quat_wxyz

    def compute_arm_observation(self, ee_pos: np.ndarray, ee_quat_wxyz: np.ndarray) -> np.ndarray:
        assert self.current_joint_pos is not None
        assert self.current_joint_vel is not None
        assert self.mouth_pos is not None
        assert self.mouth_quat_wxyz is not None

        spoon_center = ee_pos + quat_apply_wxyz(ee_quat_wxyz, self.SPOON_OFFSET)
        mouth_axis = quat_apply_wxyz(self.mouth_quat_wxyz, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        mouth_feature_points = self.build_mouth_feature_points()

        # 观测顺序必须与仿真 _get_observations() 的 obs_arm_policy_input 完全一致。
        obs = np.concatenate(
            [
                self.current_joint_pos - self.DEFAULT_JOINT_POS,
                self.current_joint_vel,
                ee_pos,
                ee_quat_wxyz,
                spoon_center,
                self.mouth_pos,
                mouth_feature_points.reshape(-1),
                mouth_axis,
                self.previous_arm_action,
                self.wrist_wrench_obs,
            ],
            dtype=np.float32,
        )
        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)

    def build_mouth_feature_points(self) -> np.ndarray:
        assert self.mouth_pos is not None
        assert self.mouth_quat_wxyz is not None
        rot = R.from_quat([self.mouth_quat_wxyz[1], self.mouth_quat_wxyz[2], self.mouth_quat_wxyz[3], self.mouth_quat_wxyz[0]])
        # 实机视觉只有嘴部中心位姿，仿真里的 6 个 mouth link 质心用嘴部局部小结构近似。
        return self.mouth_pos.reshape(1, 3) + rot.apply(self.MOUTH_FEATURE_OFFSETS).astype(np.float32)

    def delta_is_safe(self, ee_pos: np.ndarray, arm_delta: np.ndarray) -> bool:
        target_pos = ee_pos + arm_delta[:3]
        x_ok = self.args.min_x <= target_pos[0] <= self.args.max_x
        y_ok = self.args.min_y <= target_pos[1] <= self.args.max_y
        z_ok = self.args.min_z <= target_pos[2] <= self.args.max_z
        step_ok = np.linalg.norm(arm_delta[:3]) <= self.args.max_cartesian_step
        rot_ok = np.linalg.norm(arm_delta[3:]) <= self.args.max_rotation_step
        return bool(x_ok and y_ok and z_ok and step_ok and rot_ok)

    def publish_joint_target(self, joint_target: np.ndarray) -> None:
        assert self.current_joint_pos is not None
        point = JointTrajectoryPoint()
        point.positions = [float(joint_target[self.JOINT_NAME_TO_IDX[name]]) for name in self.JOINT_NAMES]

        max_move = float(np.max(np.abs(joint_target - self.current_joint_pos)))
        duration_s = max(self.args.min_trajectory_duration, max_move / max(self.args.max_joint_velocity, 1.0e-6))
        sec = int(duration_s)
        nanosec = int((duration_s - sec) * 1.0e9)
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)

        traj = JointTrajectory()
        traj.joint_names = list(self.JOINT_NAMES)
        traj.points.append(point)
        self.pub.publish(traj)

    def data_is_stale(self) -> bool:
        now_ns = self.get_clock().now().nanoseconds
        if self.last_face_time_ns is None or (now_ns - self.last_face_time_ns) * 1.0e-9 > self.args.target_timeout:
            self.get_logger().warn("视觉目标超时，暂停发布策略动作。", throttle_duration_sec=1.0)
            return True
        if self.last_joint_state_time_ns is None or (now_ns - self.last_joint_state_time_ns) * 1.0e-9 > self.args.joint_state_timeout:
            self.get_logger().warn("关节状态超时，暂停发布策略动作。", throttle_duration_sec=1.0)
            return True
        return False


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Deploy only the HARL arm policy on a real UR5e.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="actor_agent_arm.pt checkpoint path.")
    parser.add_argument("--algo-cfg", default=str(DEFAULT_ALGO_CFG), help="HARL yaml used to reconstruct the actor.")
    parser.add_argument("--device", default="cpu", help="Torch device for policy inference; only used by torch backend.")
    parser.add_argument("--policy-backend", default="auto", choices=["auto", "torchscript", "torch", "numpy"], help="Policy inference backend. auto prefers the exported TorchScript model, then HARL+torch, then .npz.")
    parser.add_argument("--dry-run", action="store_true", help="Only print policy outputs; do not publish joint commands.")
    parser.add_argument("--target-frame", default="base_link", help="Robot base frame used by the policy observation.")
    parser.add_argument("--ee-frame", default="wrist_3_link", help="End-effector frame matching the IsaacLab ee_link_name.")
    parser.add_argument("--wrench-topic", default="", help="Optional WrenchStamped topic; empty means zero wrench observation.")
    parser.add_argument("--control-hz", type=float, default=30.0, help="Policy/control frequency, matching sim decimation.")
    parser.add_argument("--target-timeout", type=float, default=0.5, help="Stop commanding if /face_pose is older than this many seconds.")
    parser.add_argument("--joint-state-timeout", type=float, default=0.5, help="Stop commanding if controller_state is older than this many seconds.")
    parser.add_argument("--action-scale-arm", type=float, default=0.02, help="Scale from policy action to delta pose.")
    parser.add_argument("--action-limit", type=float, default=1.0, help="Clamp raw policy action before scaling.")
    parser.add_argument("--ik-damping", type=float, default=0.05, help="DLS IK damping.")
    parser.add_argument("--max-joint-step", type=float, default=math.radians(2.0), help="Maximum joint change per command.")
    parser.add_argument("--max-joint-velocity", type=float, default=0.4, help="Trajectory duration velocity limit in rad/s.")
    parser.add_argument("--min-trajectory-duration", type=float, default=0.08, help="Minimum JointTrajectory point duration.")
    parser.add_argument("--max-cartesian-step", type=float, default=0.025, help="Maximum TCP translation step per policy action.")
    parser.add_argument("--max-rotation-step", type=float, default=0.04, help="Maximum TCP rotation-vector step per policy action.")
    parser.add_argument("--min-x", type=float, default=0.15)
    parser.add_argument("--max-x", type=float, default=0.95)
    parser.add_argument("--min-y", type=float, default=-0.65)
    parser.add_argument("--max-y", type=float, default=0.65)
    parser.add_argument("--min-z", type=float, default=0.06)
    parser.add_argument("--max-z", type=float, default=0.75)
    return parser.parse_known_args()


def main() -> None:
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = ArmPolicyDeployNode(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # 允许 Ctrl-C 或外部 shutdown 安静退出，避免实机调试时刷出无意义 traceback。
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
