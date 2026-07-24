import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import sys

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rclpy.duration import Duration as RclpyDuration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_geometry_msgs  # noqa: F401  # 让 tf2_ros 能自动转换 PoseStamped。
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.controllers.harl_arm_policy import HarlArmPolicy
from python.utils.ur5e_kinematics import UR5eDLSIK, quat_apply_wxyz, quat_xyzw_to_wxyz


DEFAULT_MODEL_PATH = PROJECT_ROOT / "sample/actor_agent_arm_torchscript.pt"
WRENCH_RECORD_DIR = PROJECT_ROOT / "logs" / "wrench_records"


class ArmPolicyDeployNode(Node):
    """ROS2 node that deploys only the trained HARL arm policy on UR5e."""

    STATE_TOPIC = "/scaled_joint_trajectory_controller/controller_state"
    CMD_TOPIC = "/scaled_joint_trajectory_controller/joint_trajectory"
    SPEED_SCALING_TOPIC = "/speed_scaling_state_broadcaster/speed_scaling"
    WRENCH_TOPIC = "/force_torque_sensor_broadcaster/wrench"
    DEFAULT_MOUTH_TOPIC = "/mouth_pose"

    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    JOINT_NAME_TO_IDX = {name: idx for idx, name in enumerate(JOINT_NAMES)}
    # 与仿真执行器一致: 肩/肘允许20度/秒，三个腕部关节限制为15度/秒。
    JOINT_MAX_VELOCITIES_RAD_S = np.deg2rad(
        np.array([20.0, 20.0, 20.0, 15.0, 15.0, 15.0], dtype=np.float32)
    )
    ARM_KINEMATIC_OBS_DIM = 52  # 观测布局: 前52维保持原有运动学、嘴部几何和历史动作顺序不变。
    WRIST_WRENCH_OBS_DIM = 6  # 力觉观测: 尾部追加三维力和三维力矩。
    ARM_POLICY_OBS_DIM = ARM_KINEMATIC_OBS_DIM + WRIST_WRENCH_OBS_DIM
    DEFAULT_JOINT_POS = np.deg2rad(np.array([-45.0, -80.0, 110.0, -90.0, 0.0, 60.0], dtype=np.float32))
    # 与当前长勺仿真一致: wrist_3_link 到勺心的局部偏移包含新增的 16 cm 延长段。
    SPOON_OFFSET = np.array([0.0, -0.02, 0.2785], dtype=np.float32)
    SPOON_FORWARD_AXIS_LOCAL = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # 与仿真一致: 勺柄/前送方向取 wrist_3_link 局部+Z。
    SPOON_UP_AXIS_LOCAL = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # 姿态诊断: 勺面法向取 wrist_3_link 局部+Y，但不参与实机成功门控。
    WRIST_WRENCH_OBS_SCALE = np.array([0.1, 0.1, 0.1, 1.0, 1.0, 1.0], dtype=np.float32)
    WRIST_FORCE_OBS_CLIP_N = 8.0
    WRIST_TORQUE_OBS_CLIP_NM = 1.0
    WRIST_WRENCH_FILTER_ALPHA = 0.2
    # 固定嘴模式直接复用一帧真实仿真 OBS_DEBUG 样本，确保 mouth_pos/mouth_axis/6个嘴部观测点来自同一个可出现状态。
    DEFAULT_FIXED_MOUTH_POS = np.array([0.77805, 0.55456, 0.21979], dtype=np.float32)
    DEFAULT_FIXED_MOUTH_AXIS = np.array([0.25625, 0.96661, 0.0], dtype=np.float32)
    DEFAULT_FIXED_MOUTH_FEATURE_CENTER = np.array([0.77805, 0.55456, 0.21979], dtype=np.float32)
    DEFAULT_FIXED_MOUTH_FEATURE_POINTS = np.array(
        [
            [0.77758, 0.55479, 0.24045],
            [0.80132, 0.54916, 0.22869],
            [0.80090, 0.54753, 0.20462],
            [0.77797, 0.55429, 0.19233],
            [0.75468, 0.55979, 0.20423],
            [0.75518, 0.56139, 0.22830],
        ],
        dtype=np.float32,
    )
    FEED_INSERT_RADIUS_M = 0.05
    FEED_INSERT_DEPTH_MIN_M = 0.005
    FEED_INSERT_DEPTH_MAX_M = 0.035
    FEED_INSERT_ALIGN_DOT_MIN = 0.8
    FEED_STABLE_STEPS_REQUIRED = 5
    ARM_ACTION_ROTATION_SCALE = 0.002  # 与仿真一致: 旋转增量小于平移增量，避免真机将策略姿态动作放大四倍。
    RESET_ACTION_FREEZE_STEPS = 10  # 与仿真一致: reset 后先冻结10个控制步，避免初始饱和动作立刻改变勺子姿态。
    RESET_ACTION_RAMP_STEPS = 12  # 与仿真一致: 冻结结束后用12步把实际位姿增量线性放大到完整值。
    WORKSPACE_MIN = np.array([0.15, -0.65, 0.06], dtype=np.float32)
    WORKSPACE_MAX = np.array([0.95, 0.75, 0.75], dtype=np.float32)
    SAFETY_EPS = 1.0e-5
    MOUTH_FEATURE_OFFSETS = np.array(
        [
            [0.0000, 0.0180, 0.0],
            [0.0140, 0.0120, 0.0],
            [-0.0140, 0.0120, 0.0],
            [0.0000, -0.0180, 0.0],
            [-0.0140, -0.0120, 0.0],
            [0.0140, -0.0120, 0.0],
        ],
        dtype=np.float32,
    )

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("arm_policy_deploy_node")
        self.args = args
        self.target_frame = args.target_frame
        self.ee_frame = args.ee_frame
        self.wrench_source_frame = args.wrench_source_frame.lstrip("/")
        self.wrench_sign = float(args.wrench_sign)
        self.mouth_topic = args.mouth_topic
        self.control_period = 1.0 / args.control_hz

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.policy = HarlArmPolicy(
            args.model_path,
            obs_dim=self.ARM_POLICY_OBS_DIM,
            device=args.device,
            backend=args.policy_backend,
        )
        self.ik = UR5eDLSIK(damping=args.ik_damping)

        self.current_joint_pos: np.ndarray | None = None
        self.current_joint_vel: np.ndarray | None = None
        self.mouth_pos: np.ndarray | None = None
        self.mouth_quat_wxyz: np.ndarray | None = None
        self.fixed_mouth_axis: np.ndarray | None = None
        self.previous_arm_action = np.zeros(6, dtype=np.float32)
        self.wrist_wrench_sensor_raw = np.zeros(6, dtype=np.float32)
        # 策略原始力觉: 默认沿用tool0_controller/wrist_3_link方向，先扣传感器零偏再做符号对齐，仍保持 N/Nm。
        self.wrist_wrench_raw = np.zeros(6, dtype=np.float32)
        self.wrist_wrench_raw_before_tare = np.zeros(6, dtype=np.float32)
        self.wrist_wrench_sensor_bias = np.zeros(6, dtype=np.float32)
        self.wrench_bias_sum = np.zeros(6, dtype=np.float64)
        self.wrench_bias_count = 0
        self.wrist_wrench_filtered = np.zeros(6, dtype=np.float32)
        self.wrist_wrench_policy_obs = np.zeros(6, dtype=np.float32)
        self.wrench_filter_step = 0
        self.wrench_frame_id: str | None = None
        self.last_wrench_time_ns: int | None = None
        self.wrench_record_file = None
        self.wrench_record_writer = None
        self.wrench_record_count = 0
        self.last_joint_target: np.ndarray | None = None
        self.target_received = False
        self.warned_missing_tf = False
        self.last_face_time_ns: int | None = None
        self.last_joint_state_time_ns: int | None = None
        self.last_speed_scaling_time_ns: int | None = None
        self.speed_scaling: float | None = None
        self.policy_step = 0
        self.episode_start_time_ns: int | None = None
        self.episode_stopped = False
        self.fsm_state = "APPROACH"
        self.feed_stable_steps = 0
        self.last_fsm_metrics: dict[str, float | bool] = {}
        self.max_episode_steps = (
            args.max_episode_steps
            if args.max_episode_steps > 0
            else max(1, int(round(args.episode_length_s * args.control_hz)))
        )

        if args.fixed_mouth_pose:
            self.mouth_pos = self.DEFAULT_FIXED_MOUTH_POS.copy()
            self.fixed_mouth_axis = self.DEFAULT_FIXED_MOUTH_AXIS.copy()
            self.mouth_quat_wxyz = self.mouth_quat_from_policy_axis(self.fixed_mouth_axis)
            self.target_received = True
            self.last_face_time_ns = self.get_clock().now().nanoseconds

        self.create_subscription(JointTrajectoryControllerState, self.STATE_TOPIC, self.joint_state_callback, 10)
        if not args.fixed_mouth_pose:
            self.create_subscription(PoseStamped, self.mouth_topic, self.face_pose_callback, 10)
        self.create_subscription(Float64, self.SPEED_SCALING_TOPIC, self.speed_scaling_callback, 10)
        # 58维策略始终依赖 wrench；debug_wrench 只控制额外日志和CSV记录。
        self.create_subscription(WrenchStamped, self.WRENCH_TOPIC, self.wrench_callback, 10)

        self.pub = self.create_publisher(JointTrajectory, self.CMD_TOPIC, 10)
        self.timer = self.create_timer(self.control_period, self.step_callback)
        if args.debug_wrench:
            self.open_wrench_record()

        mode = "dry-run，只打印不发关节轨迹" if args.dry_run else "motion enabled，会发布关节轨迹"
        self.get_logger().info(f"arm policy 部署节点已启动: {mode}")
        self.get_logger().info(f"policy request: {Path(args.model_path).expanduser().resolve()}")
        self.get_logger().info(f"policy loaded: {self.policy.loaded_path}")
        self.get_logger().info(f"policy backend: {self.policy.backend}")
        self.get_logger().info(
            f"arm action: scale={self.args.action_scale_arm:.4f}, limit={self.args.action_limit:.2f}, "
            f"rotation_scale={self.ARM_ACTION_ROTATION_SCALE:.4f}, action_filter=disabled"
        )
        self.get_logger().info(
            "joint velocity limits (deg/s): "
            f"{np.round(np.rad2deg(self.JOINT_MAX_VELOCITIES_RAD_S), 1).tolist()}"
        )
        self.get_logger().info(
            "arm startup: "
            f"freeze={self.RESET_ACTION_FREEZE_STEPS} steps, "
            f"ramp={self.RESET_ACTION_RAMP_STEPS} steps"
        )
        self.get_logger().info(
            "arm policy输入为58维: 52维运动学观测 + 6维腕部force/torque；"
            "wrench使用8N/1Nm模长限幅、仿真同款缩放和alpha=0.2低通。"
        )
        self.get_logger().info(
            "wrench部署对齐: "
            f"source_frame={self.wrench_source_frame or '<msg.header>'}, "
            f"sign={self.wrench_sign:+.0f}, "
            f"tare_samples={self.args.wrench_bias_samples}；"
            "默认认为UR力话题frame与wrist_3_link一致，只统一符号/静态零偏后进入clip/scale/filter。"
        )
        if args.debug_wrench:
            self.get_logger().info(f"wrench debug 已开启: {self.WRENCH_TOPIC} 将额外输出诊断并记录CSV。")
        if args.fixed_mouth_pose:
            self.get_logger().info(
                "使用固定嘴巴目标: "
                f"mouth_pos={np.round(self.mouth_pos, 4)}, "
                f"mouth_axis_w={np.round(self.compute_mouth_axis(), 4)}；姿态和最大张嘴特征在整个 episode 保持不变。"
            )
            self.get_logger().info(
                f"等待 {self.STATE_TOPIC}、{self.SPEED_SCALING_TOPIC}、{self.WRENCH_TOPIC} "
                f"和 TF {self.target_frame}->{self.ee_frame} ..."
            )
        else:
            self.get_logger().info(
                f"等待 {self.mouth_topic}、{self.STATE_TOPIC}、{self.SPEED_SCALING_TOPIC}、{self.WRENCH_TOPIC} "
                f"和 TF {self.target_frame}->{self.ee_frame} ..."
            )

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

    def speed_scaling_callback(self, msg: Float64) -> None:
        self.speed_scaling = float(msg.data)
        self.last_speed_scaling_time_ns = self.get_clock().now().nanoseconds

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
            self.get_logger().info("mouth_axis 已默认取反校正，仅影响52维观测中的 mouth_axis_w。")

    def wrench_callback(self, msg: WrenchStamped) -> None:
        sensor_raw = np.array(
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
        self.wrist_wrench_sensor_raw = np.nan_to_num(sensor_raw, nan=0.0, posinf=250.0, neginf=-250.0)
        frame_id = msg.header.frame_id.lstrip("/") or self.wrench_source_frame
        if self.wrench_frame_id is None:
            self.wrench_frame_id = frame_id
            self.get_logger().info(
                f"已接收到第一个wrench，frame_id={frame_id or '<empty>'}；"
                "将按部署端wrench对齐参数转换为策略力觉输入。"
            )
        elif frame_id != self.wrench_frame_id:
            self.get_logger().warn(
                f"wrench frame发生变化: {self.wrench_frame_id or '<empty>'} -> {frame_id or '<empty>'}；"
                "仍按tool0_controller/wrist_3_link同向假设处理，请确认TF一致。",
                throttle_duration_sec=1.0,
            )
            self.wrench_frame_id = frame_id

        self.last_wrench_time_ns = self.get_clock().now().nanoseconds
        self.wrist_wrench_raw_before_tare = self.align_wrist_wrench_direction(self.wrist_wrench_sensor_raw)
        if self.args.wrench_bias_samples > 0 and self.wrench_bias_count < self.args.wrench_bias_samples:
            # 静态tare采集传感器原始零偏；后面显式计算 sign * (sensor - bias)，避免方向和偏置约定混在一起。
            self.wrench_bias_sum += self.wrist_wrench_sensor_raw.astype(np.float64)
            self.wrench_bias_count += 1
            if self.wrench_bias_count == self.args.wrench_bias_samples:
                self.wrist_wrench_sensor_bias = (self.wrench_bias_sum / float(self.wrench_bias_count)).astype(np.float32)
                self.wrist_wrench_filtered.fill(0.0)
                self.wrist_wrench_policy_obs.fill(0.0)
                self.wrench_filter_step = 0
                self.get_logger().info(
                    "wrench静态tare完成: "
                    f"bias_sensor_raw={self.format_debug_array(self.wrist_wrench_sensor_bias)}, "
                    f"bias_policy_equiv={self.format_debug_array(self.align_wrist_wrench_direction(self.wrist_wrench_sensor_bias))}"
                )
        # 策略看到的是扣除静态传感器零偏后的wrench，再按仿真/实机作用方向差异统一符号。
        self.wrist_wrench_raw = self.align_wrist_wrench_direction(
            self.wrist_wrench_sensor_raw - self.wrist_wrench_sensor_bias
        )

    def open_wrench_record(self) -> None:
        try:
            WRENCH_RECORD_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = "dryrun" if self.args.dry_run else "motion"
            path = WRENCH_RECORD_DIR / f"real_wrench_{stamp}_{mode}.csv"
            self.wrench_record_file = path.open("w", newline="")
            self.wrench_record_writer = csv.writer(self.wrench_record_file)
            self.wrench_record_writer.writerow(
                [
                    "time_s",
                    "policy_step",
                    "fsm_state",
                    "speed_scaling",
                    "sensor_fx",
                    "sensor_fy",
                    "sensor_fz",
                    "sensor_tx",
                    "sensor_ty",
                    "sensor_tz",
                    "policy_raw_fx",
                    "policy_raw_fy",
                    "policy_raw_fz",
                    "policy_raw_tx",
                    "policy_raw_ty",
                    "policy_raw_tz",
                    "policy_fx_scaled",
                    "policy_fy_scaled",
                    "policy_fz_scaled",
                    "policy_tx_scaled",
                    "policy_ty_scaled",
                    "policy_tz_scaled",
                    "mouth_minus_spoon_x",
                    "mouth_minus_spoon_y",
                    "mouth_minus_spoon_z",
                    "mouth_spoon_dist",
                    "spoon_z",
                    "raw_action_x",
                    "raw_action_y",
                    "raw_action_z",
                    "raw_action_rx",
                    "raw_action_ry",
                    "raw_action_rz",
                    "policy_action_x",
                    "policy_action_y",
                    "policy_action_z",
                    "policy_action_rx",
                    "policy_action_ry",
                    "policy_action_rz",
                    "command_action_x",
                    "command_action_y",
                    "command_action_z",
                    "command_action_rx",
                    "command_action_ry",
                    "command_action_rz",
                    "arm_delta_x",
                    "arm_delta_y",
                    "arm_delta_z",
                    "arm_delta_rx",
                    "arm_delta_ry",
                    "arm_delta_rz",
                ]
            )
            self.get_logger().info(f"真实运行 wrench CSV 记录: {path}")
        except OSError as exc:
            self.wrench_record_file = None
            self.wrench_record_writer = None
            self.get_logger().warn(f"无法创建 wrench CSV 记录文件: {exc}")

    def record_wrench_sample(
        self,
        spoon_center: np.ndarray,
        raw_action: np.ndarray,
        policy_action: np.ndarray,
        command_action: np.ndarray,
        arm_delta: np.ndarray,
    ) -> None:
        if self.wrench_record_writer is None:
            return
        assert self.mouth_pos is not None
        if self.episode_start_time_ns is None:
            time_s = 0.0
        else:
            time_s = (self.get_clock().now().nanoseconds - self.episode_start_time_ns) * 1.0e-9
        mouth_delta = self.mouth_pos - spoon_center
        dist = float(np.linalg.norm(mouth_delta))
        speed_scaling = float(self.speed_scaling) if self.speed_scaling is not None else float("nan")
        row = (
            [time_s, self.policy_step, self.fsm_state, speed_scaling]
            + self.wrist_wrench_sensor_raw.astype(float).tolist()
            + self.wrist_wrench_raw.astype(float).tolist()
            + self.wrist_wrench_policy_obs.astype(float).tolist()
            + mouth_delta.astype(float).tolist()
            + [dist, float(spoon_center[2])]
            + np.asarray(raw_action, dtype=np.float32).astype(float).tolist()
            + np.asarray(policy_action, dtype=np.float32).astype(float).tolist()
            + np.asarray(command_action, dtype=np.float32).astype(float).tolist()
            + np.asarray(arm_delta, dtype=np.float32).astype(float).tolist()
        )
        self.wrench_record_writer.writerow(row)
        self.wrench_record_count += 1
        if self.wrench_record_count % 30 == 0 and self.wrench_record_file is not None:
            self.wrench_record_file.flush()

    def close_wrench_record(self) -> None:
        if self.wrench_record_file is None:
            return
        self.wrench_record_file.flush()
        self.wrench_record_file.close()
        self.wrench_record_file = None
        self.wrench_record_writer = None

    def step_callback(self) -> None:
        if not self.target_received:
            self.get_logger().warn(
                f"等待视觉嘴部目标 {self.mouth_topic}，尚未开始策略推理。",
                throttle_duration_sec=1.0,
            )
            return
        if self.current_joint_pos is None or self.current_joint_vel is None:
            self.get_logger().warn(
                f"等待关节状态 {self.STATE_TOPIC}，尚未开始策略推理。",
                throttle_duration_sec=1.0,
            )
            return
        if self.episode_stopped:
            return
        if self.data_is_stale():
            return
        if not self.args.dry_run and not self.ros_control_is_ready():
            return

        ee_pose = self.lookup_ee_pose()
        if ee_pose is None:
            return
        ee_pos, ee_quat_wxyz = ee_pose

        if self.episode_start_time_ns is None:
            self.episode_start_time_ns = self.get_clock().now().nanoseconds
            self.get_logger().info(
                f"episode FSM started: timeout={self.args.episode_length_s:.2f}s/"
                f"{self.max_episode_steps} steps, state={self.fsm_state}, "
                f"insert_radius={self.FEED_INSERT_RADIUS_M:.3f}m, "
                f"insert_depth=[{self.FEED_INSERT_DEPTH_MIN_M:.3f}, {self.FEED_INSERT_DEPTH_MAX_M:.3f}]m"
            )

        spoon_center = ee_pos + quat_apply_wxyz(ee_quat_wxyz, self.SPOON_OFFSET)
        if self.update_episode_fsm(spoon_center, ee_quat_wxyz):
            return

        obs = self.compute_arm_observation(ee_pos, ee_quat_wxyz)
        if self.args.debug_wrench:
            # 在30Hz策略步记录，确保日志中的policy_wrench就是本次推理实际使用的值。
            self.log_wrench_debug()
        self.policy_step += 1
        raw_action = self.policy.act(obs)
        policy_action = np.clip(raw_action, -self.args.action_limit, self.args.action_limit).astype(np.float32)
        # 与仿真一致: 策略动作只经过硬限幅和单步安全限制，不加入额外低通滤波。
        command_action = self.limit_command_action(policy_action)
        startup_action_scale = self.startup_action_scale()
        # 与仿真一致: 平移和旋转使用独立尺度，启动阶段只缩放真正执行的位姿增量。
        arm_delta = command_action.copy()
        arm_delta[:3] *= self.args.action_scale_arm
        arm_delta[3:] *= self.ARM_ACTION_ROTATION_SCALE
        arm_delta *= startup_action_scale
        arm_delta_ik = self.ros_delta_to_ik_delta(arm_delta)
        mouth_axis_raw = self.compute_mouth_axis(invert=False)
        mouth_axis = self.compute_mouth_axis()
        self.record_wrench_sample(spoon_center, raw_action, policy_action, command_action, arm_delta)

        if self.args.debug_obs:
            self.log_observation_debug(
                obs,
                raw_action,
                policy_action,
                command_action,
                arm_delta,
                arm_delta_ik,
                startup_action_scale,
                mouth_axis_raw,
                mouth_axis,
                spoon_center,
            )

        delta_safe, delta_safety_msg = self.delta_safety_status(ee_pos, ee_quat_wxyz, arm_delta)
        if not delta_safe:
            self.stop_episode(
                "FAILURE",
                f"策略目标越界: arm_delta={np.round(arm_delta, 4)}, {delta_safety_msg}",
                spoon_center,
            )
            return

        joint_target = self.ik.joint_target_from_delta(
            self.current_joint_pos,
            arm_delta_ik,
            max_joint_step=self.args.max_joint_step,
        ).astype(np.float32)
        joint_step = joint_target - self.current_joint_pos
        # 与训练观测保持一致: 历史项记录硬限幅后真正下发的动作，启动缩放只作用于本步实际位姿增量。
        self.previous_arm_action = command_action.copy()

        if self.args.dry_run:
            self.get_logger().info(
                f"dry-run policy_action={np.round(policy_action, 3)}, "
                f"command_action={np.round(command_action, 3)}, "
                f"joint_target(deg)={np.round(np.rad2deg(joint_target), 1)}",
                throttle_duration_sec=0.5,
            )
            return

        self.publish_joint_target(joint_target)
        if self.args.debug_obs:
            self.get_logger().info(
                "[policy command debug]\n"
                f"speed_scaling={self.speed_scaling}\n"
                f"joint_step(deg)={np.round(np.rad2deg(joint_step), 4)}\n"
                f"joint_target(deg)={np.round(np.rad2deg(joint_target), 3)}",
                throttle_duration_sec=0.5,
            )

    def update_episode_fsm(self, spoon_center: np.ndarray, ee_quat_wxyz: np.ndarray) -> bool:
        metrics = self.compute_fsm_metrics(spoon_center, ee_quat_wxyz)
        self.last_fsm_metrics = metrics

        if metrics["spoon_z"] < self.args.min_spoon_z:
            self.stop_episode("FAILURE", f"勺子中心过低: z={metrics['spoon_z']:.4f}m", spoon_center)
            return True
        if self.args.max_mouth_spoon_distance > 0.0 and metrics["mouth_spoon_dist"] > self.args.max_mouth_spoon_distance:
            self.stop_episode(
                "FAILURE",
                f"嘴和勺距离过大: dist={metrics['mouth_spoon_dist']:.4f}m",
                spoon_center,
            )
            return True

        elapsed_s = 0.0
        if self.episode_start_time_ns is not None:
            elapsed_s = (self.get_clock().now().nanoseconds - self.episode_start_time_ns) * 1.0e-9
        if elapsed_s >= self.args.episode_length_s or self.policy_step >= self.max_episode_steps:
            self.stop_episode("TIMEOUT", f"超过 episode 限制: elapsed={elapsed_s:.2f}s, steps={self.policy_step}", spoon_center)
            return True

        if metrics["feed_reached"]:
            self.feed_stable_steps += 1
            if self.feed_stable_steps >= self.FEED_STABLE_STEPS_REQUIRED:
                self.fsm_state = "DONE"
                # 实机暂时没有球位置观测，因此用“勺心稳定入嘴”作为仿真球+勺成功条件的代理。
                self.stop_episode("SUCCESS", "勺心在嘴内满足径向、深度和对齐条件并稳定保持。", spoon_center)
                return True
        else:
            self.feed_stable_steps = 0
        return False

    def compute_fsm_metrics(self, spoon_center: np.ndarray, ee_quat_wxyz: np.ndarray) -> dict[str, float | bool]:
        assert self.mouth_pos is not None
        mouth_axis = self.compute_mouth_axis()
        mouth_axis = mouth_axis / max(float(np.linalg.norm(mouth_axis)), 1.0e-6)
        mouth_to_spoon = spoon_center - self.mouth_pos
        mouth_spoon_dist = float(np.linalg.norm(mouth_to_spoon))
        mouth_depth = float(np.dot(mouth_to_spoon, mouth_axis))
        radial_vec = mouth_to_spoon - mouth_depth * mouth_axis
        radial = float(np.linalg.norm(radial_vec))
        # 姿态诊断: 使用 wrist_3_link 局部+Z/+Y计算倾斜角，仅输出日志，不限制实机成功。
        spoon_axis = quat_apply_wxyz(ee_quat_wxyz, self.SPOON_FORWARD_AXIS_LOCAL)
        spoon_axis = spoon_axis / max(float(np.linalg.norm(spoon_axis)), 1.0e-6)
        spoon_up = quat_apply_wxyz(ee_quat_wxyz, self.SPOON_UP_AXIS_LOCAL)
        spoon_up = spoon_up / max(float(np.linalg.norm(spoon_up)), 1.0e-6)
        handle_tilt = math.asin(float(np.clip(abs(spoon_axis[2]), 0.0, 1.0)))
        bowl_tilt = math.acos(float(np.clip(spoon_up[2], -1.0, 1.0)))
        align_dot = float(np.dot(spoon_axis, mouth_axis))
        radial_ok = radial < self.FEED_INSERT_RADIUS_M
        depth_ok = self.FEED_INSERT_DEPTH_MIN_M < mouth_depth < self.FEED_INSERT_DEPTH_MAX_M
        align_ok = align_dot > self.FEED_INSERT_ALIGN_DOT_MIN
        feed_reached = bool(radial_ok and depth_ok and align_ok)
        return {
            "radial": radial,
            "mouth_depth": mouth_depth,
            "mouth_spoon_dist": mouth_spoon_dist,
            "spoon_z": float(spoon_center[2]),
            "align_dot": align_dot,
            "radial_ok": radial_ok,
            "depth_ok": depth_ok,
            "align_ok": align_ok,
            "handle_tilt_deg": math.degrees(handle_tilt),
            "bowl_tilt_deg": math.degrees(bowl_tilt),
            "feed_reached": feed_reached,
            "inserted": feed_reached,
        }

    def stop_episode(self, reason_type: str, reason: str, spoon_center: np.ndarray) -> None:
        if self.episode_stopped:
            return
        self.episode_stopped = True
        self.previous_arm_action = np.zeros(6, dtype=np.float32)
        metrics = self.last_fsm_metrics
        mouth_delta = self.mouth_pos - spoon_center if self.mouth_pos is not None else None
        metrics_msg = ""
        if metrics:
            metrics_msg = (
                f", dist={metrics['mouth_spoon_dist']:.4f}m, feed_reached={metrics['feed_reached']}, "
                f"radial={metrics['radial']:.4f}m, depth={metrics['mouth_depth']:.4f}m, "
                f"align={metrics['align_dot']:.4f}, radial_ok={metrics['radial_ok']}, "
                f"depth_ok={metrics['depth_ok']}, align_ok={metrics['align_ok']}, "
                f"handle_tilt={metrics['handle_tilt_deg']:.2f}deg, "
                f"bowl_tilt={metrics['bowl_tilt_deg']:.2f}deg"
            )
        self.get_logger().warn(
            f"[episode {reason_type}] {reason}; state={self.fsm_state}, steps={self.policy_step}, "
            f"mouth-spoon={np.round(mouth_delta, 4) if mouth_delta is not None else 'n/a'}{metrics_msg}"
        )
        if not self.args.dry_run and self.current_joint_pos is not None:
            self.publish_joint_target(self.current_joint_pos)

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
        mouth_axis = self.compute_mouth_axis()
        mouth_feature_points = self.build_mouth_feature_points()
        wrist_wrench_obs = self.compute_wrist_wrench_observation()

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
                wrist_wrench_obs,
            ],
            dtype=np.float32,
        )
        if obs.size != self.ARM_POLICY_OBS_DIM:
            raise RuntimeError(f"arm观测维度错误: expected={self.ARM_POLICY_OBS_DIM}, actual={obs.size}")
        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)

    def align_wrist_wrench_direction(self, wrench_source: np.ndarray) -> np.ndarray:
        """在同一tool0_controller/wrist_3_link坐标系内同步真实力和仿真力的方向约定。"""
        wrench_source = np.nan_to_num(np.asarray(wrench_source, dtype=np.float32), nan=0.0, posinf=250.0, neginf=-250.0)
        # 真实传感器与仿真关节反力可能是作用/反作用关系；用--wrench-sign做最小必要的方向同步。
        policy_raw = self.wrench_sign * wrench_source.astype(np.float32)
        return np.nan_to_num(policy_raw, nan=0.0, posinf=250.0, neginf=-250.0)

    @staticmethod
    def clip_vector_norm(vector: np.ndarray, limit: float) -> np.ndarray:
        """按向量模长同比缩放，保留真实力或力矩的方向。"""
        clipped = np.asarray(vector, dtype=np.float32).copy()
        norm = float(np.linalg.norm(clipped))
        if limit > 0.0 and norm > limit:
            clipped *= limit / max(norm, 1.0e-6)
        return clipped

    def compute_wrist_wrench_observation(self) -> np.ndarray:
        """复现仿真的 wrench 限幅、缩放和每控制步一次的一阶低通。"""
        raw = np.nan_to_num(self.wrist_wrench_raw, nan=0.0, posinf=250.0, neginf=-250.0)
        force = self.clip_vector_norm(raw[:3], self.WRIST_FORCE_OBS_CLIP_N)
        torque = self.clip_vector_norm(raw[3:], self.WRIST_TORQUE_OBS_CLIP_NM)
        scaled = np.concatenate([force, torque]).astype(np.float32) * self.WRIST_WRENCH_OBS_SCALE
        scaled = np.clip(np.nan_to_num(scaled, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)

        if self.wrench_filter_step == 0:
            # 仿真 reset 的第一个观测直接读取当前 wrench，但滤波历史仍从零开始。
            policy_obs = scaled
        else:
            alpha = float(np.clip(self.WRIST_WRENCH_FILTER_ALPHA, 0.0, 1.0))
            self.wrist_wrench_filtered = (
                (1.0 - alpha) * self.wrist_wrench_filtered + alpha * scaled
            ).astype(np.float32)
            policy_obs = self.wrist_wrench_filtered

        self.wrench_filter_step += 1
        self.wrist_wrench_policy_obs = policy_obs.astype(np.float32, copy=True)
        return self.wrist_wrench_policy_obs.copy()

    def compute_mouth_axis(self, invert: bool = True) -> np.ndarray:
        assert self.mouth_quat_wxyz is not None
        if self.args.fixed_mouth_pose:
            mouth_axis = self.current_fixed_mouth_axis().astype(np.float32)
            return mouth_axis if invert else -mouth_axis
        mouth_axis = quat_apply_wxyz(self.mouth_quat_wxyz, np.array([0.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)
        if invert:
            mouth_axis = -mouth_axis
        return mouth_axis

    @classmethod
    def mouth_quat_from_policy_axis(cls, policy_axis: np.ndarray) -> np.ndarray:
        axis = np.asarray(policy_axis, dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm < 1.0e-6:
            raise ValueError("DEFAULT_FIXED_MOUTH_AXIS 不能是零向量。")
        raw_z_axis = -axis / norm
        source_z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        cross = np.cross(source_z_axis, raw_z_axis)
        cross_norm = np.linalg.norm(cross)
        dot = float(np.clip(np.dot(source_z_axis, raw_z_axis), -1.0, 1.0))
        if cross_norm < 1.0e-8:
            rot = R.identity() if dot > 0.0 else R.from_rotvec(np.array([math.pi, 0.0, 0.0], dtype=np.float64))
        else:
            rot_axis = cross / cross_norm
            rot_angle = math.atan2(cross_norm, dot)
            rot = R.from_rotvec(rot_axis * rot_angle)
        quat_xyzw = rot.as_quat()
        return quat_xyzw_to_wxyz(quat_xyzw.astype(np.float32))

    def log_observation_debug(
        self,
        obs: np.ndarray,
        raw_action: np.ndarray,
        policy_action: np.ndarray,
        command_action: np.ndarray,
        arm_delta: np.ndarray,
        arm_delta_ik: np.ndarray,
        startup_action_scale: float,
        mouth_axis_raw: np.ndarray,
        mouth_axis: np.ndarray,
        spoon_center: np.ndarray,
    ) -> None:
        assert self.mouth_pos is not None
        assert self.mouth_quat_wxyz is not None
        mouth_feature_points = obs[25:43].reshape(6, 3)
        point_lines = "\n".join(
            f"    p{point_idx}: {self.format_debug_array(point)}"
            for point_idx, point in enumerate(mouth_feature_points)
        )
        self.get_logger().info(
            "\n"
            f"[OBS_DEBUG] step={self.policy_step} arm_obs_dim={obs.size} deploy_prefix_dim=58\n"
            "[OBS_DEBUG] 52 kinematic dims are followed by the 6-dim policy wrench observation.\n"
            f"  current_stage = {self.fsm_state}\n"
            f"  joint_pos_minus_default_rad[0:6] = {self.format_debug_array(obs[0:6])}\n"
            f"  joint_vel_rad_s[6:12] = {self.format_debug_array(obs[6:12])}\n"
            f"  ee_pos_local_m[12:15] = {self.format_debug_array(obs[12:15])}\n"
            f"  ee_quat_wxyz[15:19] = {self.format_debug_array(obs[15:19])}\n"
            f"  spoon_center_local_m[19:22] = {self.format_debug_array(obs[19:22])}\n"
            f"  mouth_pos_local_m[22:25] = {self.format_debug_array(obs[22:25])}\n"
            "  mouth_feature_points_local_m[25:43] =\n"
            f"{point_lines}\n"
            f"  mouth_axis_w[43:46] = {self.format_debug_array(obs[43:46])}\n"
            f"  previous_arm_action[46:52] = {self.format_debug_array(obs[46:52])}\n"
            f"  wrist_wrench_scaled_filtered[52:58] = {self.format_debug_array(obs[52:58])}\n"
            f"  deploy_obs_full[0:58] = {self.format_debug_array(obs[0:58])}\n"
            "[POLICY_DEBUG]\n"
            f"  mouth_quat_wxyz = {self.format_debug_array(self.mouth_quat_wxyz)}\n"
            f"  mouth_axis_raw(+Z) = {self.format_debug_array(mouth_axis_raw)}\n"
            f"  mouth_axis_policy = {self.format_debug_array(mouth_axis)}\n"
            f"  mouth_minus_spoon_m = {self.format_debug_array(self.mouth_pos - spoon_center)}\n"
            f"  raw_action = {self.format_debug_array(raw_action)}\n"
            f"  policy_action_for_obs = {self.format_debug_array(policy_action)}\n"
            f"  command_action_limited = {self.format_debug_array(command_action)}\n"
            f"  startup_action_scale = {startup_action_scale:.5f}\n"
            f"  arm_delta_ros = {self.format_debug_array(arm_delta)}\n"
            f"  arm_delta_ik = {self.format_debug_array(arm_delta_ik)}\n"
            f"  fsm_state = {self.fsm_state}\n"
            f"  fsm_metrics = {self.format_fsm_metrics()}",
            throttle_duration_sec=0.5,
        )

    @staticmethod
    def format_debug_array(values: np.ndarray) -> str:
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        return "[" + ", ".join(f"{float(value):.5f}" for value in flat) + "]"

    def log_wrench_debug(self) -> None:
        sensor_force = self.wrist_wrench_sensor_raw[:3]
        sensor_torque = self.wrist_wrench_sensor_raw[3:]
        policy_force = self.wrist_wrench_raw[:3]
        policy_torque = self.wrist_wrench_raw[3:]
        self.get_logger().info(
            "[WRENCH_DEBUG] "
            f"sensor_force={self.format_debug_array(sensor_force)}, "
            f"sensor_torque={self.format_debug_array(sensor_torque)}, "
            f"policy_raw_force={self.format_debug_array(policy_force)}, "
            f"policy_raw_torque={self.format_debug_array(policy_torque)}, "
            f"policy_wrench={self.format_debug_array(self.wrist_wrench_policy_obs)}, "
            f"frame={self.wrench_frame_id or '<empty>'}",
            throttle_duration_sec=0.5,
        )

    def format_fsm_metrics(self) -> str:
        if not self.last_fsm_metrics:
            return "{}"
        return (
            "{"
            f"dist={self.last_fsm_metrics['mouth_spoon_dist']:.4f}, "
            f"spoon_z={self.last_fsm_metrics['spoon_z']:.4f}, "
            f"feed_reached={self.last_fsm_metrics['feed_reached']}, "
            f"radial={self.last_fsm_metrics['radial']:.4f}, "
            f"depth={self.last_fsm_metrics['mouth_depth']:.4f}, "
            f"align={self.last_fsm_metrics['align_dot']:.4f}, "
            f"radial_ok={self.last_fsm_metrics['radial_ok']}, "
            f"depth_ok={self.last_fsm_metrics['depth_ok']}, "
            f"align_ok={self.last_fsm_metrics['align_ok']}, "
            f"handle_tilt={self.last_fsm_metrics['handle_tilt_deg']:.2f}deg, "
            f"bowl_tilt={self.last_fsm_metrics['bowl_tilt_deg']:.2f}deg"
            "}"
        )

    @staticmethod
    def ros_delta_to_ik_delta(delta_pose: np.ndarray) -> np.ndarray:
        """Convert a base_link policy delta into the DH frame used by UR5eDLSIK.

        UR5eDLSIK's DH FK has x/y opposite to ROS base_link on this setup
        (equivalent to a 180 degree rotation around Z). Observations and safety
        checks stay in ROS base_link; only the IK input needs this conversion.
        """
        delta = np.asarray(delta_pose, dtype=np.float32).copy()
        delta[[0, 1, 3, 4]] *= -1.0
        return delta

    def build_mouth_feature_points(self) -> np.ndarray:
        assert self.mouth_pos is not None
        assert self.mouth_quat_wxyz is not None
        if self.args.fixed_mouth_pose:
            # 固定嘴复用 play 日志中的 6 个 link 形状，但跟随当前 fixed mouth center 平移。
            fixed_offsets = self.DEFAULT_FIXED_MOUTH_FEATURE_POINTS - self.DEFAULT_FIXED_MOUTH_FEATURE_CENTER
            return self.mouth_pos.reshape(1, 3) + fixed_offsets
        quat_wxyz = self.mouth_quat_wxyz
        rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        # 实机视觉只有嘴部中心位姿，仿真里的 6 个 mouth link 质心用嘴部局部小结构近似。
        return self.mouth_pos.reshape(1, 3) + rot.apply(self.MOUTH_FEATURE_OFFSETS).astype(np.float32)

    def current_fixed_mouth_axis(self) -> np.ndarray:
        assert self.fixed_mouth_axis is not None
        axis = self.fixed_mouth_axis
        norm = max(float(np.linalg.norm(axis)), 1.0e-6)
        return (axis / norm).astype(np.float32)

    def startup_action_scale(self) -> float:
        """复现仿真 reset 后的冻结和线性启动，减小第一批饱和动作造成的姿态突变。"""
        ramp_steps = max(float(self.RESET_ACTION_RAMP_STEPS), 1.0)
        return float(np.clip((self.policy_step - self.RESET_ACTION_FREEZE_STEPS) / ramp_steps, 0.0, 1.0))

    def limit_command_action(self, policy_action: np.ndarray) -> np.ndarray:
        command_action = np.asarray(policy_action, dtype=np.float32).copy()
        translation_scale = max(float(self.args.action_scale_arm), 1.0e-6)
        rotation_scale = max(float(self.ARM_ACTION_ROTATION_SCALE), 1.0e-6)

        trans_delta = command_action[:3] * translation_scale
        trans_norm = float(np.linalg.norm(trans_delta))
        if trans_norm > self.args.max_cartesian_step:
            command_action[:3] *= self.args.max_cartesian_step / max(trans_norm, 1.0e-6)

        rot_delta = command_action[3:] * rotation_scale
        rot_norm = float(np.linalg.norm(rot_delta))
        if rot_norm > self.args.max_rotation_step:
            command_action[3:] *= self.args.max_rotation_step / max(rot_norm, 1.0e-6)

        return command_action

    def delta_safety_status(self, ee_pos: np.ndarray, ee_quat_wxyz: np.ndarray, arm_delta: np.ndarray) -> tuple[bool, str]:
        target_ee_pos = ee_pos + arm_delta[:3]
        current_rot = R.from_quat([ee_quat_wxyz[1], ee_quat_wxyz[2], ee_quat_wxyz[3], ee_quat_wxyz[0]])
        target_rot = R.from_rotvec(arm_delta[3:]) * current_rot
        target_spoon_center = target_ee_pos + target_rot.apply(self.SPOON_OFFSET)
        # 长勺安全边界同时检查 wrist_3_link 和勺心，避免只看手腕而让 27.85 cm 延长段越界。
        ee_workspace_ok = np.all(target_ee_pos >= self.WORKSPACE_MIN - self.SAFETY_EPS) and np.all(
            target_ee_pos <= self.WORKSPACE_MAX + self.SAFETY_EPS
        )
        spoon_workspace_ok = np.all(target_spoon_center >= self.WORKSPACE_MIN - self.SAFETY_EPS) and np.all(
            target_spoon_center <= self.WORKSPACE_MAX + self.SAFETY_EPS
        )
        step_ok = np.linalg.norm(arm_delta[:3]) <= self.args.max_cartesian_step + self.SAFETY_EPS
        rot_ok = np.linalg.norm(arm_delta[3:]) <= self.args.max_rotation_step + self.SAFETY_EPS
        safety_ok = bool(ee_workspace_ok and spoon_workspace_ok and step_ok and rot_ok)
        if safety_ok:
            return True, "delta safety ok"

        reasons = []
        if not ee_workspace_ok:
            reasons.append(
                f"ee_target={np.round(target_ee_pos, 4)} outside "
                f"[{np.round(self.WORKSPACE_MIN, 3)}, {np.round(self.WORKSPACE_MAX, 3)}]"
            )
        if not spoon_workspace_ok:
            reasons.append(
                f"spoon_target={np.round(target_spoon_center, 4)} outside "
                f"[{np.round(self.WORKSPACE_MIN, 3)}, {np.round(self.WORKSPACE_MAX, 3)}]"
            )
        if not step_ok:
            reasons.append(
                f"translation_step={np.linalg.norm(arm_delta[:3]):.4f}m > {self.args.max_cartesian_step:.4f}m"
            )
        if not rot_ok:
            reasons.append(
                f"rotation_step={np.linalg.norm(arm_delta[3:]):.4f}rad > {self.args.max_rotation_step:.4f}rad"
            )
        return False, "; ".join(reasons)

    def delta_is_safe(self, ee_pos: np.ndarray, ee_quat_wxyz: np.ndarray, arm_delta: np.ndarray) -> bool:
        return self.delta_safety_status(ee_pos, ee_quat_wxyz, arm_delta)[0]

    def publish_joint_target(self, joint_target: np.ndarray) -> None:
        assert self.current_joint_pos is not None
        point = JointTrajectoryPoint()
        point.positions = [float(joint_target[self.JOINT_NAME_TO_IDX[name]]) for name in self.JOINT_NAMES]

        joint_move = np.abs(joint_target - self.current_joint_pos)
        # 每个关节按自己的速度上限计算所需时间，整条轨迹采用其中最大值以保证所有关节都不超速。
        joint_durations = joint_move / np.maximum(self.JOINT_MAX_VELOCITIES_RAD_S, 1.0e-6)
        duration_s = max(self.args.min_trajectory_duration, float(np.max(joint_durations)))
        sec = int(duration_s)
        nanosec = int((duration_s - sec) * 1.0e9)
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)

        traj = JointTrajectory()
        traj.joint_names = list(self.JOINT_NAMES)
        traj.points.append(point)
        self.pub.publish(traj)

    def ros_control_is_ready(self) -> bool:
        if self.pub.get_subscription_count() == 0:
            self.get_logger().warn(f"等待 {self.CMD_TOPIC} 出现订阅者，暂停发布策略动作。", throttle_duration_sec=1.0)
            return False

        now_ns = self.get_clock().now().nanoseconds
        if self.last_speed_scaling_time_ns is None:
            self.get_logger().warn(f"等待 {self.SPEED_SCALING_TOPIC}，暂停发布策略动作。", throttle_duration_sec=1.0)
            return False
        speed_age_s = (now_ns - self.last_speed_scaling_time_ns) * 1.0e-9
        if speed_age_s > self.args.speed_scaling_timeout:
            self.get_logger().warn(
                f"speed_scaling 超时 {speed_age_s:.2f}s，暂停发布策略动作。",
                throttle_duration_sec=1.0,
            )
            return False
        assert self.speed_scaling is not None
        if self.speed_scaling < self.args.min_speed_scaling:
            self.get_logger().warn(
                f"speed_scaling={self.speed_scaling:.3f} 过低，暂停发布策略动作。"
                "请确认示教器正在运行 External Control 程序且速度滑条非零。",
                throttle_duration_sec=1.0,
            )
            return False
        return True

    def data_is_stale(self) -> bool:
        now_ns = self.get_clock().now().nanoseconds
        if not self.args.fixed_mouth_pose:
            if self.last_face_time_ns is None or (now_ns - self.last_face_time_ns) * 1.0e-9 > self.args.target_timeout:
                self.get_logger().warn("视觉目标超时，暂停发布策略动作。", throttle_duration_sec=1.0)
                return True
        if self.last_joint_state_time_ns is None or (now_ns - self.last_joint_state_time_ns) * 1.0e-9 > self.args.joint_state_timeout:
            self.get_logger().warn("关节状态超时，暂停发布策略动作。", throttle_duration_sec=1.0)
            return True
        if self.last_wrench_time_ns is None:
            self.get_logger().warn(f"等待 {self.WRENCH_TOPIC}，暂停策略推理。", throttle_duration_sec=1.0)
            return True
        if self.args.wrench_bias_samples > 0 and self.wrench_bias_count < self.args.wrench_bias_samples:
            self.get_logger().warn(
                f"wrench静态tare采集中 {self.wrench_bias_count}/{self.args.wrench_bias_samples}，暂停策略推理。",
                throttle_duration_sec=1.0,
            )
            return True
        wrench_age_s = (now_ns - self.last_wrench_time_ns) * 1.0e-9
        if wrench_age_s > self.args.wrench_timeout:
            self.get_logger().warn(
                f"wrench超时 {wrench_age_s:.2f}s，暂停策略推理和动作发布。",
                throttle_duration_sec=1.0,
            )
            return True
        return False


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Deploy only the HARL arm policy on a real UR5e.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="TorchScript actor path or exported .npz policy base path.")
    parser.add_argument("--device", default="cpu", help="Torch device for TorchScript policy inference.")
    parser.add_argument("--policy-backend", default="auto", choices=["auto", "torchscript", "numpy"], help="Policy inference backend. auto prefers the exported TorchScript model, then .npz.")
    parser.add_argument("--dry-run", action="store_true", help="Only print policy outputs; do not publish joint commands.")
    parser.add_argument("--debug-obs", action="store_true", help="Print low-rate policy observation and action diagnostics.")
    parser.add_argument("--debug-wrench", action="store_true", help="Print and record raw/processed wrist wrench diagnostics; wrench is always part of the 58-dim policy input.")
    parser.add_argument("--invert-mouth-axis", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mouth-topic", default=ArmPolicyDeployNode.DEFAULT_MOUTH_TOPIC, help="PoseStamped topic containing the fused raw mouth pose.")
    parser.set_defaults(fixed_mouth_pose=True)
    parser.add_argument("--fixed-mouth-pose", dest="fixed_mouth_pose", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--vision-mouth", dest="fixed_mouth_pose", action="store_false", help="Use /mouth_pose instead of the fixed sim-like mouth target.")
    parser.add_argument("--target-frame", default="base_link", help="Robot base frame used by the policy observation.")
    parser.add_argument("--ee-frame", default="wrist_3_link", help="End-effector frame matching the IsaacLab ee_link_name.")
    parser.add_argument("--wrench-source-frame", default="tool0_controller", help="Fallback frame for the UR F/T wrench if the message header is empty.")
    parser.add_argument("--wrench-sign", type=float, default=-1.0, choices=[-1.0, 1.0], help="Sign used to align real F/T direction with the simulation joint-reaction convention.")
    parser.add_argument("--wrench-bias-samples", type=int, default=30, help="Stationary wrench samples used for tare after sign/frame alignment; 0 disables tare.")
    parser.add_argument("--control-hz", type=float, default=30.0, help="Policy/control frequency, matching sim decimation.")
    parser.add_argument("--target-timeout", type=float, default=0.5, help="Stop commanding if the mouth pose topic is older than this many seconds.")
    parser.add_argument("--joint-state-timeout", type=float, default=0.5, help="Stop commanding if controller_state is older than this many seconds.")
    parser.add_argument("--wrench-timeout", type=float, default=0.5, help="Stop policy inference if the wrist wrench topic is older than this many seconds.")
    parser.add_argument("--speed-scaling-timeout", type=float, default=0.5, help="Stop commanding if speed_scaling is older than this many seconds.")
    parser.add_argument("--min-speed-scaling", type=float, default=0.01, help="Do not publish motion commands unless UR speed scaling is at least this value.")
    parser.add_argument("--episode-length-s", type=float, default=240.0, help="Deployment episode timeout in seconds, matching simulation.")
    parser.add_argument("--max-episode-steps", type=int, default=0, help="Deployment episode timeout in policy steps; <=0 uses episode_length_s * control_hz.")
    parser.add_argument("--feed-distance", type=float, default=0.10, help=argparse.SUPPRESS)
    parser.add_argument("--withdraw-distance", type=float, default=0.10, help=argparse.SUPPRESS)
    parser.add_argument("--feed-hold-steps", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--bite-release-steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--min-spoon-z", type=float, default=0.06, help="Episode failure threshold for spoon center height in target-frame meters.")
    parser.add_argument("--max-mouth-spoon-distance", type=float, default=0.75, help="Episode failure threshold for mouth-spoon distance; <=0 disables this guard.")
    parser.add_argument("--action-scale-arm", type=float, default=0.008, help="Scale from policy action to delta pose, matching simulation.")
    parser.add_argument("--action-limit", type=float, default=3.0, help="Clamp policy action before filtering and scaling, matching simulation.")
    parser.add_argument("--ik-damping", type=float, default=0.05, help="DLS IK damping.")
    parser.add_argument("--max-joint-step", type=float, default=math.radians(2.0), help="Maximum joint change per command.")
    parser.add_argument("--min-trajectory-duration", type=float, default=0.08, help="Minimum JointTrajectory point duration.")
    parser.add_argument("--max-cartesian-step", type=float, default=0.042, help="Vector translation guard; slightly above sqrt(3)*3*0.008 so it does not further clip valid simulation-scale actions.")
    parser.add_argument("--max-rotation-step", type=float, default=0.042, help="Vector rotation guard; slightly above sqrt(3)*3*0.008 so it does not further clip valid simulation-scale actions.")
    argv = sys.argv[1:]
    if "--ros-args" in argv:
        ros_args_start = argv.index("--ros-args")
        app_args = argv[:ros_args_start]
        ros_args = argv[ros_args_start:]
    else:
        app_args = argv
        ros_args = []
    args = parser.parse_args(app_args)
    if args.bite_release_steps is not None and args.feed_hold_steps == 5:
        args.feed_hold_steps = args.bite_release_steps
    args.wrench_bias_samples = max(0, int(args.wrench_bias_samples))
    return args, ros_args


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
        node.close_wrench_record()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
