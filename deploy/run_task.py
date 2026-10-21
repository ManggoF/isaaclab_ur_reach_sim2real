import argparse
import csv
from collections import deque
from datetime import datetime
import math
from pathlib import Path
import sys

import numpy as np
try:
    import cv2

    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False
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
from python.utils.joint_angle_recorder import JointAngleRecorder
from python.utils.ur5e_kinematics import UR5eDLSIK, quat_apply_wxyz, quat_xyzw_to_wxyz


DEFAULT_MODEL_PATH = PROJECT_ROOT / "sample/actor_agent_arm_torchscript.pt"
DEFAULT_SAFETY_MODEL_PATH = PROJECT_ROOT / "sample/actor_agent_safety_torchscript.pt"
WRENCH_RECORD_DIR = PROJECT_ROOT / "logs" / "wrench_records"
SAFETY_GAIN_VIS_DIR = PROJECT_ROOT / "logs" / "safety_gain_visualization"
JOINT_ANGLE_RECORD_DIR = PROJECT_ROOT / "logs" / "joint_angle_records"


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
    WRENCH_SIM_REAL_SIGN = -1.0  # 力方向对齐: -1 表示实机传感器读数与仿真关节反力方向相反。
    # 固定嘴模式直接复用一帧真实仿真 OBS_DEBUG 样本，确保 mouth_pos/mouth_axis/6个嘴部观测点来自同一个可出现状态。
    DEFAULT_FIXED_MOUTH_POS = np.array([0.77805, 0.55456, 0.21979], dtype=np.float32)
    # DEFAULT_FIXED_MOUTH_POS = np.array([0.66026, 0.41026, 0.26], dtype=np.float32)
    DEFAULT_FIXED_MOUTH_AXIS = np.array([0.25625, 0.96661, 0.0], dtype=np.float32)
    DEFAULT_FIXED_MOUTH_FEATURE_CENTER = np.array([0.77805, 0.55456, 0.21979], dtype=np.float32)
    # DEFAULT_FIXED_MOUTH_FEATURE_CENTER = np.array([0.66026, 0.41026, 0.26], dtype=np.float32)
    # DEFAULT_FIXED_MOUTH_FEATURE_CENTER = np.array([0.665, 0.52, 0.25], dtype=np.float32)
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
    FEED_INSERT_RADIUS_M = 0.04
    FEED_INSERT_DEPTH_MIN_M = 0.008
    FEED_INSERT_DEPTH_MAX_M = 0.080
    FEED_INSERT_ALIGN_DOT_MIN = 0.8
    FEED_STABLE_STEPS_REQUIRED = 5
    SUCCESS_HOLD_STEPS_AFTER_REACHED = 30
    SUCCESS_WITHDRAW_SPEED_M_S = 0.05  # 平均撤回速度：0.10 m / 0.05 m/s = 2 s；仍受关节限速约束。
    SUCCESS_WITHDRAW_PATH_RESOLUTION_M = 0.0025
    SUCCESS_WITHDRAW_TIMEOUT_S = 30.0
    SUCCESS_WITHDRAW_POSITION_TOL_M = 0.002
    SUCCESS_WITHDRAW_ORIENTATION_TOL_RAD = math.radians(2.0)
    SUCCESS_WITHDRAW_STABLE_STEPS = 5
    SUCCESS_WITHDRAW_MAX_LATERAL_ERROR_M = 0.010
    SUCCESS_WITHDRAW_MAX_ORIENTATION_ERROR_RAD = math.radians(5.0)
    RESET_AFTER_SUCCESS = True
    RESET_MAX_JOINT_VELOCITY_RAD_S = 0.25
    RESET_MIN_DURATION_S = 4.0
    RESET_GOAL_TOL_RAD = math.radians(0.1)
    RESET_SETTLE_TIME_S = 0.5
    RESET_TIMEOUT_MARGIN_S = 8.0
    LATCH_FIRST_VISION_MOUTH_POSE = True
    ARM_ACTION_ROTATION_SCALE = 0.002  # 与仿真一致: 旋转增量小于平移增量，避免真机将策略姿态动作放大四倍。
    RESET_ACTION_FREEZE_STEPS = 10  # 与仿真一致: reset 后先冻结10个控制步，避免初始饱和动作立刻改变勺子姿态。
    RESET_ACTION_RAMP_STEPS = 12  # 与仿真一致: 冻结结束后用12步把实际位姿增量线性放大到完整值。
    WORKSPACE_MIN = np.array([0.15, -0.65, 0.06], dtype=np.float32)
    WORKSPACE_MAX = np.array([0.95, 0.95, 0.75], dtype=np.float32)
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
    # 规则安全层配置集中写在代码里，实机调参时直接改这里，避免部署命令行越来越长。
    # 开关与可视化: 控制规则层是否接管动作，以及是否显示/保存三通道增益图像。
    RULE_SAFETY_ENABLED = False
    safety_strenthen = True  # True: 有效 grec > 0 后锁定撤回 10 cm 再归位；False: 保持原安全策略。
    safety_grec_force_one = False  # 加强模式记录：True 固定 grec=1；False 记录原安全策略增益。
    SAFETY_STRENGTHEN_WITHDRAW_DISTANCE_M = 0.10
    SAFETY_GAIN_VISUALIZATION = True
    SAVE_SAFETY_GAIN_IMAGES = True
    SAFETY_GAIN_HISTORY = 300
    SAFETY_GAIN_IMAGE_INTERVAL = 10
    SAVE_FULL_SAFETY_GAIN_CURVE = True
    SAFETY_GAIN_CURVE_DPI = 220

    # 几何增益: 控制近嘴减速、横向居中和危险撤回在什么距离开始生效。
    RULE_SAFETY_ZONE_DIST = 0.12
    RULE_SLOW_START_DIST = 0.12
    RULE_SLOW_FULL_DIST = 0.05
    RULE_CENTER_START_DIST = 0.10
    RULE_CENTER_FULL_DIST = 0.03
    RULE_CENTER_DEADBAND = 0.006
    RULE_CENTER_FULL_RADIAL = 0.035
    RULE_LATERAL_STEP = 0.010
    RULE_RETREAT_STEP = 0.012
    RULE_RETREAT_START_DIST = 0.040
    RULE_RETREAT_RELEASE_DIST = 0.080
    RULE_MIN_RETREAT_GAIN = 0.45
    RULE_MOUTH_CHANNEL_RADIUS = FEED_INSERT_RADIUS_M
    RULE_MOUTH_CHANNEL_FRONT = 0.060
    RULE_MOUTH_CHANNEL_BACK = 0.030

    # 风险阈值: 控制嘴唇/嘴部连杆/腕部力觉何时触发减速或撤回。
    RULE_LIP_MARGIN = 0.040
    RULE_LINK_THRESHOLD = 0.018
    RULE_FORCE_THRESHOLD_N = 8.0
    RULE_FORCE_RELEASE_N = 5.0
    RULE_TORQUE_THRESHOLD_NM = 0.8
    RULE_AXIAL_FORCE_THRESHOLD_N = 4.0
    RULE_TRIGGER_SCORE = 0.05
    RULE_CLEAR_SCORE = 0.01

    # 学习安全策略部署: 关闭规则安全后自动加载 27 维 safety actor，参数仍放在代码里便于实机调试。
    LOAD_SAFETY_POLICY_WHEN_RULE_OFF = True
    SAFETY_POLICY_MODEL_PATH = DEFAULT_SAFETY_MODEL_PATH
    SAFETY_POLICY_OBS_DIM = 27
    SAFETY_POLICY_ACTION_DIM = 3
    SAFETY_POLICY_MOUTH_HOLE_RADIUS = 0.020
    SAFETY_POLICY_MOUTH_CHANNEL_RADIUS = SAFETY_POLICY_MOUTH_HOLE_RADIUS * 1.25
    SAFETY_POLICY_MOUTH_CHANNEL_FRONT = 0.050
    SAFETY_POLICY_MOUTH_CHANNEL_BACK = 0.030
    SAFETY_POLICY_LINK_THRESHOLD = 0.005
    SAFETY_POLICY_FORCE_SOFT_LIMIT = 0.80
    SAFETY_POLICY_TORQUE_SOFT_LIMIT = 0.80
    SAFETY_POLICY_AX_FORCE_LIMIT = 0.50
    SAFETY_POLICY_CLEAR_AXIAL_DIST = 0.150
    SAFETY_POLICY_RESET_CLEARANCE = 0.75
    SAFETY_POLICY_RETRACT_GAIN_MIN = 0.35
    SAFETY_POLICY_RETREAT_STEP = RULE_RETREAT_STEP
    SAFETY_POLICY_LATERAL_STEP = RULE_LATERAL_STEP
    SAFETY_POLICY_FACE_PROXY_ENABLED = True
    SAFETY_POLICY_FACE_PROXY_MIN_DEPTH = 0.0
    SAFETY_POLICY_FACE_PROXY_START_DIST = 0.085
    SAFETY_POLICY_FACE_PROXY_FULL_DIST = 0.010
    # 安全语义速记:
    # - lip_risk: 近嘴唇/入口边缘的减速信号，主要触发 slow / center，不直接进 RETREAT
    # - face/link/force/poke: 直接触发 RETREAT 锁存
    # - mouth_depth > 0: 才表示勺心沿嘴轴真正进入嘴内，成功门控也依赖这个正向 depth

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
        self.vision_mouth_pose_latched = False
        self.warned_missing_tf = False
        self.last_face_time_ns: int | None = None
        self.last_joint_state_time_ns: int | None = None
        self.last_speed_scaling_time_ns: int | None = None
        self.speed_scaling: float | None = None
        self.policy_step = 0
        self.episode_start_time_ns: int | None = None
        self.episode_stopped = False
        self.episode_stop_reason: str | None = None
        self.reset_started = False
        self.reset_finished = False
        self.reset_start_time_ns: int | None = None
        self.reset_reached_since_ns: int | None = None
        self.reset_timeout_s = 0.0
        self.fsm_state = "APPROACH"
        self.feed_stable_steps = 0
        self.success_hold_steps = 0
        self.success_hold_joint_pos: np.ndarray | None = None
        self.success_withdraw_start_spoon: np.ndarray | None = None
        self.success_withdraw_direction: np.ndarray | None = None
        self.success_withdraw_quat_wxyz: np.ndarray | None = None
        self.success_withdraw_start_time_ns: int | None = None
        self.success_withdraw_stable_steps = 0
        self.success_withdraw_complete = False
        self.success_withdraw_trajectory_sent = False
        self.safety_strengthen_active = False
        self.withdrawal_reason = "SUCCESS"
        self.withdraw_distance_m = float(args.withdraw_distance)
        self.last_fsm_metrics: dict[str, float | bool] = {}
        self.max_episode_steps = (
            args.max_episode_steps
            if args.max_episode_steps > 0
            else max(1, int(round(args.episode_length_s * args.control_hz)))
        )
        self.rule_safety_enabled = bool(self.RULE_SAFETY_ENABLED)
        # 规则安全关闭时才启用学习 safety actor，避免同一帧同时出现两套 safety 接管逻辑。
        self.learned_safety_enabled = bool(
            self.LOAD_SAFETY_POLICY_WHEN_RULE_OFF and not self.rule_safety_enabled
        )
        self.safety_policy: HarlArmPolicy | None = None
        if self.learned_safety_enabled:
            self.safety_policy = HarlArmPolicy(
                str(self.SAFETY_POLICY_MODEL_PATH),
                obs_dim=self.SAFETY_POLICY_OBS_DIM,
                action_dim=self.SAFETY_POLICY_ACTION_DIM,
                device=args.device,
                backend="auto",
            )
        self.previous_safety_action = np.zeros(self.SAFETY_POLICY_ACTION_DIM, dtype=np.float32)
        self.safety_slow_active = False  # 学习 safety actor 复用仿真中的 slow flag，作为下一帧安全观测的一部分。
        self.save_safety_gain_images = bool(self.SAVE_SAFETY_GAIN_IMAGES)
        self.save_full_safety_gain_curve = bool(self.SAVE_FULL_SAFETY_GAIN_CURVE)
        self.safety_gain_history_len = max(2, int(self.SAFETY_GAIN_HISTORY))
        self.safety_gain_image_interval = max(1, int(self.SAFETY_GAIN_IMAGE_INTERVAL))
        self.safety_gain_visualization_step = 0
        self.safety_gain_curve_rows: list[dict[str, float | str | bool]] = []  # 论文图: 保存完整 episode 的三增益和风险量，用于结束时画全程曲线。
        self.safety_gain_curve_saved = False  # 论文图: 防止 stop_episode 和 Ctrl-C 清理时重复写同一条曲线。
        self.rule_safety_retreat_active = False  # 规则/策略安全层共用: 危险触发后锁存撤回，直到净空和力觉都恢复安全。
        self.rule_safety_retreat_steps = 0  # 规则/策略安全层共用: 统计连续撤回步数，便于图像面板显示当前安全接管时长。
        self.rule_safety_last_state = self.default_rule_safety_state()  # 规则/策略安全层共用: 保存最近一次实际生效的三增益和风险指标。
        self.safety_gain_history = {
            "slow": deque(maxlen=self.safety_gain_history_len),
            "center": deque(maxlen=self.safety_gain_history_len),
            "retreat": deque(maxlen=self.safety_gain_history_len),
        }  # 可视化: 三个增益分别保留短历史，用于绘制实时曲线。
        self.safety_gain_window = "Safety Gains"
        self.safety_gain_visualization_enabled = bool(self.SAFETY_GAIN_VISUALIZATION and _HAS_CV2)
        if self.SAFETY_GAIN_VISUALIZATION and not _HAS_CV2:
            self.get_logger().warn("未能导入 cv2，安全增益图像可视化将自动关闭。")
        if self.safety_gain_visualization_enabled:
            try:
                cv2.namedWindow(self.safety_gain_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.safety_gain_window, 760, 460)
            except Exception as exc:
                self.safety_gain_visualization_enabled = False
                self.get_logger().warn(f"无法创建安全增益可视化窗口，已关闭图像显示: {exc}")
        if self.save_safety_gain_images or self.save_full_safety_gain_curve:
            SAFETY_GAIN_VIS_DIR.mkdir(parents=True, exist_ok=True)

        if args.fixed_mouth_pose:
            self.mouth_pos = self.DEFAULT_FIXED_MOUTH_POS.copy()
            self.fixed_mouth_axis = self.DEFAULT_FIXED_MOUTH_AXIS.copy()
            self.mouth_quat_wxyz = self.mouth_quat_from_policy_axis(self.fixed_mouth_axis)
            self.target_received = True
            self.last_face_time_ns = self.get_clock().now().nanoseconds

        self.joint_angle_recorder: JointAngleRecorder | None = None
        try:
            self.joint_angle_recorder = JointAngleRecorder(
                JOINT_ANGLE_RECORD_DIR,
                self.JOINT_NAMES,
                sample_period_s=self.control_period,
                mode="dry_run" if args.dry_run else "motion",
            )
            self.get_logger().info(
                f"六关节实际角度自动记录: csv={self.joint_angle_recorder.csv_path}；"
                "任务结束或 Ctrl-C 时保存曲线图，纵轴范围 [-2π, 2π] rad。"
            )
        except OSError as exc:
            self.get_logger().warn(f"无法创建关节角记录文件: {exc}")

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
        self.get_logger().info(
            f"safety_strenthen={self.safety_strenthen}: "
            "开启时有效 grec > 0 即锁定撤回 10 cm 再归位；"
            f"safety_grec_force_one={self.safety_grec_force_one} "
            f"（grec记录模式: {'固定为1' if self.safety_grec_force_one else '原安全策略输出'}）。"
        )
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
        if self.rule_safety_enabled:
            self.get_logger().info(
                "规则安全层已启用: "
                f"slow=[{self.RULE_SLOW_FULL_DIST:.3f}, {self.RULE_SLOW_START_DIST:.3f}]m, "
                f"center=[{self.RULE_CENTER_FULL_DIST:.3f}, {self.RULE_CENTER_START_DIST:.3f}]m, "
                f"retreat_step={self.RULE_RETREAT_STEP:.3f}m；"
                "安全增益会在策略 arm_delta 后、IK 前生效。"
            )
        elif self.learned_safety_enabled:
            assert self.safety_policy is not None
            self.get_logger().info(
                "规则安全层已关闭，改用学习 safety actor: "
                f"obs_dim={self.SAFETY_POLICY_OBS_DIM}, action_dim={self.SAFETY_POLICY_ACTION_DIM}, "
                f"retreat_step={self.SAFETY_POLICY_RETREAT_STEP:.3f}m, "
                f"lateral_step={self.SAFETY_POLICY_LATERAL_STEP:.3f}m。"
            )
            self.get_logger().info(f"safety policy request: {Path(self.SAFETY_POLICY_MODEL_PATH).expanduser().resolve()}")
            self.get_logger().info(f"safety policy loaded: {self.safety_policy.loaded_path}")
            self.get_logger().info(f"safety policy backend: {self.safety_policy.backend}")
            self.get_logger().info(
                "safety policy输入为27维: 嘴局部几何 + 风险量 + 阶段flag + 上一步safety动作；"
                "实机暂无完整face点云时，face_intrusion_risk先由6个嘴部特征点近似。"
            )
        else:
            self.get_logger().warn("规则安全层和学习 safety actor 均未启用: arm_delta 将不经过 near-mouth safety filter。")
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
        if self.joint_angle_recorder is not None:
            try:
                self.joint_angle_recorder.record(msg.joint_names, msg.feedback.positions)
            except OSError as exc:
                self.get_logger().warn(f"关节角 CSV 写入失败: {exc}", throttle_duration_sec=2.0)

    def save_joint_angle_curve(self, close_record: bool = False) -> None:
        recorder = self.joint_angle_recorder
        if recorder is None:
            return
        try:
            png_path = recorder.close() if close_record else recorder.save_plot()
            if png_path is not None:
                self.get_logger().info(
                    f"六关节角度曲线已保存: png={png_path}, csv={recorder.csv_path}, "
                    f"samples={recorder.sample_count}"
                )
            elif close_record:
                self.get_logger().warn(f"未收到完整有效的六关节反馈，关节角 CSV 只有表头: {recorder.csv_path}")
        except Exception as exc:
            self.get_logger().warn(f"关节角曲线保存失败: {exc}；已记录的数据见 {recorder.csv_path}")

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
        self.last_face_time_ns = self.get_clock().now().nanoseconds

        if self.LATCH_FIRST_VISION_MOUTH_POSE and self.vision_mouth_pose_latched:
            return

        self.mouth_pos = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
        self.mouth_quat_wxyz = quat_xyzw_to_wxyz(np.array([ori.x, ori.y, ori.z, ori.w], dtype=np.float32))
        if self.LATCH_FIRST_VISION_MOUTH_POSE:
            self.vision_mouth_pose_latched = True
            self.get_logger().info(
                "已锁定第一帧视觉嘴巴目标，本 episode 后续视觉消息只刷新时间戳，不再改变 mouth_pos/mouth_axis: "
                f"mouth_pos={np.round(self.mouth_pos, 4)}, mouth_axis={np.round(self.compute_mouth_axis(), 4)}"
            )
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
                    "policy_arm_delta_x",
                    "policy_arm_delta_y",
                    "policy_arm_delta_z",
                    "policy_arm_delta_rx",
                    "policy_arm_delta_ry",
                    "policy_arm_delta_rz",
                    "safety_mode",
                    "safety_slow_gain",
                    "safety_center_gain",
                    "safety_grec",
                    "safety_danger_score",
                    "safety_spoon_radial",
                    "safety_mouth_depth",
                    "safety_force_norm",
                    "safety_axial_force",
                    "safety_signed_axial_force",
                    "safety_wrench_sign",
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
        safety_state: dict[str, object] | None = None,
        policy_arm_delta: np.ndarray | None = None,
    ) -> None:
        if self.wrench_record_writer is None:
            return
        assert self.mouth_pos is not None
        if safety_state is None:
            safety_state = self.default_rule_safety_state()
        if policy_arm_delta is None:
            policy_arm_delta = np.asarray(arm_delta, dtype=np.float32)
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
            + np.asarray(policy_arm_delta, dtype=np.float32).astype(float).tolist()
            + [
                str(safety_state["mode"]),
                float(safety_state["slow_gain"]),
                float(safety_state["center_gain"]),
                float(safety_state["grec"]),
                float(safety_state["danger_score"]),
                float(safety_state["spoon_radial"]),
                float(safety_state["mouth_depth"]),
                float(safety_state["force_norm"]),
                float(safety_state["axial_force"]),
                float(safety_state["signed_axial_force"]),
                float(safety_state["wrench_sign"]),
            ]
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

    def default_rule_safety_state(self) -> dict[str, float | str | bool]:
        """生成安全层的默认状态，供未启用或尚未运行时的日志和可视化使用。"""
        return {
            "mode": "OFF",
            "slow_gain": 0.0,
            "center_gain": 0.0,
            "retreat_gain": 0.0,
            "grec": 0.0,
            "danger_score": 0.0,
            "lip_risk": 0.0,
            "link_risk": 0.0,
            "force_risk": 0.0,
            "poke_risk": 0.0,
            "spoon_mouth_dist": 0.0,
            "spoon_radial": 0.0,
            "mouth_depth": 0.0,
            "force_norm": 0.0,
            "torque_norm": 0.0,
            "axial_force": 0.0,
            "signed_axial_force": 0.0,
            "wrench_sign": 0.0,
            "safe_clearance": 0.0,
            "retreat_active": False,
            "retreat_steps": 0,
            "inside_mouth_channel": False,
            "safety_zone": False,
            "policy_delta_norm": 0.0,
            "final_delta_norm": 0.0,
        }

    @staticmethod
    def normalize_vector(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
        """把向量归一化，退化时使用 fallback，避免安全方向出现 NaN。"""
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 1.0e-6:
            return (vector / norm).astype(np.float32)
        if fallback is None:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.asarray(fallback, dtype=np.float32).copy()

    @staticmethod
    def inverse_ramp(value: float, start: float, full: float) -> float:
        """把越小越危险的距离映射到 0~1，start 外为 0，full 内为 1。"""
        if start <= full:
            return float(value <= full)
        return float(np.clip((start - value) / max(start - full, 1.0e-6), 0.0, 1.0))

    def build_mouth_axis_frame(self, mouth_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """围绕嘴巴开口轴构造局部坐标系，用于计算横向偏心和轴向深度。"""
        z_axis = self.normalize_vector(mouth_axis, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(ref, z_axis))) > 0.90:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        x_axis = self.normalize_vector(np.cross(ref, z_axis), np.array([0.0, 1.0, 0.0], dtype=np.float32))
        y_axis = self.normalize_vector(np.cross(z_axis, x_axis), np.array([0.0, 0.0, 1.0], dtype=np.float32))
        return x_axis, y_axis, z_axis

    def compute_safety_stage_flags(self) -> dict[str, float]:
        """把实机简化任务的 FSM 状态映射到 safety actor 训练时保留的阶段槽位。"""
        feed_reached = bool(self.last_fsm_metrics.get("feed_reached", False))
        entered_or_holding = feed_reached or self.feed_stable_steps > 0 or self.fsm_state in {"SUCCESS_HOLD", "DONE"}
        # 实机当前没有球/咬合/撤出观测，兼容旧槽位时只保留“已经入嘴”和“当前成功”两个可构建信号。
        return {
            "stage_entered": float(entered_or_holding),
            "stage_bite": 0.0,
            "withdraw_done": 0.0,
            "full_success": float(feed_reached or self.fsm_state in {"SUCCESS_HOLD", "DONE"}),
        }

    def compute_safety_policy_state(
        self,
        ee_quat_wxyz: np.ndarray,
        spoon_center: np.ndarray,
    ) -> dict[str, object]:
        """构建与仿真 _compute_safety_state 对齐的实机安全状态，供 learning safety actor 使用。"""
        assert self.mouth_pos is not None
        assert self.mouth_quat_wxyz is not None

        stage_flags = self.compute_safety_stage_flags()
        spoon_rel_w = np.nan_to_num(spoon_center - self.mouth_pos, nan=0.0, posinf=5.0, neginf=-5.0)
        # run_task 中的 mouth_quat 来自真实视觉约定，compute_mouth_axis() 才是已经对齐仿真策略观测的嘴巴开口轴。
        mouth_axis_w = self.normalize_vector(self.compute_mouth_axis(), np.array([1.0, 0.0, 0.0], dtype=np.float32))
        mouth_x, mouth_y, mouth_z = self.build_mouth_axis_frame(mouth_axis_w)
        spoon_rel_local = np.array(
            [
                float(np.dot(spoon_rel_w, mouth_x)),
                float(np.dot(spoon_rel_w, mouth_y)),
                float(np.dot(spoon_rel_w, mouth_z)),
            ],
            dtype=np.float32,
        )
        spoon_rel_local = np.nan_to_num(spoon_rel_local, nan=0.0, posinf=5.0, neginf=-5.0)
        retract_dir_w = -mouth_axis_w
        retract_dir_local = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        spoon_dir_w = self.normalize_vector(
            quat_apply_wxyz(ee_quat_wxyz, self.SPOON_FORWARD_AXIS_LOCAL),
            mouth_axis_w,
        )

        spoon_radial = float(np.linalg.norm(spoon_rel_local[:2]))
        spoon_mouth_dist = float(np.linalg.norm(spoon_rel_local))
        mouth_depth = float(spoon_rel_local[2])
        inside_mouth_channel = (
            spoon_radial < float(self.SAFETY_POLICY_MOUTH_CHANNEL_RADIUS)
            and mouth_depth > -float(self.SAFETY_POLICY_MOUTH_CHANNEL_BACK)
            and mouth_depth < float(self.SAFETY_POLICY_MOUTH_CHANNEL_FRONT)
        )
        safety_zone = (
            spoon_mouth_dist < float(self.RULE_SAFETY_ZONE_DIST)
            or inside_mouth_channel
            or stage_flags["stage_entered"] > 0.5
        )
        pre_insert = stage_flags["stage_entered"] <= 0.5

        lateral_to_center_local = np.array([-spoon_rel_local[0], -spoon_rel_local[1], 0.0], dtype=np.float32)
        lateral_to_center_local = self.normalize_vector(lateral_to_center_local, np.zeros(3, dtype=np.float32))
        lateral_to_center_w = (
            lateral_to_center_local[0] * mouth_x
            + lateral_to_center_local[1] * mouth_y
            + lateral_to_center_local[2] * mouth_z
        ).astype(np.float32)

        # lip_risk 关注两件事:
        # 1) spoon_radial 是否贴近嘴孔半径
        # 2) mouth_depth 是否接近 0，也就是还在嘴唇入口附近
        # 它用于近嘴减速/居中，不是 learned safety 的直接撤回条件。
        lip_gap = abs(spoon_radial - float(self.SAFETY_POLICY_MOUTH_HOLE_RADIUS))
        lip_z = abs(mouth_depth)
        lip_risk = (
            float(np.clip(1.0 - lip_gap / max(float(self.RULE_LIP_MARGIN), 1.0e-6), 0.0, 1.0))
            * float(np.clip(1.0 - lip_z / max(float(self.SAFETY_POLICY_MOUTH_CHANNEL_FRONT), 1.0e-6), 0.0, 1.0))
            * float(pre_insert)
        )

        mouth_feature_points = self.build_mouth_feature_points()
        feature_dists = np.linalg.norm(mouth_feature_points - spoon_center.reshape(1, 3), axis=1)
        min_feature_dist = float(np.min(feature_dists)) if feature_dists.size > 0 else float("inf")
        # 这个阈值是“预入嘴阶段的嘴部连杆/点云接触”判断，满足就直接算 hazard -> RETREAT。
        linkage_contact_before_insert = bool(min_feature_dist < float(self.SAFETY_POLICY_LINK_THRESHOLD) and pre_insert)
        linkage_intrusion_risk = (
            float(np.clip((float(self.SAFETY_POLICY_LINK_THRESHOLD) - min_feature_dist) / max(float(self.SAFETY_POLICY_LINK_THRESHOLD), 1.0e-6), 0.0, 1.0))
            * float(linkage_contact_before_insert)
        )

        # coarse face proxy: 这里比 lip_risk 更像“已经碰到脸/口周区域”的撤回信号，
        # 只在嘴前方且不在 mouth channel 内时启用。
        face_intrusion_risk = 0.0
        inside_face_proxy = False
        if (
            self.SAFETY_POLICY_FACE_PROXY_ENABLED
            and feature_dists.size > 0
            and safety_zone
            and not inside_mouth_channel
            and mouth_depth > float(self.SAFETY_POLICY_FACE_PROXY_MIN_DEPTH)
        ):
            face_intrusion_risk = self.inverse_ramp(
                min_feature_dist,
                float(self.SAFETY_POLICY_FACE_PROXY_START_DIST),
                float(self.SAFETY_POLICY_FACE_PROXY_FULL_DIST),
            )
            inside_face_proxy = face_intrusion_risk > 0.0

        policy_wrench = np.nan_to_num(self.wrist_wrench_policy_obs, nan=0.0, posinf=5.0, neginf=-5.0)
        force_obs = policy_wrench[:3].astype(np.float32)
        torque_obs = policy_wrench[3:].astype(np.float32)
        force_obs_norm = float(np.linalg.norm(force_obs))
        torque_obs_norm = float(np.linalg.norm(torque_obs))
        force_obs_w = quat_apply_wxyz(ee_quat_wxyz, force_obs)
        ax_force_obs = abs(float(np.dot(force_obs_w, spoon_dir_w)))
        # 力觉风险: 过大的腕力/力矩、或沿勺前进方向的异常轴向力。
        force_risk = (
            float(np.clip((force_obs_norm - float(self.SAFETY_POLICY_FORCE_SOFT_LIMIT)) / max(float(self.SAFETY_POLICY_FORCE_SOFT_LIMIT), 1.0e-6), 0.0, 1.0))
            + 0.5
            * float(np.clip((torque_obs_norm - float(self.SAFETY_POLICY_TORQUE_SOFT_LIMIT)) / max(float(self.SAFETY_POLICY_TORQUE_SOFT_LIMIT), 1.0e-6), 0.0, 1.0))
        )
        force_risk = float(np.clip(force_risk, 0.0, 1.0)) * float(safety_zone) * float(pre_insert)
        # poke_risk: 轴向顶入过大时触发，通常比 lip_risk 更接近“戳入喉咙”的语义。
        poke_risk = (
            float(np.clip((ax_force_obs - float(self.SAFETY_POLICY_AX_FORCE_LIMIT)) / max(float(self.SAFETY_POLICY_AX_FORCE_LIMIT), 1.0e-6), 0.0, 1.0))
            * float(pre_insert)
            * float(safety_zone)
        )

        safety_wrench_raw = self.get_sim_aligned_safety_wrench_raw()
        force_norm = float(np.linalg.norm(safety_wrench_raw[:3]))
        torque_norm = float(np.linalg.norm(safety_wrench_raw[3:]))
        raw_force_w = quat_apply_wxyz(ee_quat_wxyz, safety_wrench_raw[:3].astype(np.float32))
        signed_axial_force = float(np.dot(raw_force_w, spoon_dir_w))
        axial_force = abs(signed_axial_force)
        safe_clearance = max(float(np.dot(spoon_rel_w, retract_dir_w)), 0.0)
        safe_clearance_norm = float(
            np.clip(safe_clearance / max(float(self.SAFETY_POLICY_CLEAR_AXIAL_DIST), 1.0e-6), 0.0, 1.0)
        )
        danger_score = float(
            np.clip(max(face_intrusion_risk, linkage_intrusion_risk, force_risk, poke_risk), 0.0, 1.0)
        )

        return {
            "mode": "TRACK",
            "slow_gain": 0.0,
            "center_gain": 0.0,
            "retreat_gain": 0.0,
            "danger_score": danger_score,
            "lip_risk": float(lip_risk),
            "face_intrusion_risk": float(face_intrusion_risk),
            "link_risk": float(linkage_intrusion_risk),
            "linkage_intrusion_risk": float(linkage_intrusion_risk),
            "force_risk": float(force_risk),
            "poke_risk": float(poke_risk),
            "spoon_mouth_dist": spoon_mouth_dist,
            "spoon_radial": spoon_radial,
            "mouth_depth": mouth_depth,
            "force_norm": force_norm,
            "torque_norm": torque_norm,
            "axial_force": axial_force,
            "signed_axial_force": signed_axial_force,
            "wrench_sign": float(self.wrench_sign),
            "safe_clearance": safe_clearance,
            "safe_clearance_norm": safe_clearance_norm,
            "retreat_active": bool(self.rule_safety_retreat_active),
            "retreat_steps": int(self.rule_safety_retreat_steps),
            "inside_mouth_channel": bool(inside_mouth_channel),
            "inside_face_proxy": bool(inside_face_proxy),
            "linkage_contact_before_insert": bool(linkage_contact_before_insert),
            "safety_zone": bool(safety_zone),
            "policy_delta_norm": 0.0,
            "final_delta_norm": 0.0,
            "spoon_rel_local": spoon_rel_local,
            "retract_dir_local": retract_dir_local,
            "retract_dir_w": retract_dir_w,
            "spoon_dir_w": spoon_dir_w,
            "mouth_axis_w": mouth_axis_w,
            "lateral_to_center_w": lateral_to_center_w,
            **stage_flags,
        }

    def compute_safety_policy_observation(self, safety_state: dict[str, object]) -> np.ndarray:
        """按仿真 obs_safety_policy 的 27 维顺序拼接实机 safety actor 观测。"""
        scalar_obs = np.array(
            [
                float(safety_state["lip_risk"]),
                float(safety_state["face_intrusion_risk"]),
                float(safety_state["linkage_intrusion_risk"]),
                float(safety_state["force_risk"]),
                float(safety_state["poke_risk"]),
                float(safety_state["safe_clearance_norm"]),
                float(safety_state["stage_entered"]),
                float(safety_state["stage_bite"]),
                float(safety_state["withdraw_done"]),
                float(safety_state["full_success"]),
                float(self.safety_slow_active),
                float(self.rule_safety_retreat_active),
            ],
            dtype=np.float32,
        )
        obs = np.concatenate(
            [
                np.asarray(safety_state["spoon_rel_local"], dtype=np.float32).reshape(3),
                np.asarray(safety_state["retract_dir_local"], dtype=np.float32).reshape(3),
                np.asarray(safety_state["spoon_dir_w"], dtype=np.float32).reshape(3),
                np.asarray(safety_state["mouth_axis_w"], dtype=np.float32).reshape(3),
                scalar_obs,
                self.previous_safety_action.astype(np.float32).reshape(self.SAFETY_POLICY_ACTION_DIM),
            ],
            dtype=np.float32,
        )
        if obs.size != self.SAFETY_POLICY_OBS_DIM:
            raise RuntimeError(f"safety观测维度错误: expected={self.SAFETY_POLICY_OBS_DIM}, actual={obs.size}")
        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)

    def learned_safety_hazard_now(self, safety_state: dict[str, object]) -> bool:
        """复现仿真 safety hazard mask，决定学习增益是否进入撤回锁存。"""
        # 这里是“直接进入 RETREAT”的四类 hazard。
        # lip_risk 不在这里，它只会在后面的 slow_hazard 里触发减速/居中。
        face_hazard = (
            bool(safety_state["safety_zone"])
            and not bool(safety_state["inside_mouth_channel"])
            and float(safety_state["face_intrusion_risk"]) > float(self.RULE_TRIGGER_SCORE)
        )
        link_hazard = bool(safety_state["linkage_contact_before_insert"])
        force_hazard = float(safety_state["force_risk"]) > float(self.RULE_TRIGGER_SCORE)
        poke_hazard = float(safety_state["poke_risk"]) > float(self.RULE_TRIGGER_SCORE)
        return bool(face_hazard or link_hazard or force_hazard or poke_hazard)

    def apply_learned_safety_filter(
        self,
        ee_pos: np.ndarray,
        ee_quat_wxyz: np.ndarray,
        spoon_center: np.ndarray,
        arm_delta: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """关闭规则安全后，使用学习 safety actor 输出的三通道增益融合 arm_delta。"""
        del ee_pos  # 学习 safety 观测以勺心和嘴局部几何为主，ee_pos 由外层 delta safety 继续检查。
        arm_delta = np.asarray(arm_delta, dtype=np.float32).copy()
        if not self.learned_safety_enabled or self.safety_policy is None or self.mouth_pos is None:
            safety_state = self.default_rule_safety_state()
            safety_state["policy_delta_norm"] = float(np.linalg.norm(arm_delta[:3]))
            safety_state["final_delta_norm"] = float(np.linalg.norm(arm_delta[:3]))
            self.previous_safety_action.fill(0.0)
            self.safety_slow_active = False
            self.rule_safety_last_state = safety_state
            return arm_delta, safety_state

        safety_state = self.compute_safety_policy_state(ee_quat_wxyz, spoon_center)
        safety_obs = self.compute_safety_policy_observation(safety_state)
        raw_safety_action = np.asarray(self.safety_policy.act(safety_obs), dtype=np.float32).reshape(-1)
        if raw_safety_action.size != self.SAFETY_POLICY_ACTION_DIM:
            raise RuntimeError(
                f"safety policy 动作维度错误: expected={self.SAFETY_POLICY_ACTION_DIM}, actual={raw_safety_action.size}"
            )
        safety_action = np.clip(raw_safety_action, -1.0, 1.0).astype(np.float32)
        safety_gain = np.clip((safety_action + 1.0) * 0.5, 0.0, 1.0)
        safety_slow_gain = float(safety_gain[0])
        safety_ret_gain = float(safety_gain[1])
        safety_lateral_gain = float(safety_gain[2])

        hazard_now = self.learned_safety_hazard_now(safety_state)
        if hazard_now:
            self.rule_safety_retreat_active = True
        elif self.rule_safety_retreat_active:
            retreat_clear = float(safety_state["safe_clearance_norm"]) > float(self.SAFETY_POLICY_RESET_CLEARANCE)
            risk_clear = float(safety_state["danger_score"]) < float(self.RULE_CLEAR_SCORE)
            proxy_clear = not bool(safety_state["inside_face_proxy"]) and not bool(safety_state["linkage_contact_before_insert"])
            if retreat_clear and risk_clear and proxy_clear:
                self.rule_safety_retreat_active = False

        # lip_risk 只负责减速/居中，不会单独把状态打进 RETREAT。
        slow_hazard = float(safety_state["lip_risk"]) > float(self.RULE_TRIGGER_SCORE) and not hazard_now
        self.safety_slow_active = bool(slow_hazard and not self.rule_safety_retreat_active)
        if self.rule_safety_retreat_active:
            self.rule_safety_retreat_steps += 1
        else:
            self.rule_safety_retreat_steps = 0

        arm_lin = arm_delta[:3].copy()
        lin_norm = float(np.linalg.norm(arm_lin))
        retract_dir_w = np.asarray(safety_state["retract_dir_w"], dtype=np.float32).reshape(3)
        fwd_dir = -retract_dir_w
        fwd_mag = max(float(np.dot(arm_lin, fwd_dir)), 0.0)
        lateral_need = float(
            np.clip(
                (float(safety_state["spoon_radial"]) - float(self.SAFETY_POLICY_MOUTH_HOLE_RADIUS) * 0.5)
                / max(float(self.RULE_LIP_MARGIN), 1.0e-6),
                0.0,
                1.0,
            )
        )
        lateral_risk = max(
            float(safety_state["lip_risk"]),
            float(safety_state["face_intrusion_risk"]),
            float(safety_state["linkage_intrusion_risk"]),
        )
        lateral_gain = (
            safety_lateral_gain
            * max(lateral_need, lateral_risk)
            * float(bool(safety_state["safety_zone"]))
        )
        lateral_vec = (
            np.asarray(safety_state["lateral_to_center_w"], dtype=np.float32).reshape(3)
            * (float(self.SAFETY_POLICY_LATERAL_STEP) * lateral_gain)
        )
        slow_gain = float(
            np.clip((0.25 + 0.75 * safety_slow_gain) * float(safety_state["lip_risk"]), 0.0, 0.95)
        )

        effective_slow_gain = 0.0
        effective_center_gain = 0.0
        effective_retreat_gain = 0.0
        mode = "TRACK"
        if self.rule_safety_retreat_active:
            effective_retreat_gain = max(safety_ret_gain, float(self.SAFETY_POLICY_RETRACT_GAIN_MIN))
            effective_center_gain = lateral_gain
            arm_lin = retract_dir_w * (float(self.SAFETY_POLICY_RETREAT_STEP) * effective_retreat_gain) + lateral_vec
            arm_delta[3:] = 0.0
            mode = "RETREAT"
        elif self.safety_slow_active:
            effective_slow_gain = slow_gain
            effective_center_gain = lateral_gain
            arm_lin = arm_lin - effective_slow_gain * fwd_mag * fwd_dir + lateral_vec
            if effective_center_gain > 0.05 and effective_slow_gain > 0.05:
                mode = "SLOW_CENTER"
            elif effective_center_gain > 0.05:
                mode = "CENTER"
            else:
                mode = "SLOW"

        # 学习策略只给增益，实机实际位移仍走 max_cartesian_step 保护，避免 safety 补偿绕过原有硬边界。
        filtered_norm = float(np.linalg.norm(arm_lin))
        if filtered_norm > float(self.args.max_cartesian_step):
            arm_lin = arm_lin * (float(self.args.max_cartesian_step) / max(filtered_norm, 1.0e-6))
            filtered_norm = float(np.linalg.norm(arm_lin))
        arm_delta[:3] = arm_lin.astype(np.float32)

        safety_state.update(
            {
                "mode": mode,
                "slow_gain": float(effective_slow_gain),
                "center_gain": float(effective_center_gain),
                "retreat_gain": float(effective_retreat_gain),
                "retreat_active": bool(self.rule_safety_retreat_active),
                "retreat_steps": int(self.rule_safety_retreat_steps),
                "policy_delta_norm": lin_norm,
                "final_delta_norm": filtered_norm,
            }
        )
        self.previous_safety_action = safety_action.copy()
        self.publish_safety_state(safety_state)
        return arm_delta, safety_state

    def apply_rule_based_safety_filter(
        self,
        ee_pos: np.ndarray,
        ee_quat_wxyz: np.ndarray,
        spoon_center: np.ndarray,
        arm_delta: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float | str | bool]]:
        """在策略 arm_delta 之后施加确定性安全规则，模拟 safety agent 的三通道增益。"""
        arm_delta = np.asarray(arm_delta, dtype=np.float32).copy()
        if not self.rule_safety_enabled or self.mouth_pos is None:
            safety_state = self.default_rule_safety_state()
            safety_state["policy_delta_norm"] = float(np.linalg.norm(arm_delta[:3]))
            safety_state["final_delta_norm"] = float(np.linalg.norm(arm_delta[:3]))
            self.rule_safety_last_state = safety_state
            return arm_delta, safety_state

        mouth_axis = self.normalize_vector(self.compute_mouth_axis(), np.array([1.0, 0.0, 0.0], dtype=np.float32))
        mouth_x, mouth_y, mouth_z = self.build_mouth_axis_frame(mouth_axis)
        rel_spoon = np.nan_to_num(spoon_center - self.mouth_pos, nan=0.0, posinf=5.0, neginf=-5.0)
        spoon_local = np.array(
            [
                float(np.dot(rel_spoon, mouth_x)),
                float(np.dot(rel_spoon, mouth_y)),
                float(np.dot(rel_spoon, mouth_z)),
            ],
            dtype=np.float32,
        )
        spoon_radial_vec_w = spoon_local[0] * mouth_x + spoon_local[1] * mouth_y
        spoon_radial = float(np.linalg.norm(spoon_radial_vec_w))
        spoon_mouth_dist = float(np.linalg.norm(rel_spoon))
        mouth_depth = float(spoon_local[2])
        lateral_to_center_w = (
            -spoon_radial_vec_w / max(spoon_radial, 1.0e-6)
            if spoon_radial > 1.0e-6
            else np.zeros(3, dtype=np.float32)
        )

        channel_radius = float(self.RULE_MOUTH_CHANNEL_RADIUS)
        inside_mouth_channel = (
            spoon_radial < channel_radius
            and mouth_depth > -float(self.RULE_MOUTH_CHANNEL_BACK)
            and mouth_depth < float(self.RULE_MOUTH_CHANNEL_FRONT)
        )
        safety_zone = spoon_mouth_dist < float(self.RULE_SAFETY_ZONE_DIST) or inside_mouth_channel

        # lip_risk 只表示接近嘴唇环带的预防性减速需求，不直接等同于必须撤回。
        lip_gap = abs(spoon_radial - channel_radius)
        lip_z = abs(mouth_depth)
        lip_risk = (
            self.inverse_ramp(lip_gap, float(self.RULE_LIP_MARGIN), 0.0)
            * self.inverse_ramp(lip_z, float(self.RULE_MOUTH_CHANNEL_FRONT), 0.0)
            * float(safety_zone)
        )

        mouth_feature_points = self.build_mouth_feature_points()
        feature_dists = np.linalg.norm(mouth_feature_points - spoon_center.reshape(1, 3), axis=1)
        min_feature_dist = float(np.min(feature_dists)) if feature_dists.size > 0 else float("inf")
        link_risk = (
            self.inverse_ramp(min_feature_dist, float(self.RULE_LINK_THRESHOLD), 0.0)
            * float(not inside_mouth_channel)
            * float(safety_zone)
        )

        safety_wrench_raw = self.get_sim_aligned_safety_wrench_raw()
        force_norm = float(np.linalg.norm(safety_wrench_raw[:3]))
        torque_norm = float(np.linalg.norm(safety_wrench_raw[3:]))
        spoon_axis_w = self.normalize_vector(
            quat_apply_wxyz(ee_quat_wxyz, self.SPOON_FORWARD_AXIS_LOCAL),
            mouth_axis,
        )
        # safety 与策略共用仿真方向约定的 wrench，避免实机传感器作用力方向和仿真关节反力方向相反时把轴向戳碰判反。
        force_w = quat_apply_wxyz(ee_quat_wxyz, safety_wrench_raw[:3].astype(np.float32))
        signed_axial_force = float(np.dot(force_w, spoon_axis_w))
        axial_force = abs(signed_axial_force)
        force_risk = float(np.clip((force_norm - float(self.RULE_FORCE_THRESHOLD_N)) / max(float(self.RULE_FORCE_THRESHOLD_N), 1.0e-6), 0.0, 1.0))
        torque_risk = float(np.clip((torque_norm - float(self.RULE_TORQUE_THRESHOLD_NM)) / max(float(self.RULE_TORQUE_THRESHOLD_NM), 1.0e-6), 0.0, 1.0))
        force_risk = max(force_risk, 0.5 * torque_risk)
        poke_risk = float(np.clip((axial_force - float(self.RULE_AXIAL_FORCE_THRESHOLD_N)) / max(float(self.RULE_AXIAL_FORCE_THRESHOLD_N), 1.0e-6), 0.0, 1.0))
        force_risk *= float(safety_zone)
        poke_risk *= float(safety_zone)
        danger_score = float(np.clip(max(link_risk, force_risk, poke_risk), 0.0, 1.0))

        lin_delta = arm_delta[:3].copy()
        lin_norm = float(np.linalg.norm(lin_delta))
        inward_dir_w = mouth_axis
        retract_dir_w = -mouth_axis
        inward_mag = max(float(np.dot(lin_delta, inward_dir_w)), 0.0)
        inward_ratio = inward_mag / max(lin_norm, 1.0e-6)

        slow_near_gain = self.inverse_ramp(
            spoon_mouth_dist,
            float(self.RULE_SLOW_START_DIST),
            float(self.RULE_SLOW_FULL_DIST),
        )
        slow_gain = max(slow_near_gain, lip_risk)
        slow_gain = float(np.clip(slow_gain * (0.35 + 0.65 * inward_ratio) * float(safety_zone), 0.0, 0.95))

        radial_need = float(
            np.clip(
                (spoon_radial - float(self.RULE_CENTER_DEADBAND))
                / max(float(self.RULE_CENTER_FULL_RADIAL) - float(self.RULE_CENTER_DEADBAND), 1.0e-6),
                0.0,
                1.0,
            )
        )
        center_near_gain = self.inverse_ramp(
            spoon_mouth_dist,
            float(self.RULE_CENTER_START_DIST),
            float(self.RULE_CENTER_FULL_DIST),
        )
        center_gain = float(np.clip(center_near_gain * radial_need * float(safety_zone), 0.0, 1.0))

        geometry_danger = (
            safety_zone
            and spoon_mouth_dist < float(self.RULE_RETREAT_START_DIST)
            and not inside_mouth_channel
        )
        hazard_now = danger_score >= float(self.RULE_TRIGGER_SCORE) or geometry_danger
        safe_clearance = max(float(np.dot(rel_spoon, retract_dir_w)), 0.0)
        if hazard_now:
            self.rule_safety_retreat_active = True
        elif self.rule_safety_retreat_active:
            clear_enough = safe_clearance > float(self.RULE_RETREAT_RELEASE_DIST)
            force_clear = force_norm < float(self.RULE_FORCE_RELEASE_N)
            danger_clear = danger_score < float(self.RULE_CLEAR_SCORE)
            if clear_enough and force_clear and danger_clear:
                self.rule_safety_retreat_active = False

        if self.rule_safety_retreat_active:
            self.rule_safety_retreat_steps += 1
        else:
            self.rule_safety_retreat_steps = 0

        retreat_gain = 0.0
        filtered_lin_delta = lin_delta
        mode = "TRACK"
        if self.rule_safety_retreat_active:
            retreat_gain = float(
                np.clip(
                    max(float(self.RULE_MIN_RETREAT_GAIN), danger_score, float(geometry_danger)),
                    0.0,
                    1.0,
                )
            )
            center_vec = lateral_to_center_w * (float(self.RULE_LATERAL_STEP) * center_gain)
            filtered_lin_delta = retract_dir_w * (float(self.RULE_RETREAT_STEP) * retreat_gain) + center_vec
            arm_delta[3:] = 0.0
            mode = "RETREAT"
        else:
            if slow_gain > 0.0 and inward_mag > 0.0:
                filtered_lin_delta = filtered_lin_delta - slow_gain * inward_mag * inward_dir_w
            if center_gain > 0.0:
                filtered_lin_delta = filtered_lin_delta + lateral_to_center_w * (float(self.RULE_LATERAL_STEP) * center_gain)
            if center_gain > 0.05 and slow_gain > 0.05:
                mode = "SLOW_CENTER"
            elif center_gain > 0.05:
                mode = "CENTER"
            elif slow_gain > 0.05:
                mode = "SLOW"

        # 规则层叠加后再次限制平移步长，保证不会绕过原有单步笛卡尔安全边界。
        filtered_norm = float(np.linalg.norm(filtered_lin_delta))
        if filtered_norm > float(self.args.max_cartesian_step):
            filtered_lin_delta = filtered_lin_delta * (float(self.args.max_cartesian_step) / max(filtered_norm, 1.0e-6))
            filtered_norm = float(np.linalg.norm(filtered_lin_delta))
        arm_delta[:3] = filtered_lin_delta.astype(np.float32)

        safety_state = {
            "mode": mode,
            "slow_gain": float(slow_gain),
            "center_gain": float(center_gain),
            "retreat_gain": float(retreat_gain),
            "danger_score": float(danger_score),
            "lip_risk": float(lip_risk),
            "link_risk": float(link_risk),
            "force_risk": float(force_risk),
            "poke_risk": float(poke_risk),
            "spoon_mouth_dist": spoon_mouth_dist,
            "spoon_radial": spoon_radial,
            "mouth_depth": mouth_depth,
            "force_norm": force_norm,
            "torque_norm": torque_norm,
            "axial_force": axial_force,
            "signed_axial_force": signed_axial_force,
            "wrench_sign": float(self.wrench_sign),
            "safe_clearance": safe_clearance,
            "retreat_active": bool(self.rule_safety_retreat_active),
            "retreat_steps": int(self.rule_safety_retreat_steps),
            "inside_mouth_channel": bool(inside_mouth_channel),
            "safety_zone": bool(safety_zone),
            "policy_delta_norm": lin_norm,
            "final_delta_norm": filtered_norm,
        }
        self.publish_safety_state(safety_state)
        return arm_delta, safety_state

    def publish_safety_state(self, safety_state: dict[str, object]) -> None:
        """记录值与触发值分开，切换 grec 记录模式不会改变撤回/归位控制。"""
        original_gain = float(safety_state["retreat_gain"])
        safety_state["grec"] = original_gain
        pending_strengthen = self.safety_strenthen and not self.episode_stopped and original_gain > 0.0
        if self.safety_strengthen_active or pending_strengthen:
            # 保留原策略的危险锁存和 retreat_gain，供后续推理和首次接管判断使用。
            safety_state.update({
                "mode": "SAFETY_RESET" if self.reset_started else "SAFETY_WITHDRAW",
                "slow_gain": 0.0, "center_gain": 0.0,
                "grec": 1.0 if self.safety_grec_force_one else original_gain,
                "retreat_active": True,
            })
        self.rule_safety_last_state = safety_state
        self.record_safety_gain_curve_sample(safety_state)
        self.update_safety_gain_visualization(safety_state)

    def maybe_start_safety_strengthen(
        self, spoon_center: np.ndarray, ee_quat_wxyz: np.ndarray, safety_state: dict[str, object]
    ) -> bool:
        """有效 grec 首次触发后接管整条轨迹，当前策略动作不再下发。"""
        if not self.safety_strenthen or float(safety_state["retreat_gain"]) <= 0.0:
            return False
        if self.safety_strengthen_active:
            return True
        self.get_logger().warn(
            "安全加强模式触发: 有效 grec > 0，停止策略接近，立即撤回 10 cm 后归位；"
            f"grec记录模式={'固定为1' if self.safety_grec_force_one else '原安全策略输出'}。"
        )
        self.start_success_withdraw(spoon_center, ee_quat_wxyz, safety_triggered=True)
        if not self.episode_stopped:
            self.update_success_withdraw()
        return True

    def update_strengthened_safety_state(self, spoon_center: np.ndarray, ee_quat_wxyz: np.ndarray) -> None:
        """持续记录增益；原输出模式只运行 safety 推理，忽略其运动动作。"""
        if not self.safety_strengthen_active:
            return
        if self.learned_safety_enabled:
            if self.safety_grec_force_one:
                safety_state = self.compute_safety_policy_state(ee_quat_wxyz, spoon_center)
            else:
                self.compute_wrist_wrench_observation()
                ee_pos = spoon_center - quat_apply_wxyz(ee_quat_wxyz, self.SPOON_OFFSET)
                self.apply_learned_safety_filter(ee_pos, ee_quat_wxyz, spoon_center, np.zeros(6, dtype=np.float32))
                return
        elif self.rule_safety_enabled:
            ee_pos = spoon_center - quat_apply_wxyz(ee_quat_wxyz, self.SPOON_OFFSET)
            # 仅计算当前风险/记录 grec，不发布该规则动作。
            self.apply_rule_based_safety_filter(ee_pos, ee_quat_wxyz, spoon_center, np.zeros(6, dtype=np.float32))
            return
        else:
            safety_state = self.rule_safety_last_state.copy()
            metrics = self.compute_fsm_metrics(spoon_center, ee_quat_wxyz)
            safety_state.update({
                "spoon_mouth_dist": metrics["mouth_spoon_dist"],
                "spoon_radial": metrics["radial"], "mouth_depth": metrics["mouth_depth"],
            })
        self.rule_safety_retreat_steps += 1
        safety_state["retreat_steps"] = self.rule_safety_retreat_steps
        self.publish_safety_state(safety_state)

    @staticmethod
    def draw_gain_bar(canvas: np.ndarray, top_left: tuple[int, int], width: int, label: str, value: float, color: tuple[int, int, int]) -> None:
        """在增益面板上画一条 0~1 的条形图。"""
        assert cv2 is not None
        x, y = top_left
        value = float(np.clip(value, 0.0, 1.0))
        cv2.putText(canvas, label, (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (235, 235, 235), 1, cv2.LINE_AA)
        bar_x = x + 110
        cv2.rectangle(canvas, (bar_x, y), (bar_x + width, y + 24), (80, 80, 80), 1)
        fill_w = int(round(width * value))
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x, y), (bar_x + fill_w, y + 24), color, -1)
        cv2.putText(canvas, f"{value:.2f}", (bar_x + width + 14, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)

    def record_safety_gain_curve_sample(self, safety_state: dict[str, float | str | bool]) -> None:
        """记录完整 episode 的三通道安全增益，供结束时生成论文曲线。"""
        if not self.save_full_safety_gain_curve:
            return
        if self.episode_start_time_ns is None:
            time_s = float(self.policy_step) * self.control_period
        else:
            time_s = (self.get_clock().now().nanoseconds - self.episode_start_time_ns) * 1.0e-9
        self.safety_gain_curve_rows.append(
            {
                "step": float(self.policy_step),
                "time_s": float(time_s),
                "gslow": float(safety_state["slow_gain"]),
                "glat": float(safety_state["center_gain"]),
                "grec": float(safety_state["grec"]),
                "danger_score": float(safety_state["danger_score"]),
                "lip_risk": float(safety_state["lip_risk"]),
                "link_risk": float(safety_state["link_risk"]),
                "force_risk": float(safety_state["force_risk"]),
                "poke_risk": float(safety_state["poke_risk"]),
                "spoon_mouth_dist": float(safety_state["spoon_mouth_dist"]),
                "spoon_radial": float(safety_state["spoon_radial"]),
                "mouth_depth": float(safety_state["mouth_depth"]),
                "force_norm": float(safety_state["force_norm"]),
                "axial_force": float(safety_state["axial_force"]),
                "signed_axial_force": float(safety_state["signed_axial_force"]),
                "mode": str(safety_state["mode"]),
                "safety_zone": bool(safety_state["safety_zone"]),
                "retreat_active": bool(safety_state["retreat_active"]),
            }
        )

    def save_safety_gain_curve_outputs(self, reason_label: str = "shutdown") -> None:
        """保存完整三增益曲线 PNG 和 CSV，便于论文制图和后续复现实机曲线。"""
        if not self.save_full_safety_gain_curve or self.safety_gain_curve_saved or len(self.safety_gain_curve_rows) == 0:
            return
        SAFETY_GAIN_VIS_DIR.mkdir(parents=True, exist_ok=True)
        label = "".join(ch if ch.isalnum() else "_" for ch in reason_label.lower()).strip("_") or "episode"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = SAFETY_GAIN_VIS_DIR / f"safety_gain_curve_{stamp}_{label}"
        csv_path = base_path.with_suffix(".csv")
        png_path = base_path.with_suffix(".png")
        fieldnames = [
            "step",
            "time_s",
            "gslow",
            "glat",
            "grec",
            "danger_score",
            "lip_risk",
            "link_risk",
            "force_risk",
            "poke_risk",
            "spoon_mouth_dist",
            "spoon_radial",
            "mouth_depth",
            "force_norm",
            "axial_force",
            "signed_axial_force",
            "mode",
            "safety_zone",
            "retreat_active",
        ]
        try:
            with csv_path.open("w", newline="") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.safety_gain_curve_rows)
        except OSError as exc:
            self.get_logger().warn(f"安全增益完整曲线 CSV 保存失败: {exc}")

        plotted = self.plot_safety_gain_curve_matplotlib(png_path)
        if not plotted:
            plotted = self.plot_safety_gain_curve_cv2(png_path)
        if plotted:
            self.get_logger().info(f"安全增益完整曲线已保存: png={png_path}, csv={csv_path}")
        else:
            self.get_logger().warn(f"安全增益完整曲线 PNG 保存失败，但 CSV 仍可用于重画: {csv_path}")
        self.safety_gain_curve_saved = True

    @staticmethod
    def _rows_to_gain_array(rows: list[dict[str, float | str | bool]]) -> np.ndarray:
        """把三条安全增益整理成 N x 3 数组，便于分段和异常点检测。"""
        if len(rows) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(
            [
                [float(row["gslow"]), float(row["glat"]), float(row["grec"])]
                for row in rows
            ],
            dtype=np.float32,
        )

    def build_safety_gain_curve_plot_rows(self) -> list[dict[str, float | str | bool]]:
        """剔除一个明显的孤立跳点后再绘图，避免曲线首段出现不自然的折点。"""
        rows = list(self.safety_gain_curve_rows)
        outlier_idx = self.find_safety_gain_curve_outlier_index(rows)
        if outlier_idx is None:
            return rows
        return rows[:outlier_idx] + rows[outlier_idx + 1 :]

    def find_safety_gain_curve_outlier_index(self, rows: list[dict[str, float | str | bool]] | None = None) -> int | None:
        """找出一个最像“孤立异常点”的样本索引，只用于画图，不改 CSV 原始记录。"""
        plot_rows = self.safety_gain_curve_rows if rows is None else rows
        row_count = len(plot_rows)
        if row_count < 3:
            return None

        gains = self._rows_to_gain_array(plot_rows)
        best_idx = -1
        best_score = 0.0
        for idx in range(1, row_count - 1):
            local_score = 0.0
            for col in range(gains.shape[1]):
                neighborhood = gains[idx - 1 : idx + 2, col]
                local_median = float(np.median(neighborhood))
                local_score += abs(float(gains[idx, col]) - local_median)
            if local_score > best_score:
                best_score = local_score
                best_idx = idx

        # 只把足够明显的单点跳变当成异常点，避免误删真实峰值。
        if best_score < 0.15:
            return None
        return best_idx

    def build_safety_gain_curve_stage_axis(self, rows: list[dict[str, float | str | bool]] | None = None) -> np.ndarray:
        """把完整曲线样本压缩到 approach / feeding / retreat 三段横轴上。"""
        plot_rows = self.safety_gain_curve_rows if rows is None else rows
        row_count = len(plot_rows)
        if row_count == 0:
            return np.zeros(0, dtype=np.float32)
        if row_count < 3:
            return np.linspace(0.5, 2.5, row_count, dtype=np.float32)

        approach_end = next(
            (idx for idx, row in enumerate(plot_rows) if float(row.get("mouth_depth", 0.0)) > 0.0),
            max(1, row_count // 3),
        )
        approach_end = min(max(int(approach_end), 1), row_count - 1)

        gains = self._rows_to_gain_array(plot_rows)
        peak_idx = int(np.argmax(np.max(gains, axis=1)))
        feeding_end = min(row_count, max(peak_idx + 1, approach_end + 1))

        stage_sizes = (
            max(approach_end, 1),
            max(feeding_end - approach_end, 1),
            max(row_count - feeding_end, 1),
        )
        stage_x = np.empty(row_count, dtype=np.float32)
        for idx in range(row_count):
            if idx < approach_end:
                stage_idx = 0
                stage_pos = idx
                stage_size = stage_sizes[0]
            elif idx < feeding_end:
                stage_idx = 1
                stage_pos = idx - approach_end
                stage_size = stage_sizes[1]
            else:
                stage_idx = 2
                stage_pos = idx - feeding_end
                stage_size = stage_sizes[2]
            stage_x[idx] = stage_idx + (float(stage_pos) + 0.5) / float(stage_size)
        return stage_x

    def plot_safety_gain_curve_matplotlib(self, png_path: Path) -> bool:
        """优先用 matplotlib 输出论文可用的高分辨率三增益曲线。"""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.get_logger().warn(f"matplotlib 不可用，将尝试 OpenCV 兜底绘制安全增益曲线: {exc}")
            return False

        try:
            plot_rows = self.build_safety_gain_curve_plot_rows()
            stage_x = self.build_safety_gain_curve_stage_axis(plot_rows)
            fig, axis = plt.subplots(figsize=(9.0, 4.8))
            axis.plot(stage_x, [float(row["gslow"]) for row in plot_rows], color="#1f77b4", linewidth=2.1, label="gslow")
            axis.plot(stage_x, [float(row["glat"]) for row in plot_rows], color="#2ca02c", linewidth=2.1, label="glat")
            axis.plot(stage_x, [float(row["grec"]) for row in plot_rows], color="#d62728", linewidth=2.1, label="grec")
            axis.legend()
            for boundary in (1.0, 2.0):
                axis.axvline(boundary, color="#b0b0b0", linestyle="--", linewidth=1.0, alpha=0.45)
            axis.set_xlim(0.0, 3.0)
            axis.set_xticks([0.5, 1.5, 2.5], ["approach", "feeding", "retreat"])
            axis.set_xlabel("Stage")
            axis.set_ylabel("Gain")
            axis.set_ylim(-0.03, 1.05)
            axis.grid(True, alpha=0.25)
            axis.set_title("Safety Gain Profiles by Stage")
            fig.tight_layout()
            fig.savefig(png_path, dpi=max(80, int(self.SAFETY_GAIN_CURVE_DPI)))
            plt.close(fig)
            return True
        except Exception as exc:
            self.get_logger().warn(f"matplotlib 绘制安全增益完整曲线失败: {exc}")
            return False

    def plot_safety_gain_curve_cv2(self, png_path: Path) -> bool:
        """OpenCV 兜底保存三增益曲线，避免缺少 matplotlib 时完全没有图片。"""
        if cv2 is None:
            return False
        try:
            canvas = np.full((700, 1200, 3), 255, dtype=np.uint8)
            plot_x, plot_y, plot_w, plot_h = 90, 90, 1040, 500
            cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (35, 35, 35), 2)
            cv2.putText(canvas, "Safety Gain Profiles by Stage", (90, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25, 25, 25), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Stage", (560, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (25, 25, 25), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Gain", (18, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (25, 25, 25), 2, cv2.LINE_AA)
            for frac in (0.25, 0.5, 0.75):
                yy = plot_y + int(round(plot_h * (1.0 - frac)))
                cv2.line(canvas, (plot_x, yy), (plot_x + plot_w, yy), (220, 220, 220), 1)
                cv2.putText(canvas, f"{frac:.2f}", (38, yy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv2.LINE_AA)
            plot_rows = self.build_safety_gain_curve_plot_rows()
            stage_x = self.build_safety_gain_curve_stage_axis(plot_rows)
            for boundary_frac in (1.0 / 3.0, 2.0 / 3.0):
                xx = plot_x + int(round(plot_w * boundary_frac))
                cv2.line(canvas, (xx, plot_y), (xx, plot_y + plot_h), (205, 205, 205), 1, cv2.LINE_AA)
            for idx, stage_name in enumerate(("approach", "feeding", "retreat")):
                center_x = plot_x + int(round(plot_w * ((idx + 0.5) / 3.0)))
                text_size = cv2.getTextSize(stage_name, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 1)[0]
                cv2.putText(
                    canvas,
                    stage_name,
                    (center_x - text_size[0] // 2, 658),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (60, 60, 60),
                    1,
                    cv2.LINE_AA,
                )
            for key, color in (
                ("gslow", (180, 119, 31)),
                ("glat", (44, 160, 44)),
                ("grec", (40, 39, 214)),
            ):
                values = np.asarray([float(row[key]) for row in plot_rows], dtype=np.float32)
                if values.size < 2:
                    continue
                xs = plot_x + np.clip(stage_x / 3.0, 0.0, 1.0) * plot_w
                ys = plot_y + plot_h - 1 - np.clip(values, 0.0, 1.0) * (plot_h - 2)
                pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], False, color, 3, cv2.LINE_AA)
            return bool(cv2.imwrite(str(png_path), canvas))
        except Exception as exc:
            self.get_logger().warn(f"OpenCV 绘制安全增益完整曲线失败: {exc}")
            return False

    def update_safety_gain_visualization(self, safety_state: dict[str, float | str | bool]) -> None:
        """把安全层三个增益画成实时条形图和曲线，体现 safety filter 的作用。"""
        self.safety_gain_visualization_step += 1
        self.safety_gain_history["slow"].append(float(safety_state["slow_gain"]))
        self.safety_gain_history["center"].append(float(safety_state["center_gain"]))
        self.safety_gain_history["retreat"].append(float(safety_state["grec"]))
        if not self.safety_gain_visualization_enabled and not self.save_safety_gain_images:
            return
        if cv2 is None:
            return

        try:
            canvas = np.full((460, 760, 3), 18, dtype=np.uint8)
            cv2.putText(canvas, "Safety Gains: gslow / glat / grec", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"mode={safety_state['mode']}  zone={'ON' if safety_state['safety_zone'] else 'OFF'}  "
                f"retreat={'ON' if safety_state['retreat_active'] else 'OFF'}  steps={safety_state['retreat_steps']}",
                (24, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            self.draw_gain_bar(canvas, (24, 108), 440, "gslow", float(safety_state["slow_gain"]), (0, 215, 255))
            self.draw_gain_bar(canvas, (24, 145), 440, "glat", float(safety_state["center_gain"]), (80, 220, 120))
            self.draw_gain_bar(canvas, (24, 182), 440, "grec", float(safety_state["grec"]), (70, 90, 255))
            cv2.putText(
                canvas,
                f"dist={safety_state['spoon_mouth_dist']*1000.0:.1f}mm  radial={safety_state['spoon_radial']*1000.0:.1f}mm  "
                f"depth={safety_state['mouth_depth']*1000.0:.1f}mm  channel={'YES' if safety_state['inside_mouth_channel'] else 'NO'}",
                (24, 244),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"danger={safety_state['danger_score']:.2f}  lip={safety_state['lip_risk']:.2f}  link={safety_state['link_risk']:.2f}  "
                f"force={safety_state['force_norm']:.1f}N  axial={safety_state['axial_force']:.1f}N  "
                f"signed={safety_state['signed_axial_force']:.1f}N  sign={safety_state['wrench_sign']:+.0f}",
                (24, 272),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"policy_delta={safety_state['policy_delta_norm']*1000.0:.1f}mm  final_delta={safety_state['final_delta_norm']*1000.0:.1f}mm  "
                f"clearance={safety_state['safe_clearance']*1000.0:.1f}mm",
                (24, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

            plot_x, plot_y, plot_w, plot_h = 24, 340, 700, 92
            cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (75, 75, 75), 1)
            for frac in (0.25, 0.5, 0.75):
                yy = plot_y + int(round(plot_h * (1.0 - frac)))
                cv2.line(canvas, (plot_x, yy), (plot_x + plot_w, yy), (42, 42, 42), 1)
            for key, color in (("slow", (0, 215, 255)), ("center", (80, 220, 120)), ("retreat", (70, 90, 255))):
                history = np.asarray(self.safety_gain_history[key], dtype=np.float32)
                if history.size < 2:
                    continue
                xs = np.linspace(plot_x, plot_x + plot_w, history.size, dtype=np.float32)
                ys = plot_y + plot_h - 1 - history * (plot_h - 2)
                pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)

            if self.safety_gain_visualization_enabled:
                try:
                    cv2.imshow(self.safety_gain_window, canvas)
                    cv2.waitKey(1)
                except Exception as exc:
                    self.safety_gain_visualization_enabled = False
                    self.get_logger().warn(f"安全增益窗口显示失败，已关闭窗口显示: {exc}", throttle_duration_sec=2.0)
            if self.save_safety_gain_images and self.safety_gain_visualization_step % self.safety_gain_image_interval == 0:
                cv2.imwrite(str(SAFETY_GAIN_VIS_DIR / "latest_safety_gains.png"), canvas)
        except Exception as exc:
            self.safety_gain_visualization_enabled = False
            self.get_logger().warn(f"安全增益图像可视化失败，已关闭窗口显示: {exc}", throttle_duration_sec=2.0)

    def close_safety_gain_visualization(self) -> None:
        """关闭安全增益窗口，避免 Ctrl-C 后 OpenCV 窗口残留。"""
        if cv2 is None:
            return
        try:
            cv2.destroyWindow(self.safety_gain_window)
        except Exception:
            pass

    def step_callback(self) -> None:
        if self.reset_started and not self.reset_finished:
            self.update_reset_after_success()
            return
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
        if self.fsm_state in {"SUCCESS_WITHDRAW", "SAFETY_WITHDRAW"}:
            self.update_success_withdraw()
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
        policy_arm_delta = arm_delta.copy()
        if self.learned_safety_enabled:
            arm_delta, safety_state = self.apply_learned_safety_filter(
                ee_pos,
                ee_quat_wxyz,
                spoon_center,
                arm_delta,
            )
        else:
            arm_delta, safety_state = self.apply_rule_based_safety_filter(
                ee_pos,
                ee_quat_wxyz,
                spoon_center,
                arm_delta,
            )
        if self.maybe_start_safety_strengthen(spoon_center, ee_quat_wxyz, safety_state):
            return
        arm_delta_ik = self.ros_delta_to_ik_delta(arm_delta)
        mouth_axis_raw = self.compute_mouth_axis(invert=False)
        mouth_axis = self.compute_mouth_axis()
        self.record_wrench_sample(
            spoon_center,
            raw_action,
            policy_action,
            command_action,
            arm_delta,
            safety_state,
            policy_arm_delta,
        )

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
                safety_state,
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
                f"safety_gain=[slow={safety_state['slow_gain']:.2f}, "
                f"center={safety_state['center_gain']:.2f}, "
                f"grec={safety_state['grec']:.2f}], "
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

        if self.fsm_state == "SUCCESS_HOLD":
            self.update_success_hold(spoon_center, ee_quat_wxyz)
            return True

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
                self.start_success_hold()
                return True
        else:
            self.feed_stable_steps = 0
        return False

    def start_success_hold(self) -> None:
        self.fsm_state = "SUCCESS_HOLD"
        self.success_hold_steps = 0
        if self.current_joint_pos is not None:
            self.success_hold_joint_pos = self.current_joint_pos.copy()
            if not self.args.dry_run:
                self.publish_joint_target(self.success_hold_joint_pos)
        self.previous_arm_action = np.zeros(6, dtype=np.float32)
        self.previous_safety_action.fill(0.0)
        self.safety_slow_active = False
        self.rule_safety_retreat_active = False
        self.rule_safety_retreat_steps = 0
        self.get_logger().warn(
            "成功条件已连续稳定满足，进入成功后保持阶段: "
            f"hold_steps={self.SUCCESS_HOLD_STEPS_AFTER_REACHED}"
        )

    def update_success_hold(self, spoon_center: np.ndarray, ee_quat_wxyz: np.ndarray) -> None:
        self.success_hold_steps += 1
        if self.success_hold_joint_pos is not None and not self.args.dry_run:
            self.publish_joint_target(self.success_hold_joint_pos)
        if self.success_hold_steps >= self.SUCCESS_HOLD_STEPS_AFTER_REACHED:
            self.start_success_withdraw(spoon_center, ee_quat_wxyz)

    def start_success_withdraw(
        self, spoon_center: np.ndarray, ee_quat_wxyz: np.ndarray, *, safety_triggered: bool = False
    ) -> None:
        """成功或安全接管时锁定进入轴和姿态，在归位前先沿原轴直线撤出。"""
        self.safety_strengthen_active = safety_triggered
        self.withdrawal_reason = "SAFETY" if safety_triggered else "SUCCESS"
        self.withdraw_distance_m = (
            self.SAFETY_STRENGTHEN_WITHDRAW_DISTANCE_M if safety_triggered else float(self.args.withdraw_distance)
        )
        withdraw_label = "安全加强" if safety_triggered else "成功后"
        self.previous_arm_action.fill(0.0)
        if not safety_triggered:
            self.previous_safety_action.fill(0.0)
        if not (
            np.all(np.isfinite(spoon_center))
            and np.all(np.isfinite(ee_quat_wxyz))
            and float(np.linalg.norm(ee_quat_wxyz)) > 1.0e-6
        ):
            self.stop_episode("FAILURE", f"{withdraw_label}撤回起始位姿无效，取消自动归位。", spoon_center)
            return
        direction = -np.asarray(self.compute_mouth_axis(), dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction))
        if not np.isfinite(direction_norm) or direction_norm < 1.0e-6:
            self.stop_episode("FAILURE", f"{withdraw_label}撤回失败: 嘴巴轴向无效，取消自动归位。", spoon_center)
            return
        self.success_withdraw_start_spoon = np.asarray(spoon_center, dtype=np.float64).copy()
        self.success_withdraw_direction = direction / direction_norm
        self.success_withdraw_quat_wxyz = np.asarray(ee_quat_wxyz, dtype=np.float64).copy()
        self.success_withdraw_start_time_ns = self.get_clock().now().nanoseconds
        self.success_withdraw_stable_steps = 0
        self.success_withdraw_complete = False
        self.success_withdraw_trajectory_sent = False
        self.fsm_state = "SAFETY_WITHDRAW" if safety_triggered else "SUCCESS_WITHDRAW"

        goal_spoon = self.success_withdraw_start_spoon + self.success_withdraw_direction * self.withdraw_distance_m
        goal_ee = goal_spoon - quat_apply_wxyz(ee_quat_wxyz, self.SPOON_OFFSET)
        if not (
            np.all(np.isfinite(goal_ee))
            and np.all(goal_ee >= self.WORKSPACE_MIN)
            and np.all(goal_ee <= self.WORKSPACE_MAX)
            and np.all(goal_spoon >= self.WORKSPACE_MIN)
            and np.all(goal_spoon <= self.WORKSPACE_MAX)
        ):
            self.stop_episode("FAILURE", f"{withdraw_label}撤回目标超出工作空间，取消自动归位。", spoon_center)
            return
        self.get_logger().warn(
            f"{withdraw_label}先直线撤回再归位: "
            f"distance={self.withdraw_distance_m:.3f}m, speed={self.SUCCESS_WITHDRAW_SPEED_M_S:.3f}m/s, "
            f"direction={np.round(self.success_withdraw_direction, 4)}, goal_spoon={np.round(goal_spoon, 4)}；"
            "一次下发完整连续轨迹，保持末端姿态，等待实际反馈到位。"
        )
        if self.args.dry_run:
            self.fsm_state = "DONE"
            self.stop_episode(
                self.withdrawal_reason,
                f"dry-run: 已预览{withdraw_label}撤回目标，不发布撤回或归位轨迹。",
                spoon_center,
            )

    def update_success_withdraw(self) -> None:
        """反馈闭环撤回；完成前不运行策略，也不启动关节归位。"""
        assert self.success_withdraw_start_spoon is not None
        assert self.success_withdraw_direction is not None
        assert self.success_withdraw_quat_wxyz is not None
        assert self.success_withdraw_start_time_ns is not None
        withdraw_label = "安全加强" if self.withdrawal_reason == "SAFETY" else "成功后"
        now_ns = self.get_clock().now().nanoseconds
        elapsed_s = (now_ns - self.success_withdraw_start_time_ns) * 1.0e-9
        if elapsed_s >= self.SUCCESS_WITHDRAW_TIMEOUT_S:
            self.stop_episode(
                "FAILURE", f"{withdraw_label}撤回超时，取消自动归位。", self.success_withdraw_start_spoon
            )
            return
        if self.data_is_stale() or (not self.args.dry_run and not self.ros_control_is_ready()):
            self.success_withdraw_stable_steps = 0
            return
        ee_pose = self.lookup_ee_pose(require_fresh=True)
        if ee_pose is None:
            self.success_withdraw_stable_steps = 0
            return
        ee_pos, ee_quat_wxyz = ee_pose
        if not (
            np.all(np.isfinite(ee_pos))
            and np.all(np.isfinite(ee_quat_wxyz))
            and float(np.linalg.norm(ee_quat_wxyz)) > 1.0e-6
            and self.current_joint_pos is not None
            and np.all(np.isfinite(self.current_joint_pos))
        ):
            self.stop_episode("FAILURE", f"{withdraw_label}撤回反馈无效，取消自动归位。", self.success_withdraw_start_spoon)
            return
        spoon_center = ee_pos + quat_apply_wxyz(ee_quat_wxyz, self.SPOON_OFFSET)
        self.last_fsm_metrics = self.compute_fsm_metrics(spoon_center, ee_quat_wxyz)
        direction = self.success_withdraw_direction
        displacement = spoon_center - self.success_withdraw_start_spoon
        progress = float(np.dot(displacement, direction))
        lateral_error = float(np.linalg.norm(displacement - progress * direction))
        goal_spoon = self.success_withdraw_start_spoon + direction * self.withdraw_distance_m
        position_error = float(np.linalg.norm(goal_spoon - spoon_center))
        current_rot = R.from_quat(ee_quat_wxyz[[1, 2, 3, 0]])
        goal_rot = R.from_quat(self.success_withdraw_quat_wxyz[[1, 2, 3, 0]])
        rotation_error = (goal_rot * current_rot.inv()).as_rotvec()
        rotation_error_norm = float(np.linalg.norm(rotation_error))
        if not np.all(np.isfinite([progress, lateral_error, position_error, rotation_error_norm])):
            self.stop_episode("FAILURE", f"{withdraw_label}撤回反馈无效，取消自动归位。", spoon_center)
            return
        if (
            lateral_error > self.SUCCESS_WITHDRAW_MAX_LATERAL_ERROR_M
            or rotation_error_norm > self.SUCCESS_WITHDRAW_MAX_ORIENTATION_ERROR_RAD
        ):
            self.stop_episode(
                "FAILURE",
                f"{withdraw_label}撤回偏离直线路径或姿态: lateral={lateral_error:.4f}m, "
                f"rotation={math.degrees(rotation_error_norm):.2f}deg，取消自动归位。",
                spoon_center,
            )
            return

        self.update_strengthened_safety_state(spoon_center, ee_quat_wxyz)

        pose_reached = (
            position_error <= self.SUCCESS_WITHDRAW_POSITION_TOL_M
            and progress >= self.withdraw_distance_m - self.SUCCESS_WITHDRAW_POSITION_TOL_M
            and rotation_error_norm <= self.SUCCESS_WITHDRAW_ORIENTATION_TOL_RAD
            and float(self.last_fsm_metrics["mouth_depth"]) < 0.0
        )
        settled = self.current_joint_vel is not None and float(np.max(np.abs(self.current_joint_vel))) < 0.03
        self.success_withdraw_stable_steps = self.success_withdraw_stable_steps + 1 if pose_reached and settled else 0
        if self.success_withdraw_stable_steps >= self.SUCCESS_WITHDRAW_STABLE_STEPS:
            self.success_withdraw_complete = True
            self.fsm_state = "DONE"
            self.stop_episode(
                self.withdrawal_reason, f"{withdraw_label}已沿进入方向的反方向撤回 {progress:.3f}m，开始归位。", spoon_center
            )
            return
        if pose_reached:
            return

        self.get_logger().info(
            f"{withdraw_label}撤回中: actual={progress:.3f}/{self.withdraw_distance_m:.3f}m, "
            f"lateral={lateral_error:.4f}m, elapsed={elapsed_s:.1f}s, "
            f"trajectory_sent={self.success_withdraw_trajectory_sent}, speed_scaling_raw={self.speed_scaling}",
            throttle_duration_sec=1.0,
        )
        if self.success_withdraw_trajectory_sent or self.args.dry_run:
            return
        try:
            trajectory = self.plan_success_withdraw_trajectory(ee_pos, ee_quat_wxyz)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.stop_episode("FAILURE", f"{withdraw_label}撤回轨迹规划失败: {exc}，取消自动归位。", spoon_center)
            return
        # 整条轨迹一次发布，中间点带连续速度，不再逐段等待反馈或重发目标。
        self.pub.publish(trajectory)
        self.success_withdraw_trajectory_sent = True
        end_time = trajectory.points[-1].time_from_start
        self.get_logger().info(
            f"已一次下发完整撤回轨迹: distance={self.withdraw_distance_m:.3f}m, "
            f"duration={end_time.sec + end_time.nanosec * 1e-9:.2f}s（UR速度缩放前）"
        )

    def plan_success_withdraw_trajectory(self, ee_pos: np.ndarray, ee_quat_wxyz: np.ndarray) -> JointTrajectory:
        """预先求解整条直线路径，生成平滑起停、途中不停顿的单条轨迹。"""
        assert self.current_joint_pos is not None
        assert self.success_withdraw_start_spoon is not None
        assert self.success_withdraw_direction is not None
        assert self.success_withdraw_quat_wxyz is not None
        initial_rot = R.from_quat(ee_quat_wxyz[[1, 2, 3, 0]])
        goal_rot = R.from_quat(self.success_withdraw_quat_wxyz[[1, 2, 3, 0]])
        rotation_delta = (goal_rot * initial_rot.inv()).as_rotvec()
        initial_spoon = ee_pos + initial_rot.apply(self.SPOON_OFFSET)
        goal_spoon = self.success_withdraw_start_spoon + self.success_withdraw_direction * self.withdraw_distance_m
        distance = float(np.linalg.norm(goal_spoon - initial_spoon))
        resolution = min(self.SUCCESS_WITHDRAW_PATH_RESOLUTION_M, self.args.max_cartesian_step / 1.5)
        if not np.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("撤回路径采样间距必须大于零")
        sample_count = max(20, int(math.ceil(distance / resolution)))
        joint_path = [self.current_joint_pos.astype(np.float64).copy()]
        initial_model_pos, initial_model_rot = self.ik.fk(joint_path[0])
        ros_to_ik = R.from_rotvec([0.0, 0.0, math.pi])
        previous_ee, previous_rot = ee_pos, initial_rot
        for index in range(1, sample_count + 1):
            fraction = index / sample_count
            # 整段使用统一的平滑时间律，只有起点和终点速度为零。
            fraction = fraction * fraction * (3.0 - 2.0 * fraction)
            target_spoon = initial_spoon + fraction * (goal_spoon - initial_spoon)
            target_rot = R.from_rotvec(fraction * rotation_delta) * initial_rot
            target_ee = target_spoon - target_rot.apply(self.SPOON_OFFSET)
            arm_delta = np.concatenate([target_ee - previous_ee, (target_rot * previous_rot.inv()).as_rotvec()])
            previous_quat = previous_rot.as_quat()[[3, 0, 1, 2]]
            safe, reason = self.delta_safety_status(previous_ee, previous_quat, arm_delta)
            if not safe:
                raise ValueError(reason)
            target_model_pos = initial_model_pos + ros_to_ik.apply(target_ee - ee_pos)
            target_model_rot = ros_to_ik * target_rot * initial_rot.inv() * ros_to_ik.inv() * initial_model_rot
            joint_target = joint_path[-1].copy()
            # 小增量 IK 只用于发布前的数值求解，不会逐步驱动实机。
            for _ in range(40):
                model_pos, model_rot = self.ik.fk(joint_target)
                error = np.concatenate([target_model_pos - model_pos, (target_model_rot * model_rot.inv()).as_rotvec()])
                if np.linalg.norm(error[:3]) <= 0.00005 and np.linalg.norm(error[3:]) <= 0.0001:
                    break
                joint_target = self.ik.joint_target_from_delta(
                    joint_target, error, max_joint_step=self.args.max_joint_step
                )
                if not np.all(np.isfinite(joint_target)):
                    raise ValueError("IK 关节目标无效")
            else:
                raise ValueError("直线撤回 IK 未收敛")
            if np.max(np.abs(joint_target - joint_path[-1])) > self.args.max_joint_step + self.SAFETY_EPS:
                raise ValueError("撤回路径相邻采样点关节变化过大")
            joint_path.append(joint_target)
            previous_ee, previous_rot = target_ee, target_rot

        positions = np.asarray(joint_path)
        derivatives = np.gradient(positions, axis=0)
        derivatives[[0, -1]] = 0.0
        # 三次 Hermite 插值的速度是二次 Bezier 曲线；用其控制点上界
        # 计算整段时长，保证采样点之间也满足每个关节的速度限制。
        velocity_bounds = np.maximum.reduce([
            np.abs(derivatives[:-1]), np.abs(derivatives[1:]),
            np.abs(3.0 * np.diff(positions, axis=0) - derivatives[:-1] - derivatives[1:]),
        ])
        duration_s = max(
            self.args.min_trajectory_duration,
            distance / self.SUCCESS_WITHDRAW_SPEED_M_S,
            sample_count * float(np.max(velocity_bounds / self.JOINT_MAX_VELOCITIES_RAD_S)),
        )
        interval_s = duration_s / sample_count
        trajectory = JointTrajectory()
        trajectory.joint_names = list(self.JOINT_NAMES)
        for index, (position, derivative) in enumerate(zip(positions, derivatives)):
            point = JointTrajectoryPoint()
            point.positions = position.tolist()
            point.velocities = (derivative / interval_s).tolist()
            time_ns = round(index * interval_s * 1e9)
            point.time_from_start = Duration(sec=time_ns // 1_000_000_000, nanosec=time_ns % 1_000_000_000)
            trajectory.points.append(point)

        # 下发前检查控制器在采样点之间的实际关节插值，避免长勺走弧线或越界。
        for index in range(sample_count):
            for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                h00 = 2.0 * fraction**3 - 3.0 * fraction**2 + 1.0
                h10 = fraction**3 - 2.0 * fraction**2 + fraction
                h01 = -2.0 * fraction**3 + 3.0 * fraction**2
                h11 = fraction**3 - fraction**2
                joints = (h00 * positions[index] + h10 * derivatives[index]
                          + h01 * positions[index + 1] + h11 * derivatives[index + 1])
                model_pos, model_rot = self.ik.fk(joints)
                predicted_ee = ee_pos + ros_to_ik.inv().apply(model_pos - initial_model_pos)
                predicted_rot = ros_to_ik.inv() * model_rot * initial_model_rot.inv() * ros_to_ik * initial_rot
                predicted_spoon = predicted_ee + predicted_rot.apply(self.SPOON_OFFSET)
                displacement = predicted_spoon - self.success_withdraw_start_spoon
                progress = float(displacement @ self.success_withdraw_direction)
                lateral = np.linalg.norm(displacement - progress * self.success_withdraw_direction)
                if not (
                    np.all(joints >= self.ik.lower_limits) and np.all(joints <= self.ik.upper_limits)
                    and np.all(predicted_ee >= self.WORKSPACE_MIN) and np.all(predicted_ee <= self.WORKSPACE_MAX)
                    and np.all(predicted_spoon >= self.WORKSPACE_MIN) and np.all(predicted_spoon <= self.WORKSPACE_MAX)
                    and lateral <= self.SUCCESS_WITHDRAW_MAX_LATERAL_ERROR_M
                    and (goal_rot * predicted_rot.inv()).magnitude() <= self.SUCCESS_WITHDRAW_MAX_ORIENTATION_ERROR_RAD
                ):
                    raise ValueError("撤回轨迹插值超出关节/工作空间限制或偏离直线路径")
        return trajectory

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
        # depth_ok 这里的正负号是关键: mouth_depth > 0 才代表勺心已经进入嘴内。
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
        self.episode_stop_reason = reason_type
        self.previous_arm_action = np.zeros(6, dtype=np.float32)
        if not self.safety_strengthen_active:
            self.previous_safety_action.fill(0.0)
            self.safety_slow_active = False
            self.rule_safety_retreat_active = False
            self.rule_safety_retreat_steps = 0
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
        if reason_type in {"SUCCESS", "SAFETY"} and self.start_reset_after_success():
            # 撤回确认到位后立即归位；绘图延后，避免阻塞轨迹衔接和归位期间的关节记录。
            return
        if not self.args.dry_run and self.current_joint_pos is not None and np.all(np.isfinite(self.current_joint_pos)):
            self.publish_joint_target(self.current_joint_pos)
        self.save_safety_gain_curve_outputs(reason_type)
        self.save_joint_angle_curve()

    def start_reset_after_success(self) -> bool:
        if ((not self.RESET_AFTER_SUCCESS and not self.safety_strengthen_active)
                or self.args.dry_run or self.reset_started or not self.success_withdraw_complete):
            return False
        if self.current_joint_pos is None or not np.all(np.isfinite(self.current_joint_pos)):
            self.get_logger().error("归位起始关节反馈无效，取消自动归位。")
            return False
        # 使用当前已连接的 publisher，保持 reset_arm.py 默认的 0.25 rad/s、最短 4 s。
        joint_move = np.abs(self.DEFAULT_JOINT_POS - self.current_joint_pos)
        duration_s = max(
            self.RESET_MIN_DURATION_S, self.args.min_trajectory_duration,
            float(np.max(joint_move)) / self.RESET_MAX_JOINT_VELOCITY_RAD_S,
        )
        scaling = float(self.speed_scaling) if self.speed_scaling is not None else 1.0
        if scaling > 1.0:
            scaling /= 100.0
        self.reset_timeout_s = duration_s / max(scaling, 0.01) + self.RESET_TIMEOUT_MARGIN_S
        self.reset_started = True
        self.reset_finished = False
        self.reset_start_time_ns = self.get_clock().now().nanoseconds
        self.reset_reached_since_ns = None
        self.fsm_state = "RESET"
        self.publish_joint_target(self.DEFAULT_JOINT_POS, min_duration_s=duration_s)
        self.get_logger().info(f"撤回已确认，已直接下发归位轨迹: duration={duration_s:.2f}s（UR速度缩放前）。")
        return True

    def update_reset_after_success(self) -> None:
        """只监测归位，不重发轨迹；结束后再绘图，关节 CSV 全程继续记录。"""
        assert self.reset_start_time_ns is not None
        now_ns = self.get_clock().now().nanoseconds
        elapsed_s = (now_ns - self.reset_start_time_ns) * 1e-9
        if self.safety_strengthen_active:
            ee_pose = self.lookup_ee_pose(require_fresh=True)
            if ee_pose is not None:
                ee_pos, ee_quat = ee_pose
                spoon_center = ee_pos + quat_apply_wxyz(ee_quat, self.SPOON_OFFSET)
                self.update_strengthened_safety_state(spoon_center, ee_quat)
        feedback_valid = (
            self.current_joint_pos is not None
            and np.all(np.isfinite(self.current_joint_pos))
            and self.last_joint_state_time_ns is not None
            and (now_ns - self.last_joint_state_time_ns) * 1e-9 <= self.args.joint_state_timeout
        )
        settled = (
            self.current_joint_vel is not None and np.all(np.isfinite(self.current_joint_vel))
            and float(np.max(np.abs(self.current_joint_vel))) < 0.03
        )
        reached = (
            feedback_valid
            and float(np.max(np.abs(self.DEFAULT_JOINT_POS - self.current_joint_pos))) <= self.RESET_GOAL_TOL_RAD
        )
        if reached and settled:
            if self.reset_reached_since_ns is None:
                self.reset_reached_since_ns = now_ns
            if (now_ns - self.reset_reached_since_ns) * 1e-9 < self.RESET_SETTLE_TIME_S:
                return
            self.reset_finished = True
            self.fsm_state = "DONE"
            self.get_logger().info("自动归位完成，开始保存曲线。")
        elif elapsed_s >= self.reset_timeout_s:
            self.reset_finished = True
            self.fsm_state = "RESET_FAILED"
            if feedback_valid:
                self.publish_joint_target(self.current_joint_pos)
            self.get_logger().error(f"自动归位超时: elapsed={elapsed_s:.1f}s，停止等待并保存曲线。")
        else:
            self.reset_reached_since_ns = None
            return
        if self.safety_strengthen_active:
            self.safety_strengthen_active = False
            self.rule_safety_retreat_active = False
            safety_state = self.rule_safety_last_state.copy()
            safety_state.update({"mode": self.fsm_state, "retreat_active": False})
            if self.safety_grec_force_one:
                safety_state["retreat_gain"] = 0.0
            self.publish_safety_state(safety_state)
        self.save_safety_gain_curve_outputs(self.episode_stop_reason or "SUCCESS")
        self.save_joint_angle_curve()

    def lookup_ee_pose(self, require_fresh: bool = False) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            transform = self.tf_buffer.lookup_transform(self.target_frame, self.ee_frame, rclpy.time.Time(), timeout=RclpyDuration(seconds=0.05))
        except tf2_ros.TransformException as exc:
            if not self.warned_missing_tf:
                self.get_logger().warn(f"等待 TF {self.target_frame}->{self.ee_frame}: {exc}")
                self.warned_missing_tf = True
            return None

        if require_fresh:
            stamp_ns = transform.header.stamp.sec * 1_000_000_000 + transform.header.stamp.nanosec
            age_s = (self.get_clock().now().nanoseconds - stamp_ns) * 1.0e-9
            if abs(age_s) > self.args.joint_state_timeout:
                self.get_logger().warn(f"撤回阶段 TF 超时 {age_s:.2f}s，暂停撤回。", throttle_duration_sec=1.0)
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
        # 真实传感器与仿真关节反力可能是作用/反作用关系；用 wrench_sign 做最小必要的方向同步。
        policy_raw = self.wrench_sign * wrench_source.astype(np.float32)
        return np.nan_to_num(policy_raw, nan=0.0, posinf=250.0, neginf=-250.0)

    def get_sim_aligned_safety_wrench_raw(self) -> np.ndarray:
        """读取 safety 专用 wrench，显式使用与仿真/策略一致的方向约定。"""
        # 这里重新从 sensor-bias 计算一次，而不是直接读名字较泛的缓存，确保 safety 的力方向始终跟随 wrench_sign。
        sensor_minus_bias = self.wrist_wrench_sensor_raw - self.wrist_wrench_sensor_bias
        return self.align_wrist_wrench_direction(sensor_minus_bias)

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
        safety_state: dict[str, object] | None = None,
    ) -> None:
        assert self.mouth_pos is not None
        assert self.mouth_quat_wxyz is not None
        if safety_state is None:
            safety_state = self.default_rule_safety_state()
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
            f"  safety = mode:{safety_state['mode']}, "
            f"slow:{safety_state['slow_gain']:.3f}, "
            f"center:{safety_state['center_gain']:.3f}, "
            f"grec:{safety_state['grec']:.3f}, "
            f"danger:{safety_state['danger_score']:.3f}, "
            f"radial:{safety_state['spoon_radial']:.4f}, "
            f"depth:{safety_state['mouth_depth']:.4f}, "
            f"signed_axial_force:{safety_state['signed_axial_force']:.3f}, "
            f"wrench_sign:{safety_state['wrench_sign']:+.0f}\n"
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

    def publish_joint_target(self, joint_target: np.ndarray, min_duration_s: float = 0.0) -> None:
        assert self.current_joint_pos is not None
        point = JointTrajectoryPoint()
        point.positions = [float(joint_target[self.JOINT_NAME_TO_IDX[name]]) for name in self.JOINT_NAMES]

        joint_move = np.abs(joint_target - self.current_joint_pos)
        # 每个关节按自己的速度上限计算所需时间，整条轨迹采用其中最大值以保证所有关节都不超速。
        joint_durations = joint_move / np.maximum(self.JOINT_MAX_VELOCITIES_RAD_S, 1.0e-6)
        duration_s = max(self.args.min_trajectory_duration, min_duration_s, float(np.max(joint_durations)))
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
        if not self.args.fixed_mouth_pose and self.fsm_state not in {"SUCCESS_WITHDRAW", "SAFETY_WITHDRAW"}:
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
    parser.add_argument("--wrench-sign", type=float, default=ArmPolicyDeployNode.WRENCH_SIM_REAL_SIGN, choices=[-1.0, 1.0], help="Sign used to align real F/T direction with the simulation joint-reaction convention.")
    parser.add_argument("--wrench-bias-samples", type=int, default=30, help="Stationary wrench samples used for tare after sign/frame alignment; 0 disables tare.")
    parser.add_argument("--control-hz", type=float, default=30.0, help="Policy/control frequency, matching sim decimation.")
    parser.add_argument("--target-timeout", type=float, default=0.5, help="Stop commanding if the mouth pose topic is older than this many seconds.")
    parser.add_argument("--joint-state-timeout", type=float, default=0.5, help="Stop commanding if controller_state is older than this many seconds.")
    parser.add_argument("--wrench-timeout", type=float, default=0.5, help="Stop policy inference if the wrist wrench topic is older than this many seconds.")
    parser.add_argument("--speed-scaling-timeout", type=float, default=0.5, help="Stop commanding if speed_scaling is older than this many seconds.")
    parser.add_argument("--min-speed-scaling", type=float, default=0.01, help="Do not publish motion commands unless UR speed scaling is at least this value.")
    parser.add_argument("--episode-length-s", type=float, default=240.0, help="Deployment episode timeout in seconds, matching simulation.")
    parser.add_argument("--max-episode-steps", type=int, default=0, help="Deployment episode timeout in policy steps; <=0 uses episode_length_s * control_hz.")
    parser.add_argument("--withdraw-distance", type=float, default=0.10, help="Straight withdrawal distance in meters before automatic reset after success.")
    parser.add_argument("--feed-hold-steps", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--bite-release-steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--min-spoon-z", type=float, default=0.06, help="Episode failure threshold for spoon center height in target-frame meters.")
    parser.add_argument("--max-mouth-spoon-distance", type=float, default=3.00, help="Episode failure threshold for mouth-spoon distance; <=0 disables this guard.")
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
    if not math.isfinite(args.withdraw_distance) or args.withdraw_distance < ArmPolicyDeployNode.FEED_INSERT_DEPTH_MAX_M:
        parser.error("--withdraw-distance 必须是有限数且不小于最大入嘴深度 0.08m，确保归位前勺心已退出嘴平面。")
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
        node.save_joint_angle_curve(close_record=True)
        node.close_wrench_record()
        node.save_safety_gain_curve_outputs("shutdown")
        node.close_safety_gain_visualization()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
