import rclpy
from collections import OrderedDict
from copy import deepcopy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.duration import Duration as RclpyDuration
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs


class PoseFusionNode(Node):
    CAMERA_MODE = 'wrist'  # 保留当前腕部相机选择；可改 'fixed' 或 'auto'（固定优先）。
    MAX_DETECTION_AGE_S = 0.4
    POSE_HISTORY_SIZE = 30
    MAX_PUBLISHED_POSITION_STEP_M = 0.02
    MIN_PUBLISHED_AXIS_DOT = 0.95
    POSE_FILTER_ALPHA = 0.35
    OPENING_HISTORY_SIZE = 30

    def __init__(self):
        super().__init__('pose_fusion_node')
        
        # --- 1. TF2 监听器 ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_frame = 'base_link' # RL 策略观测使用的目标坐标系
        self.mouth_openings = {camera: OrderedDict() for camera in ('wrist', 'fixed')}
        self.pending_poses = {camera: OrderedDict() for camera in ('wrist', 'fixed')}
        self.camera_pose_cache = {}
        self.last_selected_key = None

        # --- 2. 最近接受的目标，保持时不刷新检测时间戳 ---
        self.last_published_pose = None
        self.last_published_axis = None
        self.last_accepted_time = None

        # --- 3. 订阅两个相机的话题 ---
        self.sub1 = self.create_subscription(
            PoseStamped, '/wrist_face_pose', self.pose1_callback, 10
        )
        self.sub2 = self.create_subscription(
            PoseStamped, '/fixed_face_pose', self.pose2_callback, 10
        )
        
        # --- 4. 发布最终融合后的原始嘴部位姿，供 RL 策略作为 mouth pose 观测 ---
        self.publisher = self.create_publisher(PoseStamped, '/mouth_pose', 10)
        self.opening_publisher = self.create_publisher(Vector3Stamped, '/mouth_opening', 10)
        self.opening_subs = [
            self.create_subscription(
                Vector3Stamped, f'/{camera}_face_pose_node/mouth_opening',
                lambda msg, camera=camera: self.mouth_opening_callback(camera, msg), 10,
            )
            for camera in ('wrist', 'fixed')
        ]
        
        # --- 5. 定时器进行融合计算 ---
        # 每 20ms (50Hz) 进行一次融合计算
        self.timer = self.create_timer(0.02, self.fusion_timer_callback)
        
        self.get_logger().info(
            f"Pose Fusion 节点已就绪，camera_mode={self.CAMERA_MODE}，按检测时刻 TF 转换；"
            f"发布前过滤: max_step={self.MAX_PUBLISHED_POSITION_STEP_M:.3f}m, "
            f"min_axis_dot={self.MIN_PUBLISHED_AXIS_DOT:.2f}, alpha={self.POSE_FILTER_ALPHA:.2f}, "
            "hold_last_pose=unlimited"
        )
        self.get_logger().info('视觉跟随直接使用 /mouth_pose 和同相机同帧 /mouth_opening；保持目标时保留原检测时间戳。')

    @staticmethod
    def stamp_ns(msg):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def mouth_opening_callback(self, camera, msg):
        history = self.mouth_openings[camera]
        history[self.stamp_ns(msg)] = msg
        while len(history) > self.OPENING_HISTORY_SIZE:
            history.popitem(last=False)

    def publish_matching_opening(self, camera, raw):
        """Only forward the selected camera's opening from this acquisition."""
        opening = self.mouth_openings[camera].get(self.stamp_ns(raw))
        if opening is not None and opening.header.frame_id == raw.header.frame_id:
            # Vector length is rotation invariant; retain its camera frame.
            self.opening_publisher.publish(opening)
        
    def pose1_callback(self, msg: PoseStamped):
        self.remember_pose('wrist', msg)

    def pose2_callback(self, msg: PoseStamped):
        self.remember_pose('fixed', msg)

    def remember_pose(self, camera, msg):
        history = self.pending_poses[camera]
        history[self.stamp_ns(msg)] = self.copy_pose(msg)
        while len(history) > self.POSE_HISTORY_SIZE:
            history.popitem(last=False)

    def latest_transformable_pose(self, camera, now):
        """Keep pending frames until TF catches up; never block or use newer TF."""
        history = self.pending_poses[camera]
        cached = self.camera_pose_cache.get(camera)
        last_stamp = self.stamp_ns(cached[0]) if cached else 0
        for stamp in sorted(history, reverse=True):
            age = (now.nanoseconds - stamp) * 1e-9
            if stamp <= last_stamp or stamp <= 0 or age > self.MAX_DETECTION_AGE_S or age < -0.05:
                del history[stamp]
                continue
            raw = history[stamp]
            pose = self.get_pose_in_base(raw)
            if pose is not None:
                cached = (raw, pose)
                self.camera_pose_cache[camera] = cached
                # Newer frames whose TF has not arrived remain queued.
                for old in list(history):
                    if old <= stamp:
                        del history[old]
                break
        if cached is not None:
            age = (now.nanoseconds - self.stamp_ns(cached[0])) * 1e-9
            if -0.05 <= age <= self.MAX_DETECTION_AGE_S:
                return cached
        return None
        
    def get_pose_in_base(self, raw_pose_stamped: PoseStamped) -> PoseStamped or None:
        """将 PoseStamped 从其 frame_id 转换到策略观测坐标系下"""
        if raw_pose_stamped is None:
            return None
            
        try:
            # 腕部相机随机器人移动，必须使用检测时刻的 TF。
            # 最新 TF 会把图像处理延迟变成虚假的嘴部移动。
            acquisition_time = rclpy.time.Time.from_msg(raw_pose_stamped.header.stamp)
            # 检查 TF 变换是否可用
            if not self.tf_buffer.can_transform(self.target_frame, raw_pose_stamped.header.frame_id, 
                                                acquisition_time):
                self.get_logger().warn(
                    f'等待检测时刻从 {raw_pose_stamped.header.frame_id} 到 {self.target_frame} 的变换（缓存重试，不阻塞）...',
                    throttle_duration_sec=2.0
                )
                return None
    
            # 执行 TF 变换
            transformed_pose = self.tf_buffer.transform(
                self.copy_pose(raw_pose_stamped),
                self.target_frame,
                timeout=RclpyDuration()
            )
            transformed_pose.header.stamp = deepcopy(raw_pose_stamped.header.stamp)
            return transformed_pose
        
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f'TF 变换失败 ({raw_pose_stamped.header.frame_id} -> {self.target_frame}): {ex}', 
                                   throttle_duration_sec=1.0)
            return None
        
    def fusion_timer_callback(self):
        """定时器回调，执行融合逻辑"""
        
        now = self.get_clock().now()
        final_pose_msg = None
        holding_last_pose = False
        cameras = {'wrist': ('wrist',), 'fixed': ('fixed',), 'auto': ('fixed', 'wrist')}[self.CAMERA_MODE]
        for camera in cameras:
            selected = self.latest_transformable_pose(camera, now)
            if selected is not None:
                raw, final_pose_msg = selected
                self.get_logger().info(f'使用相机: {camera}', throttle_duration_sec=1.0)
                break
        if final_pose_msg is None:
            final_pose_msg = self.hold_last_pose(now, '所选相机无有效检测或检测时刻 TF 尚未就绪')
            if final_pose_msg is None:
                return
            holding_last_pose = True

        # RL 策略需要真实嘴部 pose 作为观测，不需要投喂姿态修正或 TCP 偏移。
        if not holding_last_pose:
            key = (camera, self.stamp_ns(raw))
            if key != self.last_selected_key:
                final_pose_msg = self.filter_pose_before_publish(final_pose_msg, now)
                self.last_selected_key = key
            else:
                final_pose_msg = self.copy_pose(self.last_published_pose)
            if final_pose_msg is None:
                return
        self.publisher.publish(final_pose_msg)
        if not holding_last_pose:
            self.publish_matching_opening(camera, raw)
        self.get_logger().info(
            f"发布 /mouth_pose ({self.target_frame}) -> "
            f"X:{final_pose_msg.pose.position.x:.3f}, "
            f"Y:{final_pose_msg.pose.position.y:.3f}, "
            f"Z:{final_pose_msg.pose.position.z:.3f}",
            throttle_duration_sec=0.5,
        )
        return

    def filter_pose_before_publish(self, pose_msg: PoseStamped, now) -> PoseStamped or None:
        pos = self.pose_position_array(pose_msg)
        axis = self.pose_axis_array(pose_msg)

        if self.last_published_pose is None:
            self.last_published_pose = self.copy_pose(pose_msg)
            self.last_published_axis = axis
            self.last_accepted_time = now
            return self.copy_pose(pose_msg)

        last_pos = self.pose_position_array(self.last_published_pose)
        pos_step = float(np.linalg.norm(pos - last_pos))
        axis_dot = float(np.dot(axis, self.last_published_axis))

        limited_pos = pos
        if pos_step > self.MAX_PUBLISHED_POSITION_STEP_M:
            direction = (pos - last_pos) / max(pos_step, 1.0e-6)
            limited_pos = last_pos + direction * self.MAX_PUBLISHED_POSITION_STEP_M
            self.get_logger().warn(
                f"/mouth_pose 位置变化过大，限速连续更新: "
                f"step={pos_step:.4f}m > {self.MAX_PUBLISHED_POSITION_STEP_M:.4f}m",
                throttle_duration_sec=0.5,
            )
        use_candidate_quat = True
        if axis_dot < self.MIN_PUBLISHED_AXIS_DOT:
            use_candidate_quat = False
            self.get_logger().warn(
                f"/mouth_pose 姿态轴跳变，本帧沿用上一帧姿态: "
                f"dot={axis_dot:.4f} < {self.MIN_PUBLISHED_AXIS_DOT:.4f}",
                throttle_duration_sec=0.5,
            )

        filtered_pose = PoseStamped()
        filtered_pose.header = deepcopy(pose_msg.header)
        alpha = self.POSE_FILTER_ALPHA
        filtered_pos = (1.0 - alpha) * last_pos + alpha * limited_pos
        filtered_pose.pose.position.x = float(filtered_pos[0])
        filtered_pose.pose.position.y = float(filtered_pos[1])
        filtered_pose.pose.position.z = float(filtered_pos[2])

        last_quat = self.pose_quat_array(self.last_published_pose)
        if use_candidate_quat:
            quat = self.pose_quat_array(pose_msg)
            if float(np.dot(last_quat, quat)) < 0.0:
                quat = -quat
        else:
            quat = last_quat
        filtered_quat = (1.0 - alpha) * last_quat + alpha * quat
        filtered_quat /= max(float(np.linalg.norm(filtered_quat)), 1.0e-6)
        filtered_pose.pose.orientation.x = float(filtered_quat[0])
        filtered_pose.pose.orientation.y = float(filtered_quat[1])
        filtered_pose.pose.orientation.z = float(filtered_quat[2])
        filtered_pose.pose.orientation.w = float(filtered_quat[3])

        self.last_published_pose = filtered_pose
        self.last_published_axis = self.pose_axis_array(filtered_pose)
        self.last_accepted_time = now
        return filtered_pose

    def hold_last_pose(self, now, reason: str) -> PoseStamped or None:
        if self.last_published_pose is None or self.last_accepted_time is None:
            self.get_logger().debug(f"{reason}，且没有上一帧可信嘴巴可保持")
            return None

        hold_age_s = (now - self.last_accepted_time).nanoseconds * 1.0e-9
        held_pose = self.copy_pose(self.last_published_pose)
        self.get_logger().warn(
            f"{reason}，继续发布上一帧可信嘴巴: hold_age={hold_age_s:.2f}s",
            throttle_duration_sec=0.5,
        )
        return held_pose

    @staticmethod
    def copy_pose(pose_msg: PoseStamped) -> PoseStamped:
        copied = PoseStamped()
        copied.header = deepcopy(pose_msg.header)
        copied.pose.position.x = pose_msg.pose.position.x
        copied.pose.position.y = pose_msg.pose.position.y
        copied.pose.position.z = pose_msg.pose.position.z
        copied.pose.orientation.x = pose_msg.pose.orientation.x
        copied.pose.orientation.y = pose_msg.pose.orientation.y
        copied.pose.orientation.z = pose_msg.pose.orientation.z
        copied.pose.orientation.w = pose_msg.pose.orientation.w
        return copied

    @staticmethod
    def pose_position_array(pose_msg: PoseStamped) -> np.ndarray:
        return np.array(
            [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z],
            dtype=np.float64,
        )

    @staticmethod
    def pose_quat_array(pose_msg: PoseStamped) -> np.ndarray:
        return np.array(
            [
                pose_msg.pose.orientation.x,
                pose_msg.pose.orientation.y,
                pose_msg.pose.orientation.z,
                pose_msg.pose.orientation.w,
            ],
            dtype=np.float64,
        )

    def pose_axis_array(self, pose_msg: PoseStamped) -> np.ndarray:
        quat = self.pose_quat_array(pose_msg)
        quat /= max(float(np.linalg.norm(quat)), 1.0e-6)
        axis = R.from_quat(quat).apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
        axis /= max(float(np.linalg.norm(axis)), 1.0e-6)
        return axis

    def average_poses(self, pose_a: PoseStamped, pose_b: PoseStamped) -> PoseStamped:
        """对两个策略观测坐标系下的 PoseStamped 进行平均"""
        
        # --- 位置平均 ---
        pos_avg = np.array([
            (pose_a.pose.position.x + pose_b.pose.position.x) / 2.0,
            (pose_a.pose.position.y + pose_b.pose.position.y) / 2.0,
            (pose_a.pose.position.z + pose_b.pose.position.z) / 2.0
        ])
        
        # --- 姿态平均 (旋转向量近似平均) ---
        quat_a = [pose_a.pose.orientation.x, pose_a.pose.orientation.y, 
                  pose_a.pose.orientation.z, pose_a.pose.orientation.w]
        quat_b = [pose_b.pose.orientation.x, pose_b.pose.orientation.y, 
                  pose_b.pose.orientation.z, pose_b.pose.orientation.w]
                  
        r_a = R.from_quat(quat_a)
        r_b = R.from_quat(quat_b)

        # 旋转向量平均
        rotvec_avg = (r_a.as_rotvec() + r_b.as_rotvec()) / 2.0
        r_avg = R.from_rotvec(rotvec_avg)
        quat_avg = r_avg.as_quat()

        # --- 组合结果 ---
        fused_pose = PoseStamped()
        fused_pose.header.frame_id = self.target_frame
        
        fused_pose.pose.position.x = pos_avg[0]
        fused_pose.pose.position.y = pos_avg[1]
        fused_pose.pose.position.z = pos_avg[2]
        
        fused_pose.pose.orientation.x = quat_avg[0]
        fused_pose.pose.orientation.y = quat_avg[1]
        fused_pose.pose.orientation.z = quat_avg[2]
        fused_pose.pose.orientation.w = quat_avg[3]
        
        return fused_pose

def main(args=None):
    rclpy.init(args=args)
    node = PoseFusionNode()
    # TF subscriptions use a reentrant group; keep receiving them while the
    # mutually exclusive fusion/camera callbacks run in the other worker.
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        rclpy.spin(node, executor=executor)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
