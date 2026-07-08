import rclpy
from rclpy.node import Node
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import PoseStamped 
from rclpy.duration import Duration as RclpyDuration
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs


class PoseFusionNode(Node):
    def __init__(self):
        super().__init__('pose_fusion_node')
        
        # --- 1. TF2 监听器 ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_frame = 'base_link' # RL 策略观测使用的目标坐标系

        # --- 2. 状态变量和时间戳 ---
        self.pose1_raw = None
        self.pose2_raw = None
        self.time1 = self.get_clock().now()
        self.time2 = self.get_clock().now()
        self.timeout_duration = RclpyDuration(seconds=0.5) # 0.5秒内未收到，则认为该相机目标无效

        # --- 3. 订阅两个相机的话题 ---
        self.sub1 = self.create_subscription(
            PoseStamped, '/wrist_face_pose', self.pose1_callback, 10
        )
        self.sub2 = self.create_subscription(
            PoseStamped, '/fixed_face_pose', self.pose2_callback, 10
        )
        
        # --- 4. 发布最终融合后的原始嘴部位姿，供 RL 策略作为 mouth pose 观测 ---
        self.publisher = self.create_publisher(PoseStamped, '/mouth_pose', 10)
        
        # --- 5. 定时器进行融合计算 ---
        # 每 20ms (50Hz) 进行一次融合计算
        self.timer = self.create_timer(0.02, self.fusion_timer_callback)
        
        self.get_logger().info("Pose Fusion 节点已就绪...")
        
    def pose1_callback(self, msg: PoseStamped):
        self.pose1_raw = msg
        self.time1 = self.get_clock().now()

    def pose2_callback(self, msg: PoseStamped):
        self.pose2_raw = msg
        self.time2 = self.get_clock().now()
        
    def get_pose_in_base(self, raw_pose_stamped: PoseStamped) -> PoseStamped or None:
        """将 PoseStamped 从其 frame_id 转换到策略观测坐标系下"""
        if raw_pose_stamped is None:
            return None
            
        try:
            # --- 修改点 1: 检查变换时使用 Time(nanoseconds=0) ---
            # rclpy.time.Time() 等同于 Time(nanoseconds=0)，代表获取该链路上最新的数据
            latest_time = rclpy.time.Time()
            # 检查 TF 变换是否可用
            if not self.tf_buffer.can_transform(self.target_frame, raw_pose_stamped.header.frame_id, 
                                                latest_time):
                self.get_logger().warn(
                    f'等待从 {raw_pose_stamped.header.frame_id} 到 {self.target_frame} 的变换...', 
                    throttle_duration_sec=2.0
                )
                return None
    
            # --- 修改点 2: 转换时不使用消息里的 stamp，而是强制用最新时间 ---
            # 创建一个新的消息副本，把它的时间戳改为 0
            # 这样 tf2_geometry_msgs 内部会去查找最新的变换，而不是查找“半小时前”的变换
            lookup_pose = PoseStamped()
            lookup_pose.header.frame_id = raw_pose_stamped.header.frame_id
            lookup_pose.header.stamp = latest_time.to_msg() # 关键：强制设为最新
            lookup_pose.pose = raw_pose_stamped.pose
            # 执行 TF 变换
            transformed_pose = self.tf_buffer.transform(
                lookup_pose,
                self.target_frame,
                timeout=RclpyDuration(seconds=0.1)
            )
            
            return transformed_pose
        
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f'TF 变换失败 ({raw_pose_stamped.header.frame_id} -> {self.target_frame}): {ex}', 
                                   throttle_duration_sec=1.0)
            return None
        
    def fusion_timer_callback(self):
        """定时器回调，执行融合逻辑"""
        
        now = self.get_clock().now()
        # 1. 检查数据新鲜度
        is_pose1_valid = self.pose1_raw is not None and (now - self.time1) < self.timeout_duration
        is_pose2_valid = self.pose2_raw is not None and (now - self.time2) < self.timeout_duration
        
        # 2. 将有效位姿转换到策略观测坐标系
        pose1_base = self.get_pose_in_base(self.pose1_raw) if is_pose1_valid else None
        pose2_base = self.get_pose_in_base(self.pose2_raw) if is_pose2_valid else None
        
        is_pose1_ready = pose1_base is not None
        is_pose2_ready = pose2_base is not None

        final_pose_msg = None
        
        # 3. 融合逻辑 (T_base^face)
        # if is_pose1_ready and is_pose2_ready:
        #     # final_pose_msg = self.average_poses(pose1_base, pose2_base)
        #     final_pose_msg = pose1_base
        #     self.get_logger().debug("双相机目标融合：使用腕部相机 1")
        
        if is_pose2_ready:
            # 策略：只要腕部相机就绪，就使用它的结果，实现最高优先级。
            final_pose_msg = pose2_base
            self.get_logger().info("双相机目标融合：使用固定相机 2")
            
        elif is_pose1_ready:
            final_pose_msg = pose1_base
            self.get_logger().info("双相机目标融合：使用腕部相机 1")
            
        else:
            self.get_logger().debug("双相机目标融合：当前所有人脸目标均无效或被遮挡")
            return 

        # RL 策略需要真实嘴部 pose 作为观测，不需要投喂姿态修正或 TCP 偏移。
        final_pose_msg.header.stamp = now.to_msg()
        self.publisher.publish(final_pose_msg)
        self.get_logger().info(
            f"发布 /mouth_pose ({self.target_frame}) -> "
            f"X:{final_pose_msg.pose.position.x:.3f}, "
            f"Y:{final_pose_msg.pose.position.y:.3f}, "
            f"Z:{final_pose_msg.pose.position.z:.3f}",
            throttle_duration_sec=0.5,
        )
        return

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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
