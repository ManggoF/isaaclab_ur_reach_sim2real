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

# -----------------------------------------------------------------
# --- 1. TCP 变换定义 (T_tool^gripper) ---
# T_tool^gripper: 夹爪的位姿相对于法兰 (tool frame)
# 
# 假设: 夹爪中心沿着法兰的 Z 轴延伸 0.15 米 (150 mm)，无旋转。
# 请根据您的机械臂和夹爪的实际安装情况，准确测量并修改这个矩阵！
# -----------------------------------------------------------------
TCP_TRANSFORM = np.array([
    [1, 0, 0, 0.0],  # R1 | X (平移)
    [0, 1, 0, 0.0],  # R2 | Y (平移)
    [0, 0, 1, 0.15], # R3 | Z (平移) -> 0.15 米 (m)
    [0, 0, 0, 1]     # 0 0 0 | 1
], dtype=np.float64) 

# --- 投喂姿态修正函数 ---
def apply_feeding_pose_correction(rotation_matrix_raw: np.ndarray) -> np.ndarray:
    """
    根据投喂姿态要求，对人脸坐标系转换后的旋转矩阵进行修正。
    此修正适用于 Base 坐标系下的旋转矩阵。
    """
    corrected_matrix = rotation_matrix_raw.copy()
    
    # X 轴取反 (第一列)
    corrected_matrix[:,0] = -corrected_matrix[:,0]
    # Z 轴取反 (第三列)
    corrected_matrix[:,2] = -corrected_matrix[:,2]
    
    return corrected_matrix

# --- 辅助函数：PoseStamped <-> 4x4 Matrix ---
def pose_to_matrix(pose_msg: PoseStamped) -> np.ndarray:
    """将 PoseStamped 转换为 4x4 齐次变换矩阵"""
    pos = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])
    quat = np.array([pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, 
                     pose_msg.pose.orientation.z, pose_msg.pose.orientation.w])
    
    R_matrix = R.from_quat(quat).as_matrix()
    
    T = np.identity(4)
    T[:3, :3] = R_matrix
    T[:3, 3] = pos
    return T

def matrix_to_pose(matrix: np.ndarray, frame_id: str, stamp) -> PoseStamped:
    """将 4x4 齐次变换矩阵转换回 PoseStamped"""
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = frame_id
    pose_msg.header.stamp = stamp
    
    # 提取位置
    pose_msg.pose.position.x = matrix[0, 3]
    pose_msg.pose.position.y = matrix[1, 3]
    pose_msg.pose.position.z = matrix[2, 3]
    
    # 提取旋转并转换为四元数
    R_matrix = matrix[:3, :3]
    # 确保旋转矩阵是有效的（正交且行列式为1），否则 SciPy 可能报错
    # 这一步通常是为了防止浮点误差导致矩阵不完全正交
    r_check = R.from_matrix(R_matrix) 
    quat = r_check.as_quat()
    
    pose_msg.pose.orientation.x = quat[0]
    pose_msg.pose.orientation.y = quat[1]
    pose_msg.pose.orientation.z = quat[2]
    pose_msg.pose.orientation.w = quat[3]
    
    return pose_msg
# --- 辅助函数 END ---


class PoseFusionNode(Node):
    def __init__(self):
        super().__init__('pose_fusion_node')
        
        # --- 1. TF2 监听器 ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_frame = 'base' # 融合后的目标坐标系

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
        
        # --- 4. 发布最终融合结果 ---
        self.publisher = self.create_publisher(PoseStamped, '/face_pose', 10)
        
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
        """将 PoseStamped 从其 frame_id 转换为 base 坐标系下"""
        if raw_pose_stamped is None:
            return None
            
        try:
            # 检查 TF 变换是否可用
            if not self.tf_buffer.can_transform(self.target_frame, raw_pose_stamped.header.frame_id, 
                                                rclpy.time.Time()):
                self.get_logger().warn(
                    f'等待从 {raw_pose_stamped.header.frame_id} 到 {self.target_frame} 的变换...', 
                    throttle_duration_sec=2.0
                )
                return None
            
            # 执行 TF 变换
            transformed_pose = self.tf_buffer.transform(
                raw_pose_stamped,
                self.target_frame,
                timeout=RclpyDuration(seconds=0.1)
            )
            
            return transformed_pose
        
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f'TF 变换失败 ({raw_pose_stamped.header.frame_id} -> base): {ex}', 
                                   throttle_duration_sec=1.0)
            return None
        
    def fusion_timer_callback(self):
        """定时器回调，执行融合逻辑"""
        
        now = self.get_clock().now()
        
        # 1. 检查数据新鲜度
        is_pose1_valid = self.pose1_raw is not None and (now - self.time1) < self.timeout_duration
        is_pose2_valid = self.pose2_raw is not None and (now - self.time2) < self.timeout_duration
        
        # 2. 将有效位姿转换到 base 坐标系
        pose1_base = self.get_pose_in_base(self.pose1_raw) if is_pose1_valid else None
        pose2_base = self.get_pose_in_base(self.pose2_raw) if is_pose2_valid else None
        
        is_pose1_ready = pose1_base is not None
        is_pose2_ready = pose2_base is not None

        final_pose_msg = None
        
        # 3. 融合逻辑 (T_base^face)
        if is_pose1_ready and is_pose2_ready:
            # final_pose_msg = self.average_poses(pose1_base, pose2_base)
            final_pose_msg = pose1_base
            self.get_logger().debug("双相机目标融合：使用腕部相机 1")
            
        elif is_pose1_ready:
            final_pose_msg = pose1_base
            self.get_logger().debug("双相机目标融合：使用腕部相机 1")
            
        elif is_pose2_ready:
            final_pose_msg = pose2_base
            self.get_logger().debug("双相机目标融合：使用固定相机 2")
            
        else:
            self.get_logger().warn("双相机目标融合：当前所有人脸目标均无效或被遮挡")
            return 
            
        # ------------------------------------------------------------------
        # 4. 【关键步骤】应用 TCP 偏移 (计算机器人法兰的目标位姿 T_base^tool)
        # T_base^tool = T_base^face * (T_tool^gripper)^-1
        # ------------------------------------------------------------------
        try:
            # a. 转换融合后的目标 (T_base^face) 为矩阵
            T_base_face = pose_to_matrix(final_pose_msg)
            
            # b. 计算夹爪变换的逆矩阵 (T_tool^gripper)^-1
            T_tool_gripper_inv = np.linalg.inv(TCP_TRANSFORM)
            
            # c. 计算法兰的目标位姿 (T_base^tool)
            T_base_tool = T_base_face @ T_tool_gripper_inv
            
            # d. 转换回 PoseStamped 消息
            final_pose_msg = matrix_to_pose(T_base_tool, self.target_frame, now.to_msg())
            self.get_logger().debug("已成功应用 TCP 变换。")
            
        except np.linalg.LinAlgError:
            self.get_logger().error("TCP 变换矩阵不可逆，请检查 TCP_TRANSFORM 的定义!")
            return
        except Exception as e:
            self.get_logger().error(f"应用 TCP 变换时发生错误: {e}")
            return
            
        # --- 5. 应用投喂姿态修正 (对 T_base^tool 进行修正) ---
        quat_ros = final_pose_msg.pose.orientation
        quat_list = [quat_ros.x, quat_ros.y, quat_ros.z, quat_ros.w]
        rotation_matrix_raw = R.from_quat(quat_list).as_matrix()
        
        rotation_matrix_corrected = apply_feeding_pose_correction(rotation_matrix_raw)
        
        # 转换回四元数并更新消息
        r_corrected = R.from_matrix(rotation_matrix_corrected)
        quat_corrected = r_corrected.as_quat()

        final_pose_msg.pose.orientation.x = quat_corrected[0]
        final_pose_msg.pose.orientation.y = quat_corrected[1]
        final_pose_msg.pose.orientation.z = quat_corrected[2]
        final_pose_msg.pose.orientation.w = quat_corrected[3]

        # 6. 发布最终结果
        final_pose_msg.header.stamp = now.to_msg() 
        self.publisher.publish(final_pose_msg)
        self.get_logger().info(f"发布融合位姿: {final_pose_msg.pose.position.x:.3f}, {final_pose_msg.pose.position.y:.3f}, {final_pose_msg.pose.position.z:.3f}")


    def average_poses(self, pose_a: PoseStamped, pose_b: PoseStamped) -> PoseStamped:
        """对两个 base 坐标系下的 PoseStamped 进行平均"""
        
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
        fused_pose.header.frame_id = self.target_frame # 'base'
        
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