# reach_rtde_node.py (简化版)
import rclpy
from rclpy.node import Node
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import PoseStamped 
from rclpy.duration import Duration as RclpyDuration
from std_msgs.msg import Float32
# --- 引入 ur_rtde ---
import rtde_control
import rtde_receive

class ReachRTDE(Node):
    def __init__(self):
        super().__init__('reach_rtde_node')

        # --- 1. 配置机器人连接 ---
        self.ROBOT_IP = "192.168.56.10" 
        
        self.get_logger().info(f"正在连接机器人 RTDE ({self.ROBOT_IP})...")
        try:
            self.rtde_c = rtde_control.RTDEControlInterface(self.ROBOT_IP)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ROBOT_IP)
            self.get_logger().info("机器人连接成功！")

        except Exception as e:
            self.get_logger().error(f"无法连接到机器人: {e}")
            raise e

        # --- 2. 状态和控制参数 ---
        self.target_frame = 'base' # 目标基坐标系
        self.vel = 0.2  # servoL 速度 m/s
        self.acc = 0.1  # servoL 加速度 m/s^2
        self.blend = 0.05 # 混合半径 
        
        self.last_target_pos = None # 用于简单的防抖动

        # --- 3. 嘴部状态控制参数 【新增】---
        self.mouth_open = False # 机器人的当前停止/运动状态
        # 嘴巴张开的归一化阈值 (0.02是一个经验值，可能需要根据您的相机和光线条件调整)
        self.MAR_THRESHOLD = 0.02 
        # 我们只使用腕部相机 ('wrist_face_pose_node') 作为停止控制的来源
        self.PRIMARY_CAM_NODE = 'wrist_face_pose_node'
        
        # --- 4. 订阅人脸位姿 (接收的是融合节点处理好的 Base 目标) ---
        self.face_pose_subscriber = self.create_subscription(
            PoseStamped, '/face_pose', self.face_pose_callback, 10
        )
        # --- 5. 订阅嘴部状态 【新增】---
        self.mouth_state_subscriber = self.create_subscription(
            Float32, 
            f'/{self.PRIMARY_CAM_NODE}/mouth_dist', 
            self.mouth_state_callback, 
            10
        )
        
        self.get_logger().info("ReachRTDE 节点已就绪，等待已修正的融合目标...")

    def mouth_state_callback(self, msg: Float32):
        """接收嘴部垂直距离并判断是否张开"""
        mar = msg.data # 嘴部垂直距离（Mouth Aspect Ratio的近似值）
        if mar > self.MAR_THRESHOLD:
            if not self.mouth_open:
                # 仅在状态变化时记录警告，防止日志刷屏
                self.get_logger().warn(f"检测到嘴巴张开 ({mar:.3f} > {self.MAR_THRESHOLD:.3f})。机器人停止跟随。", throttle_duration_sec=1.0)
            self.mouth_open = True
        else:
            if self.mouth_open:
                # 仅在状态变化时记录信息
                self.get_logger().info(f"检测到嘴巴闭合 ({mar:.3f})。机器人恢复跟随。")
            self.mouth_open = False

    def face_pose_callback(self, msg: PoseStamped):
        """
        1. 接收已转换到 Base 坐标系且已修正的 PoseStamped
        2. 转换为 UR 格式 (x, y, z, rx, ry, rz)
        3. 发送 servoL
        """
        try:
            # 检查：确保接收到的位姿已经在 base 坐标系下
            if msg.header.frame_id != self.target_frame:
                 self.get_logger().error(f"警告：融合节点发布的位姿不在目标 '{self.target_frame}' 坐标系下！")
                 return
                 
            # --- A. 移除坐标系变换 (已由 PoseFusionNode 完成) ---
            pos = msg.pose.position
            quat_ros = msg.pose.orientation
            quat_list = [quat_ros.x, quat_ros.y, quat_ros.z, quat_ros.w]
            
            # --- B. 移除投喂姿态修正 (已由 PoseFusionNode 完成) ---
            # rotation_matrix[:,2] = -rotation_matrix[:,2]
            # rotation_matrix[:,0] = -rotation_matrix[:,0]
            
            # --- C. 姿态转换 (Quaternion -> Rotation Vector) ---
            r = R.from_quat(quat_list)
            rot_vec = r.as_rotvec() # 返回 [rx, ry, rz]

            # 组合目标点 [x, y, z, rx, ry, rz]
            target_tcp = [pos.x, pos.y, pos.z, rot_vec[0], rot_vec[1], rot_vec[2]]

            # --- D. 安全检查与修正 (可选) ---
            if target_tcp[2] < 0.05: # 防止撞击桌面
                self.get_logger().warn("目标点过低，已自动抬高保护")
                target_tcp[2] = 0.05

            # --- E. 控制指令发送 (Gated by mouth state) 【核心修改】---
            if self.mouth_open:
                self.get_logger().warn("跳过 servoL：嘴巴张开，保持当前位置。", throttle_duration_sec=1.0)
                return # 跳过所有运动指令，机器人将停留在上一个 servoL 位置

            # 只有嘴巴闭合时，才检查是否需要移动
            if self.should_move(target_tcp):
                self.get_logger().info(
                    f"执行 servoL -> Pos: [{target_tcp[0]:.3f}, {target_tcp[1]:.3f}, {target_tcp[2]:.3f}]"
                )
                
                success = self.rtde_c.servoL(
                    target_tcp,         
                    self.vel,           
                    self.acc,           
                    self.blend, # 使用 self.blend 
                    0.04,              
                    100 # lookahead_time (推荐 0.03 到 0.2)               
                )

                if not success:
                    self.get_logger().error(f"servoL 调用失败 (返回 False)")

                self.last_target_pos = target_tcp

        except Exception as e:
            self.get_logger().error(f'控制指令发送失败: {e}')

    def should_move(self, new_target):
        """简单的滤波器：只有当目标移动距离超过一定阈值才发送新指令"""
        if self.last_target_pos is None:
            return True
            
        dist = np.linalg.norm(np.array(new_target[:3]) - np.array(self.last_target_pos[:3]))
        
        if dist > 0.01: 
            return True
        return False

def main(args=None):
    rclpy.init(args=args)
    node = ReachRTDE()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if hasattr(node, 'rtde_c'):
            node.rtde_c.stopScript()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()