# reach_rtde_node.py (简化版)
import time
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
        self.vel = 0.05  # servoL 速度 m/s
        self.acc = 0.05  # servoL 加速度 m/s^2
        self.last_target_pos = None # 用于简单的防抖动
        self.is_stopping = False # 【新增】逻辑锁：标记是否正在执行停止动作
        self.mouth_open = False # 机器人的当前停止/运动状态
        self.MAR_THRESHOLD = 0.01 # 嘴部垂直距离阈值 (单位: 米) 
        
        # --- 3. 订阅人脸位姿 (接收的是融合节点处理好的 Base 目标) ---
        self.face_pose_subscriber = self.create_subscription(
            PoseStamped, '/face_pose', self.face_pose_callback, 10
        )
        
        # --- 4. 订阅嘴部状态 (两个相机都订阅) ---
        self.sub_mouth_wrist = self.create_subscription(
            Float32, '/wrist_face_pose_node/mouth_dist', self.mouth_state_callback, 10
        ) # 订阅腕部相机
        self.sub_mouth_fixed = self.create_subscription(
            Float32, '/fixed_face_pose_node/mouth_dist', self.mouth_state_callback, 10
        ) # 订阅固定相机
        
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
                # 闭嘴瞬间，清除停止锁，允许重新发送运动指令
                self.is_stopping = False
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
                # stopL(decereation rate)
                self.rtde_c.stopL(2.0)
                self.is_stopping = True # 加锁
                self.get_logger().warn("检测到张嘴：已发送 stopL 强制停止。", throttle_duration_sec=1.0)
                return # 跳过所有运动指令，机器人将停留在当前位置
            
            move_state = self.check_movement(target_tcp)

            if move_state == "TOO_LARGE":
                # 【关键】检测到异常跳变，立即刹车，保护实验者
                self.rtde_c.stopL(2.0)
                self.get_logger().error(
                    f"检测到异常位置跳变 (距离过大)！已触发紧急停止。目标位置: {target_tcp[:3]}"
                )
                return

            elif move_state == "NORMAL" and not self.is_stopping:
            # 只有嘴巴闭合时，才检查是否需要移动
            # if self.should_move(target_tcp):
                # self.get_logger().info(
                #     f"执行 servoL -> Pos: [{target_tcp[0]:.3f}, {target_tcp[1]:.3f}, {target_tcp[2]:.3f}]"
                # )
                # self.rtde_c.moveL(target_tcp, self.vel, self.acc, asynchronous=True)

                if not self.rtde_c.isProgramRunning():
                    self.get_logger().error("RTDE 脚本未运行，正在尝试重新上传...")
                    self.rtde_c.reuploadScript()
                    time.sleep(0.1) # 给脚本启动留出微量时间
                    
                success = self.rtde_c.servoL(
                    target_tcp,         
                    self.vel,           
                    self.acc,           
                    2.0, # dt
                    0.04,# lookahead_time       
                    100 # gain             
                )

                if not success:
                    self.get_logger().error(f"servoL 调用失败 (返回 False)")

                self.last_target_pos = target_tcp
            else: # TOO_SMALL
                # 忽略微小抖动，不做任何动作
                pass


        except Exception as e:
            self.get_logger().error(f'控制指令发送失败: {e}')

    
    def check_movement(self, new_target):
        """
        判断移动距离：
        - 小于 2mm: 忽略 (防止震动)
        - 2mm ~ 30cm: 正常运动
        - 大于 30cm: 判定为跳变/误识别，触发保护
        """
        if self.last_target_pos is None:
            return "NORMAL"
            
        dist = np.linalg.norm(np.array(new_target[:3]) - np.array(self.last_target_pos[:3]))
        if dist > 0.30: return "TOO_LARGE"
        if dist < 0.002: return "TOO_SMALL"
            
        return "NORMAL"

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