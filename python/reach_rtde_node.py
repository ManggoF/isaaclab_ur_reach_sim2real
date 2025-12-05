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

# --- 引入 ur_rtde ---
import rtde_control
import rtde_receive

class ReachRTDE(Node):
    def __init__(self):
        super().__init__('reach_rtde_node')

        # --- 1. 配置机器人连接 ---
        # 请修改为你的实体机器人IP地址
        self.ROBOT_IP = "192.168.56.10" 
        
        self.get_logger().info(f"正在连接机器人 RTDE ({self.ROBOT_IP})...")
        try:
            # 初始化控制接口
            self.rtde_c = rtde_control.RTDEControlInterface(self.ROBOT_IP)
            # 初始化接收接口 (可选，用于获取当前状态)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ROBOT_IP)
            self.get_logger().info("机器人连接成功！")




        except Exception as e:
            self.get_logger().error(f"无法连接到机器人: {e}")
            raise e

        # --- 2. TF2 监听器 (用于坐标变换) ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # --- 3. 订阅人脸位姿 ---
        self.face_pose_subscriber = self.create_subscription(
            PoseStamped, '/face_pose', self.face_pose_callback, 10
        )
        
        # 控制参数
        self.target_frame = 'base' # 目标基坐标系
        self.vel = 0.2  # moveL 速度 m/s
        self.acc = 0.1  # moveL 加速度 m/s^2
        self.blend = 0.05 # 混合半径 (0表示不平滑过渡，直接点对点)
        
        self.last_target_pos = None # 用于简单的防抖动
        
        self.get_logger().info("ReachRTDE 节点已就绪，等待人脸目标...")

    def face_pose_callback(self, msg: PoseStamped):
        """
        1. 接收人脸 Pose
        2. 转换到 base 坐标系，这是实际相机检测到的坐标系
        3. 转换为 UR 格式 (x, y, z, rx, ry, rz)
        4. 发送 moveL
        """
        try:
            # --- A. 坐标系变换 (Camera -> Base) ---
            # 确保 buffer 中有变换关系
            if not self.tf_buffer.can_transform(self.target_frame, msg.header.frame_id, rclpy.time.Time()):
                self.get_logger().warn(f'等待从 {msg.header.frame_id} 到 {self.target_frame} 的变换...', throttle_duration_sec=2.0)
                return

            transformed_pose_stamped = self.tf_buffer.transform(
                msg,
                self.target_frame,
                timeout=RclpyDuration(seconds=0.1)
            )
            
            pos = transformed_pose_stamped.pose.position
            quat_ros = transformed_pose_stamped.pose.orientation
            quat_list = [quat_ros.x, quat_ros.y, quat_ros.z, quat_ros.w]
            rotation_matrix = R.from_quat(quat_list).as_matrix()
            # self.get_logger().info(f'\nqian的 Rotation Matrix:\n{rotation_matrix}')
            
            rotation_matrix[:,2] = -rotation_matrix[:,2]
            rotation_matrix[:,0] = -rotation_matrix[:,0]
            self.get_logger().info(f'\n后的 Rotation Matrix:\n{rotation_matrix}')


            ori= R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]
            # --- B. 姿态转换 (Quaternion -> Rotation Vector) ---
            # UR 的 moveL 接受 [x, y, z, rx, ry, rz]
            # 其中 rx, ry, rz 是旋转向量 (Rotation Vector)
            
            r = R.from_quat([ori[0], ori[1], ori[2], ori[3]])
            rot_vec = r.as_rotvec() # 返回 [rx, ry, rz]

            # 组合目标点 [x, y, z, rx, ry, rz]
            target_tcp = [pos.x, pos.y, pos.z, rot_vec[0], rot_vec[1], rot_vec[2]]

            # --- C. 安全检查与修正 (可选) ---
            # 在这里可以添加限制，例如：Z轴不能低于桌面，X轴不能太远等
            if target_tcp[2] < 0.05: # 防止撞击桌面
                self.get_logger().warn("目标点过低，已自动抬高保护")
                target_tcp[2] = 0.05

            # --- D. 发送控制指令 moveL ---
            
            # 简单的距离检查，避免微小的抖动导致机器人频繁规划
            if self.should_move(target_tcp):
                self.get_logger().info(
                    f"执行 moveL -> Pos: [{target_tcp[0]:.3f}, {target_tcp[1]:.3f}, {target_tcp[2]:.3f}]"
                )
                self.get_logger().info(f'\n修正后的 Rotation Matrix:\n{rotation_matrix}')
                
                # asynchronous=True 非常重要！
                # 如果是 False，程序会卡在这里直到运动结束，会导致 ROS 回调阻塞，丢失后续的视觉帧。
                # 设置为 True 后，机器人会在后台移动，你可以随时发送新的指令覆盖旧的（取决于UR控制器的设置）
                # self.rtde_c.moveL(target_tcp, self.vel, self.acc, asynchronous=True)
                

                success = self.rtde_c.servoL(
                    target_tcp,         # arg0 (List[float])
                    self.vel,           # arg1 (float) -> v
                    self.acc,           # arg2 (float) -> a
                    1.0,                # arg3 (float) -> blend (r)
                    0.04,              # arg4 (float) -> t (伺服周期)
                    100                 # arg5 (float) -> lookahead_time
                )

                if not success:
                    self.get_logger().error(f"servoL 调用失败 (返回 False)")
                # ----------------------------------------------------

                self.last_target_pos = target_tcp

        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f'TF变换失败: {ex}', throttle_duration_sec=1.0)
        except Exception as e:
            self.get_logger().error(f'控制指令发送失败: {e}')

    def should_move(self, new_target):
        """简单的滤波器：只有当目标移动距离超过一定阈值才发送新指令"""
        if self.last_target_pos is None:
            return True
            
        # 计算位置欧氏距离
        dist = np.linalg.norm(np.array(new_target[:3]) - np.array(self.last_target_pos[:3]))
        
        # 阈值：例如 1cm (0.01m)。如果人脸移动小于 1cm，机器人不动作，避免高频抖动
        if dist > 0.01: 
            return True
        return False

def main(args=None):
    rclpy.init(args=args)
    node = ReachRTDE()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 退出时停止机器人
        if hasattr(node, 'rtde_c'):
            node.rtde_c.stopScript()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()