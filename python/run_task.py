import rclpy
from rclpy.node import Node
import numpy as np
import math

from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped 

from robots.ur import URReachPolicy
from rclpy.duration import Duration as RclpyDuration
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs # <<-- 非常重要，用于自动变换PoseStamped

class ReachPolicy(Node):
    SIM_DOF_ANGLE_LIMITS = [(-360, 360, False), (-360, 360, False), (-360, 360, False), (-360, 360, False), (-360, 360, False), (-360, 360, False)]
    PI = math.pi
    SERVO_ANGLE_LIMITS = [(-2 * PI, 2 * PI), (-2 * PI, 2 * PI), (-2 * PI, 2 * PI), (-2 * PI, 2 * PI), (-2 * PI, 2 * PI), (-2 * PI, 2 * PI)]
    STATE_TOPIC = '/scaled_joint_trajectory_controller/controller_state'
    CMD_TOPIC = '/scaled_joint_trajectory_controller/joint_trajectory'
    JOINT_NAMES = ['elbow_joint', 'shoulder_lift_joint', 'shoulder_pan_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    JOINT_NAME_TO_IDX = {'elbow_joint': 2, 'shoulder_lift_joint': 1, 'shoulder_pan_joint': 0, 'wrist_1_joint': 3, 'wrist_2_joint': 4, 'wrist_3_joint': 5}

    def __init__(self, fail_quietly: bool = False, verbose: bool = False):
        super().__init__('reach_policy_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.robot = URReachPolicy()
        
        self.target_command = np.array([0.5, 0.4, 0.3, 0.7071, 0.7071, 0.0, 0.0])
        self.target_received = False
        
        self.step_size = 1.0 / 100
        self.timer = self.create_timer(self.step_size, self.step_callback)
        self.current_pos = None
        self.target_pos = None

        self.create_subscription(
            JointTrajectoryControllerState, self.STATE_TOPIC, self.sub_callback, 10
        )
        
        self.face_pose_subscriber = self.create_subscription(
            PoseStamped, '/face_pose', self.face_pose_callback, 10
        )
        
        self.pub = self.create_publisher(JointTrajectory, self.CMD_TOPIC, 10)
        self.min_traj_dur = 0.01
        
        self.get_logger().info("ReachPolicy节点已初始化, 等待人脸位姿目标...")

    def face_pose_callback(self, msg: PoseStamped):
        if not self.target_received:
            self.get_logger().info("已接收到第一个人脸位姿目标，机器人开始运动。")
            self.target_received = True

        target_frame = 'base'

        try:
            # --- 核心：使用TF2进行坐标变换 ---
            transformed_pose_stamped = self.tf_buffer.transform(
                msg,
                target_frame,
                timeout=RclpyDuration(seconds=0.1)
            )
            
            # 从变换后的位姿中提取位置和方向
            pos = transformed_pose_stamped.pose.position
            ori = transformed_pose_stamped.pose.orientation

            self.target_command = np.array([
                pos.x, pos.y, pos.z,
                ori.x, ori.y, ori.z, ori.w
            ])

            # <<< 在这里添加新的日志打印行 <<<
            self.get_logger().info(
                f"转换后目标指令 ({target_frame}): "
                f"Pos(x={self.target_command[0]:.3f}, y={self.target_command[1]:.3f}, z={self.target_command[2]:.3f})"
                f"Ori(x={self.target_command[3]:.3f}, y={self.target_command[4]:.3f}, z={self.target_command[5]:.3f}, w={self.target_command[6]:.3f})"
            )
            
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f'无法将位姿从 {msg.header.frame_id} 变换到 {target_frame}: {ex}', throttle_duration_sec=1.0)
            return
    
    # ... (其余方法保持不变) ...
    def sub_callback(self, msg: JointTrajectoryControllerState):
        actual_pos = {}
        for i, joint_name in enumerate(msg.joint_names):
            joint_pos = msg.reference.positions[i] # joint_pos = msg.actual.positions[i] 22.04
            actual_pos[joint_name] = joint_pos
        self.current_pos = actual_pos
        self.robot.update_joint_state(msg.reference.positions, msg.reference.velocities)

    def map_joint_angle(self, pos: float, index: int) -> float:
        L, U, inversed = self.SIM_DOF_ANGLE_LIMITS[index]
        A, B = self.SERVO_ANGLE_LIMITS[index]
        angle_deg = np.rad2deg(float(pos))
        angle_deg = np.clip(angle_deg, L, U)
        mapped = (angle_deg - L) * ((B - A) / (U - L)) + A
        if inversed: mapped = (B - A) - (mapped - A) + A
        return mapped

    def step_callback(self):
        if not self.target_received:
            return

        # target_frame = 'base_link'
        # source_frame = 'tool0'
        # if self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time(), timeout=RclpyDuration(seconds=0.1)):
        #     try:
        #         trans = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
        #         tcp_position = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
        #         self.robot.update_tcp_position(tcp_position)
        #     except tf2_ros.TransformException as ex:
        #         self.get_logger().warn(f'无法获取TF变换: {ex}')
        #         return
        # else:
        #     self.get_logger().info(f'等待从 {source_frame} 到 {target_frame} 的TF变换...', throttle_duration_sec=1.0)
        #     return
        
        joint_pos = self.robot.forward(self.step_size, self.target_command)
        
        if joint_pos is not None:
            if len(joint_pos) != 6:
                raise Exception(f"期望6个关节位置,但得到了 {len(joint_pos)}!")
            
            self.target_pos = [self.map_joint_angle(pos, i) for i, pos in enumerate(joint_pos)]
            
            if self.current_pos is None or self.target_pos is None: return
            
            traj = JointTrajectory()
            traj.joint_names = self.JOINT_NAMES
            point = JointTrajectoryPoint()
            dur_list = []
            moving_average = 1
            
            for joint_name in traj.joint_names:
                pos = self.current_pos[joint_name]
                target = self.target_pos[self.JOINT_NAME_TO_IDX[joint_name]]
                cmd = pos * (1 - moving_average) + target * moving_average
                max_vel = 0.7
                duration = abs(cmd - pos) / max_vel if max_vel else self.min_traj_dur
                dur_list.append(max(duration, self.min_traj_dur))
                point.positions.append(cmd)
            
            max_duration = max(dur_list) if dur_list else 0.0
            sec = int(max_duration)
            nanosec = int((max_duration - sec) * 1e9)
            point.time_from_start = Duration(sec=sec, nanosec=nanosec)
            traj.points.append(point)
            
            self.pub.publish(traj)

def main(args=None):
    rclpy.init(args=args)
    node = ReachPolicy()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()