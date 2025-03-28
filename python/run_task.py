import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np

import math

from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from robots.ur import URReachPolicy

class ReachPolicy(Node):
    sim_dof_angle_limits = [
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
    ]
    pi = math.pi
    servo_angle_limits = [
        (-2 * pi, 2 * pi),
        (-2 * pi, 2 * pi),
        (-2 * pi, 2 * pi),
        (-2 * pi, 2 * pi),
        (-2 * pi, 2 * pi),
        (-2 * pi, 2 * pi),
    ]
    # ROS-related strings
    state_topic = '/scaled_joint_trajectory_controller/state'
    cmd_topic = '/scaled_joint_trajectory_controller/joint_trajectory'
    joint_names = [
        'elbow_joint',
        'shoulder_lift_joint',
        'shoulder_pan_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]
    # Joint name mapping to simulation action index
    joint_name_to_idx = {
        'elbow_joint': 2,
        'shoulder_lift_joint': 1,
        'shoulder_pan_joint': 0,
        'wrist_1_joint': 3,
        'wrist_2_joint': 4,
        'wrist_3_joint': 5
    }

    def __init__(self, fail_quietly=False, verbose=False):
        super().__init__('reach_policy_node')

        self.robot = URReachPolicy()
        self.target_command = np.zeros(7)
        self.step_size = 1.0 / 100  # 10 ms = 100 Hz

        self.timer = self.create_timer(self.step_size, self.step_callback)
        self.i = 0

        self.fail_quietly = fail_quietly
        self.verbose = verbose
        self.pub_freq = 100.0  # Hz
        # Current and target positions (dictionaries/lists)
        self.current_pos = None
        self.target_pos = None

        # Subscriber for the controller state
        self.create_subscription(
            JointTrajectoryControllerState,
            self.state_topic,
            self.sub_callback,
            10
        )
        # Publisher for sending joint trajectory commands
        self.pub = self.create_publisher(JointTrajectory, self.cmd_topic, 10)
        self.min_traj_dur = 0  # Minimum trajectory duration

        self.get_logger().info("ReachPolicy node initialized.")

    def sub_callback(self, msg):
        # msg has type: JointTrajectoryControllerState
        actual_pos = {}
        for i, joint_name in enumerate(msg.joint_names):
            joint_pos = msg.actual.positions[i]
            actual_pos[joint_name] = joint_pos
        self.current_pos = actual_pos

        #### Logging ####
        #self.get_logger().info(f'(sub) {actual_pos}')

        self.robot.update_joint_state(msg.actual.positions, msg.actual.velocities)


    def step_callback(self):
        self.target_command = np.array([0.5, 0.0, 0.2, 0.7071, 0.0, 0.7071, 0.0])

        joint_pos = self.robot.forward(self.step_size, self.target_command)

        if joint_pos is not None:

            if len(joint_pos) != 6:
                raise Exception(f"The length of UR joint_pos is {len(joint_pos)}, but should be 6!")

            # Convert simulation angles to real angles.
            target_pos = [0] * 6
            for i, pos in enumerate(joint_pos):
                if i == 5:
                    # Ignore gripper joints for this task
                    continue
                L, U, inversed = self.sim_dof_angle_limits[i]
                A, B = self.servo_angle_limits[i]
                angle = np.rad2deg(float(pos))
                if not L <= angle <= U:
                    self.get_logger().warn(
                        f"The {i}-th simulation joint angle ({angle}) is out of range! Should be in [{L}, {U}]"
                    )
                    angle = np.clip(angle, L, U)
                mapped = (angle - L) * ((B - A) / (U - L)) + A  # Map [L, U] to [A, B]
                if inversed:
                    mapped = (B - A) - (mapped - A) + A  # Inverse mapping if needed
                if not A <= mapped <= B:
                    raise Exception(
                        f"(Should Not Happen) The {i}-th real world joint angle ({mapped}) is out of range! Should be in [{A}, {B}]"
                    )
                target_pos[i] = mapped
            self.target_pos = target_pos

            if self.current_pos is None or self.target_pos is None:
                # Wait until both the current state and a command are available.
                return

            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            point = JointTrajectoryPoint()
            dur_list = []
            moving_average = 1  # Weight factor (1 = full target value)

            for joint_name in traj.joint_names:
                pos = self.current_pos[joint_name]
                target = self.target_pos[self.joint_name_to_idx[joint_name]]
                # Apply a moving average: here, with a weight of 1 the command equals target.
                cmd = pos * (1 - moving_average) + target * moving_average
                max_vel = 0.7
                duration = abs(cmd - pos) / max_vel  # time = distance / velocity

                dur_list.append(max(duration, self.min_traj_dur))
                point.positions.append(cmd)

            max_duration = max(dur_list) if dur_list else 0.0
            # Set time_from_start as a builtin_interfaces Duration message
            sec = int(max_duration)
            nanosec = int((max_duration - sec) * 1e9)
            point.time_from_start = Duration(sec=sec, nanosec=nanosec)
            traj.points.append(point)

            self.pub.publish(traj)
            # self.get_logger().info(f'(pub) {point.positions}')



        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = ReachPolicy()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
