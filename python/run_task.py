import rclpy
from rclpy.node import Node
import numpy as np
import math

from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from robots.ur import URReachPolicy


class ReachPolicy(Node):
    """ROS2 node for controlling a UR robot's reach policy."""
    
    # Define simulation degree-of-freedom angle limits: (Lower limit, Upper limit, Inversed flag)
    SIM_DOF_ANGLE_LIMITS = [
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
    ]
    
    # Define servo angle limits (in radians)
    PI = math.pi
    SERVO_ANGLE_LIMITS = [
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
    ]
    
    # ROS topics and joint names
    STATE_TOPIC = '/scaled_joint_trajectory_controller/state'
    CMD_TOPIC = '/scaled_joint_trajectory_controller/joint_trajectory'
    JOINT_NAMES = [
        'elbow_joint',
        'shoulder_lift_joint',
        'shoulder_pan_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]
    
    # Mapping from joint name to simulation action index
    JOINT_NAME_TO_IDX = {
        'elbow_joint': 2,
        'shoulder_lift_joint': 1,
        'shoulder_pan_joint': 0,
        'wrist_1_joint': 3,
        'wrist_2_joint': 4,
        'wrist_3_joint': 5
    }

    def __init__(self, fail_quietly: bool = False, verbose: bool = False):
        """Initialize the ReachPolicy node."""
        super().__init__('reach_policy_node')
        
        self.robot = URReachPolicy()
        self.target_command = np.zeros(7)
        self.step_size = 1.0 / 100  # 10 ms period = 100 Hz
        self.timer = self.create_timer(self.step_size, self.step_callback)
        self.i = 0
        self.fail_quietly = fail_quietly
        self.verbose = verbose
        self.pub_freq = 100.0  # Hz
        self.current_pos = None  # Dictionary of current joint positions
        self.target_pos = None   # List of target joint positions

        # Subscriber for controller state messages
        self.create_subscription(
            JointTrajectoryControllerState,
            self.STATE_TOPIC,
            self.sub_callback,
            10
        )
        
        # Publisher for joint trajectory commands
        self.pub = self.create_publisher(JointTrajectory, self.CMD_TOPIC, 10)
        self.min_traj_dur = 0  # Minimum trajectory duration in seconds
        
        self.get_logger().info("ReachPolicy node initialized.")

    def sub_callback(self, msg: JointTrajectoryControllerState):
        """
        Callback for receiving controller state messages.
        Updates the current joint positions and passes the state to the robot model.
        """
        actual_pos = {}
        for i, joint_name in enumerate(msg.joint_names):
            joint_pos = msg.actual.positions[i]
            actual_pos[joint_name] = joint_pos
        self.current_pos = actual_pos
        
        # Update the robot's state with current joint positions and velocities.
        self.robot.update_joint_state(msg.actual.positions, msg.actual.velocities)

    def map_joint_angle(self, pos: float, index: int) -> float:
        """
        Map a simulation joint angle (in radians) to the real-world servo angle (in radians).
        
        Args:
            pos: Joint angle from simulation (in radians).
            index: Index of the joint.
        
        Returns:
            Mapped joint angle within the servo limits.
        """
        L, U, inversed = self.SIM_DOF_ANGLE_LIMITS[index]
        A, B = self.SERVO_ANGLE_LIMITS[index]
        angle_deg = np.rad2deg(float(pos))
        # Check if the simulation angle is within limits
        if not L <= angle_deg <= U:
            self.get_logger().warn(
                f"Simulation joint {index} angle ({angle_deg}) out of range [{L}, {U}]. Clipping."
            )
            angle_deg = np.clip(angle_deg, L, U)
        # Map the angle from the simulation range to the servo range
        mapped = (angle_deg - L) * ((B - A) / (U - L)) + A
        if inversed:
            mapped = (B - A) - (mapped - A) + A
        # Verify the mapped angle is within servo limits
        if not A <= mapped <= B:
            raise Exception(
                f"Mapped joint {index} angle ({mapped}) out of servo range [{A}, {B}]."
            )
        return mapped

    def step_callback(self):
        """
        Timer callback to compute and publish the next joint trajectory command.
        """
        # Set a constant target command for the robot (example values)
        self.target_command = np.array([0.5, 0.0, 0.2, 0.7071, 0.0, 0.7071, 0.0])
        
        # Get simulation joint positions from the robot's forward model
        joint_pos = self.robot.forward(self.step_size, self.target_command)
        
        if joint_pos is not None:
            if len(joint_pos) != 6:
                raise Exception(f"Expected 6 joint positions, got {len(joint_pos)}!")
            
            target_pos = [0] * 6
            for i, pos in enumerate(joint_pos):
                # Skip gripper joints for this task (assuming index 5 is gripper)
                if i == 5:
                    continue
                target_pos[i] = self.map_joint_angle(pos, i)
            self.target_pos = target_pos
            
            # Ensure both current and target positions are available before publishing
            if self.current_pos is None or self.target_pos is None:
                return
            
            traj = JointTrajectory()
            traj.joint_names = self.JOINT_NAMES
            point = JointTrajectoryPoint()
            dur_list = []
            # The moving_average factor here is set to 1, meaning full target command is used.
            moving_average = 1  
            
            for joint_name in traj.joint_names:
                pos = self.current_pos[joint_name]
                target = self.target_pos[self.JOINT_NAME_TO_IDX[joint_name]]
                # Compute the command using a weighted average (with weight = 1, it equals the target)
                cmd = pos * (1 - moving_average) + target * moving_average
                max_vel = 0.7  # maximum velocity (units per second)
                duration = abs(cmd - pos) / max_vel if max_vel else self.min_traj_dur
                dur_list.append(max(duration, self.min_traj_dur))
                point.positions.append(cmd)
            
            max_duration = max(dur_list) if dur_list else 0.0
            sec = int(max_duration)
            nanosec = int((max_duration - sec) * 1e9)
            point.time_from_start = Duration(sec=sec, nanosec=nanosec)
            traj.points.append(point)
            
            self.pub.publish(traj)
            
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = ReachPolicy()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
