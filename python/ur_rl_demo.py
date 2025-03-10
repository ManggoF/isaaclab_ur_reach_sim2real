#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# Common ROS message types
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker

import numpy as np
import os

from isaaclab.isaaclab import IsaacLab 
from ur.ur import UR 

###############################################################################
# Main Node: URONNXController
###############################################################################
class URONNXController(Node):
    def __init__(self):
        super().__init__('ur_onnx_controller')  # Node name

        # (1) Load your ONNX model via IsaacLab
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'utils', 'reach_ur10', 'policy.onnx')
        self.isaaclab = IsaacLab(model_path=model_path)

        # (2) Initialize UR transformations
        self.ur = UR()
        self.ur.action_scales *= 0.5
        self.ur.action_offsets = [0.0, -1.712, 1.712, 0.0, 0.0, 0.0]
        self.ur.position_offsets = [0.0, -1.712, 1.712, 0.0, 0.0, 0.0]

        self.num_joints = 6

        # (3) Initialize placeholders for data
        self.current_joint_positions = np.zeros(self.num_joints, dtype=np.float32)
        self.current_joint_velocities = np.zeros(self.num_joints, dtype=np.float32)
        self.target_pose = np.zeros(7, dtype=np.float32)   # e.g. [x, y, z, qx, qy, qz, qw]
        self.last_actions = np.zeros(self.num_joints, dtype=np.float32)

        self.has_joint_data = False
        self.has_pose_data = False

        # (4) Subscribers
        # 4.1) JointState: read current joint angles and velocities
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # 4.2) Target end-effector pose
        self.target_pose_sub = self.create_subscription(
            PoseStamped,           # or Pose
            '/target_pose',
            self.target_pose_callback,
            10
        )

        # (5) Publishers
        # 5.1) Desired joint commands
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/forward_position_controller/commands',
            10
        )

        # 5.2) Marker to show commanded pose in RViz
        self.marker_pub = self.create_publisher(
            Marker,
            'visualization_marker',
            10
        )
        
        # (6) Timer for the control loop (e.g., 60 Hz)
        self.timer = self.create_timer(1.0 / 60.0, self.control_loop)

    def joint_state_callback(self, msg: JointState):
        """
        Callback that updates current joint positions and velocities.
        """
        if len(msg.position) >= self.num_joints:
            self.current_joint_positions = np.array(msg.position[:self.num_joints], dtype=np.float32)
        if len(msg.velocity) >= self.num_joints:
            self.current_joint_velocities = np.array(msg.velocity[:self.num_joints], dtype=np.float32)

        self.has_joint_data = True

    def target_pose_callback(self, msg: PoseStamped):
        """
        Callback that updates the desired end-effector pose.
        """
        pos = msg.pose.position
        ori = msg.pose.orientation
        # Store in the 7-element array [x, y, z, qx, qy, qz, qw]
        self.target_pose[:] = [
            pos.x, pos.y, pos.z,
            ori.x, ori.y, ori.z, ori.w
        ]
        self.has_pose_data = True

    def control_loop(self):
        """
        Periodic timer callback for sending commands to the UR arm.
        We use the IsaacLab model to compute new joint setpoints.
        """
        # Only proceed if we have both the joint data and the target pose
        if not (self.has_joint_data and self.has_pose_data):
            return

        # (7) Build the model observation using the UR class
        observation = self.ur.build_observation(
            self.current_joint_positions,
            self.current_joint_velocities,
            self.target_pose,
            self.last_actions
        )
        print("obs")
        print(observation)

        # (8) Run inference via IsaacLab
        model_action = self.isaaclab.compute_action(observation)

        print("model action")
        print(model_action)
        # (9) Transform the model's action to Gazebo format
        desired_joints = self.ur.transform_action_to_gazebo(model_action)
        
        print("desired joints")
        print(desired_joints)
        # (10) Publish joint commands
        cmd_msg = Float64MultiArray()
        cmd_msg.data = desired_joints.tolist()
        self.joint_cmd_pub.publish(cmd_msg)

        # self.get_logger().info(f"Publishing joint command: {desired_joints}")

        # (11) Update last_actions
        self.last_actions = model_action # desired_joints.astype(np.float32)

        # (12) Publish a marker in RViz to indicate the commanded pose
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "commanded_pose"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        # Position
        marker.pose.position.x = float(self.target_pose[0])
        marker.pose.position.y = float(self.target_pose[1])
        marker.pose.position.z = float(self.target_pose[2])
        # Orientation
        marker.pose.orientation.x = float(self.target_pose[3])
        marker.pose.orientation.y = float(self.target_pose[4])
        marker.pose.orientation.z = float(self.target_pose[5])
        marker.pose.orientation.w = float(self.target_pose[6])
        # Scale
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        # Color
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        # Lifetime (0 means forever)
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = URONNXController()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
