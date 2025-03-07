#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# Common ROS message types
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped   # or Pose if you prefer
from std_msgs.msg import Float64MultiArray  # or custom message for commands

# For publishing markers in RViz
from visualization_msgs.msg import Marker

import numpy as np

from ament_index_python.packages import get_package_share_directory
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
        self.ur.action_scales = [0.5, -0.5, 0.5, 0.5, 0.5, 0.5]

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

        # (8) Run inference via IsaacLab
        model_action = self.isaaclab.compute_action(observation)

        # (9) Transform the model's action to Gazebo format
        desired_joints = self.ur.transform_action_to_gazebo(model_action)

        # (10) Publish joint commands
        cmd_msg = Float64MultiArray()
        cmd_msg.data = desired_joints.tolist()
        self.joint_cmd_pub.publish(cmd_msg)

        self.get_logger().info(f"Publishing joint command: {desired_joints}")

        # (11) Update last_actions (in real or model order – depends on your choice)
        # Here, let's store the real command we just published as last_actions,
        # but if you prefer the model-space action as "last_actions", you can keep that.
        self.last_actions = model_action # desired_joints.astype(np.float32)

        # (12) Publish a marker in RViz to indicate the commanded pose
        marker = Marker()
        marker.header.frame_id = "base_link"  # or your chosen frame
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


#################################################################################################################################

# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node

# # Common ROS message types you might need:
# from sensor_msgs.msg import JointState
# from geometry_msgs.msg import PoseStamped   # or Pose if you prefer
# from std_msgs.msg import Float64MultiArray  # or custom message for commands

# # For publishing markers in RViz
# from visualization_msgs.msg import Marker

# # ONNX Runtime
# import onnxruntime as ort
# import numpy as np

# from ament_index_python.packages import get_package_share_directory
# import os

# class URONNXController(Node):
#     def __init__(self):
#         super().__init__('ur_onnx_controller')  # Node name

#         # (1) Load your ONNX model
#         # model_path = os.path.join(get_package_share_directory('ur_onnx_control'), 'resource', 'reach_ur10', 'policy.onnx')
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         model_path = os.path.join(script_dir, 'utils', 'reach_ur10', 'policy.onnx')
#         self.ort_session = ort.InferenceSession(model_path)
#         print(f"ONNX model path: {model_path}")

#         # (2) Initialize placeholders for data
#         self.num_joints = 6

#         self.current_joint_positions = np.zeros(self.num_joints, dtype=np.float32)
#         self.current_joint_velocities = np.zeros(self.num_joints, dtype=np.float32)
#         self.target_pose = np.zeros(7, dtype=np.float32)   # e.g., [x, y, z, qx, qy, qz, qw]
#         self.last_actions = np.zeros(self.num_joints, dtype=np.float32)

#         self.has_joint_data = False
#         self.has_pose_data = False

#         # (3) Subscribers
#         # 3.1) JointState: read current joint angles and velocities
#         self.joint_state_sub = self.create_subscription(
#             JointState,
#             '/joint_states',
#             self.joint_state_callback,
#             10
#         )

#         # 3.2) Target end-effector pose
#         self.target_pose_sub = self.create_subscription(
#             PoseStamped,           # or Pose
#             '/target_pose',
#             self.target_pose_callback,
#             10
#         )

#         # (4) Publishers
#         # 4.1) Desired joint commands
#         self.joint_cmd_pub = self.create_publisher(
#             Float64MultiArray,
#             '/forward_position_controller/commands',
#             10
#         )

#         # 4.2) Marker to show commanded pose in RViz
#         self.marker_pub = self.create_publisher(
#             Marker,
#             'visualization_marker',
#             10
#         )
        
#         # (5) Timer for the control loop (e.g., 60 Hz)
#         self.timer = self.create_timer(1.0/60.0, self.control_loop)

#     def joint_state_callback(self, msg: JointState):
#         """
#         Callback that updates current joint positions and velocities.
#         Assumes the msg contains exactly the UR's 6 joints in order.
#         """
#         if len(msg.position) >= self.num_joints:
#             self.current_joint_positions = np.array(msg.position[:self.num_joints], dtype=np.float32)
#         if len(msg.velocity) >= self.num_joints:
#             self.current_joint_velocities = np.array(msg.velocity[:self.num_joints], dtype=np.float32)

#         self.has_joint_data = True

#     def target_pose_callback(self, msg: PoseStamped):
#         """
#         Callback that updates the desired end-effector pose.
#         """
#         pos = msg.pose.position
#         ori = msg.pose.orientation
#         # Store in the 7-element array [x, y, z, qx, qy, qz, qw]
#         self.target_pose[:] = [
#             pos.x, pos.y, pos.z,
#             ori.x, ori.y, ori.z, ori.w
#         ]
#         self.has_pose_data = True

#     def control_loop(self):
#         """
#         Periodic timer callback for sending commands to the UR arm.
#         We use the ONNX model to compute new joint setpoints.
#         Model input = [joint_pos + joint_vel + pose_command + last_actions].
#         """
#         # Only proceed if we have both the joint data and the target pose
#         if not (self.has_joint_data and self.has_pose_data):
#             return

#         # (6) Build the model input:
#         #     joint_pos (6) + joint_vel (6) + pose_command (7) + last_actions (6) = 25 elements (example).
#         model_input = np.concatenate([
#             self.current_joint_positions,
#             self.current_joint_velocities,
#             self.target_pose,
#             self.last_actions
#         ])
#         # Reshape to [batch_size, input_dim] as required by ONNX (usually [1, n])
#         model_input = model_input.reshape(1, -1).astype(np.float32)

#         # (7) Run inference
#         if model_input.shape != (1, 25):
#             print(model_input)
#             return
#         outputs = self.ort_session.run(None, {"obs": model_input})
#         # Apply action scale
#         outputs = [x * 0.5 for x in outputs]

#         # Suppose the model returns a 6D array for next joint commands
#         desired_joints = outputs[0].squeeze()

#         # (8) Publish joint commands
#         cmd_msg = Float64MultiArray()
#         cmd_msg.data = desired_joints.tolist()
#         self.joint_cmd_pub.publish(cmd_msg)

#         self.get_logger().info(f"Publishing joint command: {desired_joints}")

#         # (9) Update last_actions with newly published commands
#         self.last_actions = desired_joints.astype(np.float32)

#         # (10) Publish a marker in RViz to indicate the commanded pose
#         marker = Marker()
#         marker.header.frame_id = "base_link"  # or whatever frame you want to use
#         marker.header.stamp = self.get_clock().now().to_msg()
#         marker.ns = "commanded_pose"
#         marker.id = 0
#         marker.type = Marker.SPHERE
#         marker.action = Marker.ADD
#         # Position
#         marker.pose.position.x = float(self.target_pose[0])
#         marker.pose.position.y = float(self.target_pose[1])
#         marker.pose.position.z = float(self.target_pose[2])
#         # Orientation
#         marker.pose.orientation.x = float(self.target_pose[3])
#         marker.pose.orientation.y = float(self.target_pose[4])
#         marker.pose.orientation.z = float(self.target_pose[5])
#         marker.pose.orientation.w = float(self.target_pose[6])
#         # Scale
#         marker.scale.x = 0.05
#         marker.scale.y = 0.05
#         marker.scale.z = 0.05
#         # Color
#         marker.color.a = 1.0
#         marker.color.r = 0.0
#         marker.color.g = 1.0
#         marker.color.b = 0.0
#         # Lifetime (0 means forever)
#         marker.lifetime.sec = 0
#         marker.lifetime.nanosec = 0
#         # Publish
#         self.marker_pub.publish(marker)

# def main(args=None):
#     rclpy.init(args=args)
#     node = URONNXController()
#     rclpy.spin(node)

#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()