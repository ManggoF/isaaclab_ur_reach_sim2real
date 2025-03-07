ros2 launch ur_simulation_gazebo ur_sim_control.launch.py initial_joint_controller:=forward_position_controller

ros2 topic pub /forward_position_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.0, -1.712, 1.712, 0.0, 0.0, 0.0]}"
ros2 topic pub /forward_position_controller/commands std_msgs/msg/Float64MultiArray "{data: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]}"
ros2 topic pub /forward_position_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"

ros2 topic pub /target_pose geometry_msgs/PoseStamped "
header:
  stamp:
    sec: 0
    nanosec: 0
  frame_id: 'base_link'
pose:
  position:
    x: 0.5
    y: 0.0
    z: 0.2
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0
"



reach ur10 policy onnx infos :

Input details:
Name: obs, Shape: [1, 25]

Output details:
Name: actions, Shape: [1, 6]





HOW TO GET THESE INFOS :
place this script in the policy.onnx model file dir

import onnx

# Load the ONNX model
model_path = "policy.onnx"
model = onnx.load(model_path)

# Get the input details
print("Input details:")
for input_tensor in model.graph.input:
    print(f"Name: {input_tensor.name}, Shape: {[(dim.dim_value if dim.dim_value > 0 else 'dynamic') for dim in input_tensor.type.tensor_type.shape.dim]}")

# Get the output details
print("\nOutput details:")
for output_tensor in model.graph.output:
    print(f"Name: {output_tensor.name}, Shape: {[(dim.dim_value if dim.dim_value > 0 else 'dynamic') for dim in output_tensor.type.tensor_type.shape.dim]}")
