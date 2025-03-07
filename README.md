# ur_onnx_controller

This repository provides a ROS2-based control framework for Universal Robots using an ONNX policy model. It integrates the UR controller with a Gazebo simulation environment. The setup has been tested with ROS2 Humble, and it should also work with ROS2 Foxy.

> **Note:** This project is still a work in progress. Policy testing is working but exhibit unexpected behavior.

## Prerequisites

- **ROS2** (tested on Humble; Foxy support is expected)
  - **UR Controller**
  - **UR Simulation in Gazebo**
- **Python** (with the onnx package installed)

## Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd ur_onnx_control
   ```

2. **Source your ROS2 setup and other necessary environments:**

   ```bash
   cd repo-path/
   source /opt/ros/<distro>/setup.bash
   ```

## Launching the Simulation and Controller

### 1. Launch the Simulation

Open a terminal and run:

```bash
ros2 launch ur_simulation_gazebo ur_sim_control.launch.py initial_joint_controller:=forward_position_controller
```

### 2. Test the Forward Position Controller

Open a second terminal.

*Tests:*

- Send test commands:

  ```bash
  ros2 topic pub /forward_position_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"
  ```

- Test the positional demo:

  ```bash
  python3 python/ur_pos_demo.py
  ```

Try out the reinforcement learning (policy) demo:

```bash
python3 python/ur_rl_demo.py
```

### 3. Send a Target Pose Command

Open a third terminal and publish a target pose:

```bash
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
```

## Policy ONNX Model Details

The ONNX policy model for the UR10 has the following input and output specifications:

- **Input:**
  - **Name:** `obs`
  - **Shape:** `[1, 25]`

- **Output:**
  - **Name:** `actions`
  - **Shape:** `[1, 6]`

## How to Retrieve ONNX Model Information

To inspect the input and output details of the ONNX model (`policy.onnx`), navigate to the model’s directory and run the following Python script:

```python
import onnx

# Load the ONNX model
model_path = "policy.onnx"
model = onnx.load(model_path)

# Get the input details
print("Input details:")
for input_tensor in model.graph.input:
    shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in input_tensor.type.tensor_type.shape.dim]
    print(f"Name: {input_tensor.name}, Shape: {shape}")

# Get the output details
print("\nOutput details:")
for output_tensor in model.graph.output:
    shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in output_tensor.type.tensor_type.shape.dim]
    print(f"Name: {output_tensor.name}, Shape: {shape}")
```
