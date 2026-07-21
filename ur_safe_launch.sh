#!/bin/bash

set -e

echo "=============================="
echo " UR + Camera START (STABLE VERSION)"
echo "=============================="

# ===== 0. ROS环境 =====
source /opt/ros/humble/setup.bash

# ⚠️ 只保留一个 workspace（避免污染）
source ~/ros2_ws/install/setup.bash

# ===== 1. 等UR端口 ready =====
echo "[1/3] Waiting for UR robot..."

until nc -z 192.168.56.101 29999; do
  echo "waiting dashboard..."
done

until nc -z 192.168.56.101 30004; do
  echo "waiting rtde..."
done

echo "UR ports ready ✔"
# ===== 2. 启动UR driver =====
echo "[2/3] Launching UR driver..."

ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5e \
  robot_ip:=192.168.56.101 \
  initialization_timeout:=15.0 &
UR_PID=$!

# 👉 等 controller 真起来（关键优化）
echo "Waiting UR driver stable..."

# ===== 3. 启动 camera / TF =====
echo "[3/3] Launching camera..."

ros2 launch my_robot dual_cam.launch.py &
CAMERA_PID=$!


echo "=============================="
echo "SYSTEM STARTED ✔"
echo "UR PID: $UR_PID"
echo "CAMERA PID: $CAMERA_PID"
echo "Run ./camera_perception_launch.sh in another terminal for face tracking/fusion."
echo "=============================="

wait
