#!/bin/bash

set -e

echo "=============================="
echo " UR + Camera + Perception START (STABLE VERSION)"
echo "=============================="

# ===== 0. ROS环境 =====
source /opt/ros/humble/setup.bash

# ⚠️ 只保留一个 workspace（避免污染）
source ~/ros2_ws/install/setup.bash

# ===== 1. 等UR端口 ready =====
echo "[1/4] Waiting for UR robot..."

until nc -z 192.168.56.101 29999; do
  echo "waiting dashboard..."
  sleep 1
done

until nc -z 192.168.56.101 30004; do
  echo "waiting rtde..."
  sleep 1
done

echo "UR ports ready ✔"

# 👉 比 sleep 更靠谱：加 handshake buffer
sleep 5

# ===== 2. 启动UR driver =====
echo "[2/4] Launching UR driver..."

ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5e \
  robot_ip:=192.168.56.101 \
  initialization_timeout:=15.0 &
UR_PID=$!

# 👉 等 controller 真起来（关键优化）
echo "Waiting UR driver stable..."
sleep 12

# ===== 3. 启动 camera / TF =====
echo "[3/4] Launching camera..."

ros2 launch my_robot dual_cam.launch.py &
CAMERA_PID=$!

sleep 1

# ===== 4. 启动 perception =====
echo "[4/4] Starting face tracker..."

cd ~/isaaclab_ur_reach_sim2real/dual_cam
python3 dual_camera_face_pose.py &
PERCEPTION_PID=$!

echo "=============================="
echo "SYSTEM STARTED ✔"
echo "UR PID: $UR_PID"
echo "CAMERA PID: $CAMERA_PID"
echo "=============================="

wait