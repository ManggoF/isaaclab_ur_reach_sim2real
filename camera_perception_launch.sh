#!/bin/bash

set -e

echo "=============================="
echo " Camera Perception START"
echo "=============================="

# ===== 0. ROS环境 =====
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

cd ~/isaaclab_ur_reach_sim2real/dual_cam

cleanup() {
  echo ""
  echo "Stopping camera perception..."
  kill "$FACE_TRACKER_PID" "$FACE_FUSION_PID" 2>/dev/null || true
  wait "$FACE_TRACKER_PID" "$FACE_FUSION_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ===== 1. 启动双目人脸/嘴部检测 =====
echo "[1/2] Starting dual camera face tracker..."
python3 dual_camera_face_pose.py &
FACE_TRACKER_PID=$!

sleep 1

# ===== 2. 启动人脸位姿融合 =====
echo "[2/2] Starting face pose fusion..."
python3 face_pose_fusion.py &
FACE_FUSION_PID=$!

echo "=============================="
echo "CAMERA PERCEPTION STARTED ✔"
echo "FACE TRACKER PID: $FACE_TRACKER_PID"
echo "FACE FUSION PID: $FACE_FUSION_PID"
echo "=============================="

wait
