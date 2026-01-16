#!/bin/bash

# ================= 配置区域 =================
# 第一个 Python 脚本的路径
PYTHON_SCRIPT_1="play.py"
PKL_PATH="mydemo.pkl"

# 第二个任务：你可以选择执行 Python 或者 roslaunch
# 如果是 Python:
NEXT_PYTHON_SCRIPT="./reach_rtde_node.py"

# 如果是 ROS Launch (取消下面的注释即可)
# ROS_PKG="your_package_name"
# ROS_LAUNCH_FILE="your_launch_file.launch"
# ===========================================

echo "开始执行第一个任务: 轨迹播放..."

# 执行第一个脚本
# --path 是你之前 Python 脚本需要的参数
python $PYTHON_SCRIPT_1 --path $PKL_PATH

# $? 是上一个命令的退出状态码。0 表示成功。
if [ $? -eq 0 ]; then
    echo "------------------------------------------"
    echo "第一个任务成功完成。正在启动下一个任务..."
    echo "------------------------------------------"
    
    # 方式 A: 执行另一个 Python 脚本
    python3 $NEXT_PYTHON_SCRIPT
    
    # 方式 B: 执行 roslaunch (如果要用这个，请把上面的 python 行注释掉)
    # source /opt/ros/noetic/setup.bash  # 确保环境变量已加载
    # source ~/catkin_ws/devel/setup.bash
    # roslaunch $ROS_PKG $ROS_LAUNCH_FILE

else
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "警告: 第一个脚本执行出错或被手动中断，将不执行后续任务。"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1
fi

echo "所有任务执行完毕。"
