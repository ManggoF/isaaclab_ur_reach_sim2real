import os
import time
import pickle
import numpy as np
import argparse
import sys
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import data_models

# --- 项目路径配置 ---
project_path = '/home/xry/isaaclab_ur_reach_sim2real/dual_cam'
if project_path not in sys.path:
    sys.path.append(project_path)



try:
    from data_models import SensorMessageList
except ImportError as e:
    print(f"无法导入项目模块: {e}")
    sys.exit(1)

try:
    import rtde_control
except ImportError:
    print("请安装: pip install ur_rtde")
    sys.exit(1)

sys.modules['reactive_diffusion_policy.common.data_models'] = sys.modules['data_models']
sys.modules['reactive_diffusion_policy'] = sys.modules['data_models']

ROBOT_IP = "192.168.56.10" 

DT = 0.005

LOOKAHEAD_TIME = 0.2
GAIN = 100            


GRIPPER_PORT = 63352
MAX_GRIPPER_WIDTH = 0.085

def convert_pose_to_ur_format(pose):
    pose = np.array(pose)
    pos = pose[:3]
    r = R.from_euler('xyz', pose[3:], degrees=False)
    return np.concatenate([pos, r.as_rotvec()])


def play_trajectory_servo(file_path, rtde_c):
    print(f"\n>> 正在加载数据: {file_path}")
    with open(file_path, 'rb') as f:
        data_object = pickle.load(f)
    
    messages = data_object.sensorMessages
    num_frames = len(messages)
    
    original_poses = []
    original_gripper_widths = []
    
    for msg in messages:
        original_poses.append(convert_pose_to_ur_format(msg.leftRobotTCP))
        
        if hasattr(msg, 'leftGripperState'):
            width = msg.leftGripperState[0]
        elif hasattr(msg, 'leftRobotGripperWidth'):
            width = msg.leftRobotGripperWidth
        else:
            width = MAX_GRIPPER_WIDTH
        original_gripper_widths.append(width)

    original_poses = np.array(original_poses)


    record_freq = 10.0
    total_time = num_frames / record_freq
    t_original = np.linspace(0, total_time, num_frames)
    t_new = np.arange(0, total_time, DT)
    

    interp_robot = interp1d(t_original, original_poses, axis=0, kind='linear', fill_value="extrapolate")
    smoothed_path = interp_robot(t_new)
    
    # 插值夹爪 (0阶保持或线性插值均可，线性更平滑)
    #interp_gripper = interp1d(t_original, original_gripper_widths, kind='linear', fill_value="extrapolate")
    #smoothed_gripper_widths = interp_gripper(t_new)

    # 3. 移动到起始点
    print("正在移动到起始位置...")
    #start_gripper_val = width_to_gripper_val(smoothed_gripper_widths[0])
    
    # 移动夹爪到初始位置
    #gripper.move_and_wait_for_pos(start_gripper_val, 255, 255)
    # 移动机械臂到初始位置
    rtde_c.moveL(smoothed_path[0], speed=0.1, acceleration=0.2)
    time.sleep(0.5)

    # 4. 进入 servoL 实时循环
    #print(f"开始 servoL 播放 (频率: {CONTROL_FREQ}Hz)...")
    
    last_gripper_val = -1
    
    for i, target_pose in enumerate(smoothed_path):
        t_start = rtde_c.initPeriod() 
        
        # --- A. 发送机械臂指令 ---
        rtde_c.servoL(target_pose, 0.0, 0.0, DT, LOOKAHEAD_TIME, GAIN)
        
        # --- B. 发送夹爪指令 ---
        # 计算当前的夹爪目标值 (0-255)
        #target_gripper_val = width_to_gripper_val(smoothed_gripper_widths[i])
        
        # 为了减少 Socket 通信压力，只有当值变化超过阈值时才发送指令
        # 且使用非阻塞的 move() 方法 (不要用 move_and_wait)
        #if abs(target_gripper_val - last_gripper_val) > 2:
            # 参数: position(0-255), speed(0-255), force(0-255)
            #gripper.move(target_gripper_val, 255, 255) 
            #last_gripper_val = target_gripper_val

        
        rtde_c.waitPeriod(t_start)

    rtde_c.servoStop()
    print("播放完成！")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    # 初始化机械臂
    print("正在连接机械臂 RTDE...")
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)

    # 初始化夹爪
    #print("正在连接 Robotiq 夹爪...")
    #gripper = RobotiqGripper()
    #gripper.connect(ROBOT_IP, GRIPPER_PORT) # 夹爪通常映射在机器人 IP 的 63352 端口
    
    # 激活夹爪 (如果还没激活)
    #if not gripper.is_active():
        #print("激活夹爪中...")
        #gripper.activate()
    
    print("设备连接就绪。")

    try:
        play_trajectory_servo(args.path, rtde_c)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        print("停止设备...")
        rtde_c.servoStop()
        rtde_c.stopScript()
        # 可以在这里松开夹爪或保持现状

if __name__ == "__main__":
    main()
