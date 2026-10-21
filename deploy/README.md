# UR5e 部署说明

本文对应当前 `deploy/` 实机部署代码，环境为 Ubuntu 22.04、ROS 2 Humble、UR5e。
项目根目录示例为 `~/isaaclab_ur_reach_sim2real`，ROS 工作区为 `~/ros2_ws`，
Python 环境为 `~/ros_env`。更换电脑时需要替换路径、机器人 IP、相机序列号和标定。

**直接执行 `python3 run_task.py` 会发布运动指令，且默认使用固定嘴部目标。
使用相机识别的嘴部目标必须加 `--vision-mouth`；只检查推理则加 `--dry-run`。**

阅读入口：[启动流程](#3-完整启动流程) · [参数与记录](#4-常用参数与输出) ·
[常见问题](#5-常见问题与解决方法) · [离线测试](#6-离线回归检查)。

## 1. 文件和数据流

| 文件/组件 | 作用 |
| --- | --- |
| [`../ur_safe_launch.sh`](../ur_safe_launch.sh) | 启动 UR driver 和相机标定 TF；当前机器人 IP 写在脚本内 |
| [`../camera_perception_launch.sh`](../camera_perception_launch.sh) | 启动双相机识别与位姿融合；使用当前终端的 `python3` |
| [`dual_camera_face_pose.py`](dual_camera_face_pose.py) | 通过 RealSense SDK 直接打开两台相机，识别人脸并发布嘴部位姿 |
| [`face_pose_fusion.py`](face_pose_fusion.py) | 将两台相机的结果转换到 `base_link`，选择、过滤后发布 `/mouth_pose` |
| [`reset_arm.py`](reset_arm.py) | 独立归位工具，默认只预览，`--execute` 才发布轨迹 |
| [`run_task.py`](run_task.py) | arm/safety 推理、任务状态机、撤回、自动归位和曲线记录 |
| [`tests/`](tests/) | 离线测试，正常执行任务时不会加载 |
| 外部工作区的 `my_robot/launch/dual_cam.launch.py` | 发布腕部、固定相机的标定 TF，不包含在本仓库中 |

当前相机链路如下：

```text
腕部相机 → /wrist_face_pose ─┐
                           ├→ face_pose_fusion.py → /mouth_pose（base_link）
固定相机 → /fixed_face_pose ─┘                          ↓ --vision-mouth
关节反馈 + 速度缩放 + wrench + 机器人 TF → run_task.py → 轨迹控制器 → UR5e
```

当前工作区中的 `my_robot dual_cam.launch.py` **只发布 TF，不打开相机图像流**。
相机图像由 `dual_camera_face_pose.py` 读取，不依赖 ROS 图像话题。
因此，不需要再为这两台相机启动 `realsense2_camera`；同时打开同一设备可能导致占用冲突。

## 2. 每个新终端的环境准备

执行 Python 脚本或 ROS 检查命令之前，先加载相同环境：

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/ros_env/bin/activate
cd ~/isaaclab_ur_reach_sim2real
```

检查解释器和主要依赖：

```bash
command -v python3
python3 --version
python3 -c 'import rclpy, numpy, scipy, yaml, torch, cv2, matplotlib; print("task dependencies OK")'
python3 -c 'import mediapipe as mp, pyrealsense2; print("camera dependencies OK", hasattr(mp, "solutions"))'
ros2 pkg prefix ur_robot_driver
ros2 pkg prefix my_robot
```

当前 `ros_env` 使用 Python 3.10，第一条应指向 `/home/gxr/ros_env/bin/python3`。
ROS 2 Humble 的 Python 扩展需要匹配的解释器；不要误用 Conda 中其他版本的 Python。
第二个导入检查还应输出 `True`，因为识别脚本使用 `mp.solutions.face_mesh`。

如果有多个 ROS 工作区，只加载本次使用的工作区，避免同名包被其他安装覆盖。

## 3. 完整启动流程

下面按实际启动顺序操作。各终端均先完成第 2 节的环境准备。
除明确写出 `cd` 的示例外，命令均在项目根目录执行。

### 3.1 示教器与网络

1. 确认机器人型号为 UR5e，机器人上电、释放制动，停止状态已按现场流程解除。
2. 示教器中准备包含 **External Control** 节点的程序；URCap 中填写的是机器人能访问的**电脑 IP**。
3. 确认电脑能访问机器人。当前脚本使用 `192.168.56.101`，它也常用于 URSim；
   实机地址不同就修改 `ur_safe_launch.sh` 中的所有对应地址，或使用下文手动启动方式。

```bash
ping -c 3 192.168.56.101
nc -zv 192.168.56.101 29999
nc -zv 192.168.56.101 30004
```

`29999` 是 Dashboard 端口，`30004` 是 RTDE 端口。端口能连通仅代表网络服务可达，
还需要后续运行 External Control 并检查控制器。

### 3.2 终端 A：启动 UR driver 和相机 TF

在项目根目录执行：

```bash
bash ur_safe_launch.sh
```

该脚本加载 ROS 环境、等待机器人端口，然后启动：

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5e robot_ip:=192.168.56.101 initialization_timeout:=15.0
```

以及外部工作区的：

```bash
ros2 launch my_robot dual_cam.launch.py
```

以上两条是脚本内部操作的说明，**已经运行脚本就不要再重复启动**。
脚本虽然打印 `Waiting UR driver stable...` / `SYSTEM STARTED`，但当前没有等待控制器就绪的检查；
不能据此直接开始任务。

也可以不用脚本，分别在两个终端手动启动。手动方式便于明确初始控制器：

```bash
# 终端 A1：机器人地址按实际修改
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5e \
  robot_ip:=192.168.56.101 \
  initialization_timeout:=15.0 \
  initial_joint_controller:=scaled_joint_trajectory_controller \
  activate_joint_controller:=true
```

```bash
# 终端 A2：相机标定 TF
ros2 launch my_robot dual_cam.launch.py
```

驱动启动后，在示教器上运行 External Control 程序，并将速度滑条设为经过验证的非零值。
驱动重连或机器人停止后，可能需要重新按播放。

### 3.3 终端 B：确认控制器与机器人反馈

先看控制器状态：

```bash
ros2 control list_controllers -c /controller_manager
```

本任务依赖以下组件处于 `active`：

| 组件 | 用途 |
| --- | --- |
| `joint_state_broadcaster` | 发布关节状态，供机器人 TF 等使用 |
| `scaled_joint_trajectory_controller` | 发布任务使用的关节反馈，接收关节轨迹 |
| `speed_scaling_state_broadcaster` | 发布 UR 速度缩放 |
| `force_torque_sensor_broadcaster` | 发布六维力/力矩 |

若 `scaled_joint_trajectory_controller` 为 `inactive`，按第 5.1 节激活；
若其他运动控制器占用了相同关节，按第 5.2 节切换。**只有 `/joint_states` 并不足以让任务开始。**

检查消息确实到达，而不只是话题名存在：

```bash
ros2 topic echo /scaled_joint_trajectory_controller/controller_state --once
ros2 topic echo /speed_scaling_state_broadcaster/speed_scaling --once
ros2 topic echo /force_torque_sensor_broadcaster/wrench --once
ros2 run tf2_ros tf2_echo base_link wrist_3_link
```

`tf2_echo` 持续输出，用 `Ctrl+C` 结束检查。如果 `echo --once` 一直等待，按 `Ctrl+C` 后排查对应发布者。
不同 driver 版本可能发布 `/force_torque_sensor_broadcaster/ft_data`，见第 5.4 节的重映射方式。

### 3.4 终端 C：启动双相机识别和融合

```bash
bash camera_perception_launch.sh
```

正常会出现腕部、固定相机两个 MediaPipe 窗口，并发布：

```text
/wrist_face_pose                    腕部相机坐标系下的嘴部位姿
/fixed_face_pose                    固定相机坐标系下的嘴部位姿
/mouth_pose                         base_link 下的嘴部位姿
/wrist_face_pose_node/mouth_dist     嘴唇开合距离（辅助观察）
/fixed_face_pose_node/mouth_dist     嘴唇开合距离（辅助观察）
```

两台设备的序列号写在 `deploy/dual_camera_face_pose.py` 的 `main()` 中：

```python
WRIST_CAM_SN = "335122270893"
FIXED_CAM_SN = "043322071261"
```

更换相机后用 `rs-enumerate-devices` 查询序列号，再修改这两个值。
更换安装位置还需要更新手眼/固定相机标定，不能只改序列号。

在终端 B 检查视觉消息和 TF：

```bash
ros2 topic echo /wrist_face_pose --once
ros2 topic echo /fixed_face_pose --once
ros2 topic echo /mouth_pose --once
ros2 topic hz /mouth_pose
ros2 run tf2_ros tf2_echo base_link wrist_cam_optical_frame
ros2 run tf2_ros tf2_echo base_link fixed_cam_optical_frame
```

后面三条是持续检查，逐条执行，用 `Ctrl+C` 结束后执行下一条。
`/mouth_pose.header.frame_id` 应为 `base_link`，位置单位为米。

当前融合逻辑**优先使用固定相机**，固定相机无有效目标时使用腕部相机，并非两个位姿取平均。
两路目标超过 0.5 秒无更新时，融合节点会持续重发最后一个可信目标，日志中出现
`继续发布上一帧可信嘴巴`；没有保持时长上限。因此，`/mouth_pose` 仍有频率不能证明相机仍在识别，
还要检查两路原始位姿和识别窗口。

### 3.5 终端 D：归位到策略初始姿态

初次运行或需要恢复初始姿态时，先预览：

```bash
python3 deploy/reset_arm.py
```

默认目标角度依次为：

```text
shoulder_pan_joint   -45°
shoulder_lift_joint  -80°
elbow_joint          110°
wrist_1_joint        -90°
wrist_2_joint          0°
wrist_3_joint         60°
```

确认当前位置到目标的关节运动路径畅通后，执行归位：

```bash
python3 deploy/reset_arm.py --execute
```

该工具发送关节空间归位轨迹，不包含先退出嘴部的逻辑。
勺子仍在嘴部附近时应先撤离，再独立归位。等待归位完成后再启动任务，避免两个节点同时发布轨迹。

### 3.6 终端 D：先检查推理，再执行任务

使用视觉目标进行不发布运动的检查：

```bash
python3 deploy/run_task.py --vision-mouth --dry-run --debug-obs --debug-wrench
```

检查模型已加载、关节/力数据和 TF 正常、嘴部坐标与嘴轴方向合理。
`--dry-run` 不发布策略、撤回或归位运动，但仍需要真实反馈才能推理；它会跳过运动发布前的
控制器订阅者/速度缩放就绪检查，所以正式执行前仍要完成第 3.3 节。
检查结束用 `Ctrl+C` 退出，再执行：

```bash
python3 deploy/run_task.py --vision-mouth
```

也可以进入 `deploy/` 后执行同样命令：

```bash
cd ~/isaaclab_ur_reach_sim2real/deploy
python3 run_task.py --vision-mouth
```

如果要复现固定目标实验，回到项目根目录，使用不带 `--vision-mouth` 的命令：

```bash
cd ~/isaaclab_ur_reach_sim2real
python3 deploy/run_task.py
```

固定目标由 `DEFAULT_FIXED_MOUTH_POS`、`DEFAULT_FIXED_MOUTH_AXIS` 等常量设置，
不随相机结果更新。当前目标位置约为 `[0.77805, 0.55456, 0.21979] m`。

视觉模式也有锁定行为：`LATCH_FIRST_VISION_MOUTH_POSE = True` 默认锁定首帧目标，
后续视觉消息只刷新时间戳，不改变本次 episode 的嘴位置/姿态。
需要持续跟随视觉时才将其改为 `False`，并先以 `--dry-run` 检查抖动和嘴轴方向。
锁定模式重新选目标需要退出并重新运行任务。

### 3.7 完成任务和退出

正常成功流程为：满足成功条件 → 保持 30 个控制步 → 沿进入嘴巴的反方向连续撤回 10 cm
→ 实际反馈确认到位、稳定 → 当前节点直接下发归位轨迹 → 归位完成后保存曲线。
默认 30 Hz 下保持阶段约为 1 秒。

安全加强开启时，有效 `grec > 0` 会跳过成功保持，直接执行撤回 10 cm 再归位。
归位完成后不会自动开始下一次 episode；需要退出并重新运行任务。

通常先在任务终端按 `Ctrl+C` 保存并退出，再停止相机识别和底层启动终端。
完整轨迹已经下发后，退出 Python 不保证控制器立即停止；需要立即停止运动时使用示教器停止或急停。

## 4. 常用参数与输出

### 4.1 命令行参数

| 参数 | 当前默认值/作用 |
| --- | --- |
| `--vision-mouth` | 使用视觉目标；不传时默认固定目标 |
| `--dry-run` | 只推理/预览，不发布运动 |
| `--debug-obs` | 打印低频观测、动作、安全状态诊断 |
| `--debug-wrench` | 打印力觉诊断，并额外保存 wrench CSV |
| `--model-path` | 默认 `sample/actor_agent_arm_torchscript.pt` |
| `--policy-backend` | 默认 `auto`，可选 `torchscript`、`numpy` |
| `--mouth-topic` | 默认 `/mouth_pose`，消息类型 `geometry_msgs/msg/PoseStamped` |
| `--control-hz` | 默认 `30` Hz |
| `--withdraw-distance` | 成功撤回距离，默认 `0.10` m，不能小于 `0.08` m；不改变安全加强的固定 10 cm |
| `--wrench-bias-samples` | 默认静态采集 `30` 个样本做 tare；`0` 表示关闭 tare |
| `--wrench-sign` | 默认 `-1`，统一实机与仿真力方向 |
| `--target-timeout` / `--joint-state-timeout` / `--wrench-timeout` / `--speed-scaling-timeout` | 默认均为 `0.5` 秒 |
| `--episode-length-s` | 默认 `240` 秒 |

更多选项见 `python3 deploy/run_task.py --help`。
应用参数要写在 `--ros-args` **之前**；话题重映射等 ROS 参数写在它之后。

### 4.2 `ArmPolicyDeployNode` 配置区

以下是 `deploy/run_task.py` 中的类属性，修改保存后重新运行生效，不是命令行参数：

```python
safety_strenthen = True       # 安全加强：有效 grec 触发后，撤回 10 cm 再归位
safety_grec_force_one = False # False：记录原有效安全增益；True：接管期间记录 grec=1
SUCCESS_WITHDRAW_SPEED_M_S = 0.05  # 标称平均撤回速度，0.10/0.05 = 2 秒
RESET_AFTER_SUCCESS = True
LATCH_FIRST_VISION_MOUTH_POSE = True
SAFETY_GAIN_VISUALIZATION = True
SAVE_SAFETY_GAIN_IMAGES = True
SAVE_FULL_SAFETY_GAIN_CURVE = True
```

开关拼写保留为 `safety_strenthen`，布尔值使用 Python 的 `True` / `False`。
关闭安全加强会恢复原安全策略的逐步干预，不会关闭原安全策略。
`safety_grec_force_one=False` 保留危险门控、撤回锁存和最小增益逻辑，记录的是原有效增益，
不是未经处理的 actor 原始动作；撤回/归位期间仍可更新诊断增益，但不会让 arm actor 重新接管。
安全加强自己的归位不受 `RESET_AFTER_SUCCESS=False` 影响。

### 4.3 自动记录

路径均相对于项目根目录，与从哪个目录启动脚本无关：

| 输出位置 | 保存内容与时机 |
| --- | --- |
| `logs/joint_angle_records/joint_angles_<时间>_<模式>.csv` | 默认自动记录六关节实际反馈，包括撤回和归位；运行中定期刷新到磁盘 |
| 同名 `.png` | 六条曲线画在同一张图中，纵轴为 `-2π` 到 `2π` rad；结束/归位完成及 `Ctrl+C` 时生成或更新 |
| `logs/safety_gain_visualization/safety_gain_curve_<时间>_<原因>.csv` / `.png` | 默认自动保存完整 `gslow`、`glat`、`grec` 和风险数据；episode 停止、自动归位结束/超时或 `Ctrl+C` 时导出 |
| `logs/safety_gain_visualization/latest_safety_gains.png` | 定期覆盖的实时面板截图，不是完整运行曲线 |
| `logs/wrench_records/real_wrench_<时间>_<模式>.csv` | 需要 `--debug-wrench` 才生成的额外诊断记录 |

关节 CSV 保留原始实测角度，不做 `±2π` 裁剪或取模。
完整 safety 曲线先保存在内存中；一直未进入推理、没有 safety 样本时不会导出。
强制杀进程可能来不及绘图或导出，因此日常用 `Ctrl+C` 正常退出。

## 5. 常见问题与解决方法

### 5.1 一直提示“等待关节状态”，如何激活控制器？

典型日志：

```text
等待关节状态 /scaled_joint_trajectory_controller/controller_state，尚未开始策略推理。
```

这是关节反馈未到达，不是模型没有加载。能收到 wrench 也不代表轨迹控制器已就绪。
先退出正在等待的任务，再检查控制器，避免激活后任务突然开始发布运动：

```bash
ros2 control list_controllers -c /controller_manager
```

根据实际状态选择操作，以下命令不是要求全部依次执行：

**已加载、状态为 `inactive`：**

```bash
ros2 control set_controller_state scaled_joint_trajectory_controller active -c /controller_manager
```

**状态为 `unconfigured`：先配置，再激活。**

```bash
ros2 control set_controller_state scaled_joint_trajectory_controller inactive -c /controller_manager
ros2 control set_controller_state scaled_joint_trajectory_controller active -c /controller_manager
```

**列表中根本没有这个控制器：**

```bash
ros2 run controller_manager spawner scaled_joint_trajectory_controller -c /controller_manager
```

这要求 UR driver 已经为该控制器加载类型、关节等参数；若 spawner 报参数缺失，
检查 UR driver 的控制器 YAML 和启动参数。若发生资源冲突，按下一节切换占用者。

完成后确认 `active`，并再次执行：

```bash
ros2 topic echo /scaled_joint_trajectory_controller/controller_state --once
```

若已 `active` 仍无反馈，检查 External Control 是否播放、driver 终端有无断连，
以及实际话题名称是否带命名空间或属于另一控制器：

```bash
ros2 topic list -t
ros2 topic info /scaled_joint_trajectory_controller/controller_state -v
ros2 topic echo /joint_states --once
```

当前代码固定使用六个无前缀的 UR 关节名。出现 `controller_state 缺少关节` 时检查消息里的
`joint_names` 和 driver 的 `tf_prefix`/关节前缀配置。
若消息字段报 `feedback`/`actual` 的 `AttributeError`，用
`ros2 interface show control_msgs/msg/JointTrajectoryControllerState` 核对接口版本；
当前代码读取 `feedback.positions` 和 `feedback.velocities`，需要匹配的控制器消息实现。

### 5.2 激活失败：其他控制器占用了关节

常见提示包括 `Error activating controller`、`Error switching controllers`、
`resource ... already claimed`。先用 `list_controllers` 确认哪个运动控制器为 `active`。
例如实际占用者是 `joint_trajectory_controller`，可在同一次切换中停用它并激活本任务控制器：

```bash
ros2 control switch_controllers -c /controller_manager \
  --deactivate joint_trajectory_controller \
  --activate scaled_joint_trajectory_controller \
  --strict
```

如果占用者是 `forward_position_controller`、`forward_velocity_controller` 等，
把 `--deactivate` 后的名称替换为实际占用者。切换前停止任务及其他发送运动的程序。
不要停用 `joint_state_broadcaster`、速度缩放或力传感器 broadcaster 来解决运动接口冲突。

### 5.3 没有订阅者、速度缩放过低，或者机器人不动

| 日志/现象 | 检查和处理 |
| --- | --- |
| `等待 .../joint_trajectory 出现订阅者` | 用 `ros2 topic info /scaled_joint_trajectory_controller/joint_trajectory -v` 检查订阅者，再检查控制器状态和命名空间 |
| `等待 .../speed_scaling` | 检查 `speed_scaling_state_broadcaster` 是否 `active`，并检查实际话题 |
| `speed_scaling=0.000 过低` | 确认示教器 External Control 正在播放、速度滑条非零，机器人没有停止/保护停止 |
| `speed_scaling 超时` | 检查 driver 是否仍连通、RTDE 是否稳定及 ROS 环境是否一致 |
| 发出了命令但一直不动 | 依次确认控制器 `active`、External Control 播放、速度缩放非零、没有其他发布者覆盖命令 |

```bash
ros2 topic echo /speed_scaling_state_broadcaster/speed_scaling --once
ros2 topic info /scaled_joint_trajectory_controller/joint_trajectory -v
```

标准速度缩放通常为 `0～1`，`0.55` 约代表 55%；日志有时会显示原始百分数形式。
不要通过降低 `--min-speed-scaling` 绕过机器人停止状态。

### 5.4 一直等待 wrench，或 `wrench超时`

58 维 arm 观测始终包含六维力/力矩，`--debug-wrench` 不是启用力传感器的开关。
先检查 `force_torque_sensor_broadcaster` 状态及实际话题：

```bash
ros2 topic list -t
ros2 topic echo /force_torque_sensor_broadcaster/wrench --once
```

若该 broadcaster 已配置但为 `inactive`，激活它：

```bash
ros2 control set_controller_state force_torque_sensor_broadcaster active -c /controller_manager
```

部分 UR driver 配置使用 `topic_name: ft_data`。如果实际有数据的是
`/force_torque_sensor_broadcaster/ft_data`，启动任务时重映射：

```bash
python3 deploy/run_task.py --vision-mouth --ros-args \
  -r /force_torque_sensor_broadcaster/wrench:=/force_torque_sensor_broadcaster/ft_data
```

固定目标实验去掉 `--vision-mouth` 即可；干运行把 `--dry-run` 加在 `--ros-args` 前面。
当前 `run_task.py` 没有 `--wrench-topic` 参数，不能套用标定工具的同名选项。

`wrench静态tare采集中 .../30` 在启动时短暂出现是正常的。
启动期间保持机械臂静止、工具无外部接触，等待 `wrench静态tare完成`。
若持续不完成，先解决消息中断；不要用关闭 tare 代替排查传感器。
更换载荷或姿态变化后存在明显静态残差时，检查示教器 payload/TCP 设置，必要时参考
[`TOOL_COM_CALIBRATION.md`](../TOOL_COM_CALIBRATION.md)。

当前任务的力对齐按 `tool0_controller` 与 `wrist_3_link` 同向的部署假设处理，
`--wrench-source-frame` 只提供空消息头的回退名称，不会自动转换外置传感器的坐标系或力矩原点。

### 5.5 `controller_manager` 不可达、找不到 ROS 包、一直等待端口

| 现象 | 处理 |
| --- | --- |
| `waiting dashboard...` / `waiting rtde...` | 核对脚本中的机器人 IP、网卡/路由、机器人是否开机；用第 3.1 节的端口命令检查 |
| `Could not contact service .../list_controllers` | 先看 driver 是否启动成功，再检查 ROS 环境及 controller manager 名称 |
| `Package 'my_robot' not found` | `my_robot` 来自外部 `~/ros2_ws`，确认已经构建并加载对应 `install/setup.bash` |
| `Package 'ur_robot_driver' not found` | 确认已加载 `/opt/ros/humble/setup.bash`；未安装时安装 `ros-humble-ur-robot-driver` |
| 不同终端看见的话题不同 | 核对 `ROS_DOMAIN_ID`、`RMW_IMPLEMENTATION`、`ROS_LOCALHOST_ONLY` 和所加载的工作区 |

只检查服务和环境的命令：

```bash
ros2 node list
ros2 service list
printenv ROS_DOMAIN_ID RMW_IMPLEMENTATION ROS_LOCALHOST_ONLY
ros2 pkg prefix ur_robot_driver
ros2 pkg prefix my_robot
```

如果 controller manager 使用其他命名空间，将命令中的 `-c /controller_manager`
改为实际名称，同时调整任务话题重映射。
网络连通但 External Control 无法连接时，检查 URCap 中的电脑 IP、双向路由和端口访问；
多网卡电脑还需确认 driver 选取的回连地址，可查看该版本启动参数中的 `reverse_ip`。

若只修改了外部 `my_robot` 源码而启动行为没变，重新构建并加载安装空间，然后重启对应 launch：

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot --symlink-install
source install/setup.bash
cd ~/isaaclab_ur_reach_sim2real
```

### 5.6 相机无法启动、没有图像或没有嘴部消息

| 日志/现象 | 可能原因与处理 |
| --- | --- |
| `No device connected` / 找不到指定设备 | 检查 USB 连接、供电和设备序列号；用 `rs-enumerate-devices` 核对两台设备 |
| `Device or resource busy` / pipeline 启动失败 | 关闭占用相机的 RealSense Viewer、其他识别脚本或相机 ROS 节点，只保留一套采集程序 |
| `Couldn't resolve requests` | 检查该设备是否支持脚本请求的深度和彩色 `640×480@30` 组合，以及 USB 带宽 |
| `Frame didn't arrive` / `等待帧时出错` | 检查 USB 线、端口和设备连接稳定性；识别脚本帧等待超时为 1 秒 |
| 有画面但无 `/wrist_face_pose` 或 `/fixed_face_pose` 数据 | 检查人脸是否可见、嘴部关键点/深度是否有效；脚本需要检测成功且嘴中心深度大于 0 才发布 |
| `module 'mediapipe' has no attribute 'solutions'` | 当前代码依赖旧式 FaceMesh API，使用提供 `mp.solutions.face_mesh` 的兼容环境；不要只升级包而不核对接口 |
| `qt.qpa.xcb` / `could not connect to display` | 相机脚本会创建 OpenCV 窗口，需要可用桌面会话；当前没有相机无窗口命令行开关 |

查询设备不需要再启动一份采集程序：

```bash
rs-enumerate-devices
```

当前识别启动脚本会创建两台相机；只插一台可能在另一台初始化时失败。
不应把同一序列号同时填给两个节点。

### 5.7 原始相机位姿有数据，`/mouth_pose` 没有数据或坐标不对

典型日志为 `等待从 ... 到 base_link 的变换` 或 `TF 变换失败`。
检查相机标定 TF 和机器人链路：

```bash
ros2 run tf2_ros tf2_echo base_link fixed_cam_optical_frame
ros2 run tf2_ros tf2_echo base_link wrist_cam_optical_frame
ros2 run tf2_ros tf2_echo base_link wrist_3_link
```

逐条执行并用 `Ctrl+C` 结束。腕部相机通过 `tool0` 连接机器人 TF，
机器人 TF 未就绪时也会影响腕部目标转换。
当前标定来自外部 `~/ros2_ws/src/my_robot/launch/dual_cam.launch.py`，不要用任意单位变换补齐链路。
相机移动、安装改变或标定 frame 写错时，应重新校准，再检查 `/mouth_pose` 的米制坐标是否符合实物位置。

若 `/mouth_pose` 持续有数据但已经看不到人脸，检查融合日志是否在保持历史目标；
当前保持逻辑会刷新消息时间戳，所以任务的目标超时检查无法识别这种历史目标。

### 5.8 已经启动识别，任务仍然使用旧嘴部位置

先看任务启动日志：

- `使用固定嘴巴目标`：没有传 `--vision-mouth`，重新按视觉模式启动。
- `已锁定第一帧视觉嘴巴目标`：本次目标已经锁定，移动人脸不会更新目标；重新运行选取首帧，
  或按第 3.6 节修改 `LATCH_FIRST_VISION_MOUTH_POSE`。
- `等待视觉嘴部目标`：检查 `/mouth_pose` 是否有消息、frame 是否能转换到 `base_link`。
- `视觉目标超时`：检查融合节点是否还在发布、ROS 环境和通信是否一致。

识别窗口显示嘴部开合距离，不表示当前任务直接使用这个话题判断张嘴；
当前 `run_task.py` 订阅的是嘴部 `PoseStamped`，嘴部几何特征由部署代码构造。

### 5.9 Python 依赖错误、模型无法加载

| 错误 | 处理 |
| --- | --- |
| `No module named rclpy` / `_rclpy_pybind11` 无法导入 | 确认已加载 ROS 环境，且使用 `~/ros_env/bin/python3` 的 Python 3.10，而非其他版本的 Conda Python |
| `No module named torch/cv2/mediapipe/pyrealsense2/matplotlib` | 在实际运行脚本的 Python 环境中补齐对应依赖；用 `python3 -m pip` 核对安装目标 |
| `无法加载 arm policy` / `TorchScript policy 不存在` | 检查模型文件路径和 `policy request` / `policy loaded` 日志；相对路径以当前工作目录解析 |
| 报 HARL 导入失败，但本来想用 TorchScript | 检查 TorchScript 文件是否缺失或先加载失败；`auto` 会尝试回退，查看 traceback 中最早的原因 |
| 输入维度/矩阵尺寸不匹配 | arm 模型需要 `58` 维观测、`6` 维动作；safety 模型需要 `27` 维观测、`3` 维动作，使用匹配训练版本的导出模型 |

默认模型应位于：

```text
sample/actor_agent_arm_torchscript.pt
sample/actor_agent_safety_torchscript.pt
```

arm 模型可通过 `--model-path` 指定；safety 模型路径由 `SAFETY_POLICY_MODEL_PATH` 配置。
例如在项目根目录指定 arm TorchScript 文件进行干运行：

```bash
python3 deploy/run_task.py --vision-mouth --dry-run \
  --model-path sample/actor_agent_arm_torchscript.pt --policy-backend torchscript
```

### 5.10 为什么触发安全撤回，如何定位？

当前学习 safety 的撤回危险来源包括：脸部近似区域侵入、入嘴前勺心过近嘴部特征点、
力/力矩风险和沿勺方向的异常轴向力。`lip_risk` 单独出现主要触发减速/居中，
不会单独把学习安全层切入撤回。actor 原始撤回通道为正也不代表必然触发，仍有危险门控。

使用 `--debug-obs --debug-wrench` 复查触发前的几何/力觉数据，以及 safety CSV 中的
`danger_score`、`lip_risk`、`link_risk`、`force_risk`、`poke_risk`、`grec` 等字段。
先确认嘴目标、相机标定、嘴轴方向和 `SPOON_OFFSET` 符合当前安装；再检查启动时是否静止 tare、
工具是否真的接触、payload/力方向是否匹配。脸部风险目前由嘴部特征点近似，并非完整脸部点云碰撞检测。

`safety_strenthen=True` 时，一旦有效 `grec > 0`，就锁定撤回 10 cm 再归位，风险减小也不会恢复接近。
因此 `grec` 在撤回过程中降低，但机器人继续撤回，是当前预期行为。
需要恢复原逐步干预方式时设置 `safety_strenthen=False`；
修改 `safety_grec_force_one` 只改变加强接管期间的增益显示/记录方式，不改变撤回路径。

### 5.11 撤回太慢、分段，或撤回后归位有等待

当前实现一次发布完整撤回轨迹，内部采样点不是逐段等待执行。
检查是否出现 `已一次下发完整撤回轨迹`，以及其中的 `duration`：

```text
标称平均时间 = 撤回距离 / SUCCESS_WITHDRAW_SPEED_M_S
当前 0.10 m / 0.05 m/s = 2 s
实际时间约为 轨迹 duration / UR 速度缩放 + 到位稳定确认
```

轨迹 `duration` 还会根据整条路径的关节限速增加，并非强制 2 秒完成。
例如规划时长为 2 秒、速度缩放为 55%，实际约为 3.64 秒，再加到位确认。
先检查速度缩放、日志里的轨迹时长，以及是否有另一个任务节点反复覆盖轨迹。
`--min-trajectory-duration` 不是设置完整撤回总时长的参数。

撤回末尾需要实际位置误差不超过 2 mm、姿态误差不超过 2°、勺心退出嘴平面、关节速度稳定，
连续满足 5 个控制步后才开始归位，30 Hz 下约 0.17 秒。
归位完成另有 0.5 秒稳定确认；成功后撤回前还有 30 步保持。
当前已将绘图放到归位结束后，不会在撤回与归位之间先绘图或启动新的 Python 进程。

### 5.12 撤回失败、超时，或任务停止后没有自动归位

| 日志/现象 | 含义与处理 |
| --- | --- |
| `撤回轨迹规划失败` / `直线撤回 IK 未收敛` | 检查当前姿态、撤回方向、可达性和机器人/工具标定；当前代码取消自动归位 |
| `撤回偏离直线路径或姿态` | 实测侧向误差超过 1 cm 或姿态偏差超过 5°；检查 TF、安装与反馈，不要直接扩大阈值掩盖偏差 |
| `撤回超时，取消自动归位` | 默认撤回超时 30 秒，检查实际运动、速度缩放、轨迹订阅者和反馈新鲜度 |
| `策略目标越界` / `勺子中心过低` / `嘴和勺距离过大` | 当前目标或动作触发工作空间/episode 保护，核对目标坐标、工具偏移和当前位置 |
| `自动归位超时` | 查看最终关节误差、速度缩放、是否失去反馈或被停止；曲线仍会尝试保存 |
| `[episode FAILURE]` / `[episode TIMEOUT]` 后不继续 | 当前 episode 已停止，需要处理原因、退出并重新启动；失败不会自动执行成功归位流程 |

撤回中反馈超时会阻止继续确认到位或启动归位，已发送的完整轨迹仍可能由控制器执行。
异常后先确认机械臂的实际位置，不能用独立 `reset_arm.py --execute` 跳过嘴部撤离。

### 5.13 没有生成 CSV/曲线图，或窗口无法显示

- 关节 CSV 只有表头：尚未收到完整、有效的六关节反馈，先按第 5.1 节检查。
- 没有完整 safety 曲线：检查 `SAVE_FULL_SAFETY_GAIN_CURVE=True`，是否已经有推理样本，
  是否已到归位结束、episode 停止或正常退出的保存时机。
- 只有 `latest_safety_gains.png`：这是实时截图，完整记录需等上述导出时机。
- CSV 有数据但 PNG 失败：检查日志中的 matplotlib/OpenCV 错误、目录权限和磁盘空间。
- 没有图形桌面：可设置 `SAFETY_GAIN_VISUALIZATION=False` 关闭任务的实时窗口，
  保留两个保存开关，完整 CSV/离线绘图不要求开启窗口。相机识别窗口仍需单独处理。
- 没有 wrench CSV：需要启动参数 `--debug-wrench`。

## 6. 离线回归检查

需要修改撤回/安全加强代码时，可在项目根目录运行：

```bash
python3 -m unittest deploy.tests.test_success_withdraw deploy.tests.test_safety_strengthen -v
```

这些测试使用本地模拟反馈和发布器，不创建 ROS 节点或连接机器人；
仍需可导入 ROS 消息和部署代码的 Python 环境。正常执行任务无需运行这些测试。
