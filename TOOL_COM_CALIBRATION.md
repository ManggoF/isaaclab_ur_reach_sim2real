# 夹爪和勺子质量/重心标定

`deploy/measure_tool_com.py` 使用多个静止姿态下的六维力数据，估计力传感器之后整套夹爪和勺子的：

- 总质量，单位 kg；
- 相对力数据原点的三维重心，单位 m；
- 力和力矩的固定零偏；
- 转换到仿真 `wrist_3_link` 坐标系后的等效重心；
- 拟合误差和姿态覆盖质量。

程序只订阅力数据和 TF，不会向机械臂发送任何运动命令。静态重力实验无法辨识转动惯量，惯量张量仍需使用 CAD、摆动实验或其他动态辨识方法获得。

## 测量前准备

1. 完整安装力传感器、夹爪和空勺，测量期间不要改变安装状态。
2. 停止 `deploy/run_task.py` 等自动控制程序，只通过示教器或安全的 freedrive 移动机械臂。
3. 确保勺子不接触桌面、人体或线缆，传感器预热后再测量。
4. 测量期间保持 PolyScope 的 payload/TCP 配置不变，不要在每个姿态重新清零传感器。
5. `base_link` 必须是 Z 轴向上的重力参考坐标系；力和力矩必须确实表达在 `WrenchStamped.header.frame_id` 中，且力矩原点也是该 frame 原点。

如果力话题已经减去了 PolyScope 中配置的 payload，程序测到的是“未被补偿的剩余载荷”，不是完整工具载荷。优先使用未做载荷补偿的原始力传感器话题；只有遵守机器人厂商安全流程时，才考虑临时改变 payload 配置。

先确认话题和坐标系：

```bash
ros2 topic echo --once /force_torque_sensor_broadcaster/wrench
ros2 run tf2_ros tf2_echo base_link tool0_controller
```

当前仓库部署代码使用 `/force_torque_sensor_broadcaster/wrench`。有些 UR driver 配置发布的是 `/force_torque_sensor_broadcaster/ft_data`，Robotiq 外置传感器通常使用 `/robotiq_force_torque_sensor_broadcaster/wrench`，运行时可通过 `--wrench-topic` 指定。

## 运行

启动 UR driver 后，在新终端执行：

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
cd /home/gxr/isaaclab_ur_reach_sim2real
python3 deploy/measure_tool_com.py
```

每次按 Enter 后，程序默认等待 1.5 秒，再连续采样 2 秒。至少采集 8 个明显不同的倾斜姿态，让重力分别在传感器的 `+X/-X/+Y/-Y/+Z/-Z` 附近取向。具体姿态必须服从机械臂关节、碰撞和线缆限制，不要求精确对齐坐标轴。

只绕世界竖直轴旋转末端不会改变传感器坐标系中的重力方向，对重心辨识没有帮助。程序会拒绝过于接近、仍在运动或力信号波动过大的姿态。

常用参数示例：

```bash
python3 deploy/measure_tool_com.py \
  --wrench-topic /force_torque_sensor_broadcaster/wrench \
  --base-frame base_link \
  --sim-frame wrist_3_link \
  --pose-count 8
```

如果消息的 `header.frame_id` 为空，需要显式指定力数据实际所在的 frame：

```bash
python3 deploy/measure_tool_com.py --wrench-frame tool0_controller
```

`--wrench-frame` 只声明力数据实际采用的坐标系，不会对数值进行坐标变换。错误地指定 frame 会直接导致错误的重心。

## 输出和仿真同步

默认输出：

- `calibration/tool_com_result.json`：最终质量、重心、坐标变换和误差；
- `calibration/tool_com_result_samples.csv`：每个姿态的重力、平均力/力矩和标准差。

仿真中优先读取以下字段：

```text
simulation_equivalent_payload.mass_kg
simulation_equivalent_payload.center_of_mass_xyz_m
simulation_equivalent_payload.frame
```

当夹爪和勺子在仿真中是一个独立的固定刚体时，可将这组质量和重心直接赋给该刚体。用 USD Python 修改对应 prim 的示例：

```python
from pxr import Gf, UsdPhysics

mass_api = UsdPhysics.MassAPI.Apply(tool_prim)
mass_api.CreateMassAttr(measured_mass_kg)
mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*measured_com_xyz_m))
```

如果仿真没有独立工具刚体，而是把工具合并进已有 `wrist_3_link`，不要用工具质量覆盖腕部原质量。设原 link 的质量/重心为 `m0, c0`，测得工具载荷为 `mt, ct`，合并后：

```text
m_new = m0 + mt
c_new = (m0 * c0 + mt * ct) / m_new
```

此时还必须用平行轴定理正确合并惯量张量，否则快速运动时的动力学仍不会同步。JSON 中默认给出的 `ct` 已转换到 `wrist_3_link` 坐标系。

## 结果判定

建议重测以下情况：

- `fit_quality.status` 为 `warning`；
- 力拟合 RMSE 明显大于 `0.20 N`；
- 力矩拟合 RMSE 明显大于 `0.03 Nm`；
- JSON 中没有生成 `simulation_equivalent_payload`，说明 `wrist_3_link <- wrench_frame` 的 TF 不可用；
- 测得质量与电子秤结果差异很大，通常表示力话题做过 payload 补偿、frame 声明错误或采样时存在外力。

