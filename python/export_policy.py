import torch
from rsl_rl.modules import ActorCritic
import os

# ==============================================================================
# ======================== 用户需要配置的参数 =================================
# ==============================================================================

# 1. 输入: 您训练好的检查点文件的完整路径
#    (请将其替换为您自己的文件路径)
CHECKPOINT_PATH = "/home/xry/isaaclab_ur_reach_sim2real/sample/ur_reach/model_random_1499.pt"

# 2. 输出: 您想要保存的 TorchScript 模型的路径和文件名
EXPORT_PATH = "ur5_random_1499.pt"

# 3. 环境参数 (*** 您必须根据您的环境填写这些值 ***)
#    - 您可以在您的任务配置文件中找到它们 (例如 ur5_reach_cfg.py)
NUM_OBS = 28  # <--- !! 请在此处填写正确的观察空间维度 !!
NUM_ACTIONS = 6  # <--- !! 请在此处填写正确的动作空间维度 (UR5通常是6) !!

# 4. 模型架构参数 (从您的 UR5ReachPPORunnerCfg 文件中提取)
ACTOR_HIDDEN_DIMS = [128, 128, 64]
CRITIC_HIDDEN_DIMS = [128, 128, 64]
ACTIVATION = "elu"
INIT_NOISE_STD = 1.0

# 5. 设备 (在CPU上进行转换更安全、更通用)
DEVICE = "cpu"

# ==============================================================================
# ============================ 脚本正文 (通常无需修改) ============================
# ==============================================================================

def main():
    """
    该脚本将 rsl-rl 训练的检查点文件 (.pt) 转换为
    一个可用于推理的 TorchScript 模型 (.pt)。
    """
    print("--- 开始将 RSL-RL 检查点转换为 TorchScript ---")

    # 检查输入文件是否存在
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n[错误] 检查点文件未找到: {CHECKPOINT_PATH}")
        print("请确保 CHECKPOINT_PATH 变量指向了正确的文件。")
        return

    # 1. 实例化一个与训练时结构完全相同的 ActorCritic 模型
    print("\n[步骤 1/4] 正在根据配置创建模型架构...")
    policy = ActorCritic(
        num_actor_obs=NUM_OBS,
        num_critic_obs=NUM_OBS,  # 在PPO中, critic的观察维度通常与actor相同
        num_actions=NUM_ACTIONS,
        actor_hidden_dims=ACTOR_HIDDEN_DIMS,
        critic_hidden_dims=CRITIC_HIDDEN_DIMS,
        activation=ACTIVATION,
        init_noise_std=INIT_NOISE_STD,
    ).to(DEVICE)
    print("模型架构创建成功。")

    # 2. 加载检查点文件并提取模型权重
    print(f"\n[步骤 2/4] 正在从 '{os.path.basename(CHECKPOINT_PATH)}' 加载权重...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model_state_dict = checkpoint['model_state_dict']

    # 将权重加载到我们刚刚创建的模型实例中
    policy.load_state_dict(model_state_dict)
    
    # 切换到评估模式 (这会禁用dropout等训练特有的层)
    policy.eval()
    print("模型权重加载成功。")

    # 3. 提取 actor 网络并将其转换为 TorchScript
    #    在部署时, 我们只需要 actor (策略本身), 不需要 critic (价值函数)
    print("\n[步骤 3/4] 正在将 Actor 网络转换为 TorchScript...")
    actor_model = policy.actor
    scripted_actor = torch.jit.script(actor_model)
    print("转换成功。")

    # 4. 保存转换后的 TorchScript 模型
    print(f"\n[步骤 4/4] 正在将 TorchScript 模型保存到 '{EXPORT_PATH}'...")
    scripted_actor.save(EXPORT_PATH)
    
    print("\n-------------------------------------------------")
    print("✅ 转换完成！")
    print(f"您的策略模型已成功导出到: {os.path.abspath(EXPORT_PATH)}")
    print("您现在可以拷贝此文件并在您的 `run_task.py` 中使用它。")
    print("-------------------------------------------------")


if __name__ == "__main__":
    main()