import sys
from pathlib import Path

import numpy as np
import yaml


ISAACLAB_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ALGO_CFG = ISAACLAB_ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/ur5_mouth_marl_env/agents/harl_happo_cfg.yaml"
if str(ISAACLAB_ROOT) not in sys.path:
    # 让实机部署脚本可以直接复用 IsaacLab 仓库里的 HARL 网络定义。
    sys.path.insert(0, str(ISAACLAB_ROOT))


class Box:
    """Minimal Box-like space used by HARL when reconstructing the actor."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


class HarlArmPolicy:
    """Load a HARL arm actor checkpoint and return deterministic arm actions."""

    def __init__(
        self,
        model_path: str,
        algo_cfg_path: str | None = None,
        obs_dim: int = 58,
        action_dim: int = 6,
        device: str = "cpu",
        backend: str = "auto",
    ) -> None:
        if backend not in {"auto", "torchscript", "torch", "numpy"}:
            raise ValueError(f"policy backend 只能是 auto/torchscript/torch/numpy，收到: {backend}")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.model_path = Path(model_path).expanduser().resolve()
        # Torch后端需要HARL配置重建网络；TorchScript后端也保留同一默认路径，统一部署接口。
        self.algo_cfg_path = Path(algo_cfg_path or DEFAULT_ALGO_CFG).expanduser().resolve()
        self.npz_path = self.model_path.with_suffix(".npz")
        self.scripted_path = self._resolve_scripted_path()
        self.backend: str | None = None
        self.loaded_path: Path | None = None
        script_error: Exception | None = None
        torch_error: Exception | None = None

        if backend in {"auto", "torchscript"}:
            try:
                self._load_torchscript_policy(device)
                self.backend = "torchscript"
                return
            except Exception as exc:
                script_error = exc
                if backend == "torchscript":
                    raise RuntimeError("无法使用 TorchScript 后端加载 arm policy。") from exc

        if backend in {"auto", "torch"}:
            try:
                self._load_torch_policy(device)
                self.backend = "torch"
                return
            except Exception as exc:
                torch_error = exc
                if backend == "torch":
                    raise RuntimeError("无法使用 torch/HARL 后端加载 arm policy。") from exc

        if self._load_numpy_policy():
            self.backend = "numpy"
            return

        if backend == "numpy":
            raise FileNotFoundError(f"NumPy policy 权重不存在: {self.npz_path}")
        raise RuntimeError(
            "无法加载 arm policy。请确认至少满足一种部署方式："
            f" TorchScript={self.scripted_path}，HARL checkpoint={self.model_path}，NumPy={self.npz_path}"
        ) from (script_error or torch_error)

    def _resolve_scripted_path(self) -> Path:
        if self.model_path.name.endswith("_torchscript.pt"):
            return self.model_path
        return self.model_path.with_name(f"{self.model_path.stem}_torchscript.pt")

    def _load_torchscript_policy(self, device: str) -> None:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("无法导入 torch，TorchScript 部署至少需要 PyTorch。") from exc

        if not self.scripted_path.exists():
            raise FileNotFoundError(f"TorchScript policy 不存在: {self.scripted_path}")

        self.torch = torch
        self.device = torch.device(device)
        # TorchScript 文件已经固化网络结构和权重，实机端不再需要 HARL 包。
        self.scripted_actor = torch.jit.load(str(self.scripted_path), map_location=self.device)
        self.scripted_actor.eval()
        self._validate_torchscript_dimensions()
        self.loaded_path = self.scripted_path

    def _validate_torchscript_dimensions(self) -> None:
        """在机械臂开始运动前校验导出模型的输入和动作宽度。"""
        parameter_shapes = {name: tuple(parameter.shape) for name, parameter in self.scripted_actor.named_parameters()}
        input_weight = parameter_shapes.get("actor.base.mlp.fc.0.weight")
        action_bias = parameter_shapes.get("actor.act.action_out.fc_mean.bias")
        if input_weight is not None and len(input_weight) == 2 and input_weight[1] != self.obs_dim:
            raise ValueError(
                f"TorchScript arm policy输入维度为{input_weight[1]}，部署观测维度为{self.obs_dim}；"
                "请使用与当前58维wrench观测匹配的模型。"
            )
        if action_bias is not None and len(action_bias) == 1 and action_bias[0] != self.action_dim:
            raise ValueError(
                f"TorchScript arm policy动作维度为{action_bias[0]}，部署期望{self.action_dim}。"
            )

    def _load_torch_policy(self, device: str) -> None:
        try:
            import torch
            from harl.models.policy_models.stochastic_policy import StochasticPolicy
        except Exception as exc:
            raise RuntimeError(
                "无法导入 torch/HARL。请在 IsaacLab 或安装了 torch 的 ROS Python 环境中运行 run_task.py。"
            ) from exc

        if not self.model_path.exists():
            raise FileNotFoundError(f"arm policy checkpoint 不存在: {self.model_path}")
        if not self.algo_cfg_path.exists():
            raise FileNotFoundError(f"HARL 配置文件不存在: {self.algo_cfg_path}")

        self.torch = torch
        self.device = torch.device(device)
        args = self._load_actor_args()
        state_dict = torch.load(self.model_path, map_location=self.device)
        # 当前部署没有训练期辅助标签，actor结构直接使用58维运动学+wrench输入。
        args["use_auxiliary_head"] = False
        args["policy_obs_dim"] = self.obs_dim
        args["aux_target_dim"] = 0

        obs_space = Box((self.obs_dim,))
        act_space = Box((self.action_dim,))
        self.actor = StochasticPolicy(args, obs_space, act_space, self.device)
        try:
            self.actor.load_state_dict(state_dict)
        except RuntimeError as exc:
            # 兼容性保护: 旧52/59维或带辅助头的checkpoint不能与当前58维策略混用。
            raise RuntimeError(
                f"arm checkpoint结构与当前{self.obs_dim}维运动学+wrench策略不一致；请使用匹配版本导出的模型。"
            ) from exc
        self.actor.eval()
        self.loaded_path = self.model_path

        self.rnn_state = torch.zeros((1, args["recurrent_n"], args["hidden_sizes"][-1]), dtype=torch.float32, device=self.device)
        self.mask = torch.ones((1, 1), dtype=torch.float32, device=self.device)

    def _load_numpy_policy(self) -> bool:
        if not self.npz_path.exists():
            return False

        with np.load(self.npz_path) as data:
            self.numpy_weights = {key: data[key].astype(np.float32) for key in data.files if not key.startswith("metadata_")}
            metadata_obs_dim = int(data["metadata_obs_dim"][0]) if "metadata_obs_dim" in data.files else self.obs_dim
            metadata_action_dim = int(data["metadata_action_dim"][0]) if "metadata_action_dim" in data.files else self.action_dim

        if metadata_obs_dim != self.obs_dim or metadata_action_dim != self.action_dim:
            raise ValueError(
                f"NumPy policy 维度不匹配: obs={metadata_obs_dim}, action={metadata_action_dim}, "
                f"期望 obs={self.obs_dim}, action={self.action_dim}"
            )

        required_keys = [
            "base.feature_norm.weight",
            "base.feature_norm.bias",
            "base.mlp.fc.0.weight",
            "base.mlp.fc.0.bias",
            "base.mlp.fc.2.weight",
            "base.mlp.fc.2.bias",
            "base.mlp.fc.3.weight",
            "base.mlp.fc.3.bias",
            "base.mlp.fc.5.weight",
            "base.mlp.fc.5.bias",
            "act.action_out.fc_mean.weight",
            "act.action_out.fc_mean.bias",
        ]
        missing_keys = [key for key in required_keys if key not in self.numpy_weights]
        if missing_keys:
            raise KeyError(f"NumPy policy 权重缺少必要字段: {missing_keys}")
        self.loaded_path = self.npz_path
        return True

    def _load_actor_args(self) -> dict:
        with open(self.algo_cfg_path, "r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)

        # HARL actor 构造时使用 model 与 algo 两段配置；这里与训练 runner 保持一致。
        args = {}
        args.update(cfg.get("model", {}))
        args.update(cfg.get("algo", {}))
        return args

    def act(self, obs: np.ndarray) -> np.ndarray:
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"arm policy 期望 {self.obs_dim} 维观测，但收到 {obs.shape[-1]} 维")

        obs = np.nan_to_num(obs.astype(np.float32), nan=0.0, posinf=5.0, neginf=-5.0)
        obs = np.clip(obs, -5.0, 5.0)

        if self.backend == "numpy":
            return self._act_numpy(obs)
        if self.backend == "torchscript":
            with self.torch.inference_mode():
                obs_t = self.torch.from_numpy(obs).view(1, -1).to(self.device)
                action = self.scripted_actor(obs_t)
            return action.detach().cpu().numpy().reshape(-1)

        with self.torch.inference_mode():
            obs_t = self.torch.from_numpy(obs).view(1, -1).to(self.device)
            action, _, self.rnn_state = self.actor(obs_t, self.rnn_state, self.mask, None, True)
        return action.detach().cpu().numpy().reshape(-1)

    def _act_numpy(self, obs: np.ndarray) -> np.ndarray:
        weights = self.numpy_weights
        # NumPy 后端复现 HARL MLP actor 的确定性 mode，用于 ROS Python 没有 torch 的实机部署。
        x = self._layer_norm(obs, weights["base.feature_norm.weight"], weights["base.feature_norm.bias"])
        x = self._relu(self._linear(x, weights["base.mlp.fc.0.weight"], weights["base.mlp.fc.0.bias"]))
        x = self._layer_norm(x, weights["base.mlp.fc.2.weight"], weights["base.mlp.fc.2.bias"])
        x = self._relu(self._linear(x, weights["base.mlp.fc.3.weight"], weights["base.mlp.fc.3.bias"]))
        x = self._layer_norm(x, weights["base.mlp.fc.5.weight"], weights["base.mlp.fc.5.bias"])

        if "auxiliary_head.0.weight" in weights:
            # 带 auxiliary head 的 checkpoint 会先预测意图概率，再把它融合回策略特征。
            aux = self._relu(self._linear(x, weights["auxiliary_head.0.weight"], weights["auxiliary_head.0.bias"]))
            aux = self._linear(aux, weights["auxiliary_head.2.weight"], weights["auxiliary_head.2.bias"])
            aux_context = 1.0 / (1.0 + np.exp(-aux))
            x = np.concatenate([x, aux_context.astype(np.float32)], axis=-1)
            x = self._relu(self._linear(x, weights["auxiliary_fusion.0.weight"], weights["auxiliary_fusion.0.bias"]))

        action = self._linear(x, weights["act.action_out.fc_mean.weight"], weights["act.action_out.fc_mean.bias"])
        return action.astype(np.float32, copy=False).reshape(-1)

    @staticmethod
    def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        return x @ weight.T + bias

    @staticmethod
    def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1.0e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return ((x - mean) / np.sqrt(variance + eps)) * weight + bias

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)