import ctypes
import copy
import os
import random
import sys
import time
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
import datetime

import gym


def _prepend_env_path(var_name, path):
    if not path:
        return
    current = os.environ.get(var_name, "")
    paths = [item for item in current.split(os.pathsep) if item] if current else []
    if path not in paths:
        os.environ[var_name] = path if not current else path + os.pathsep + current


def _preload_conda_libpython():
    """Help Isaac Gym find libpython when Python is started without conda activate."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    lib_names = [
        f"libpython{version}.so.1.0",
        f"libpython{version}m.so.1.0",
    ]

    prefixes = []
    for prefix in (
        os.environ.get("CONDA_PREFIX"),
        os.path.dirname(os.path.dirname(sys.executable)),
    ):
        if prefix and prefix not in prefixes:
            prefixes.append(prefix)

    for prefix in prefixes:
        lib_dir = os.path.join(prefix, "lib")
        if not os.path.isdir(lib_dir):
            continue
        _prepend_env_path("LD_LIBRARY_PATH", lib_dir)
        for lib_name in lib_names:
            lib_path = os.path.join(lib_dir, lib_name)
            if not os.path.exists(lib_path):
                continue
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                continue


_preload_conda_libpython()
import isaacgym  # noqa
import isaacgymenvs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from flow_matching import FlowMatching, Model, Model_L


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    config_path: str = str(PROJECT_DIR / "config" / "config_Ant.json")
    """配置文件路径，如果提供则从JSON文件加载参数"""
    run_name: Optional[str] = None
    gpu_id: int = 0
    """CUDA device id to use when cuda is enabled."""
    research_tag: Optional[str] = None
    metrics_jsonl: Optional[str] = None
    status_json: Optional[str] = None
    summary_json: Optional[str] = None
    reward_log_path: Optional[str] = None
    early_stop_file: Optional[str] = None
    save_minimal_logs: bool = False


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    if not path.is_absolute():
        for root in (PROJECT_DIR, SCRIPT_DIR):
            candidate = root / path
            if candidate.exists():
                return candidate
    if path.is_absolute() and str(path).startswith("/config/"):
        candidate = PROJECT_DIR / str(path).lstrip("/")
        if candidate.exists():
            return candidate
    return path


def load_config(config_path):
    """加载JSON配置文件"""
    resolved = resolve_path(config_path)
    with open(resolved, 'r') as f:
        config = json.load(f)
    return config


def to_python_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        if value.numel() == 1:
            return value.detach().item()
        return value.detach().float().mean().item()
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def tensor_stats(tensor: torch.Tensor, prefix: str) -> Dict[str, float]:
    flat = tensor.detach().reshape(-1).float()
    if flat.numel() == 0:
        return {}
    quantiles = torch.tensor([0.1, 0.5, 0.9], device=flat.device, dtype=flat.dtype)
    q_values = torch.quantile(flat, quantiles)
    return {
        f"{prefix}_mean": flat.mean().item(),
        f"{prefix}_std": flat.std(unbiased=False).item(),
        f"{prefix}_min": flat.min().item(),
        f"{prefix}_max": flat.max().item(),
        f"{prefix}_q10": q_values[0].item(),
        f"{prefix}_q50": q_values[1].item(),
        f"{prefix}_q90": q_values[2].item(),
    }


def safe_json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=to_python_scalar)
        f.write("\n")
    tmp_path.replace(path)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", buffering=1) as f:
        f.write(json.dumps(payload, sort_keys=True, default=to_python_scalar) + "\n")


def load_stop_request(path_value: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    try:
        with path.open() as f:
            payload = json.load(f)
            if isinstance(payload, dict):
                return payload
    except Exception:
        pass
    try:
        text = path.read_text().strip()
    except Exception:
        return None
    if not text:
        return None
    if text.lower() in {"1", "true", "stop", "yes"}:
        return {"stop": True, "reason": text}
    return {"stop": False, "reason": text}


def accumulate_metrics(
    sums: Dict[str, float],
    counts: Dict[str, int],
    prefix: str,
    stats: Optional[Dict[str, Any]],
) -> None:
    if not stats:
        return
    for key, value in stats.items():
        full_key = f"{prefix}{key}" if prefix else key
        scalar = to_python_scalar(value)
        if isinstance(scalar, bool) or scalar is None:
            continue
        if isinstance(scalar, (int, float)):
            sums[full_key] = sums.get(full_key, 0.0) + float(scalar)
            counts[full_key] = counts.get(full_key, 0) + 1


class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.returned_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.returned_episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones
        self.episode_lengths *= 1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        cond_dim = np.array(envs.single_observation_space.shape).prod()
        state_dim = 1
        self.rollout_mean_mode = getattr(args, "rollout_mean_mode", "remove_max")
        self.bootstrap_mean_mode = getattr(args, "bootstrap_mean_mode", "normal")
        prior_std = float(getattr(args, "prior_std", 1.0))
        target_norm = getattr(args, "target_norm", "none")
        target_norm_update = getattr(args, "target_norm_update", "cumulative")
        target_norm_ema_tau = float(getattr(args, "target_norm_ema_tau", 0.99))
        if args.model_L == True:
            model = Model_L(condition_dim=cond_dim, out_dim=state_dim, hidden_size=args.hidden_size, time_dim=32).to(
                device)
        else:
            model = Model(condition_dim=cond_dim, out_dim=state_dim, hidden_size=args.hidden_size, time_dim=32).to(device)
        self.critic = FlowMatching(
            state_dim=state_dim,
            model=model,
            device=device,
            sigma=args.flow_sigma,
            prior_std=prior_std,
            target_norm=target_norm,
            target_norm_update=target_norm_update,
            target_norm_ema_tau=target_norm_ema_tau,
        )
        self.critic_old = copy.deepcopy(self.critic)
        self.critic_old.eval()
        for param in self.critic_old.parameters():
            param.requires_grad_(False)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def _mean_from_samples(self, v_value, mean_mode):
        k_repeat = args.k_repeat
        n_remove = args.n_remove

        if mean_mode == "normal":
            mean = torch.mean(v_value, dim=1)
        elif mean_mode == "remove_max":
            if k_repeat <= n_remove or n_remove == 0:
                mean = torch.mean(v_value, dim=1)
            else:
                sorted_v, _ = torch.sort(v_value, dim=1)
                mean = torch.mean(sorted_v[:, :-n_remove], dim=1)
        elif mean_mode == "remove_max_min":
            if k_repeat <= n_remove * 2:
                mean = torch.mean(v_value, dim=1)
            else:
                sorted_v, _ = torch.sort(v_value, dim=1)
                mean = torch.mean(sorted_v[:, n_remove:-n_remove], dim=1)
        else:
            raise ValueError(f"Unknown mean_mode: {mean_mode}")
        return mean

    def _cov_temperature(self, update_step: Optional[int]) -> float:
        alpha = float(getattr(args, "exp_alpha", 0.1))
        warmup = int(getattr(args, "alpha_warmup_updates", 0) or 0)
        if warmup > 0 and update_step is not None:
            frac = min(1.0, max(0.0, float(update_step + 1) / float(warmup)))
            alpha *= frac
        return alpha

    def _apply_weight_stabilization(self, cv, update_step: Optional[int], collect_stats: bool = False):
        alpha = self._cov_temperature(update_step)
        weights = torch.exp(-alpha * cv)

        cov_weight_clip = getattr(args, "cov_weight_clip", None)
        if cov_weight_clip is not None:
            if isinstance(cov_weight_clip, dict):
                weight_min = cov_weight_clip.get("min", None)
                weight_max = cov_weight_clip.get("max", None)
            elif isinstance(cov_weight_clip, (list, tuple)) and len(cov_weight_clip) == 2:
                weight_min, weight_max = cov_weight_clip
            else:
                raise ValueError("cov_weight_clip must be a 2-item list/tuple or dict")
            if weight_min is not None:
                weights = torch.clamp(weights, min=float(weight_min))
            if weight_max is not None:
                weights = torch.clamp(weights, max=float(weight_max))

        ess_floor = float(getattr(args, "ess_floor", 0.0) or 0.0)
        stats = {"cov_alpha": alpha}
        if ess_floor > 0.0 or collect_stats:
            ess = (weights.sum() ** 2) / (weights.square().sum() + 1e-8)
            ess_ratio = ess / max(float(weights.numel()), 1.0)
            if ess_floor > 0.0 and ess_ratio.item() < ess_floor:
                blend = min(1.0, (ess_floor - ess_ratio.item()) / max(ess_floor, 1e-8))
                weights = weights * (1.0 - blend) + blend
                if collect_stats:
                    ess = (weights.sum() ** 2) / (weights.square().sum() + 1e-8)
                    ess_ratio = ess / max(float(weights.numel()), 1.0)
            stats.update(
                {
                    "weight_ess": ess.detach(),
                    "weight_ess_ratio": ess_ratio.detach(),
                }
            )
        return weights, stats

    def get_value(self, x, mean_mode=None, update_step=None, collect_stats=False):
        if mean_mode is None:
            mean_mode = self.bootstrap_mean_mode
        v_value = self.critic.generate_multiple(x, n_samples=args.k_repeat, num_steps=args.diff_step)
        mean = self._mean_from_samples(v_value, mean_mode)
        sampled = v_value[:, 0]
        stats = {}
        if collect_stats:
            stats.update(tensor_stats(v_value, "value_samples"))
            stats["value_mean_mean"] = mean.detach().mean().item()
            stats["value_mean_std"] = mean.detach().std(unbiased=False).item()
            stats["value_sample_mean"] = sampled.detach().mean().item()
            stats["value_sample_std"] = sampled.detach().std(unbiased=False).item()
        return mean, sampled, stats

    def get_action_and_value(self, x, action=None, mean_mode=None, update_step=None, collect_stats=False):
        if mean_mode is None:
            mean_mode = self.rollout_mean_mode
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        v_value = self.critic.generate_multiple(x, n_samples=args.k_repeat, num_steps=args.diff_step)
        mean = self._mean_from_samples(v_value, mean_mode)
        with torch.no_grad():
            std = torch.std(v_value, dim=1)
            cov_floor = float(getattr(args, "cov_floor", 0.0) or 0.0)
            cv = std / (torch.abs(mean) + cov_floor + 1e-8)
            weights, weight_stats = self._apply_weight_stabilization(cv, update_step, collect_stats)
        sampled = v_value[:, 0]
        std_sampled = std[0]  # 只取0号样本的std
        cv_sampled = cv[0]    # 只取0号样本的cv
        stats = {}
        if collect_stats:
            stats.update(
                {
                    "value_mean_mean": mean.detach().mean().item(),
                    "value_mean_std": mean.detach().std(unbiased=False).item(),
                    "value_sample_mean": sampled.detach().mean().item(),
                    "value_sample_std": sampled.detach().std(unbiased=False).item(),
                    "value_std_mean": std.detach().mean().item(),
                    "value_std_std": std.detach().std(unbiased=False).item(),
                    "cv_mean": cv.detach().mean().item(),
                    "cv_std": cv.detach().std(unbiased=False).item(),
                    "std_sampled": std_sampled.detach().item(),
                    "cv_sampled": cv_sampled.detach().item(),
                }
            )
            stats.update({key: to_python_scalar(value) for key, value in weight_stats.items()})
            stats.update(tensor_stats(v_value, "value_samples"))
            stats.update(tensor_stats(weights, "weights"))
            stats.update(tensor_stats(cv, "cv"))
            stats.update(tensor_stats(std, "value_std"))
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), mean, sampled, weights, std_sampled, cv_sampled, stats


    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)


class ExtractObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs["obs"]


if __name__ == "__main__":
    # 先解析命令行参数
    t_args = tyro.cli(Args)

    # 从JSON文件加载完整配置
    args = SimpleNamespace(**load_config(t_args.config_path))
    args.research_tag = t_args.research_tag if t_args.research_tag is not None else getattr(args, "research_tag", None)
    args.metrics_jsonl = t_args.metrics_jsonl if t_args.metrics_jsonl is not None else getattr(args, "metrics_jsonl", None)
    args.status_json = t_args.status_json if t_args.status_json is not None else getattr(args, "status_json", None)
    args.summary_json = t_args.summary_json if t_args.summary_json is not None else getattr(args, "summary_json", None)
    args.reward_log_path = (
        t_args.reward_log_path if t_args.reward_log_path is not None else getattr(args, "reward_log_path", None)
    )
    args.early_stop_file = t_args.early_stop_file if t_args.early_stop_file is not None else getattr(args, "early_stop_file", None)
    args.save_minimal_logs = bool(t_args.save_minimal_logs or getattr(args, "save_minimal_logs", False))
    args.rollout_mean_mode = getattr(args, "rollout_mean_mode", "remove_max")
    args.bootstrap_mean_mode = getattr(args, "bootstrap_mean_mode", "normal")
    args.prior_std = float(getattr(args, "prior_std", 1.0))
    args.target_norm = getattr(args, "target_norm", "none")
    args.target_norm_update = getattr(args, "target_norm_update", "cumulative")
    args.target_norm_ema_tau = float(getattr(args, "target_norm_ema_tau", 0.99))
    args.alpha_warmup_updates = int(getattr(args, "alpha_warmup_updates", 0) or 0)
    args.cov_floor = float(getattr(args, "cov_floor", 0.0) or 0.0)
    args.cov_weight_clip = getattr(args, "cov_weight_clip", None)
    args.ess_floor = float(getattr(args, "ess_floor", 0.0) or 0.0)
    args.old_critic_update = getattr(args, "old_critic_update", "hard")
    args.old_critic_ema_tau = float(getattr(args, "old_critic_ema_tau", 0.995))
    args.velocity_clip_delta = float(getattr(args, "velocity_clip_delta", getattr(args, "clip_coef", 0.2)))
    args.metrics_interval = max(1, int(getattr(args, "metrics_interval", 1)))
    args.save_reward_log = bool(getattr(args, "save_reward_log", True))
    args.log_tensorboard = bool(getattr(args, "log_tensorboard", False))
    args.track = bool(getattr(args, "track", False))
    if args.save_minimal_logs:
        args.track = False
        args.log_tensorboard = False

    # 如果需要，设置运行名称 - 优先使用命令行参数
    if t_args.run_name is not None:
        run_name = t_args.run_name  # 使用命令行传入的run_name
    else:
        run_name = f"{args.env_id}/{args.exp_name}/{args.seed}/{datetime.datetime.now().strftime('%m%d%H%M%S')}"

    run_dir = PROJECT_DIR / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    research_logging = any(
        getattr(args, key, None) is not None
        for key in (
            "research_tag",
            "metrics_jsonl",
            "status_json",
            "summary_json",
            "reward_log_path",
            "early_stop_file",
        )
    )
    metrics_path = Path(args.metrics_jsonl) if args.metrics_jsonl is not None else (run_dir / "metrics.jsonl" if research_logging else None)
    status_path = Path(args.status_json) if args.status_json is not None else (run_dir / "status.json" if research_logging else None)
    summary_path = Path(args.summary_json) if args.summary_json is not None else (run_dir / "summary.json" if research_logging else None)
    early_stop_path = Path(args.early_stop_file) if args.early_stop_file is not None else (run_dir / "early_stop.json" if research_logging else None)
    args.metrics_jsonl = str(metrics_path) if metrics_path is not None else None
    args.status_json = str(status_path) if status_path is not None else None
    args.summary_json = str(summary_path) if summary_path is not None else None
    args.reward_log_path = str(args.reward_log_path) if args.reward_log_path is not None else None
    args.early_stop_file = str(early_stop_path) if early_stop_path is not None else None

    # 计算运行时参数
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.effective_total_timesteps = args.num_iterations * args.batch_size

    # 修改日期格式为月日十分秒
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{datetime.datetime.now().strftime('%m%d%H%M%S')}"

    # 创建必要记录文件；默认只保留 rewards.txt，TensorBoard 需要显式打开。
    os.makedirs(run_dir, exist_ok=True)
    save_reward_log = args.save_reward_log
    reward_log = None
    if save_reward_log:
        reward_log_path = Path(args.reward_log_path) if args.reward_log_path is not None else run_dir / "rewards.txt"
        reward_log_path.parent.mkdir(parents=True, exist_ok=True)
        reward_log = open(reward_log_path, "w", buffering=1)
        reward_log.write("step,episodic_return\n")
    sync_reward_log = getattr(args, "sync_reward_log", False)
    log_tensorboard = args.log_tensorboard
    metrics_interval = args.metrics_interval
    metrics_jsonl = Path(args.metrics_jsonl) if args.metrics_jsonl is not None else None
    status_json = Path(args.status_json) if args.status_json is not None else None
    summary_json = Path(args.summary_json) if args.summary_json is not None else None
    early_stop_file = Path(args.early_stop_file) if args.early_stop_file is not None else None

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=log_tensorboard,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = None
    if log_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(run_dir))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    use_cuda = torch.cuda.is_available() and args.cuda
    cuda_device = "cuda" if t_args.gpu_id == 0 else f"cuda:{t_args.gpu_id}"
    isaac_cuda_device = f"cuda:{t_args.gpu_id}"
    device = torch.device(cuda_device if use_cuda else "cpu")

    envs = isaacgymenvs.make(
        seed=args.seed,
        task=args.env_id,
        num_envs=args.num_envs,
        sim_device=isaac_cuda_device if use_cuda else "cpu",
        rl_device=isaac_cuda_device if use_cuda else "cpu",
        graphics_device_id=-1,  # 设置为-1来禁用图形显示
        headless=True,  # 设置为True来关闭画面显示
        multi_gpu=False,
        virtual_screen_capture=args.capture_video,
        force_render=False,
    )
    if args.capture_video:
        envs.is_vector_env = True
        print(f"record_video_step_frequency={args.record_video_step_frequency}")
        envs = gym.wrappers.RecordVideo(
            envs,
            f"videos/{run_name}",
            step_trigger=lambda step: step % args.record_video_step_frequency == 0,
            video_length=100,  # for each video record up to 100 steps
        )
    envs = ExtractObsWrapper(envs)
    envs = RecordEpisodeStatisticsTorch(envs, device)
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    trainable_params = [param for param in agent.parameters() if param.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, dtype=torch.float).to(
        device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.float).to(
        device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    values_td = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    advantages_td = torch.zeros_like(rewards, dtype=torch.float).to(device)
    weights_storage = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)  # 新增权重存储
    raw_rewards = torch.zeros_like(rewards).to(device) if research_logging else None

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)
    best_episode_return = float("-inf")
    last_episode_return = None
    last_episode_length = None
    total_episode_count = 0
    last_iteration_metrics: Dict[str, Any] = {}

    if status_json is not None:
        safe_json_dump(
            status_json,
            {
                "state": "running",
                "run_name": run_name,
                "env_id": args.env_id,
                "seed": args.seed,
                "gpu_id": t_args.gpu_id,
                "global_step": global_step,
                "iteration": 0,
                "total_iterations": args.num_iterations,
                "total_timesteps": args.total_timesteps,
                "effective_total_timesteps": args.effective_total_timesteps,
                "reward_log_path": str(reward_log_path) if reward_log is not None else None,
                "start_time": datetime.datetime.now().isoformat(),
            },
        )

    iteration = 0
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        iter_metric_sums: Dict[str, float] = {}
        iter_metric_counts: Dict[str, int] = {}
        iter_episode_returns = []
        iter_episode_lengths = []

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.inference_mode():
                action, logprob, _, value_mean, value_sample, weights, std, cv, rollout_stats = agent.get_action_and_value(
                    next_obs,
                    mean_mode=args.rollout_mean_mode,
                    update_step=iteration - 1,
                    collect_stats=research_logging,
                )
                values[step] = value_mean.flatten()
                values_td[step] = value_sample.flatten()
            actions[step] = action
            logprobs[step] = logprob
            weights_storage[step] = weights.flatten()  # 保存权重
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards[step], next_done, info = envs.step(action)
            if raw_rewards is not None:
                raw_rewards[step] = rewards[step]
            rewards[step] = rewards[step] * args.reward_scaler
            if research_logging:
                accumulate_metrics(iter_metric_sums, iter_metric_counts, "rollout/", rollout_stats)
                rollout_step_stats = {
                    "reward_raw_mean": raw_rewards[step].mean().item() if raw_rewards is not None else rewards[step].mean().item(),
                    "reward_raw_std": raw_rewards[step].std(unbiased=False).item() if raw_rewards is not None else rewards[step].std(unbiased=False).item(),
                    "reward_scaled_mean": rewards[step].mean().item(),
                    "reward_scaled_std": rewards[step].std(unbiased=False).item(),
                    "done_count": next_done.sum().item(),
                }
                accumulate_metrics(iter_metric_sums, iter_metric_counts, "rollout/", rollout_step_stats)

            done_indices = torch.nonzero(next_done, as_tuple=False).flatten()
            if done_indices.numel() > 0:
                done_returns = info["r"][done_indices].detach().cpu().tolist()
                done_lengths = info["l"][done_indices].detach().cpu().tolist()
                iter_episode_returns.extend(done_returns)
                iter_episode_lengths.extend(done_lengths)
                total_episode_count += len(done_returns)
                best_episode_return = max(best_episode_return, max(done_returns))
                last_episode_return = done_returns[-1]
                last_episode_length = done_lengths[-1]

            if 0 <= step <= 2 and done_indices.numel() > 0:
                idx = done_indices[0].item()
                episodic_return = info["r"][idx].item()
                print(f"global_step={global_step}, episodic_return={episodic_return}")
                if writer is not None:
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

                if reward_log is not None:
                    reward_log.write(f"{global_step},{episodic_return}\n")
                    if sync_reward_log:
                        reward_log.flush()
                        os.fsync(reward_log.fileno())

                if writer is not None and "consecutive_successes" in info:  # ShadowHand and AllegroHand metric
                    writer.add_scalar(
                        "charts/consecutive_successes", info["consecutive_successes"].item(), global_step
                    )

        # bootstrap value if not done
        with torch.inference_mode():
            next_value_mean, next_value_sample, bootstrap_stats = agent.get_value(
                next_obs,
                mean_mode=args.bootstrap_mean_mode,
                collect_stats=research_logging,
            )
            next_value_exp = next_value_mean.reshape(1, -1)
            next_value_td = next_value_sample.reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            advantages_td.zero_()
            lastgaelam = 0
            lastgaelam_td = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues_exp = next_value_exp
                    nextvalues_td = next_value_td
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues_exp = values[t + 1]
                    nextvalues_td = values_td[t + 1]
                # policy 用期望优势
                delta_policy = rewards[t] + args.gamma * nextvalues_exp * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta_policy + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                # critic 用采样 TD
                delta_td = rewards[t] + args.gamma * nextvalues_td * nextnonterminal - values_td[t]
                advantages_td[t] = lastgaelam_td = delta_td + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam_td
            returns_td = advantages_td + values_td
        if research_logging:
            accumulate_metrics(iter_metric_sums, iter_metric_counts, "bootstrap/", bootstrap_stats)

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns_td = returns_td.reshape(-1)
        b_weights = weights_storage.reshape(-1)  # 展开权重
        # Optimizing the policy and value network
        clipfracs = []
        optimizer_step_count = 0
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy = agent.get_action(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
                    if args.target_kl is not None:
                        clipfracs += [clipfrac.item()]
                    else:
                        clipfracs += [clipfrac.detach()]

                mb_advantages = b_advantages[mb_inds]
                mb_weights = b_weights[mb_inds]  # 取出当前minibatch的权重
                # 对权重进行归一化
                if args.wadv:
                    mb_weights = mb_weights / (mb_weights.sum() + 1e-8) * mb_weights.size(0)
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                # pg_loss1 = mb_weights * pg_loss1
                # pg_loss2 = mb_weights * pg_loss2
                pg_loss = torch.max(pg_loss1, pg_loss2)
                pg_loss = (mb_weights * pg_loss).mean()
                # Value loss
                # newvalue = newvalue.view(-1)
                # if args.clip_vloss:
                #     v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                #     v_clipped = b_values[mb_inds] + torch.clamp(
                #         newvalue - b_values[mb_inds],
                #         -args.clip_coef,
                #         args.clip_coef,
                #     )
                #     v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                #     v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                #     v_loss = 0.5 * v_loss_max.mean()
                #     # v_loss = agent.critic.loss(b_returns[mb_inds].unsqueeze(1), cond=b_obs[mb_inds],
                #     #                            clip_vloss=args.clip_vloss, )
                # else:
                #     # v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                # v_loss = agent.compute_critic_loss(obs_batch=b_obs[mb_inds], returns_batch=b_returns[mb_inds])
                critic_loss_out = agent.critic.loss(
                    x=b_returns_td[mb_inds].unsqueeze(1),
                    cond=b_obs[mb_inds],
                    old_model=agent.critic_old.model,
                    clip_vloss=args.clip_vloss,
                    clip_delta=args.velocity_clip_delta,
                    return_stats=research_logging,
                )
                if research_logging:
                    v_loss, critic_loss_stats = critic_loss_out
                else:
                    v_loss = critic_loss_out
                    critic_loss_stats = None
                entropy_loss = entropy.mean()
                loss = pg_loss*args.acf_coef - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer_step_count += 1
                accumulate_metrics(
                    iter_metric_sums,
                    iter_metric_counts,
                    "train/",
                    {
                        "policy_loss": pg_loss.item(),
                        "value_loss": v_loss.item(),
                        "entropy": entropy_loss.item(),
                        "old_approx_kl": old_approx_kl.item(),
                        "approx_kl": approx_kl.item(),
                        "clipfrac": clipfrac.item(),
                        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                        "loss_total": loss.item(),
                    },
                )
                if research_logging and critic_loss_stats is not None:
                    accumulate_metrics(iter_metric_sums, iter_metric_counts, "critic/", critic_loss_stats)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        if args.old_critic_update == "ema":
            tau = float(getattr(args, "old_critic_ema_tau", 0.995))
            with torch.no_grad():
                for old_param, new_param in zip(agent.critic_old.parameters(), agent.critic.parameters()):
                    old_param.data.mul_(tau).add_(new_param.data, alpha=1.0 - tau)
                for old_buffer, new_buffer in zip(agent.critic_old.buffers(), agent.critic.buffers()):
                    old_buffer.copy_(new_buffer)
        elif args.old_critic_update == "hard":
            agent.critic_old.load_state_dict(agent.critic.state_dict())
        else:
            raise ValueError(f"Unknown old_critic_update: {args.old_critic_update}")
        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        if research_logging:
            accumulate_metrics(
                iter_metric_sums,
                iter_metric_counts,
                "run/",
                {
                    "sps": sps,
                    "global_step": global_step,
                    "iteration": iteration,
                    "optimizer_steps": optimizer_step_count,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "episode_count": total_episode_count,
                },
            )
            accumulate_metrics(iter_metric_sums, iter_metric_counts, "batch/", tensor_stats(b_returns_td, "returns_td"))
            accumulate_metrics(iter_metric_sums, iter_metric_counts, "batch/", tensor_stats(b_advantages, "advantages"))
            accumulate_metrics(iter_metric_sums, iter_metric_counts, "batch/", tensor_stats(b_weights, "weights"))
            if raw_rewards is not None:
                accumulate_metrics(iter_metric_sums, iter_metric_counts, "batch/", tensor_stats(raw_rewards, "reward_raw"))
            accumulate_metrics(iter_metric_sums, iter_metric_counts, "batch/", tensor_stats(rewards, "reward_scaled"))
            if iter_episode_returns:
                accumulate_metrics(
                    iter_metric_sums,
                    iter_metric_counts,
                    "episode/",
                    {
                        "return_mean": float(np.mean(iter_episode_returns)),
                        "return_std": float(np.std(iter_episode_returns)),
                        "return_min": float(np.min(iter_episode_returns)),
                        "return_max": float(np.max(iter_episode_returns)),
                        "length_mean": float(np.mean(iter_episode_lengths)),
                        "length_std": float(np.std(iter_episode_lengths)),
                    },
                )
            accumulate_metrics(
                iter_metric_sums,
                iter_metric_counts,
                "episode/",
                {
                    "best_return": best_episode_return if best_episode_return != float("-inf") else 0.0,
                    "last_return": last_episode_return if last_episode_return is not None else 0.0,
                    "last_length": last_episode_length if last_episode_length is not None else 0.0,
                },
            )
        last_iteration_metrics = {
            key: (iter_metric_sums[key] / max(iter_metric_counts.get(key, 1), 1))
            for key in iter_metric_sums
        }
        if writer is not None and iteration % metrics_interval == 0:
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            if args.target_kl is None:
                clipfrac_mean = torch.stack(clipfracs).mean().item()
            else:
                clipfrac_mean = np.mean(clipfracs)
            writer.add_scalar("losses/clipfrac", clipfrac_mean, global_step)
            writer.add_scalar("losses/target", b_returns_td[mb_inds].mean().item(), global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("weights/mean", b_weights.mean().item(), global_step)
            writer.add_scalar("weights/std", b_weights.std().item(), global_step)
            writer.add_scalar("weights/max", b_weights.max().item(), global_step)
            writer.add_scalar("weights/min", b_weights.min().item(), global_step)
            # 记录std和cv的统计信息（只记录0号样本）
            writer.add_scalar("critic/std_sampled", std.item(), global_step)
            writer.add_scalar("critic/cv_sampled", cv.item(), global_step)
        if metrics_jsonl is not None:
            metrics_record = {
                "run_name": run_name,
                "env_id": args.env_id,
                "seed": args.seed,
                "gpu_id": t_args.gpu_id,
                "iteration": iteration,
                "global_step": global_step,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            metrics_record.update(last_iteration_metrics)
            append_jsonl(metrics_jsonl, metrics_record)
            last_iteration_metrics = metrics_record
        if status_json is not None:
            safe_json_dump(
                status_json,
                {
                    "state": "running",
                    "run_name": run_name,
                    "env_id": args.env_id,
                    "seed": args.seed,
                    "gpu_id": t_args.gpu_id,
                    "iteration": iteration,
                    "global_step": global_step,
                    "effective_total_timesteps": args.effective_total_timesteps,
                    "sps": sps,
                    "best_episode_return": best_episode_return if best_episode_return != float("-inf") else None,
                    "last_episode_return": last_episode_return,
                    "last_episode_length": last_episode_length,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "metrics": last_iteration_metrics,
                },
            )
        stop_request = load_stop_request(str(early_stop_file) if early_stop_file is not None else None)
        if stop_request and stop_request.get("stop"):
            print(f"[early-stop] {stop_request.get('reason', 'requested')}")
            break
        # writer.add_scalar("charts/cv_mean", cv.mean().item(), global_step)
        # writer.add_scalar("charts/weights_mean", weights.mean().item(), global_step)
    # envs.close()
    if reward_log is not None:
        reward_log.close()
    if writer is not None:
        writer.close()
    final_payload = {
        "state": "finished",
        "run_name": run_name,
        "env_id": args.env_id,
        "seed": args.seed,
        "gpu_id": t_args.gpu_id,
        "global_step": global_step,
        "iterations": iteration,
        "total_iterations": args.num_iterations,
        "requested_total_timesteps": args.total_timesteps,
        "effective_total_timesteps": args.effective_total_timesteps,
        "completed_budget": global_step >= args.effective_total_timesteps,
        "reward_log_path": str(reward_log_path) if reward_log is not None else None,
        "best_episode_return": best_episode_return if best_episode_return != float("-inf") else None,
        "last_episode_return": last_episode_return,
        "last_episode_length": last_episode_length,
        "finished_at": datetime.datetime.now().isoformat(),
        "metrics": last_iteration_metrics,
    }
    if summary_json is not None:
        safe_json_dump(summary_json, final_payload)
    if status_json is not None:
        safe_json_dump(status_json, final_payload)
