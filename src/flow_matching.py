import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_ot_sampler = None


# OT
def OT_sample(x0, x1):
    global _ot_sampler
    if _ot_sampler is None:
        from torchcfm.optimal_transport import OTPlanSampler

        _ot_sampler = OTPlanSampler(method='exact')
    x0, x1 = _ot_sampler.sample_plan(x0, x1)
    return x0, x1


class FlowMatching(nn.Module):
    def __init__(
            self,
            state_dim: int,
            model=None,
            horizon: int = 1,
            device: str = "cuda",
            sigma=0.01,
            prior_std: float = 1.0,
            target_norm: str = "none",
            target_norm_eps: float = 1e-8,
            target_norm_update: str = "cumulative",
            target_norm_ema_tau: float = 0.99,
    ):
        super().__init__()

        self.model = model
        self.horizon = horizon
        self.state_dim = state_dim
        self.device = device
        self.sigma = sigma
        self.prior_std = prior_std
        self.target_norm = (target_norm or "none").lower()
        self.target_norm_eps = target_norm_eps
        self.target_norm_update = (target_norm_update or "cumulative").lower()
        self.target_norm_ema_tau = float(target_norm_ema_tau)
        self._time_grid_cache = {}
        self._time_ones_cache = {}
        self.register_buffer("target_mean", torch.zeros(state_dim))
        self.register_buffer("target_var", torch.ones(state_dim))
        self.register_buffer("target_count", torch.tensor(0.0))

    def compute_conditional_vector_field(self, x0, x1):
        return x1 - x0

    def sample_conditional_pt(self, x0, x1, t, sigma):
        t = t.reshape(-1, *([1] * (x0.dim() - 1)))
        mu_t = t * x1 + (1 - t) * x0
        epsilon = torch.randn_like(x0)
        return mu_t + sigma * epsilon

    def _target_std(self):
        return torch.sqrt(self.target_var.clamp_min(self.target_norm_eps))

    def _update_target_stats(self, x: torch.Tensor) -> None:
        if self.target_norm == "none":
            return
        x_flat = x.detach().view(-1, self.state_dim)
        if x_flat.numel() == 0:
            return
        batch_count = x_flat.shape[0]
        batch_mean = x_flat.mean(dim=0)
        batch_var = x_flat.var(dim=0, unbiased=False)

        count = self.target_count
        if count.item() == 0:
            self.target_mean.copy_(batch_mean)
            self.target_var.copy_(batch_var.clamp_min(self.target_norm_eps))
            self.target_count.fill_(float(batch_count))
            return

        if self.target_norm_update == "ema":
            tau = min(0.9999, max(0.0, self.target_norm_ema_tau))
            self.target_mean.mul_(tau).add_(batch_mean, alpha=1.0 - tau)
            self.target_var.mul_(tau).add_(batch_var.clamp_min(self.target_norm_eps), alpha=1.0 - tau)
            self.target_var.clamp_(min=self.target_norm_eps)
            self.target_count.add_(float(batch_count))
            return
        if self.target_norm_update not in {"cumulative", "running"}:
            raise ValueError(f"Unknown target_norm_update: {self.target_norm_update}")

        total_count = count + float(batch_count)
        delta = batch_mean - self.target_mean
        new_mean = self.target_mean + delta * (float(batch_count) / total_count)

        m_a = self.target_var * count
        m_b = batch_var * float(batch_count)
        correction = delta.pow(2) * count * float(batch_count) / total_count
        new_var = (m_a + m_b + correction) / total_count

        self.target_mean.copy_(new_mean)
        self.target_var.copy_(new_var.clamp_min(self.target_norm_eps))
        self.target_count.fill_(float(total_count))

    def _normalize_target(self, x: torch.Tensor, update_stats: bool = False) -> torch.Tensor:
        if self.target_norm == "none":
            return x
        if update_stats:
            self._update_target_stats(x)
        mean = self.target_mean.view(1, -1).to(device=x.device, dtype=x.dtype)
        std = self._target_std().view(1, -1).to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def _denormalize_target(self, x: torch.Tensor) -> torch.Tensor:
        if self.target_norm == "none":
            return x
        mean = self.target_mean.view(1, -1).to(device=x.device, dtype=x.dtype)
        std = self._target_std().view(1, -1).to(device=x.device, dtype=x.dtype)
        return x * std + mean

    def generate(self, bs, num_steps, cond=None) -> torch.Tensor:
        x_ini = torch.randn(bs, self.state_dim).to(self.device) * self.prior_std
        time_grid = self._get_time_grid(num_steps, x_ini)
        time_ones = self._get_time_ones(bs, x_ini)
        final = self._euler_final(x_ini, cond, time_grid, time_ones)
        return self._denormalize_target(final)

    def generate_multiple(self, cond, n_samples=10, num_steps=10) -> torch.Tensor:
        bs = cond.size(0)

        # 扩展条件：将每个条件复制n_samples次
        # 从 [bs, cond_dim] 扩展到 [bs * n_samples, cond_dim]
        cond_expanded = cond.repeat_interleave(n_samples, dim=0)

        # 生成随机初始状态
        x_ini = torch.randn(bs * n_samples, self.state_dim).to(self.device) * self.prior_std
        time_grid = self._get_time_grid(num_steps, x_ini)
        time_ones = self._get_time_ones(bs * n_samples, x_ini)
        final = self._euler_final(x_ini, cond_expanded, time_grid, time_ones)
        final = self._denormalize_target(final)
        # 重新整形为 [batch_size, n_samples, state_dim]
        result = final.view(bs, n_samples, self.state_dim)
        return result

    def _get_time_grid(self, num_steps, ref):
        key = (num_steps, ref.device, ref.dtype)
        time_grid = self._time_grid_cache.get(key)
        if time_grid is None:
            time_grid = torch.linspace(0, 1, num_steps + 1).to(device=ref.device, dtype=ref.dtype)
            self._time_grid_cache[key] = time_grid
        return time_grid

    def _get_time_ones(self, batch_size, ref):
        key = (batch_size, ref.device, ref.dtype)
        time_ones = self._time_ones_cache.get(key)
        if time_ones is None:
            time_ones = torch.ones(batch_size).to(device=ref.device, dtype=ref.dtype)
            self._time_ones_cache[key] = time_ones
        return time_ones

    def _euler_final(self, x, cond, time_grid, time_ones):
        y = x
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dt = t1 - t0
            f0 = self.model.forward(y, cond=cond, t=time_ones * t0, use_dropout=True)
            dy = dt * f0
            y = y + dy
        return y

    def generate_with_stats(self, cond, n_samples=10, num_steps=10) -> tuple:
        samples = self.generate_multiple(cond, n_samples, num_steps)

        # 计算均值和方差
        mean = torch.mean(samples, dim=1)  # [batch_size, state_dim]
        var = torch.var(samples, dim=1)  # [batch_size, state_dim]

        return mean, var

    def loss(
        self,
        x: torch.Tensor,
        cond,
        old_model,
        clip_vloss: bool = False,
        clip_delta: float = 0.2,
        clip_coef: Optional[float] = None,
        return_stats: bool = False,
    ) -> torch.Tensor:  # x--target， cond 条件

        if clip_coef is not None:
            clip_delta = clip_coef

        batch_size = x.shape[0]
        t = torch.rand(batch_size).type_as(x)
        target = self._normalize_target(x, update_stats=True)
        x0 = torch.randn_like(target) * self.prior_std
        # x0, target = OT_sample(x0, target)  # OT_sample
        xt = self.sample_conditional_pt(x0, target, t, sigma=self.sigma)
        ut = self.compute_conditional_vector_field(x0, target)
        # vt = self.model(xt, cond, t)  # cond
        # loss = F.mse_loss(vt, ut)
        if clip_vloss and old_model is not None:
            vt_new = self.model(xt, cond, t)
            with torch.no_grad():
                vt_old = old_model(xt, cond, t)
            loss_unclipped = F.mse_loss(vt_new, ut, reduction='none')
            vt_clipped = vt_old + torch.clamp(
                vt_new - vt_old,
                -clip_delta,
                clip_delta,
            )
            loss_clipped = F.mse_loss(vt_clipped, ut, reduction='none')
            final_loss = torch.max(loss_unclipped, loss_clipped).mean()
            clip_ratio = ((vt_new - vt_old).abs() > clip_delta).float().mean()
        else:
            vt = self.model(xt, cond, t)
            final_loss = F.mse_loss(vt, ut)
            loss_unclipped = final_loss
            loss_clipped = final_loss
            clip_ratio = torch.zeros((), device=x.device, dtype=x.dtype)
        if not return_stats:
            return final_loss
        stats = {
            "loss_unclipped": loss_unclipped.detach(),
            "loss_clipped": loss_clipped.detach(),
            "clip_ratio": clip_ratio.detach(),
            "target_batch_mean": x.detach().mean(),
            "target_batch_std": x.detach().std(unbiased=False),
            "target_norm_mean": self.target_mean.detach().mean(),
            "target_norm_std": self._target_std().detach().mean(),
            "prior_std": torch.tensor(float(self.prior_std), device=x.device, dtype=x.dtype),
            "target_norm_mode": self.target_norm,
            "target_norm_update": self.target_norm_update,
        }
        return final_loss, stats


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Model(nn.Module):
    def __init__(self, condition_dim, out_dim, hidden_size=512, time_dim=64):
        super(Model, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_size),

            nn.ReLU(),
            nn.Linear(hidden_size, time_dim),
        )

        input_dim = condition_dim + time_dim + out_dim
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim)
        )

        self.apply(init_weights)

    def forward(self, x, cond, t, use_dropout=True):
        t_emb = self.time_mlp(t)
        x = x.view(x.size(0), -1)
        out = torch.cat([cond, t_emb, x], dim=-1)
        out = self.layer(out)
        return out

class Model_L(nn.Module):
    def __init__(self, condition_dim, out_dim, hidden_size=512, time_dim=64):
        super(Model_L, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_size),

            nn.ReLU(),
            nn.Linear(hidden_size, time_dim),
        )

        input_dim = condition_dim + time_dim + out_dim

        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim)
        )
        self.apply(init_weights)

    def forward(self, x, cond, t, use_dropout=True):
        t_emb = self.time_mlp(t)
        x = x.view(x.size(0), -1)
        out = torch.cat([cond, t_emb, x], dim=-1)
        out = self.layer(out)
        return out
