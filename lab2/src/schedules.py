"""Beta schedules for DDPM: linear (Ho et al., 2020) and cosine (Nichol & Dhariwal, 2021)."""
from __future__ import annotations

import math
import torch


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule from the original DDPM paper."""
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule (Nichol & Dhariwal, "Improved DDPM", 2021).

    alpha_bar(t) = f(t) / f(0), where
        f(t) = cos((t/T + s) / (1 + s) * pi/2)^2
    """
    steps = torch.arange(T + 1, dtype=torch.float32)
    f = torch.cos((steps / T + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alpha_bar = f / f[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clamp(betas, 1e-8, 0.999)


def make_schedule(name: str, T: int) -> dict:
    """Return a dictionary with all derived tensors needed by DDPM and DDIM."""
    if name == "linear":
        betas = linear_beta_schedule(T)
    elif name == "cosine":
        betas = cosine_beta_schedule(T)
    else:
        raise ValueError(f"Unknown schedule: {name!r}")

    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    # For posterior sampling
    alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
    posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

    return {
        "name": name,
        "T": T,
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "alpha_bar_prev": alpha_bar_prev,
        "sqrt_alpha_bar": sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar,
        "posterior_variance": posterior_variance,
    }
