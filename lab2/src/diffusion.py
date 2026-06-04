"""DDPM forward/reverse processes and DDIM sampler."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .schedules import make_schedule


def _extract(a: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
    """Gather values from a 1-D tensor `a` at indices `t`, reshape for broadcasting."""
    out = a.gather(0, t)
    return out.view(t.size(0), *([1] * (len(shape) - 1)))


class DDPM:
    """Encapsulates the forward (q) and reverse (p) processes for DDPM.

    Stores schedule tensors on the requested device.
    """

    def __init__(self, schedule_name: str, T: int, device: torch.device):
        self.device = device
        self.T = T
        sch = make_schedule(schedule_name, T)
        self.name = sch["name"]
        self.betas = sch["betas"].to(device)
        self.alphas = sch["alphas"].to(device)
        self.alpha_bar = sch["alpha_bar"].to(device)
        self.alpha_bar_prev = sch["alpha_bar_prev"].to(device)
        self.sqrt_alpha_bar = sch["sqrt_alpha_bar"].to(device)
        self.sqrt_one_minus_alpha_bar = sch["sqrt_one_minus_alpha_bar"].to(device)
        self.posterior_variance = sch["posterior_variance"].to(device)

    # ----- Forward -----
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            _extract(self.sqrt_alpha_bar, t, x0.shape) * x0
            + _extract(self.sqrt_one_minus_alpha_bar, t, x0.shape) * noise
        )

    # ----- Training loss -----
    def p_losses(self, model, x0: torch.Tensor) -> torch.Tensor:
        b = x0.size(0)
        t = torch.randint(0, self.T, (b,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        eps_pred = model(xt, t)
        return F.mse_loss(eps_pred, noise)

    # ----- Reverse (DDPM) -----
    @torch.no_grad()
    def p_sample(self, model, xt: torch.Tensor, t: int) -> torch.Tensor:
        t_tensor = torch.full((xt.size(0),), t, device=xt.device, dtype=torch.long)
        eps = model(xt, t_tensor)

        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]
        beta_t = self.betas[t]

        mean = (xt - beta_t / torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_t)

        if t > 0:
            noise = torch.randn_like(xt)
            var = self.posterior_variance[t]
            return mean + torch.sqrt(var) * noise
        return mean

    @torch.no_grad()
    def sample(self, model, shape, *, return_intermediates: bool = False, init_noise: torch.Tensor | None = None):
        x = init_noise if init_noise is not None else torch.randn(shape, device=self.device)
        traj = []
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
            if return_intermediates and (t % max(1, self.T // 10) == 0 or t == 0):
                traj.append(x.detach().clone().cpu())
        if return_intermediates:
            return x, traj
        return x


class DDIM:
    """Deterministic / stochastic DDIM sampler with sub-grid timesteps.

    eta = 0  -> fully deterministic DDIM
    eta = 1  -> equivalent to DDPM update
    """

    def __init__(self, ddpm: DDPM):
        self.ddpm = ddpm
        self.device = ddpm.device

    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        *,
        num_steps: int = 50,
        eta: float = 0.0,
        init_noise: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ):
        T = self.ddpm.T
        # Sub-grid: uniformly select num_steps timesteps from [0, T-1]
        step_idx = torch.linspace(0, T - 1, num_steps, device=self.device).long().tolist()
        step_idx = sorted(set(step_idx))
        x = init_noise if init_noise is not None else torch.randn(shape, device=self.device)

        traj = []
        for i in reversed(range(len(step_idx))):
            t = step_idx[i]
            t_prev = step_idx[i - 1] if i > 0 else -1

            alpha_bar_t = self.ddpm.alpha_bar[t]
            alpha_bar_prev = self.ddpm.alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)

            t_tensor = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
            eps = model(x, t_tensor)

            # Predict x_0
            x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

            # Sigma
            sigma = eta * torch.sqrt(
                (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                * (1.0 - alpha_bar_t / alpha_bar_prev)
            )

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1.0 - alpha_bar_prev - sigma ** 2) * eps

            noise = torch.randn_like(x) if eta > 0 and t_prev >= 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

            if return_intermediates and (i % max(1, len(step_idx) // 10) == 0 or i == 0):
                traj.append(x.detach().clone().cpu())

        if return_intermediates:
            return x, traj
        return x
