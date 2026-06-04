"""Utility helpers: seeding, grid plotting, EMA."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_grid(samples: torch.Tensor, path: str, nrow: int = 4, title: str | None = None) -> None:
    """Save a grid of images. ``samples`` must be in [0, 1]."""
    grid = make_grid(samples, nrow=nrow, padding=2)
    np_img = grid.permute(1, 2, 0).cpu().numpy()
    if np_img.shape[-1] == 1:
        np_img = np_img.squeeze(-1)
        cmap = "gray"
    else:
        cmap = None
    fig, ax = plt.subplots(figsize=(nrow * 1.6, (samples.shape[0] / nrow) * 1.6))
    ax.imshow(np_img, cmap=cmap)
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def save_traj_grid(traj: list, path: str, sample_idx: int = 0, title: str | None = None) -> None:
    """Plot a horizontal strip showing the reverse-diffusion progression for one sample."""
    n = len(traj)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.4, 1.5))
    if n == 1:
        axes = [axes]
    for i, x in enumerate(traj):
        img = ((x[sample_idx] + 1.0) / 2.0).clamp(0.0, 1.0)
        np_img = img.permute(1, 2, 0).numpy()
        if np_img.shape[-1] == 1:
            np_img = np_img.squeeze(-1)
            axes[i].imshow(np_img, cmap="gray")
        else:
            axes[i].imshow(np_img)
        axes[i].axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


class EMA:
    """Exponential moving average of model parameters for stable sampling."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_to(self, model: torch.nn.Module) -> dict:
        """Swap in EMA parameters; return the original state for restoring."""
        backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])
        return backup

    def restore(self, model: torch.nn.Module, backup: dict) -> None:
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])
