"""Training loop for DDPM (predicts noise epsilon)."""
from __future__ import annotations

import time

import torch
from torch.optim import Adam
from tqdm.auto import tqdm

from .diffusion import DDPM
from .utils import EMA


def train_ddpm(
    model,
    train_loader,
    *,
    ddpm: DDPM,
    epochs: int = 10,
    learning_rate: float = 2e-4,
    device: torch.device,
    use_ema: bool = True,
    ema_decay: float = 0.999,
    max_batches_per_epoch: int | None = None,
):
    model.to(device).train()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    ema = EMA(model, decay=ema_decay) if use_ema else None

    history = {"epoch": [], "step": [], "train_loss": [], "epoch_mean_loss": [], "epoch_time_s": []}
    global_step = 0

    for epoch in range(epochs):
        epoch_losses = []
        epoch_start = time.time()
        bar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False)
        for i, (x, _) in enumerate(bar):
            x = x.to(device)
            loss = ddpm.p_losses(model, x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if ema is not None:
                ema.update(model)

            loss_v = float(loss.item())
            history["step"].append(global_step)
            history["train_loss"].append(loss_v)
            epoch_losses.append(loss_v)
            bar.set_postfix(loss=f"{loss_v:.4f}")
            global_step += 1
            if max_batches_per_epoch is not None and i + 1 >= max_batches_per_epoch:
                break

        mean = sum(epoch_losses) / len(epoch_losses)
        history["epoch"].append(epoch)
        history["epoch_mean_loss"].append(mean)
        history["epoch_time_s"].append(time.time() - epoch_start)
        print(f"  epoch {epoch+1}: mean loss = {mean:.4f}  ({history['epoch_time_s'][-1]:.1f}s)")

    return model, ema, history
