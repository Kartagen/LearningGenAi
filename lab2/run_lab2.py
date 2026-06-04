"""End-to-end Lab 2 pipeline.

Trains DDPM on MNIST 32x32 with linear and cosine beta schedules, generates
samples via DDPM and DDIM (200 / 50 steps), and saves all artefacts (PNG,
JSON, CSV) into ``outputs/``.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils import set_global_seed, save_grid, save_traj_grid
from src.dataloader import get_mnist, to_image
from src.unet import UNet, count_params
from src.diffusion import DDPM, DDIM
from src.schedules import make_schedule
from src.training import train_ddpm


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 32
IMG_CHANNELS = 1            # MNIST
BATCH_SIZE = 128

BASE_CHANNELS = 64           # "добре"-level: 64 base channels with attn in bottleneck
CHANNEL_MULTS = (1, 2, 4)    # 3 levels with 32->16->8 spatial; bottleneck at 8x8
NUM_RES = 1
ATTN_AT = (8,)               # self-attention at 8x8 bottleneck
DROPOUT = 0.1

T_TRAIN = 200               # diffusion timesteps for both schedules

EPOCHS = 4                   # MNIST 32x32 converges quickly
LR = 2e-4
MAX_BATCHES_PER_EPOCH = None # use full epochs

# Artefact paths
OUT = ROOT / "outputs"
CKPT_DIR = ROOT / "checkpoints"
OUT.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)


def main():
    set_global_seed(SEED)
    print(f"device = {DEVICE}, torch={torch.__version__}")

    # ---------------- Data ----------------
    train_loader, _ = get_mnist(ROOT / "data", img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_workers=0)
    print(f"train batches: {len(train_loader)}  batch={BATCH_SIZE}  img={IMG_SIZE}x{IMG_SIZE}")

    # Forward-diffusion visualisation (using LINEAR schedule for clarity)
    ddpm_lin = DDPM("linear", T_TRAIN, DEVICE)
    x0, _ = next(iter(train_loader))
    x0 = x0[:4].to(DEVICE)
    fwd_steps = [0, T_TRAIN // 10, T_TRAIN // 5, T_TRAIN // 3, T_TRAIN // 2, 2 * T_TRAIN // 3, T_TRAIN - 1]
    fig, axes = plt.subplots(len(x0), len(fwd_steps), figsize=(len(fwd_steps) * 1.4, len(x0) * 1.4))
    for r in range(len(x0)):
        for c, t in enumerate(fwd_steps):
            t_t = torch.full((1,), t, device=DEVICE, dtype=torch.long)
            xt = ddpm_lin.q_sample(x0[r:r+1], t_t)
            img = to_image(xt[0]).cpu().permute(1, 2, 0).numpy().squeeze()
            axes[r, c].imshow(img, cmap="gray")
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"t={t}", fontsize=9)
    fig.suptitle("Forward diffusion (linear schedule, T=200)")
    plt.tight_layout()
    plt.savefig(OUT / "forward_diffusion.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("saved forward_diffusion.png")

    # ---------------- Compare schedules visually ----------------
    sch_lin = make_schedule("linear", T_TRAIN)
    sch_cos = make_schedule("cosine", T_TRAIN)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(sch_lin["betas"].numpy(), label="linear")
    axes[0].plot(sch_cos["betas"].numpy(), label="cosine")
    axes[0].set_xlabel("timestep t"); axes[0].set_ylabel(r"$\beta_t$")
    axes[0].set_title(r"$\beta_t$ schedule")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(sch_lin["alpha_bar"].numpy(), label="linear")
    axes[1].plot(sch_cos["alpha_bar"].numpy(), label="cosine")
    axes[1].set_xlabel("timestep t"); axes[1].set_ylabel(r"$\bar{\alpha}_t$")
    axes[1].set_title(r"$\bar{\alpha}_t$ schedule")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "schedules.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("saved schedules.png")

    # ---------------- Training: linear schedule ----------------
    results = {}
    sample_shape = (16, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

    for sch_name in ["linear", "cosine"]:
        print(f"\n{'='*60}\nSchedule = {sch_name}\n{'='*60}")
        set_global_seed(SEED)
        model = UNet(
            img_channels=IMG_CHANNELS,
            base_channels=BASE_CHANNELS,
            channel_mults=CHANNEL_MULTS,
            num_res=NUM_RES,
            attn_at=ATTN_AT,
            dropout=DROPOUT,
            img_size=IMG_SIZE,
        )
        print(f"UNet params: {count_params(model):,}")

        ddpm = DDPM(sch_name, T_TRAIN, DEVICE)
        model, ema, history = train_ddpm(
            model, train_loader,
            ddpm=ddpm,
            epochs=EPOCHS,
            learning_rate=LR,
            device=DEVICE,
            use_ema=True,
            max_batches_per_epoch=MAX_BATCHES_PER_EPOCH,
        )

        # Save weights (EMA params for sampling)
        ckpt_path = CKPT_DIR / f"unet_{sch_name}.pt"
        torch.save({"model": model.state_dict(), "ema": ema.shadow}, ckpt_path)
        print(f"saved checkpoint {ckpt_path}")

        # Loss curve
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(history["step"], history["train_loss"], alpha=0.4, label="train (step)")
        ep_mean_steps = [(i + 1) * len(train_loader) for i in range(len(history["epoch_mean_loss"]))]
        ax.plot(ep_mean_steps, history["epoch_mean_loss"], "o-", color="red", label="epoch mean")
        ax.set_xlabel("step"); ax.set_ylabel("MSE noise loss")
        ax.set_title(f"Training loss ({sch_name})")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / f"loss_{sch_name}.png", dpi=120)
        plt.close()

        # ---------- Sampling: DDPM full (T=200) ----------
        backup = ema.apply_to(model)
        model.eval()

        # Fixed init noise so DDPM/DDIM comparisons are on equal footing
        set_global_seed(SEED + 1)
        init_noise = torch.randn(sample_shape, device=DEVICE)

        t0 = time.time()
        samples_ddpm, traj = ddpm.sample(model, sample_shape, return_intermediates=True, init_noise=init_noise.clone())
        ddpm_time = time.time() - t0
        save_grid(to_image(samples_ddpm.cpu()), OUT / f"samples_ddpm_{sch_name}.png", nrow=4,
                  title=f"DDPM ({sch_name}) — T={T_TRAIN}")
        save_traj_grid(traj, OUT / f"reverse_{sch_name}.png", sample_idx=0,
                       title=f"Reverse diffusion ({sch_name})")
        print(f"DDPM ({sch_name}) sample time = {ddpm_time:.2f}s")

        # ---------- Sampling: DDIM with 200 and 50 steps ----------
        ddim = DDIM(ddpm)

        t0 = time.time()
        samples_ddim200 = ddim.sample(model, sample_shape, num_steps=200, eta=0.0, init_noise=init_noise.clone())
        ddim200_time = time.time() - t0
        save_grid(to_image(samples_ddim200.cpu()), OUT / f"samples_ddim200_{sch_name}.png", nrow=4,
                  title=f"DDIM ({sch_name}) — 200 steps")

        t0 = time.time()
        samples_ddim50 = ddim.sample(model, sample_shape, num_steps=50, eta=0.0, init_noise=init_noise.clone())
        ddim50_time = time.time() - t0
        save_grid(to_image(samples_ddim50.cpu()), OUT / f"samples_ddim50_{sch_name}.png", nrow=4,
                  title=f"DDIM ({sch_name}) — 50 steps")

        ema.restore(model, backup)
        model.train()

        results[sch_name] = {
            "params": count_params(model),
            "epochs": EPOCHS,
            "final_loss": float(history["epoch_mean_loss"][-1]),
            "init_loss": float(history["epoch_mean_loss"][0]),
            "epoch_mean_loss": history["epoch_mean_loss"],
            "epoch_time_s": history["epoch_time_s"],
            "step_loss": history["train_loss"][::10],   # downsampled
            "sample_time": {
                "ddpm_200": ddpm_time,
                "ddim_200": ddim200_time,
                "ddim_50": ddim50_time,
            },
        }
        print(json.dumps(results[sch_name]["sample_time"], indent=2))

    # ---------------- Save consolidated outputs.json ----------------
    summary = {
        "config": {
            "seed": SEED,
            "device": str(DEVICE),
            "img_size": IMG_SIZE,
            "img_channels": IMG_CHANNELS,
            "batch_size": BATCH_SIZE,
            "base_channels": BASE_CHANNELS,
            "channel_mults": list(CHANNEL_MULTS),
            "num_res": NUM_RES,
            "attn_at": list(ATTN_AT),
            "dropout": DROPOUT,
            "T": T_TRAIN,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "ema_decay": 0.999,
        },
        "results": results,
    }
    with open(OUT / "outputs.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll artefacts saved to {OUT}")


if __name__ == "__main__":
    main()
