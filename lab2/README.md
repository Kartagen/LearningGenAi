# Лабораторна робота 2 — Denoising Diffusion Probabilistic Models (DDPM)

Реалізація DDPM з нуля на PyTorch: U-Net із sinusoidal time embedding і
self-attention у bottleneck, пряма та зворотна дифузія, порівняння linear vs
cosine beta-шедул, DDIM-sampling з підсіткою кроків. Датасет -- MNIST 32x32.

## Структура

```
lab2/
├── run_lab2.py                 # головний end-to-end пайплайн
├── requirements.txt
├── src/
│   ├── schedules.py            # linear / cosine beta schedules
│   ├── unet.py                 # U-Net з sinusoidal time embedding + attention
│   ├── diffusion.py            # DDPM + DDIM sampler
│   ├── dataloader.py           # MNIST / CIFAR-10 (32x32, [-1, 1])
│   ├── training.py             # train loop з EMA
│   └── utils.py                # seed, save_grid, EMA
├── checkpoints/
│   ├── unet_linear.pt          # ваги UNet з linear schedule
│   └── unet_cosine.pt          # ваги UNet з cosine schedule
└── outputs/
    ├── forward_diffusion.png   # q-sample на різних t
    ├── schedules.png           # beta_t та alpha_bar_t для linear/cosine
    ├── loss_linear.png         # криві навчання
    ├── loss_cosine.png
    ├── reverse_linear.png      # послідовна реверс-дифузія x_t -> x_0
    ├── reverse_cosine.png
    ├── samples_ddpm_linear.png    # 16 зразків (DDPM, 200 steps)
    ├── samples_ddpm_cosine.png
    ├── samples_ddim200_linear.png # 16 зразків (DDIM eta=0, 200 steps)
    ├── samples_ddim50_linear.png  # 16 зразків (DDIM eta=0,  50 steps)
    ├── samples_ddim200_cosine.png
    ├── samples_ddim50_cosine.png
    └── outputs.json            # усі сирі метрики
```

## Залежності

- Python 3.10+
- NVIDIA GPU >= 8 GB VRAM (повний прогін на RTX 5060 Ti ~ 10 хв)
- CUDA-збірка PyTorch

## Встановлення

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск

```bash
# Повний пайплайн: тренує 2 моделі x 4 епохи + sampling
python run_lab2.py
```

## Конфігурація

| Параметр | Значення |
|---|---:|
| Датасет | MNIST 32x32 (60 000 train), [-1, 1] |
| U-Net base channels | 64, channel_mults=(1, 2, 4) |
| Глибина | 3 рівні: 32 -> 16 -> 8 (bottleneck) |
| ResBlocks per level | 1 |
| Self-attention | тільки на 8x8 bottleneck |
| Time embedding | sinusoidal -> MLP (256-dim) |
| Diffusion T | 200 |
| Schedules | linear (Ho 2020) + cosine (Nichol 2021) |
| Optimizer | Adam, lr=2e-4 |
| Batch size | 128 |
| Epochs | 4 |
| EMA decay | 0.999 |
| Sampling | DDPM 200 steps, DDIM eta=0 з 200 і 50 кроками |

## Що реалізовано

| Завдання | Файл / комірка |
|---|---|
| Linear і cosine beta-шедули | `src/schedules.py` |
| Пряма дифузія `x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*eps` | `src/diffusion.py::q_sample` |
| U-Net із sinusoidal time embedding + attention у bottleneck | `src/unet.py::UNet` |
| Тренування з EMA + MSE-loss на шум | `src/training.py::train` |
| DDPM-sampling (повний крок-у-крок) | `src/diffusion.py::p_sample` |
| DDIM-sampling (детермінований, eta=0, підсітка timesteps) | `src/diffusion.py::ddim_sample` |
| Порівняння linear vs cosine | `run_lab2.py` |
| Заміри часу sampling-у | у `outputs/outputs.json` |

## Результати

| Метрика | linear | cosine |
|---|---:|---:|
| init epoch loss | 0.082 | 0.072 |
| final epoch loss | 0.037 | 0.032 |
| час / епоха | ~61 s | ~61 s |
| DDPM-200 sample (16 imgs) | 1.26 s | 1.20 s |
| DDIM-200 sample (16 imgs) | 1.25 s | 1.28 s |
| DDIM-50 sample (16 imgs) | 0.32 s | 0.33 s |

**Speed-up DDIM-50 vs DDPM-200 ~ 4x** при майже непомітній втраті якості.
