# Lab 2 — Denoising Diffusion Probabilistic Models (DDPM)

Повна реалізація DDPM з нуля на PyTorch для генерації цифр MNIST з порівнянням linear vs cosine schedule та DDIM-sampling-у.

## Структура

```
lab2/
├── notebook.ipynb              (не використовується, тут — скрипт)
├── run_lab2.py                 ← головний пайплайн end-to-end
├── generate_report.py          ← генератор DOCX звіту
├── requirements.txt
├── Височин_ГНМ_ЛР2.docx        ← готовий звіт
├── src/
│   ├── schedules.py            ← linear / cosine beta schedules
│   ├── unet.py                 ← U-Net з sinusoidal time embedding + attention
│   ├── diffusion.py            ← DDPM + DDIM sampler
│   ├── dataloader.py           ← MNIST / CIFAR-10 (32×32, [-1, 1])
│   ├── training.py             ← train loop з EMA
│   └── utils.py                ← seed, save_grid, EMA
├── checkpoints/
│   ├── unet_linear.pt          ← ваги UNet з linear schedule
│   └── unet_cosine.pt          ← ваги UNet з cosine schedule
└── outputs/
    ├── forward_diffusion.png   ← q-sample на різних t
    ├── schedules.png           ← βₜ та α̅ₜ для linear/cosine
    ├── loss_linear.png         ← криві навчання
    ├── loss_cosine.png
    ├── reverse_linear.png      ← послідовна реверс-дифузія xₜ→x₀
    ├── reverse_cosine.png
    ├── samples_ddpm_linear.png    ← 16 зразків (DDPM, 200 steps)
    ├── samples_ddpm_cosine.png
    ├── samples_ddim200_linear.png ← 16 зразків (DDIM η=0, 200 steps)
    ├── samples_ddim50_linear.png  ← 16 зразків (DDIM η=0, 50 steps)
    ├── samples_ddim200_cosine.png
    ├── samples_ddim50_cosine.png
    └── outputs.json            ← всі сирі метрики
```

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Повний пайплайн (тренує 2 моделі × 4 епохи + sampling). RTX 5060 Ti — ~10 хв
python run_lab2.py

# Генерація звіту з outputs/outputs.json та збережених PNG
python generate_report.py
```

## Конфігурація

| Параметр | Значення |
|---|---|
| Датасет | MNIST 32×32 (60 000 train), [-1, 1] |
| U-Net base channels | 64, channel_mults=(1, 2, 4) |
| Глибина | 3 рівні: 32 → 16 → 8 (bottleneck) |
| ResBlocks per level | 1 |
| Self-attention | тільки на 8×8 bottleneck |
| Time embedding | sinusoidal → MLP (256-dim) |
| Diffusion T | 200 |
| Schedules | linear (Ho 2020) + cosine (Nichol 2021) |
| Optimizer | Adam, lr=2e-4 |
| Batch size | 128 |
| Epochs | 4 |
| EMA decay | 0.999 |
| Sampling | DDPM 200 steps, DDIM η=0 з 200 та 50 кроками |

## Результати

| Метрика | linear | cosine |
|---|---|---|
| init epoch loss | 0.082 | 0.072 |
| final epoch loss | 0.037 | 0.032 |
| час/епоха | ~61 s | ~61 s |
| DDPM-200 sample (16 imgs) | 1.26 s | 1.20 s |
| DDIM-200 sample (16 imgs) | 1.25 s | 1.28 s |
| DDIM-50 sample (16 imgs) | 0.32 s | 0.33 s |

**Speed-up DDIM-50 vs DDPM-200 ≈ 4×** при майже непомітній втраті якості.
