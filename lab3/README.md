# Лабораторна робота 3 — Stable Diffusion pipelines + DreamBooth LoRA

Чотири pipeline Stable Diffusion v1.5 (Text-to-Image, Img2Img, Inpainting,
Depth2Img) у бібліотеці Hugging Face Diffusers + персоналізація моделі через
DreamBooth LoRA на власному датасеті з 20 augmented-зображень одного об'єкта.

Ноутбуки адаптовано з оригінальних Kaggle-версій (див. `reports/lab3/adapt_notebooks.py`)
до локального запуску у Windows з NVIDIA GPU.

## Структура

```
lab3/
├── run_lab.py                  # headless orchestrator для повного прогону
├── requirements.txt
├── notebooks/
│   ├── 01_stable_diffusion_pipelines.ipynb
│   └── 02_dreambooth_lora.ipynb
├── data/
│   └── dog_example_augmented_20/   # 20 instance images + metadata.jsonl
├── outputs/
│   ├── pipelines/              # text2img, guidance grid, img2img, inpaint, depth
│   │   └── metadata.json
│   └── dreambooth/             # baseline, lora, comparison.jpg
│       └── results.json
└── work/                       # HF cache, train_dreambooth_lora.py, LoRA weights, logs
```

## Залежності

- Python 3.10+ (тестовано на 3.14)
- NVIDIA GPU >= 8 GB VRAM (16 GB рекомендовано для DreamBooth)
- CUDA-збірка PyTorch (встановити окремо з https://pytorch.org/get-started/locally/)
- ~15 GB вільного місця на диску (моделі SD-1.5 + inpainting + depth)

## Встановлення

```powershell
cd D:\LearningMagister\Semester2\GenAIandItsUsage\lab3

python -m venv .venv
.\.venv\Scripts\Activate.ps1

# CUDA-torch встановити ПЕРШИМ
pip install torch --index-url https://download.pytorch.org/whl/cu121

# усе інше
pip install -r requirements.txt
```

Опційно (для gated моделей): `$env:HF_TOKEN = "hf_xxx"` або `huggingface-cli login`.
Для цієї лаби всі моделі публічні -- токен не обов'язковий.

## Запуск

```bash
# Варіант A -- через Jupyter (рекомендовано для покрокового вивчення)
jupyter lab notebooks/

# Варіант B -- автоматичний прогін усіх експериментів
python run_lab.py --all                       # ~60 хв + ~15 GB downloads
python run_lab.py --pipelines                 # тільки notebook 1
python run_lab.py --dreambooth                # тільки notebook 2
python run_lab.py --pipelines --no-inpaint --no-depth   # швидкий subset
```

Прапори `--no-inpaint` і `--no-depth` пропускають завантаження ~10 GB ваг
inpainting / depth моделей; решта Stage 1 (text2img + guidance + img2img)
залишається.

## Конфігурація

### Notebook 1 -- Stable Diffusion pipelines

| Параметр | Значення |
|---|---:|
| Базова модель | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Тип ваг | float16 |
| Resolution | 512x512 |
| Inference steps | 30 |
| Guidance scale (експеримент) | 2.0, 8.0, 12.0 |
| Img2Img strength (експеримент) | 0.25, 0.55, 0.85 |
| Inpaint model | `stable-diffusion-v1-5/stable-diffusion-inpainting` |
| Depth model | `sd2-community/stable-diffusion-2-depth` |
| Seed | 42 / 123 |

### Notebook 2 -- DreamBooth LoRA

| Параметр | Значення |
|---|---:|
| Instance images | 20 (`data/dog_example_augmented_20/images/`) |
| UNIQUE_TOKEN / CLASS_NOUN | `sks` / `puppy` |
| Instance prompt | `a photo of sks puppy` |
| Resolution | 512 |
| Train batch size | 1 |
| Gradient accumulation | 1 |
| Learning rate | 5e-5 |
| Max train steps | 200 |
| LoRA rank | 4 |
| Mixed precision | no |
| 8-bit Adam | False (bitsandbytes не працює на Windows out-of-the-box) |
| Gradient checkpointing | True |
| Prior preservation | False (для базового рівня) |

## Що реалізовано

| Завдання | Локація |
|---|---|
| Text-to-Image базова генерація | notebook 1, cell-21 |
| Експеримент із `guidance_scale` (3 варіанти) | notebook 1, cell-23 |
| Розбір компонентів pipeline (VAE/tokenizer/text_encoder/UNet/scheduler) | notebook 1, cells 24-43 |
| VAE encode/decode + scaling_factor | notebook 1, cells 26-29 |
| DIY sampling loop із classifier-free guidance | notebook 1, cells 44-47 |
| Img2Img із 3 значеннями `strength` | notebook 1, cells 48-55 |
| Inpainting (mask + prompt) | notebook 1, cells 56-59 |
| Depth2Img (MiDaS depth conditioning) | notebook 1, cells 60-61 |
| Підготовка instance-датасету | notebook 2, cells 16-18 |
| Baseline-генерація (до тренування) | notebook 2, cells 24-27 |
| DreamBooth LoRA training (200 steps) | notebook 2, cells 28-38 |
| Prior preservation (теорія) | notebook 2, cells 32-33 |
| Inference з LoRA-вагами | notebook 2, cells 40-41 |
| Comparison grid base vs LoRA | notebook 2, cells 42-44 |

## Результати (реальний прогін)

| Етап | Час |
|---|---:|
| SD-1.5 download | ~25 хв (5 GB) |
| Stage 1.1 text2img + guidance grid | ~25 с |
| Stage 1.2 Img2Img x 3 strengths | ~7 с |
| Stage 1.3 inpaint download + gen | 5.5 хв + 5.4 с |
| Stage 1.4 depth download + gen | 25 хв + 2.5 с |
| Stage 2 DreamBooth повний цикл | 232 с (3.9 хв), training сам -- 181 с |
| **Загалом** | **~60 хв** |

Виходи (`outputs/`):

- `pipelines/text2img_basic.png`, `guidance_cfg{2,8,12}.png`
- `pipelines/img2img_strength{0.25,0.55,0.85}.png`
- `pipelines/inpainting_result.png`, `pipelines/depth2img_result.png`
- `pipelines/metadata.json`
- `dreambooth/baseline_{01..04}.png`, `dreambooth/lora_{01..04}.png`
- `dreambooth/comparison.jpg`, `dreambooth/results.json`

## Адаптація з Kaggle

Утиліта `reports/lab3/adapt_notebooks.py` конвертує оригінальні Kaggle-ноутбуки
зі `src/` (git-ignored) у локальні версії у `lab3/notebooks/`:

| Kaggle оригінал | Локальна заміна |
|---|---|
| `/kaggle/working` | `lab3/work` (через env `LAB3_WORK_DIR`) |
| `/kaggle/input` | `lab3/data` (через env `LAB3_DATA_DIR`) |
| `kaggle_secrets.UserSecretsClient` | `os.environ['HF_TOKEN']` |
| `%%bash pip install ...` | `%pip install ...` |
| `!nvidia-smi` | `subprocess + shutil.which` guard |
| `!find ... | sort` | `Path.rglob(...) + sort` |
| `!python {script} --help` | `subprocess.run([sys.executable, ...])` |

Перегенерувати ноутбуки після оновлення оригіналів:

```bash
python ../reports/lab3/adapt_notebooks.py
```

## Windows-нюанси

- **xformers** часто не ставиться на Windows -- ноутбуки обгортають
  `enable_xformers_memory_efficient_attention()` у try/except, тож відсутність
  xformers не критична (просто трохи повільніше).
- **bitsandbytes** (потрібен для `--use_8bit_adam`) не підтримується на Windows
  out-of-the-box -- `USE_8BIT_ADAM = False`, прапор пропускається умовно.
- **CUDA OOM**: зменшити `RESOLUTION` до 384, `MAX_TRAIN_STEPS` до 100-200,
  або вимкнути inpainting/depth прапорами `--no-inpaint --no-depth`.
- **Symlinks**: HF Hub попереджає, що на Windows без Developer Mode кеш
  займає удвічі більше місця (blobs + snapshots копії). Це нормально.
