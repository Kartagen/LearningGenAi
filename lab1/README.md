# Лабораторна робота 1 — LLM Fine-tuning to Yoda speak

Fine-tuning LFM2-1.2B з LoRA-адаптером під «Yoda speak», оцінка через
LLM-as-a-judge (OpenRouter Qwen3) і власні soft-метрики стилю.

## Структура

```
lab1/
├── notebook.ipynb              # головна точка входу — запускати тут
├── run_lab1.py                 # альтернативний end-to-end скрипт
├── requirements.txt
├── .env.example                # шаблон для API-ключів
├── src/
│   ├── utils.py                # set_global_seed, chat templates
│   ├── data.py                 # create_yoda_dataloaders (70/15/15, batch>10)
│   ├── model.py                # load_base_model, apply_lora, save/load adapter
│   ├── training.py             # train_step, train, validation_loss, chat
│   ├── judge.py                # LLMClient, LLMJudgeEvaluator, 2 system prompts
│   ├── metrics.py              # soft Recall / Specificity / BalancedAccuracy / StyleGain
│   └── evaluation.py           # generate_samples, evaluate_model, yoda_loglikelihood
├── checkpoints/                # ваги LoRA після навчання
└── outputs/                    # графіки, CSV, артефакти
```

## Залежності

- Python 3.10+
- NVIDIA GPU >= 8 GB VRAM (тренування 1 епохи з `batch_size=16` ~ 10-25 хв на T4 / RTX 30xx)
- CUDA-збірка PyTorch

## Встановлення

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## API-ключі

- **Comet Opik**: https://www.comet.com/signup -- API-ключ + workspace
- **OpenRouter**: https://openrouter.ai -- API-ключ для безкоштовної моделі
  `qwen/qwen3-next-80b-a3b-instruct` (ліміт 20 req/min, 50/день)

Скопіювати `.env.example` -> `.env` і заповнити:

```bash
cp .env.example .env                # Windows: copy .env.example .env
```

## Запуск

```bash
# Варіант A -- через ноутбук (рекомендовано)
jupyter notebook notebook.ipynb

# Варіант B -- через end-to-end скрипт
python run_lab1.py
```

## Конфігурація

| Параметр | Значення |
|---|---:|
| Базова модель | LFM2-1.2B |
| Train / val / test split | 70 / 15 / 15 |
| Batch size | 16 |
| LoRA `r / alpha / dropout` | задається у `src/model.py::apply_lora` |
| Eval prompts | 20 (`N_EVAL`) |
| Sweep по `lr x rank x ctx_length` | секція 8 ноутбука |
| Multi-seed averaging | секція 9 ноутбука |

## Що реалізовано

| Вимога з лабораторної | Файл / комірка |
|---|---|
| Прибрано Leprechaun | вся логіка ноутбука працює тільки з Yoda |
| Фіксація seed + повторюваність | `src/utils.py::set_global_seed`, секція multi-seed |
| Train/val/test 70/15/15, batch_size = 16 | `src/data.py::create_yoda_dataloaders` |
| Power validation loss | `src/training.py::validation_loss` |
| LoRA `r/alpha/dropout` параметри | `src/model.py::apply_lora` |
| Завантаження/збереження ваг адаптера | `src/model.py::save_lora`, `load_lora` |
| Графік loss-кривих | секція 'Loss curve' |
| Inference до/після fine-tuning | секція 5 |
| LLM-as-a-judge через OpenRouter | `src/judge.py` |
| Два варіанти system prompt | `SYSTEM_PROMPT_BASIC`, `SYSTEM_PROMPT_DETAILED` |
| Soft Recall / BalancedAccuracy / StyleGain | `src/metrics.py::compute_style_metrics` |
| Розширене тестування на >= 20 prompts | секція 7, `N_EVAL=20` |
| Sweep по `lr x rank x ctx_length` | секція 8 |
| Multi-seed повтори | секція 9 |
| Yoda test log-likelihood (cell-70) | секція 10 (не модифікується) |

## Рівні складності

- **Задовільно**: дефолтний прогін секцій 1-7 + 10.
- **Добре**: + `RUN_SWEEP = True` (секція 8) і `SYSTEM_PROMPT_DETAILED` у
  `make_yoda_judge(..., detailed=True)`.
- **Відмінно**: + `RUN_SEEDS = True` (секція 9), більше епох, ширший набір
  конфігурацій у `SWEEP_CONFIGS`, збережені найкращі ваги LoRA з `checkpoints/`.

## Результати

Після виконання у `outputs/` зберігаються:

- `loss_curve.png` -- графік train/val loss
- `inference_before_after.csv` -- таблиця відповідей до/після fine-tuning
- `score_histogram.png`, `score_bars.png` -- розподіли судді
- `scores.csv` -- сирі бали
- `sweep_results.csv` -- підсумкова таблиця по конфігураціям
- `seed_results.csv` -- статистика по seed-ам
- `outputs.json` -- усі сирі метрики

У `checkpoints/` зберігаються LoRA-адаптери.

Підсумкова метрика змагання -- **Yoda test log-likelihood** (комірка з
`# DO NOT CHANGE / MODIFY THIS CELL.`). Чим менше, тим краще.
