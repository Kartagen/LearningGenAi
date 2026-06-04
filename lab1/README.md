# Lab 1 — LLM Fine-tuning to Yoda speak

Готова реалізація лабораторної №1 (LFM2-1.2B + LoRA → Yoda).

## Структура

```
lab1/
├── notebook.ipynb          # головний ноутбук — запускати тут
├── requirements.txt        # залежності
├── .env.example            # шаблон для API-ключів
├── src/
│   ├── utils.py            # set_global_seed, chat templates
│   ├── data.py             # create_yoda_dataloaders (70/15/15, batch>10)
│   ├── model.py            # load_base_model, apply_lora, save/load adapter
│   ├── training.py         # train_step, train, validation_loss, chat
│   ├── judge.py            # LLMClient, LLMJudgeEvaluator, 2 system prompts
│   ├── metrics.py          # soft Recall / Specificity / BalancedAccuracy / StyleGain
│   └── evaluation.py       # generate_samples, evaluate_model, yoda_loglikelihood
├── checkpoints/            # ваги LoRA після навчання
└── outputs/                # графіки, CSV, артефакти
```

## Запуск

### 1. Створити віртуальне середовище й встановити залежності
```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Зареєструватися та отримати API-ключі
- **Comet Opik**: https://www.comet.com/signup — згенерувати API-ключ + назву workspace.
- **OpenRouter**: https://openrouter.ai — згенерувати API-ключ для безкоштовної моделі `qwen/qwen3-next-80b-a3b-instruct` (ліміт 20 req/min, 50/день).

Скопіювати `.env.example` → `.env` і заповнити:
```bash
cp .env.example .env
# Windows:
# copy .env.example .env
```

### 3. Запустити ноутбук
```bash
jupyter notebook notebook.ipynb
```
…і виконувати клітинки зверху вниз. Для **GPU** (>= 8 GB) тренування 1 епохи з `batch_size=16` займає ~10–25 хвилин на T4/RTX 30xx.

## Що реалізовано

| Вимога з лабораторної | Файл / комірка |
|---|---|
| Прибрано Leprechaun | вся логіка ноутбуку працює тільки з Yoda |
| Фіксація seed + повторюваність | `src/utils.py::set_global_seed`, секція «Multi-seed averaging» |
| Train/val/test 70/15/15, batch_size = 16 | `src/data.py::create_yoda_dataloaders` |
| Power validation loss | `src/training.py::validation_loss` |
| LoRA `r/alpha/dropout` параметри | `src/model.py::apply_lora` |
| Завантаження/збереження ваг адаптера | `src/model.py::save_lora`, `load_lora` |
| Графік loss-кривих | секція **«Loss curve»** |
| Inference до/після fine-tuning | секція **5** |
| LLM-as-a-judge через OpenRouter | `src/judge.py` |
| Два варіанти system prompt | `SYSTEM_PROMPT_BASIC`, `SYSTEM_PROMPT_DETAILED` |
| Soft Recall / BalancedAccuracy / StyleGain | `src/metrics.py::compute_style_metrics` |
| Розширене тестування на ≥20 prompts | секція **7**, `N_EVAL=20` |
| Sweep по `lr × rank × ctx_length` | секція **8** |
| Multi-seed повтори | секція **9** |
| Yoda test log-likelihood (cell-70) | секція **10** (не модифікується) |

## Рівні «задов / добре / відмінно»

- **«Задовільно»**: дефолтний прогін секцій 1–7 + 10.
- **«Добре»**: додатково увімкнути `RUN_SWEEP = True` (секція 8) та `SYSTEM_PROMPT_DETAILED` у `make_yoda_judge(..., detailed=True)`.
- **«Відмінно»**: + `RUN_SEEDS = True` (секція 9), більше епох, ширший набір конфігурацій у `SWEEP_CONFIGS`, а також збережені найкращі ваги LoRA з `checkpoints/`.

## Файли результатів

Після виконання ноутбуку в `outputs/` з'являться:
- `loss_curve.png` — графік train/val loss;
- `inference_before_after.csv` — таблиця відповідей до/після fine-tuning;
- `score_histogram.png`, `score_bars.png` — розподіли судді;
- `scores.csv` — сирі бали;
- `sweep_results.csv` — підсумкова таблиця по конфігураціям;
- `seed_results.csv` — статистика по seed-ам.

А в `checkpoints/` — папки з LoRA-адаптерами.

## Підсумкова метрика

Останній рядок ноутбуку (комірка з `# DO NOT CHANGE / MODIFY THIS CELL.`) виводить **Yoda test loglikelihood** — це й є метрика змагання. Чим менше — тим краще.
