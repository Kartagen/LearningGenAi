"""End-to-end Lab 1 pipeline.

Runs the full Yoda fine-tuning experiment and saves all artefacts
(graphs, JSON, CSV) into ``outputs/``. The DOCX report is generated separately
from those artefacts.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv(ROOT / ".env")

from src.utils import set_global_seed, TEMPLATE_WITH_ANSWER, format_question  # noqa
from src.data import create_yoda_dataloaders, get_base_and_style_samples  # noqa
from src.model import (  # noqa
    load_base_model_and_tokenizer,
    apply_lora,
    count_trainable_params,
    save_lora,
)
from src.training import chat, train_step, validation_loss  # noqa
from src.judge import LLMClient, LLMJudgeEvaluator, SYSTEM_PROMPT_BASIC, YODA_EXAMPLE  # noqa
from src.metrics import compute_style_metrics, format_metrics  # noqa
from src.evaluation import yoda_loglikelihood  # noqa

from lion_pytorch import Lion
from tqdm.auto import tqdm
from torch.nn import functional as F


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 16
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
CONTEXT_LENGTH = 512
MAX_STEPS = 500           # ~500 batches of 16 samples ~= 8000 examples
PREVIEW_EVERY = 50
VAL_EVERY = 100
N_EVAL = 10               # samples per category (base / generated / style)
MAX_NEW_TOKENS_EVAL = 48
TEMPERATURE_EVAL = 0.3

OUT_DIR = ROOT / "outputs"
CKPT_DIR = ROOT / "checkpoints" / "yoda-lora-r8-lr1e-4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.parent.mkdir(parents=True, exist_ok=True)

YODA_TEST_TEXT = (
    "Wisdom, sought by many, found by few, it is. Haste not, patience have. "
    "For in stillness, answers come. Much to learn, still you have. "
    "Fear leads to anger; anger, to hate. Down the dark path, guide you it will. "
    "Trust the Force, you must. Powerful ally it is. Life it creates, surrounds, binds. "
    "Adventure, excitement, a Jedi craves not these things. Discipline, balance, seek you should. "
    "Hmm, clearer now is the path, yes? Help you more, I can, if needed it is. "
    "Endless, the journey of learning is. Stay true to your path, and clarity you will find. "
    "Remember, the Force flows through all, but your heart determines how it shapes your destiny. "
    "Much more to teach, I have. Ready, are you? Mmm."
)

BASELINE_QUESTIONS = [
    "What is the capital of France?",
    "How do I train a neural network?",
    "Tell me a short story about courage.",
    "What is the meaning of patience?",
    "Explain photosynthesis to a child.",
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
@contextmanager
def timer(name: str):
    print(f"\n>>> {name}")
    t0 = time.time()
    yield
    print(f"<<< {name}  ({time.time() - t0:.1f}s)")


def log_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


# ----------------------------------------------------------------------
# Stage 1: setup
# ----------------------------------------------------------------------
log_section("Stage 1. Setup & data")

set_global_seed(SEED)
print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

with timer("create train/val/test dataloaders 70/15/15"):
    train_loader, val_loader, test_loader = create_yoda_dataloaders(
        batch_size=BATCH_SIZE,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=SEED,
    )

n_train = len(train_loader.dataset)
n_val = len(val_loader.dataset)
n_test = len(test_loader.dataset)
print(f"train batches: {len(train_loader)} (n={n_train}, batch={BATCH_SIZE})")
print(f"val samples : {n_val}")
print(f"test samples: {n_test}")

sample = train_loader.dataset[0]
print("--- sample ---")
print("instruction   :", sample["instruction"][:200])
print("response_base :", sample["response"][:200])
print("response_yoda :", sample["response_style"][:200])


# ----------------------------------------------------------------------
# Stage 2: base model & baseline inference
# ----------------------------------------------------------------------
log_section("Stage 2. Base model + baseline inference")

with timer("load LFM2-1.2B + tokenizer"):
    model, tokenizer = load_base_model_and_tokenizer()

baseline_answers = []
for q in BASELINE_QUESTIONS:
    with torch.no_grad():
        ans = chat(model, tokenizer, q, max_new_tokens=60, temperature=0.3, only_answer=True)["answer"]
    baseline_answers.append(ans)
    print(f"Q: {q}\nA: {ans}\n")

baseline_loglik = yoda_loglikelihood(model, tokenizer, YODA_TEST_TEXT)
print(f"baseline yoda_loglikelihood = {baseline_loglik:.4f}")


# ----------------------------------------------------------------------
# Stage 3: LoRA
# ----------------------------------------------------------------------
log_section("Stage 3. Apply LoRA")

model = apply_lora(model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)
stats = count_trainable_params(model)
print(json.dumps(stats, indent=2))


# ----------------------------------------------------------------------
# Stage 4: fine-tune
# ----------------------------------------------------------------------
log_section("Stage 4. Fine-tuning")

set_global_seed(SEED)

optimizer = Lion(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
model.train()

history = {"step": [], "train_loss": [], "val_step": [], "val_loss": [], "preview": []}
global_step = 0
rolling = []

stop_training = False
with timer("train loop"):
    while not stop_training:
        for batch in train_loader:
            loss_value = train_step(model, optimizer, batch, tokenizer, context_length=CONTEXT_LENGTH)
            rolling.append(loss_value)
            history["step"].append(global_step)
            history["train_loss"].append(loss_value)

            if global_step % PREVIEW_EVERY == 0:
                with torch.no_grad():
                    p = chat(model, tokenizer, "What is the capital of France?",
                             max_new_tokens=24, only_answer=True)["answer"]
                history["preview"].append({"step": global_step, "text": p})
                avg = float(np.mean(rolling))
                rolling = []
                print(f"  step {global_step:3d}  loss={avg:.4f}  preview={p!r}")

            if global_step > 0 and global_step % VAL_EVERY == 0:
                vloss = validation_loss(model, val_loader, tokenizer, CONTEXT_LENGTH)
                history["val_step"].append(global_step)
                history["val_loss"].append(vloss)
                print(f"  step {global_step:3d}  val_loss={vloss:.4f}")

            global_step += 1
            if global_step >= MAX_STEPS:
                stop_training = True
                break

final_val = validation_loss(model, val_loader, tokenizer, CONTEXT_LENGTH)
history["val_step"].append(global_step)
history["val_loss"].append(final_val)
print(f"final val_loss = {final_val:.4f}")

with timer("save LoRA adapter"):
    save_lora(model, str(CKPT_DIR))

# Loss curve
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(history["step"], history["train_loss"], label="train", alpha=0.6)
if history["val_step"]:
    ax.plot(history["val_step"], history["val_loss"], "o-", color="red", label="val")
ax.set_xlabel("step")
ax.set_ylabel("cross-entropy")
ax.set_title("Fine-tuning loss")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "loss_curve.png", dpi=120)
plt.close()


# ----------------------------------------------------------------------
# Stage 5: inference AFTER fine-tuning
# ----------------------------------------------------------------------
log_section("Stage 5. Inference AFTER fine-tuning")

tuned_answers = []
for q in BASELINE_QUESTIONS:
    with torch.no_grad():
        ans = chat(model, tokenizer, q, max_new_tokens=60, temperature=0.3, only_answer=True)["answer"]
    tuned_answers.append(ans)
    print(f"Q: {q}\nA: {ans}\n")

compare_df = pd.DataFrame({
    "question": BASELINE_QUESTIONS,
    "base_model": baseline_answers,
    "yoda_model": tuned_answers,
})
compare_df.to_csv(OUT_DIR / "inference_before_after.csv", index=False)


# ----------------------------------------------------------------------
# Stage 6: LLM judge & evaluation
# ----------------------------------------------------------------------
log_section("Stage 6. LLM judge & evaluation")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen/qwen3-next-80b-a3b-instruct")
assert OPENROUTER_API_KEY, "OPENROUTER_API_KEY not set"

system_prompt = SYSTEM_PROMPT_BASIC.format(style="Yoda", example=YODA_EXAMPLE)
judge = LLMJudgeEvaluator(
    LLMClient(model=JUDGE_MODEL, api_key=OPENROUTER_API_KEY),
    system_prompt=system_prompt,
)

# Vibe check ------------------------------------------------------------------
probes = [
    "Tennis is a fun sport. But you must concentrate.",
    "Fun sport, tennis is. But work hard, you must.",
    "Hard to see, the dark side is.",
]
probe_scores = []
for t in probes:
    s = judge.score(t).value
    probe_scores.append((t, s))
    print(f"{s:.2f}  <-  {t}")

# Hold-out samples ------------------------------------------------------------
base_samples, style_samples = get_base_and_style_samples(test_loader, n_samples=N_EVAL)

# Generate from fine-tuned model ----------------------------------------------
generated = []
from itertools import islice
for sample in tqdm(list(islice(test_loader, N_EVAL)), desc="generating"):
    q = sample["instruction"][0]
    with torch.no_grad():
        g = chat(model, tokenizer, q,
                 only_answer=True,
                 max_new_tokens=MAX_NEW_TOKENS_EVAL,
                 temperature=TEMPERATURE_EVAL)["answer"]
    generated.append(g)

# Score all three groups ------------------------------------------------------
def score_list(texts, desc):
    out = []
    for t in tqdm(texts, desc=desc):
        out.append(judge.score(t).value)
    return out

base_scores = score_list(base_samples, "score base")
gen_scores = score_list(generated, "score generated")
style_scores = score_list(style_samples, "score style")

model_metrics = compute_style_metrics(gen_scores, base_scores)
ceiling_metrics = compute_style_metrics(style_scores, base_scores)

print("\n--- model (generated) vs base ---")
print(format_metrics(model_metrics))
print("\n--- ceiling (ground-truth Yoda) vs base ---")
print(format_metrics(ceiling_metrics))


# ----------------------------------------------------------------------
# Stage 7: plots
# ----------------------------------------------------------------------
log_section("Stage 7. Plots & saved artefacts")

scores_df = pd.DataFrame({
    "Score": [*base_scores, *gen_scores, *style_scores],
    "Type":  (["Base"] * len(base_scores)
            + ["Generated"] * len(gen_scores)
            + ["Style"] * len(style_scores)),
})
scores_df.to_csv(OUT_DIR / "scores.csv", index=False)

plt.figure(figsize=(8, 5))
sns.histplot(data=scores_df, x="Score", hue="Type", multiple="dodge", stat="probability", bins=6)
plt.xlabel("Judge score")
plt.ylabel("Probability")
plt.title("Distribution of judge scores")
plt.tight_layout()
plt.savefig(OUT_DIR / "score_histogram.png", dpi=120)
plt.close()

labels = ["Base", "Generated", "Style"]
means = [float(np.mean(base_scores)), float(np.mean(gen_scores)), float(np.mean(style_scores))]
stds  = [float(np.std(base_scores)),  float(np.std(gen_scores)),  float(np.std(style_scores))]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, means, yerr=stds, capsize=6)
plt.ylim(0, 1)
plt.ylabel("Judge score")
plt.title("Mean judge score by text type")
for bar, value in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig(OUT_DIR / "score_bars.png", dpi=120)
plt.close()


# ----------------------------------------------------------------------
# Stage 8: final competition metric
# ----------------------------------------------------------------------
log_section("Stage 8. Yoda test loglikelihood (competition metric)")

tuned_loglik = yoda_loglikelihood(model, tokenizer, YODA_TEST_TEXT)
print(f"BASELINE yoda_loglikelihood : {baseline_loglik:.4f}")
print(f"FINE-TUNED yoda_loglikelihood: {tuned_loglik:.4f}")
print(f"absolute improvement        : {baseline_loglik - tuned_loglik:+.4f}")


# ----------------------------------------------------------------------
# Save consolidated outputs.json
# ----------------------------------------------------------------------
outputs = {
    "config": {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "learning_rate": LEARNING_RATE,
        "context_length": CONTEXT_LENGTH,
        "max_steps": MAX_STEPS,
        "n_eval": N_EVAL,
        "judge_model": JUDGE_MODEL,
        "base_model_id": "LiquidAI/LFM2-1.2B",
    },
    "data": {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    },
    "trainable": stats,
    "baseline_inference": [
        {"question": q, "answer": a} for q, a in zip(BASELINE_QUESTIONS, baseline_answers)
    ],
    "tuned_inference": [
        {"question": q, "answer": a} for q, a in zip(BASELINE_QUESTIONS, tuned_answers)
    ],
    "history": history,
    "probes": [{"text": t, "score": s} for t, s in probe_scores],
    "samples": {
        "base": base_samples,
        "generated": generated,
        "style": style_samples,
    },
    "scores": {
        "base": base_scores,
        "generated": gen_scores,
        "style": style_scores,
    },
    "metrics": {
        "model_vs_base": model_metrics.__dict__,
        "ceiling_vs_base": ceiling_metrics.__dict__,
    },
    "yoda_loglikelihood": {
        "baseline": baseline_loglik,
        "fine_tuned": tuned_loglik,
        "improvement": baseline_loglik - tuned_loglik,
    },
}

with open(OUT_DIR / "outputs.json", "w", encoding="utf-8") as f:
    json.dump(outputs, f, default=jsonable, ensure_ascii=False, indent=2)

print(f"\nAll artefacts saved to {OUT_DIR}")
