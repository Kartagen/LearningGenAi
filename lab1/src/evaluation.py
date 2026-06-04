"""Evaluation helpers: generate samples, score with judge, run hyperparameter sweeps."""
from itertools import islice
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .judge import LLMJudgeEvaluator
from .metrics import compute_style_metrics, StyleMetrics
from .training import chat


def generate_samples(model, tokenizer, test_loader, n_samples: int = 20, max_new_tokens: int = 48, temperature: float = 0.3) -> list:
    samples = []
    questions = []
    for sample in tqdm(islice(test_loader, n_samples), total=n_samples, desc="Generating"):
        question = sample["instruction"][0]
        with torch.no_grad():
            generated = chat(
                model,
                tokenizer,
                question,
                only_answer=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )["answer"]
        questions.append(question)
        samples.append(generated)
    return questions, samples


def score_texts(judge: LLMJudgeEvaluator, texts: Iterable[str], desc: str = "Scoring") -> list:
    scores = []
    for text in tqdm(list(texts), desc=desc):
        score = judge.score(text).value
        scores.append(float(score))
    return scores


def evaluate_model(
    model,
    tokenizer,
    judge: LLMJudgeEvaluator,
    test_loader,
    base_samples: list,
    style_samples: list,
    n_samples: int = 20,
    max_new_tokens: int = 48,
    temperature: float = 0.3,
) -> dict:
    """Full evaluation: generate, score base/generated/style, compute metrics."""
    questions, generated = generate_samples(
        model, tokenizer, test_loader,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    base_scores = score_texts(judge, base_samples, desc="Score base")
    gen_scores = score_texts(judge, generated, desc="Score generated")
    style_scores = score_texts(judge, style_samples, desc="Score style")

    # Two views:
    #   1) Model vs base   (P = generated, N = base)
    #   2) Style vs base   (P = style,     N = base) — reference ceiling
    model_metrics = compute_style_metrics(gen_scores, base_scores)
    ceiling_metrics = compute_style_metrics(style_scores, base_scores)

    return {
        "questions": questions,
        "generated": generated,
        "base_scores": base_scores,
        "generated_scores": gen_scores,
        "style_scores": style_scores,
        "model_metrics": model_metrics,
        "ceiling_metrics": ceiling_metrics,
    }


@torch.no_grad()
def yoda_loglikelihood(model, tokenizer, yoda_test_text: str) -> float:
    """Compute the cross-entropy loss on a fixed held-out Yoda paragraph.

    This is the competition metric: lower is better.
    """
    tokens = tokenizer(yoda_test_text, return_tensors="pt").to(model.device)
    outputs = model(**tokens)
    logits = outputs.logits[:, :-1]
    targets = tokens.input_ids[:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )
    return float(loss.item())
