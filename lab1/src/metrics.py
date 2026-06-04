"""Soft confusion-matrix metrics: Recall, Specificity, BalancedAccuracy, StyleGain."""
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class StyleMetrics:
    soft_tp: float
    soft_fn: float
    soft_fp: float
    soft_tn: float
    recall: float
    specificity: float
    balanced_accuracy: float
    style_gain: float
    mean_positive: float
    mean_negative: float
    std_positive: float
    std_negative: float


def compute_style_metrics(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> StyleMetrics:
    """Compute soft-confusion-matrix-based metrics.

    Positive (P) — texts that *should* be in the target style
                   (i.e. outputs of the fine-tuned model, or ground-truth Yoda
                   responses from the dataset).
    Negative (N) — texts that should NOT be in the target style
                   (base/standard English responses).

    Scores must already be normalized to [0, 1].
    """
    pos = np.asarray(positive_scores, dtype=float)
    neg = np.asarray(negative_scores, dtype=float)

    tp = float(pos.sum())
    fn = float((1 - pos).sum())
    fp = float(neg.sum())
    tn = float((1 - neg).sum())

    recall = float(pos.mean()) if len(pos) else float("nan")
    specificity = float((1 - neg).mean()) if len(neg) else float("nan")
    balanced_acc = 0.5 * (recall + specificity)
    style_gain = recall - (float(neg.mean()) if len(neg) else 0.0)

    return StyleMetrics(
        soft_tp=tp,
        soft_fn=fn,
        soft_fp=fp,
        soft_tn=tn,
        recall=recall,
        specificity=specificity,
        balanced_accuracy=balanced_acc,
        style_gain=style_gain,
        mean_positive=float(pos.mean()) if len(pos) else float("nan"),
        mean_negative=float(neg.mean()) if len(neg) else float("nan"),
        std_positive=float(pos.std()) if len(pos) else float("nan"),
        std_negative=float(neg.std()) if len(neg) else float("nan"),
    )


def format_metrics(m: StyleMetrics) -> str:
    return (
        f"Recall                : {m.recall:.3f}\n"
        f"Specificity           : {m.specificity:.3f}\n"
        f"Balanced Accuracy     : {m.balanced_accuracy:.3f}\n"
        f"Style Gain            : {m.style_gain:.3f}\n"
        f"Mean +/- std (pos)    : {m.mean_positive:.3f} +/- {m.std_positive:.3f}\n"
        f"Mean +/- std (neg)    : {m.mean_negative:.3f} +/- {m.std_negative:.3f}\n"
        f"Soft TP/FN/FP/TN      : {m.soft_tp:.2f} / {m.soft_fn:.2f} / "
        f"{m.soft_fp:.2f} / {m.soft_tn:.2f}"
    )
