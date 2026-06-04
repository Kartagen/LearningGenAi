"""Utility functions: reproducibility, templates."""
import os
import random
import numpy as np
import torch
from transformers import set_seed as hf_set_seed


# Chat templates for LFM2
TEMPLATE_WITHOUT_ANSWER = (
    "<|startoftext|><|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
)
TEMPLATE_WITH_ANSWER = TEMPLATE_WITHOUT_ANSWER + "{answer}<|im_end|>\n"


def set_global_seed(seed: int = 42) -> None:
    """Fix all randomness sources for reproducibility."""
    hf_set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def format_question(question: str) -> str:
    return TEMPLATE_WITHOUT_ANSWER.format(question=question)


def format_qa(question: str, answer: str) -> str:
    return TEMPLATE_WITH_ANSWER.format(question=question, answer=answer)
