"""Model loading, LoRA application, save/load adapters."""
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel


MODEL_ID = "LiquidAI/LFM2-1.2B"


def load_base_model_and_tokenizer(model_id: str = MODEL_ID) -> Tuple[torch.nn.Module, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    return model, tokenizer


def apply_lora(
    model: torch.nn.Module,
    r: int = 8,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.0,
    target_modules: list | None = None,
) -> torch.nn.Module:
    """Wrap base model with a LoRA adapter."""
    if lora_alpha is None:
        lora_alpha = 2 * r
    if target_modules is None:
        target_modules = [
            "q_proj", "o_proj", "k_proj", "v_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return get_peft_model(model, lora_config)


def count_trainable_params(model: torch.nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "percent": trainable / total * 100,
    }


def save_lora(model: torch.nn.Module, path: str) -> None:
    model.save_pretrained(path)


def load_lora(base_model: torch.nn.Module, adapter_path: str) -> torch.nn.Module:
    return PeftModel.from_pretrained(base_model, adapter_path)
