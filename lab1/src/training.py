"""Training loop with LoRA fine-tuning, per-token mask on the answer span."""
from typing import Callable

import numpy as np
import torch
from torch.nn import functional as F
from lion_pytorch import Lion

from .utils import TEMPLATE_WITH_ANSWER, format_question


def forward_and_compute_loss(model, tokens, mask, context_length: int = 512):
    tokens = tokens[:, :context_length]
    mask = mask[:, :context_length]

    x = tokens[:, :-1]
    y = tokens[:, 1:]
    mask = mask[:, 1:]

    logits = model(x).logits
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1),
        reduction="none",
    )
    loss = loss[mask.view(-1)].mean()
    return loss


def _format_and_tokenize(batch, tokenizer, device):
    """Format a (possibly multi-sample) batch into per-sample tokenized inputs + answer masks.

    Batches with ``batch_size > 1`` are processed sample by sample because each
    has a different prompt length and answer span; we average the loss across the
    batch.
    """
    questions = batch["instruction"]
    answers = batch["response_style"]

    items = []
    for q, a in zip(questions, answers):
        text = TEMPLATE_WITH_ANSWER.format(question=q, answer=a)
        ids = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
        ).to(device)
        # Answer mask: tokens whose start offset is >= the index of `answer`
        answer_start = text.index(a)
        mask = ids["offset_mapping"][:, :, 0] >= answer_start
        items.append({"input_ids": ids["input_ids"], "mask": mask})
    return items


def train_step(model, optimizer, batch, tokenizer, context_length: int = 512):
    items = _format_and_tokenize(batch, tokenizer, model.device)

    losses = []
    optimizer.zero_grad()
    for item in items:
        loss = forward_and_compute_loss(
            model=model,
            tokens=item["input_ids"],
            mask=item["mask"],
            context_length=context_length,
        )
        # Average loss across batch through gradient accumulation
        (loss / len(items)).backward()
        losses.append(float(loss.item()))

    optimizer.step()
    return float(np.mean(losses))


@torch.no_grad()
def validation_loss(model, val_loader, tokenizer, context_length: int = 512) -> float:
    model.eval()
    losses = []
    for batch in val_loader:
        items = _format_and_tokenize(batch, tokenizer, model.device)
        for item in items:
            loss = forward_and_compute_loss(
                model=model,
                tokens=item["input_ids"],
                mask=item["mask"],
                context_length=context_length,
            )
            losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def train(
    model,
    train_loader,
    val_loader,
    tokenizer,
    *,
    epochs: int = 1,
    max_steps: int | None = None,
    learning_rate: float = 1e-4,
    context_length: int = 512,
    preview_every: int = 20,
    preview_fn: Callable | None = None,
    on_step: Callable | None = None,
    log_val_every: int | None = None,
):
    """Fine-tuning loop with optional validation logging."""
    optimizer = Lion(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    model.train()

    history = {"train_loss": [], "val_loss": [], "steps": []}
    rolling = []
    global_step = 0

    for epoch in range(epochs):
        for batch in train_loader:
            loss_value = train_step(
                model, optimizer, batch, tokenizer, context_length=context_length
            )
            rolling.append(loss_value)
            history["train_loss"].append(loss_value)
            history["steps"].append(global_step)

            if on_step is not None:
                on_step(global_step, loss_value)

            if global_step % preview_every == 0:
                avg = float(np.mean(rolling))
                rolling = []
                if preview_fn is not None:
                    preview_fn(global_step, avg)
                print(f"[epoch {epoch} | step {global_step}] train_loss = {avg:.4f}")

            if log_val_every is not None and global_step > 0 and global_step % log_val_every == 0:
                vloss = validation_loss(model, val_loader, tokenizer, context_length)
                history["val_loss"].append((global_step, vloss))
                print(f"  --> val_loss = {vloss:.4f}")

            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break
        if max_steps is not None and global_step >= max_steps:
            break

    # Final validation
    vloss = validation_loss(model, val_loader, tokenizer, context_length)
    history["val_loss"].append((global_step, vloss))
    print(f"[final] val_loss = {vloss:.4f}")

    return model, history


def chat(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    only_answer: bool = True,
):
    """Generate a response from the model."""
    prompt = format_question(question)
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    full_tokens = outputs[0]
    prompt_tokens = int(input_ids["input_ids"].shape[-1])
    completion_tokens = int(full_tokens.shape[-1] - prompt_tokens)
    output_tokens = full_tokens[prompt_tokens:] if only_answer else full_tokens

    return {
        "answer": tokenizer.decode(output_tokens, skip_special_tokens=True),
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
