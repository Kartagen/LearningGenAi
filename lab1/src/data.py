"""Custom dataloader with train/validation/test split for Yoda style fine-tuning."""
import os
from typing import Tuple

from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader


def _load_yoda_responses() -> list:
    """Load yoda-style responses from the mitdeeplearning package or a local file."""
    # Prefer local copy if available
    local_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "text_styles",
        "yoda.txt",
    )
    if os.path.exists(local_path):
        path = local_path
    else:
        # Fallback: take it from the installed mitdeeplearning package
        import mitdeeplearning as mdl
        pkg_dir = os.path.dirname(mdl.__file__)
        path = os.path.join(pkg_dir, "data", "text_styles", "yoda.txt")

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().replace("\\n", "\n") for line in f if line.strip()]


def create_yoda_dataloaders(
    batch_size: int = 16,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for the Yoda-style task.

    The base dataset is ``databricks/databricks-dolly-15k``. We replace the
    responses with Yoda-style ones (provided by the lab) for the first N
    samples that have stylized text, then split into train/val/test.
    """
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1
    assert train_ratio + val_ratio < 1, "train + val must leave room for test"

    raw = load_dataset("databricks/databricks-dolly-15k", split="train")
    yoda_responses = _load_yoda_responses()
    n_styled = len(yoda_responses)

    styled = raw.select(range(n_styled)).map(
        lambda x, idx: {"response_style": yoda_responses[idx]},
        with_indices=True,
    )

    # Hold-out test from rows AFTER the styled region to avoid leakage,
    # but the lab also asks for splitting the styled portion: keep both.
    # We split the styled portion into train/val/test.
    shuffled = styled.shuffle(seed=seed)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ds = shuffled.select(range(0, n_train))
    val_ds = shuffled.select(range(n_train, n_train + n_val))
    test_ds = shuffled.select(range(n_train + n_val, n_total))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


def get_base_and_style_samples(loader: DataLoader, n_samples: int) -> Tuple[list, list]:
    """Collect base-style and yoda-style samples for evaluation (positive/negative controls)."""
    base_samples, style_samples = [], []
    for i, sample in enumerate(loader):
        if i >= n_samples:
            break
        base_samples.append(sample["response"][0])
        style_samples.append(sample["response_style"][0])
    return base_samples, style_samples
