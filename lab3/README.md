# Lab 3 — Stable Diffusion + DreamBooth LoRA (Windows local run)

Adapted from the two original Kaggle notebooks in `../src/`:

- `stable-diffusion-pipelines-diffusers.ipynb` →
  `notebooks/01_stable_diffusion_pipelines.ipynb`
- `dreambooth-lora-kaggle.ipynb` →
  `notebooks/02_dreambooth_lora.ipynb`

The originals are configured for Kaggle (`/kaggle/working`, `kaggle_secrets`,
bash `%%` magics, `!nvidia-smi`, `!find`, etc.). The adapted versions in
`notebooks/` run on a local Windows machine with an NVIDIA GPU.

## Folder layout

```
lab3/
├── README.md
├── requirements.txt
├── .gitignore
├── adapt_notebooks.py            # re-runs the Kaggle→local conversion
├── notebooks/
│   ├── 01_stable_diffusion_pipelines.ipynb
│   └── 02_dreambooth_lora.ipynb
├── data/
│   └── dog_example_augmented_20/ # extracted from ../src/dog_example_augmented_20.zip
│       ├── images/
│       ├── metadata.jsonl
│       └── README.md
├── outputs/                      # generated images, comparison grids, reports
└── work/                         # HF cache, downloaded training script, LoRA weights, logs
```

`work/` and `outputs/` are created by the bootstrap cell of each notebook.

## Prerequisites

1. **NVIDIA GPU with ≥ 8 GB VRAM** (16 GB recommended for the DreamBooth notebook).
   Without CUDA the notebooks will still import, but generation will be
   prohibitively slow or fail with OOM.
2. **Python 3.10 – 3.12** in a virtual environment.
3. **CUDA-enabled PyTorch**. Install separately first, choosing the matching
   CUDA build from <https://pytorch.org/get-started/locally/>. Example:

   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Hugging Face token** (`read` scope) for gated repos. Either:
   - export it before launching Jupyter:
     ```powershell
     $env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxx"
     ```
   - or run `huggingface-cli login` once and let the notebook pick the
     cached credential up.

## Installation

```powershell
cd D:\LearningMagister\Semester2\GenAIandItsUsage\lab3

# (one-time) create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install CUDA torch FIRST (see prerequisites)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# then everything else
pip install -r requirements.txt

# launch the notebooks
jupyter lab
```

## Running the lab

Open the notebook from JupyterLab and execute cells top to bottom.

**Notebook 1** — `notebooks/01_stable_diffusion_pipelines.ipynb`
Walks through `Text-to-Image`, `Img2Img`, `Inpainting`, `Depth2Img`, the VAE /
tokenizer / UNet / scheduler components, and a manual sampling loop.

**Notebook 2** — `notebooks/02_dreambooth_lora.ipynb`
Trains a DreamBooth LoRA on the 20-image dog dataset in
`data/dog_example_augmented_20/images/`, then generates and compares against
the baseline SD 1.5 model.

The first cell of each notebook is a bootstrap cell added by
`adapt_notebooks.py`. It sets `LAB3_WORK_DIR` / `LAB3_DATA_DIR` and pins the
Hugging Face cache inside `lab3/work/`.

## What was changed vs. the Kaggle originals

| Kaggle original | Local replacement |
| --- | --- |
| `/kaggle/working` | `lab3/work` (via `LAB3_WORK_DIR`) |
| `/kaggle/input` | `lab3/data` (via `LAB3_DATA_DIR`) |
| `kaggle_secrets.UserSecretsClient` | `os.environ["HF_TOKEN"]` |
| `%%bash … pip install …` | `%pip install …` |
| `!nvidia-smi` | `subprocess.check_output(['nvidia-smi'])` guarded by `shutil.which` |
| `!find … | sort` | `Path.rglob` |
| `!python {script} --help | head -40` | `subprocess.run([..., '--help'])` |
| `RAW_DATA_DIR = Path('/kaggle/input/datasets/.../images')` | `Path(LAB3_DATA_DIR) / 'dog_example_augmented_20' / 'images'` |

The lab content itself (prompts, model IDs, hyperparameters, explanatory
markdown) is left untouched. Run `python adapt_notebooks.py` again if you
edit the originals in `../src/` and want to regenerate the adapted versions.

## Known Windows caveats

- **xformers** rarely installs cleanly on Windows. Both notebooks already wrap
  `enable_xformers_memory_efficient_attention()` in `try/except`, so missing
  xformers is non-fatal — generation just runs a bit slower.
- **bitsandbytes** (used for `--use_8bit_adam` in the DreamBooth notebook) is
  not supported on Windows out of the box. The notebook already sets
  `USE_8BIT_ADAM = False` and the conditional skips the flag if `bitsandbytes`
  is not installed.
- **CUDA OOM**: drop `RESOLUTION` to `384`, `MAX_TRAIN_STEPS` to `100–200`, or
  set `RUN_DEPTH2IMG = False` / `RUN_INPAINTING = False` in notebook 1.
- The DreamBooth training script (`train_dreambooth_lora.py`) is downloaded
  by the notebook itself from the GitHub release tag matching the installed
  `diffusers` version. It lands in `lab3/work/`.

## Dataset

`data/dog_example_augmented_20/` is the 20-image augmented dog dataset
extracted from `../src/dog_example_augmented_20.zip`. See
`data/dog_example_augmented_20/README.md` for the dataset description.

The notebook uses `data/dog_example_augmented_20/images/` directly as the
DreamBooth instance directory by setting:

```python
USE_HF_EXAMPLE_DATASET = False
RAW_DATA_DIR = Path(os.environ["LAB3_DATA_DIR"]) / "dog_example_augmented_20" / "images"
```

To use the upstream `diffusers/dog-example` dataset instead, flip
`USE_HF_EXAMPLE_DATASET = True` in cell 17.
