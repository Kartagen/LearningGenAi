"""
Headless orchestrator that runs Lab 3 end-to-end.

Two stages:
  1. STAGE_PIPELINES  -- text2img, guidance_scale grid, img2img(3 strengths),
                         inpainting, depth2img. Writes PNGs into outputs/.
  2. STAGE_DREAMBOOTH -- prepares instance images, generates baseline,
                         downloads & runs train_dreambooth_lora.py, then
                         generates LoRA images, then builds comparison grid.

Each stage can be enabled/disabled via env flags or CLI args:
    python run_lab.py --pipelines --dreambooth
    python run_lab.py --pipelines       # only stage 1
    python run_lab.py --dreambooth      # only stage 2

All generated artifacts land in:
    outputs/pipelines/   *.png  + metadata.json
    outputs/dreambooth/  baseline_*.png, lora_*.png, comparison.jpg, results.json
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request
from io import BytesIO
from pathlib import Path

# Force UTF-8 stdout/stderr (Windows console is cp1251 by default)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

LAB_DIR = Path(__file__).resolve().parent
WORK_DIR = LAB_DIR / "work"
DATA_DIR = LAB_DIR / "data"
OUTPUTS = LAB_DIR / "outputs"

WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME",            str(WORK_DIR / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE",       str(WORK_DIR / ".cache" / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(WORK_DIR / ".cache" / "huggingface" / "transformers"))
os.environ.setdefault("DIFFUSERS_CACHE",    str(WORK_DIR / ".cache" / "huggingface" / "diffusers"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"     # per-request timeout
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.setdefault("LAB3_WORK_DIR", str(WORK_DIR))
os.environ.setdefault("LAB3_DATA_DIR", str(DATA_DIR))


def fetch_model(repo_id: str, *, allow_patterns: list[str]) -> str:
    """
    Resilient snapshot_download with retries.
    Returns the local snapshot path.
    """
    from huggingface_hub import snapshot_download
    last_err = None
    for attempt in range(1, 6):
        try:
            print(f"  [fetch] {repo_id}  attempt {attempt}", flush=True)
            local = snapshot_download(
                repo_id=repo_id,
                allow_patterns=allow_patterns,
                max_workers=4,
                resume_download=True,
            )
            print(f"  [fetch] OK -> {local}", flush=True)
            return local
        except Exception as e:
            print(f"  [fetch] attempt {attempt} failed: {e}", flush=True)
            last_err = e
            time.sleep(5 * attempt)
    raise RuntimeError(f"snapshot_download failed for {repo_id}: {last_err}")


SD15_ALLOW = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "feature_extractor/*",
    "text_encoder/config.json",
    "text_encoder/*.safetensors",
    "vae/config.json",
    "vae/*.safetensors",
    "unet/config.json",
    "unet/*.safetensors",
    # Skip safety_checker (we disable it anyway) and *.bin / *.ckpt
]
SD_INPAINT_ALLOW = SD15_ALLOW
SD_DEPTH_ALLOW = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "feature_extractor/*",
    "depth_estimator/config.json",
    "depth_estimator/*.safetensors",
    "depth_estimator/*.bin",  # depth model only has bin
    "text_encoder/config.json",
    "text_encoder/*.safetensors",
    "vae/config.json",
    "vae/*.safetensors",
    "unet/config.json",
    "unet/*.safetensors",
]


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n  {msg}\n{'=' * 70}", flush=True)


def clear_gpu():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 1 — Stable Diffusion pipelines
# ---------------------------------------------------------------------------
def stage_pipelines(seed: int = 42, *,
                    do_inpaint: bool = True,
                    do_depth: bool = True) -> dict:
    """
    Run notebook 1 experiments and save results.

    Returns a dict with per-experiment file paths and timings.
    """
    import torch
    from PIL import Image, ImageDraw
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionDepth2ImgPipeline,
    )

    out_dir = OUTPUTS / "pipelines"
    out_dir.mkdir(parents=True, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
    MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    results = {
        "device": DEVICE,
        "dtype": str(DTYPE),
        "model_id": MODEL_ID,
        "experiments": {},
    }

    # ---------- 0. Pre-fetch SD-1.5 weights ---------------------------------
    banner("Stage 1.0  Pre-fetch SD-1.5 weights (safetensors only)")
    fetch_model(MODEL_ID, allow_patterns=SD15_ALLOW)

    # ---------- 1. Text2Img basic + guidance grid ---------------------------
    banner("Stage 1.1  Text-to-Image (basic + guidance_scale grid)")
    text2img_files = {
        "basic":  out_dir / "text2img_basic.png",
        "cfg2":   out_dir / "guidance_cfg2.0.png",
        "cfg8":   out_dir / "guidance_cfg8.0.png",
        "cfg12":  out_dir / "guidance_cfg12.0.png",
    }
    if all(p.exists() for p in text2img_files.values()):
        print("  All Text2Img files already exist - skipping regeneration.")
        results["experiments"]["text2img_basic"] = {
            "prompt": "Palette knife painting of an autumn cityscape",
            "negative_prompt": "Oversaturated, blurry, low quality",
            "guidance_scale": 8.0, "steps": 30, "seed": seed,
            "file": str(text2img_files["basic"].relative_to(LAB_DIR)),
            "cached": True,
        }
        results["experiments"]["guidance_grid"] = {
            "prompt": "A collie dog with a pink hat",
            "items": [
                {"cfg": 2.0,  "file": str(text2img_files["cfg2"].relative_to(LAB_DIR))},
                {"cfg": 8.0,  "file": str(text2img_files["cfg8"].relative_to(LAB_DIR))},
                {"cfg": 12.0, "file": str(text2img_files["cfg12"].relative_to(LAB_DIR))},
            ],
            "cached": True,
        }
    else:
        t0 = time.time()
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=DTYPE, use_safetensors=True,
        ).to(DEVICE)
        pipe.enable_attention_slicing()
        pipe.safety_checker = None
        pipe.requires_safety_checker = False

        image = pipe(
            prompt="Palette knife painting of an autumn cityscape",
            negative_prompt="Oversaturated, blurry, low quality",
            height=512, width=512,
            guidance_scale=8.0, num_inference_steps=30,
            generator=torch.Generator(DEVICE).manual_seed(seed),
        ).images[0]
        image.save(text2img_files["basic"])
        results["experiments"]["text2img_basic"] = {
            "prompt": "Palette knife painting of an autumn cityscape",
            "negative_prompt": "Oversaturated, blurry, low quality",
            "guidance_scale": 8.0, "steps": 30, "seed": seed,
            "file": str(text2img_files["basic"].relative_to(LAB_DIR)),
        }
        print("  saved", text2img_files["basic"].name)

        cfg_results = []
        for cfg in [2.0, 8.0, 12.0]:
            img = pipe(
                prompt="A collie dog with a pink hat",
                negative_prompt="blurry, low quality, distorted",
                height=512, width=512,
                guidance_scale=cfg, num_inference_steps=30,
                generator=torch.Generator(DEVICE).manual_seed(123),
            ).images[0]
            pp = out_dir / f"guidance_cfg{cfg}.png"
            img.save(pp)
            cfg_results.append({"cfg": cfg, "file": str(pp.relative_to(LAB_DIR))})
            print(f"  saved {pp.name}")
        results["experiments"]["guidance_grid"] = {
            "prompt": "A collie dog with a pink hat",
            "items": cfg_results,
        }
        results["experiments"]["text2img_basic"]["time_s"] = round(time.time() - t0, 2)
        del pipe
        clear_gpu()

    # ---------- 2. Img2Img (3 strengths) ------------------------------------
    banner("Stage 1.2  Img2Img with strength in {0.25, 0.55, 0.85}")
    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, use_safetensors=True,
    ).to(DEVICE)
    img2img_pipe.enable_attention_slicing()
    img2img_pipe.safety_checker = None
    img2img_pipe.requires_safety_checker = False

    # Download demo image (with synthetic fallback)
    init_image = _get_demo_image(out_dir, size=(512, 512))

    img2img_results = []
    for strength in [0.25, 0.55, 0.85]:
        t1 = time.time()
        result = img2img_pipe(
            prompt=("A front-facing person sitting on a park bench, "
                    "looking at the viewer, oil painting, warm light, high detail"),
            negative_prompt=("blurry face, distorted face, extra fingers, "
                             "low quality, blurry, ugly"),
            image=init_image,
            strength=strength,
            guidance_scale=7.5, num_inference_steps=30,
            generator=torch.Generator(DEVICE).manual_seed(seed),
        ).images[0]
        pp = out_dir / f"img2img_strength{strength:.2f}.png"
        result.save(pp)
        img2img_results.append({
            "strength": strength,
            "file": str(pp.relative_to(LAB_DIR)),
            "time_s": round(time.time() - t1, 2),
        })
        print(f"  saved {pp.name}  ({img2img_results[-1]['time_s']}s)")
    results["experiments"]["img2img"] = {"items": img2img_results}
    del img2img_pipe
    clear_gpu()

    # ---------- 3. Inpainting -----------------------------------------------
    banner("Stage 1.3  Inpainting")
    if not do_inpaint:
        print("  skipped (do_inpaint=False)")
        results["experiments"]["inpainting"] = {"skipped": True}
    else:
        try:
            fetch_model("stable-diffusion-v1-5/stable-diffusion-inpainting",
                        allow_patterns=SD_INPAINT_ALLOW)
            inpaint = StableDiffusionInpaintPipeline.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-inpainting",
                torch_dtype=DTYPE,
            ).to(DEVICE)
            inpaint.enable_attention_slicing()
            inpaint.safety_checker = None
            inpaint.requires_safety_checker = False

            mask = _get_demo_mask(out_dir, size=(512, 512))
            t1 = time.time()
            img = inpaint(
                prompt="A small friendly robot, high resolution, sitting on a park bench",
                negative_prompt="blurry, low quality, distorted, bad anatomy",
                image=init_image, mask_image=mask,
                guidance_scale=8.0, num_inference_steps=30,
                generator=torch.Generator(DEVICE).manual_seed(seed),
            ).images[0]
            pp = out_dir / "inpainting_result.png"
            img.save(pp)
            results["experiments"]["inpainting"] = {
                "file": str(pp.relative_to(LAB_DIR)),
                "time_s": round(time.time() - t1, 2),
            }
            print(f"  saved {pp.name}  ({results['experiments']['inpainting']['time_s']}s)")
            del inpaint
            clear_gpu()
        except Exception as e:
            print("  Inpainting FAILED:", repr(e))
            results["experiments"]["inpainting"] = {"error": repr(e)}

    # ---------- 4. Depth2Img ------------------------------------------------
    banner("Stage 1.4  Depth2Img")
    if not do_depth:
        print("  skipped (do_depth=False)")
        results["experiments"]["depth2img"] = {"skipped": True}
    else:
        try:
            fetch_model("sd2-community/stable-diffusion-2-depth",
                        allow_patterns=SD_DEPTH_ALLOW)
            depth = StableDiffusionDepth2ImgPipeline.from_pretrained(
                "sd2-community/stable-diffusion-2-depth", torch_dtype=DTYPE,
            ).to(DEVICE)
            depth.enable_attention_slicing()

            t1 = time.time()
            img = depth(
                prompt="An oil painting of a person sitting on a bench, cinematic lighting",
                negative_prompt="blurry, low quality, distorted",
                image=init_image, strength=0.75,
                guidance_scale=7.5, num_inference_steps=30,
                generator=torch.Generator(DEVICE).manual_seed(seed),
            ).images[0]
            pp = out_dir / "depth2img_result.png"
            img.save(pp)
            results["experiments"]["depth2img"] = {
                "file": str(pp.relative_to(LAB_DIR)),
                "time_s": round(time.time() - t1, 2),
            }
            print(f"  saved {pp.name}  ({results['experiments']['depth2img']['time_s']}s)")
            del depth
            clear_gpu()
        except Exception as e:
            print("  Depth2Img FAILED:", repr(e))
            results["experiments"]["depth2img"] = {"error": repr(e)}

    # ---------- save -----------------------------------------------------
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n[OK] Stage 1 done. Metadata:", out_dir / "metadata.json")
    return results


def _get_demo_image(out_dir: Path, size=(512, 512)):
    """Download demo image with synthetic fallback."""
    from PIL import Image, ImageDraw
    path = out_dir / "demo_init.png"
    if path.exists():
        return Image.open(path).convert("RGB").resize(size)
    try:
        url = ("https://raw.githubusercontent.com/CompVis/latent-diffusion/main/"
               "data/inpainting_examples/overture-creations-5sI6fQgYIuo.png")
        with urllib.request.urlopen(url, timeout=20) as r:
            img = Image.open(BytesIO(r.read())).convert("RGB").resize(size)
        img.save(path)
        return img
    except Exception as e:
        print("  Falling back to synthetic image:", repr(e))
        img = Image.new("RGB", size, (210, 225, 240))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, size[0], int(size[1] * 0.58)], fill=(190, 220, 245))
        d.rectangle([0, int(size[1] * 0.58), size[0], size[1]], fill=(120, 170, 120))
        d.rectangle([120, 310, 390, 330], fill=(120, 80, 45))
        d.ellipse([230, 170, 285, 225], fill=(235, 190, 150),
                  outline=(70, 60, 50), width=3)
        d.rectangle([240, 225, 275, 310], fill=(80, 110, 180))
        img.save(path)
        return img


def _get_demo_mask(out_dir: Path, size=(512, 512)):
    from PIL import Image, ImageDraw
    path = out_dir / "demo_mask.png"
    if path.exists():
        return Image.open(path).convert("RGB").resize(size)
    try:
        url = ("https://raw.githubusercontent.com/CompVis/latent-diffusion/main/"
               "data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png")
        with urllib.request.urlopen(url, timeout=20) as r:
            img = Image.open(BytesIO(r.read())).convert("RGB").resize(size)
        img.save(path)
        return img
    except Exception:
        mask = Image.new("RGB", size, (0, 0, 0))
        d = ImageDraw.Draw(mask)
        d.rectangle([215, 155, 305, 325], fill=(255, 255, 255))
        mask.save(path)
        return mask


# ---------------------------------------------------------------------------
# Stage 2 — DreamBooth LoRA
# ---------------------------------------------------------------------------
def stage_dreambooth(seed: int = 42, max_train_steps: int = 200) -> dict:
    import torch
    from PIL import Image, ImageOps, ImageDraw
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    import diffusers

    out_dir = OUTPUTS / "dreambooth"
    out_dir.mkdir(parents=True, exist_ok=True)
    instance_dir = WORK_DIR / "instance_images"
    lora_dir = WORK_DIR / "dreambooth_lora_output"
    instance_dir.mkdir(parents=True, exist_ok=True)
    lora_dir.mkdir(parents=True, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
    MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    UNIQUE_TOKEN = "sks"
    CLASS_NOUN = "puppy"
    INSTANCE_PROMPT = f"a photo of {UNIQUE_TOKEN} {CLASS_NOUN}"
    VALIDATION_PROMPT = f"a photo of {UNIQUE_TOKEN} {CLASS_NOUN} in a studio"

    results = {
        "device": DEVICE,
        "model_id": MODEL_ID,
        "unique_token": UNIQUE_TOKEN,
        "class_noun": CLASS_NOUN,
        "instance_prompt": INSTANCE_PROMPT,
        "validation_prompt": VALIDATION_PROMPT,
        "max_train_steps": max_train_steps,
    }

    # ---- 1. Prepare instance images ----------------------------------------
    banner("Stage 2.1  Prepare instance images from data/dog_example_augmented_20/")
    src = DATA_DIR / "dog_example_augmented_20" / "images"
    if not src.exists():
        raise FileNotFoundError(f"Dataset not found: {src}")

    IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    # Clean and re-fill instance_dir
    import shutil
    if instance_dir.exists():
        shutil.rmtree(instance_dir)
    instance_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted([p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXT])
    for idx, sp in enumerate(image_paths, start=1):
        img = Image.open(sp)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.save(instance_dir / f"instance_{idx:03d}.jpg", quality=95)
    n = len(list(instance_dir.glob("*.jpg")))
    print(f"  prepared {n} instance images in {instance_dir}")
    results["n_instance_images"] = n

    # ---- 2. Download train_dreambooth_lora.py ------------------------------
    banner("Stage 2.2  Download official DreamBooth LoRA training script")
    target_script = WORK_DIR / "train_dreambooth_lora.py"
    version = diffusers.__version__.split("+")[0]
    urls = [
        f"https://raw.githubusercontent.com/huggingface/diffusers/v{version}/examples/dreambooth/train_dreambooth_lora.py",
        "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora.py",
    ]
    last_err = None
    for url in urls:
        try:
            print("  trying", url)
            urllib.request.urlretrieve(url, target_script)
            print("  saved", target_script, target_script.stat().st_size, "bytes")
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"Could not download train script: {last_err}")
    results["train_script"] = str(target_script.relative_to(LAB_DIR))

    # ---- 3. Pre-train (baseline) generation --------------------------------
    banner("Stage 2.3  Baseline generation (pre-LoRA)")
    contexts = [
        "on a wooden table",
        "in the snow",
        "in a futuristic kitchen",
        "floating in space",
    ]
    baseline_prompts = [f"a photo of a {CLASS_NOUN} {c}" for c in contexts]
    lora_prompts     = [f"a photo of {UNIQUE_TOKEN} {CLASS_NOUN} {c}" for c in contexts]

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    baseline_files = []
    for i, p in enumerate(baseline_prompts, start=1):
        img = pipe(
            p, num_inference_steps=35, guidance_scale=7.5,
            generator=torch.Generator(DEVICE).manual_seed(seed),
        ).images[0]
        pp = out_dir / f"baseline_{i:02d}.png"
        img.save(pp)
        baseline_files.append(str(pp.relative_to(LAB_DIR)))
        print(f"  saved {pp.name}  | {p}")
    del pipe
    clear_gpu()

    # ---- 4. Train DreamBooth LoRA ------------------------------------------
    banner(f"Stage 2.4  Train DreamBooth LoRA ({max_train_steps} steps)")
    log_path = WORK_DIR / "training_live.log"
    cmd = [
        sys.executable, str(target_script),
        "--pretrained_model_name_or_path", MODEL_ID,
        "--instance_data_dir", str(instance_dir),
        "--output_dir", str(lora_dir),
        "--instance_prompt", INSTANCE_PROMPT,
        "--resolution", "512",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--learning_rate", "5e-5",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--max_train_steps", str(max_train_steps),
        "--checkpointing_steps", str(max_train_steps + 10),  # skip mid-checkpoints
        "--seed", str(seed),
        "--rank", "4",
        "--gradient_checkpointing",
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print("  cmd:", " ".join(cmd))
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(cmd, cwd=str(WORK_DIR),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        last_print = time.time()
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            # Throttle stdout: print every 5 s or when key keywords appear
            if (time.time() - last_print > 5
                    or any(k in line for k in ("Loss", "step", "Epoch", "saved", "Error", "error"))):
                print(" ", line.rstrip())
                last_print = time.time()
        rc = proc.wait()
    train_time = time.time() - t0
    print(f"\n  training done. return_code={rc}, time={train_time:.1f}s")
    results["training"] = {
        "return_code": rc, "time_s": round(train_time, 1),
        "log": str(log_path.relative_to(LAB_DIR)),
    }
    if rc != 0:
        raise RuntimeError(f"Training failed with return code {rc}. See {log_path}")

    # ---- 5. Post-train inference (LoRA) ------------------------------------
    banner("Stage 2.5  Generate with LoRA weights")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.load_lora_weights(str(lora_dir))
    print("  LoRA weights loaded from", lora_dir)

    lora_files = []
    for i, p in enumerate(lora_prompts, start=1):
        img = pipe(
            p, num_inference_steps=35, guidance_scale=7.5,
            generator=torch.Generator(DEVICE).manual_seed(seed),
        ).images[0]
        pp = out_dir / f"lora_{i:02d}.png"
        img.save(pp)
        lora_files.append(str(pp.relative_to(LAB_DIR)))
        print(f"  saved {pp.name}  | {p}")
    del pipe
    clear_gpu()

    # ---- 6. Comparison grid -----------------------------------------------
    banner("Stage 2.6  Build comparison grid")
    from PIL import Image, ImageDraw
    thumb_w, thumb_h, label_h, margin = 512, 512, 80, 10
    n_rows = min(len(baseline_files), len(lora_files))
    grid_w = thumb_w * 2
    grid_h = (thumb_h + label_h + margin) * n_rows
    canvas = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i in range(n_rows):
        y = i * (thumb_h + label_h + margin)
        left = Image.open(LAB_DIR / baseline_files[i]).convert("RGB").resize((thumb_w, thumb_h))
        right = Image.open(LAB_DIR / lora_files[i]).convert("RGB").resize((thumb_w, thumb_h))
        canvas.paste(left,  (0,      y + label_h))
        canvas.paste(right, (thumb_w, y + label_h))
        draw.text((10, y + 10),        "BASE: " + baseline_prompts[i], fill="black")
        draw.text((thumb_w + 10, y + 10), "LORA: " + lora_prompts[i],     fill="black")
    comparison_path = out_dir / "comparison.jpg"
    canvas.save(comparison_path, quality=95)
    print("  saved", comparison_path)
    results["baseline_files"]   = baseline_files
    results["lora_files"]       = lora_files
    results["comparison_file"]  = str(comparison_path.relative_to(LAB_DIR))
    results["baseline_prompts"] = baseline_prompts
    results["lora_prompts"]     = lora_prompts

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n[OK] Stage 2 done. Results:", out_dir / "results.json")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipelines",  action="store_true", help="Run stage 1")
    ap.add_argument("--dreambooth", action="store_true", help="Run stage 2")
    ap.add_argument("--all",        action="store_true", help="Run both stages")
    ap.add_argument("--no-inpaint", action="store_true", help="Skip inpainting (saves 4GB download)")
    ap.add_argument("--no-depth",   action="store_true", help="Skip depth2img (saves 5GB download)")
    ap.add_argument("--steps", type=int, default=200,
                    help="DreamBooth max_train_steps (default 200)")
    args = ap.parse_args()

    if args.all or (not args.pipelines and not args.dreambooth):
        args.pipelines = True
        args.dreambooth = True

    overall_t0 = time.time()
    summary = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    try:
        if args.pipelines:
            summary["pipelines"] = stage_pipelines(
                do_inpaint=not args.no_inpaint,
                do_depth=not args.no_depth,
            )
    except Exception as e:
        traceback.print_exc()
        summary["pipelines_error"] = repr(e)

    try:
        if args.dreambooth:
            summary["dreambooth"] = stage_dreambooth(max_train_steps=args.steps)
    except Exception as e:
        traceback.print_exc()
        summary["dreambooth_error"] = repr(e)

    summary["total_time_s"] = round(time.time() - overall_t0, 1)

    with open(OUTPUTS / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    banner(f"ALL DONE in {summary['total_time_s']}s. "
           f"Summary: {OUTPUTS / 'run_summary.json'}")


if __name__ == "__main__":
    main()
