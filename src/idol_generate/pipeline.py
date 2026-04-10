"""
idol_generate/pipeline.py
Shared Stable Diffusion pipeline loader and image generator.
Handles MPS float32 fix, reproducible seeding, and memory cleanup.
"""

from __future__ import annotations

import os
from pathlib import Path


def hf_login() -> None:
    """Authenticate with HuggingFace Hub if HF_TOKEN is set."""
    token = os.getenv("HF_TOKEN", "")
    if token and not token.startswith("hf_your"):
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)


def set_seed(seed: int, device: str = "mps") -> None:
    """Seed Python random, PyTorch CPU, and device RNG for reproducibility."""
    import random

    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    elif device == "cuda":
        torch.cuda.manual_seed_all(seed)


def load_pipeline(model_id: str, model_path: str, device: str) -> object:
    """
    Load StableDiffusionXLPipeline from local .safetensors or HF Hub.
    Uses bfloat16 on MPS — safe (no black-image NaN) and ~2-3x faster than float32.
    """
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    # float16 produces black/NaN on MPS; bfloat16 is safe and 2-3x faster than float32
    dtype = torch.bfloat16 if device == "mps" else torch.float16

    if model_path and Path(model_path).exists():
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path, torch_dtype=dtype, use_safetensors=True
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True
        )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    pipe = pipe.to(device)

    if device == "mps":
        pipe.enable_attention_slicing(1)
    elif device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()

    return pipe


def generate_image(
    post: dict,
    output_path: Path,
    model_id: str,
    model_path: str,
    device: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    seed: int | None = None,
) -> Path:
    """Load pipeline, run generation, save output, free device memory."""
    import torch

    hf_login()

    if seed is not None:
        set_seed(seed, device)

    pipe = load_pipeline(model_id, model_path, device)

    result = pipe(
        prompt=post["prompt"],
        prompt_2=post.get("prompt_2"),  # CLIP-G: quality/booster tags
        negative_prompt=post["negative_prompt"],
        negative_prompt_2=post.get("negative_prompt_2"),
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
    )
    image = result.images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    # Free device memory after generation
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return output_path
