"""
first_image.py - Quick first-image generator using SDXL-Turbo on MPS.
Generates a single Luna selfie with minimal setup.

Usage:
    uv run first_image.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()

# Suppress macOS HF Hub symlink warning
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

console = Console()

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from idol_generate.character import Character
from idol_generate.prompt_builder import build_post


def main() -> None:
    import torch
    from diffusers import AutoPipelineForText2Image

    # Authenticate with HuggingFace Hub if token is set
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and not hf_token.startswith("hf_your"):
        from huggingface_hub import login

        login(token=hf_token, add_to_git_credential=False)
        console.log("[green]HuggingFace Hub authenticated[/]")
    else:
        console.log("[yellow]HF_TOKEN not set — unauthenticated (slower downloads)[/]")
        console.log("[dim]Set HF_TOKEN in .env → https://huggingface.co/settings/tokens[/]")

    device = os.getenv("DEVICE", "mps")
    model_id = os.getenv("MODEL_ID", "stabilityai/sdxl-turbo")
    steps = int(os.getenv("STEPS", "4"))
    guidance = float(os.getenv("GUIDANCE_SCALE", "0.0"))
    width = int(os.getenv("WIDTH", "512"))
    height = int(os.getenv("HEIGHT", "512"))
    output_dir = Path(os.getenv("OUTPUT_DIR", "./outputs"))

    # ── build prompt from character ──────────────────────────────────────────
    character = Character.load(ROOT / "characters" / "luna.json")
    post = build_post(character, "selfie")

    console.print(Panel(post["prompt"], title="[magenta]Prompt[/]", border_style="magenta"))
    console.print(f"[dim]Caption:[/] {post['caption']}")

    # ── load pipeline ────────────────────────────────────────────────────────
    console.log(f"[cyan]Loading[/] [bold]{model_id}[/] on [yellow]{device}[/] ...")
    console.log("[dim](First run downloads ~7 GB - cached after that)[/]")

    # float16 causes black/NaN output on MPS - must use float32
    dtype = torch.float32 if device == "mps" else torch.float16
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
    ).to(device)

    # MPS: attention slicing can cause artifacts - skip it
    if device == "cuda":
        pipe.enable_attention_slicing()

    # ── generate ─────────────────────────────────────────────────────────────
    console.log(f"[cyan]Generating[/] {width}x{height} in {steps} steps ...")

    result = pipe(
        prompt=post["prompt"],
        negative_prompt=post["negative_prompt"],
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
    )
    image = result.images[0]

    # ── save ─────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = output_dir / f"luna_selfie_{ts}.png"
    image.save(img_path)

    console.print(f"\n[bold green]Done![/] Image saved → [underline]{img_path}[/]")
    console.print(f"[dim]Caption:[/] {post['caption']}")
    console.print(f"[dim]Tags:[/] {' '.join('#' + t for t in post['tags'])}")

    # open in Preview (macOS)
    os.system(f"open '{img_path}'")


if __name__ == "__main__":
    main()
