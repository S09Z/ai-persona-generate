"""
generate.py  –  AI Idol content generation CLI
Usage:
    uv run generate.py --character luna --type selfie
    uv run generate.py --character luna --type fashion --no-image
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

load_dotenv()
console = Console()

# ── resolve project root ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from idol_generate.character import Character  # noqa: E402
from idol_generate.prompt_builder import build_post  # noqa: E402

# ── config from .env ──────────────────────────────────────────────────────────
MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
MODEL_PATH = os.getenv("MODEL_PATH", "")  # local .safetensors override
DEVICE = os.getenv("DEVICE", "mps")  # mps | cuda | cpu
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))
CHAR_DIR = Path(os.getenv("CHARACTER_DIR", "./characters"))
STEPS = int(os.getenv("STEPS", "30"))
GUIDANCE = float(os.getenv("GUIDANCE_SCALE", "7.5"))
WIDTH = int(os.getenv("WIDTH", "1024"))
HEIGHT = int(os.getenv("HEIGHT", "1024"))


# ── image generation ──────────────────────────────────────────────────────────
def generate_image(post: dict, output_path: Path) -> None:
    """Load SDXL (or local model) and generate an image."""
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    console.log(f"[bold cyan]Loading model:[/] {MODEL_PATH or MODEL_ID} on [yellow]{DEVICE}[/]")

    dtype = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32

    if MODEL_PATH and Path(MODEL_PATH).exists():
        pipe = StableDiffusionXLPipeline.from_single_file(
            MODEL_PATH, torch_dtype=dtype, use_safetensors=True
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID, torch_dtype=dtype, use_safetensors=True
        )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    pipe = pipe.to(DEVICE)

    if DEVICE == "cuda":
        pipe.enable_xformers_memory_efficient_attention()

    console.log("[bold cyan]Generating image…[/]")
    result = pipe(
        prompt=post["prompt"],
        negative_prompt=post["negative_prompt"],
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        width=WIDTH,
        height=HEIGHT,
    )
    image = result.images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    console.log(f"[green]Image saved →[/] {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="AI Idol post generator")
    parser.add_argument(
        "--character", "-c", default="luna", help="Character name (matches characters/<name>.json)"
    )
    parser.add_argument(
        "--type",
        "-t",
        dest="content_type",
        choices=["daily_life", "fashion", "selfie", "story"],
        default="daily_life",
        help="Content type",
    )
    parser.add_argument(
        "--no-image", action="store_true", help="Skip image generation (prompt-only mode)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        import random

        random.seed(args.seed)

    # Load character
    char_path = CHAR_DIR / f"{args.character}.json"
    if not char_path.exists():
        console.print(f"[red]Character file not found:[/] {char_path}")
        sys.exit(1)

    character = Character.load(char_path)
    post = build_post(character, args.content_type)  # type: ignore[arg-type]

    # Display structured output
    console.print(
        Panel(
            Pretty(post),
            title=f"[bold magenta]{character.name}[/] — {args.content_type}",
            border_style="magenta",
        )
    )

    # Save JSON output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"{character.name.lower()}_{args.content_type}_{ts}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(post, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[dim]Post JSON saved → {json_path}[/]")

    # Generate image (unless skipped)
    if not args.no_image:
        img_path = OUTPUT_DIR / f"{character.name.lower()}_{args.content_type}_{ts}.png"
        generate_image(post, img_path)
    else:
        console.print("[yellow]Image generation skipped (--no-image)[/]")


if __name__ == "__main__":
    main()
