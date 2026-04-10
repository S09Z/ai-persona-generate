"""
generate.py - AI Idol content generation CLI
Usage:
    uv run generate.py --character luna --type selfie
    uv run generate.py --character luna --type fashion --batch 4
    uv run generate.py --character luna --type selfie --dry-run
    uv run generate.py --character luna --type selfie --comfyui
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Must run before any HF/diffusers import
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from idol_generate.character import Character
from idol_generate.prompt_builder import build_post

load_dotenv()
console = Console()

# ── config from .env ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
MODEL_ID = os.getenv("MODEL_ID", "SG161222/RealVisXL_V4.0")
MODEL_PATH = os.getenv("MODEL_PATH", "")
DEVICE = os.getenv("DEVICE", "mps")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))
CHAR_DIR = Path(os.getenv("CHARACTER_DIR", "./characters"))
STEPS = int(os.getenv("STEPS", "25"))
GUIDANCE = float(os.getenv("GUIDANCE_SCALE", "6.0"))
WIDTH = int(os.getenv("WIDTH", "832"))
HEIGHT = int(os.getenv("HEIGHT", "1216"))


# ── diffusers backend ─────────────────────────────────────────────────────────
def run_diffusers(post: dict, output_path: Path, seed: int | None) -> None:
    from idol_generate.pipeline import generate_image

    result = generate_image(
        post=post,
        output_path=output_path,
        model_id=MODEL_ID,
        model_path=MODEL_PATH,
        device=DEVICE,
        steps=STEPS,
        guidance=GUIDANCE,
        width=WIDTH,
        height=HEIGHT,
        seed=seed,
    )
    console.log(f"[green]Image saved →[/] {result}")


# ── ComfyUI backend ───────────────────────────────────────────────────────────
def run_comfyui(post: dict, output_path: Path, workflow_path: str, seed: int | None) -> None:
    from idol_generate.comfyui_client import (
        download_image,
        is_running,
        queue_workflow,
        wait_for_result,
    )

    if not is_running():
        console.print("[red]ComfyUI is not running![/] Start it: [bold]./comfyui.sh[/]")
        raise SystemExit(1)

    console.log("[bold cyan]Submitting workflow to ComfyUI...[/]")
    prompt_id = queue_workflow(
        workflow_path=workflow_path,
        positive_prompt=post["prompt"],
        negative_prompt=post["negative_prompt"],
        seed=seed,
        steps=STEPS,
        cfg=GUIDANCE,
        width=WIDTH,
        height=HEIGHT,
    )
    console.log(f"[dim]Prompt ID: {prompt_id} — waiting...[/]")
    images = wait_for_result(prompt_id)
    if not images:
        console.print("[red]No images returned from ComfyUI[/]")
        return
    result = download_image(images[0], output_path)
    console.log(f"[green]Image saved →[/] {result}")


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
    )
    parser.add_argument(
        "--no-image", action="store_true", help="Skip image generation, output prompt JSON only"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print prompt and exit — no torch loaded, instant"
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="Number of images to generate (each gets seed+i)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for reproducibility (seed+i used per batch item)",
    )
    parser.add_argument(
        "--comfyui", action="store_true", help="Use ComfyUI API backend instead of diffusers"
    )
    parser.add_argument(
        "--workflow",
        default="./workflows/luna_portrait.json",
        help="ComfyUI workflow JSON path (used with --comfyui)",
    )
    args = parser.parse_args()

    char_path = CHAR_DIR / f"{args.character}.json"
    if not char_path.exists():
        console.print(f"[red]Character not found:[/] {char_path}")
        sys.exit(1)

    character = Character.load(char_path)
    post = build_post(character, args.content_type)  # type: ignore[arg-type]

    console.print(
        Panel(
            Pretty(post),
            title=f"[bold magenta]{character.name}[/] — {args.content_type}",
            border_style="magenta",
        )
    )

    # --dry-run: prompt preview with zero torch overhead
    if args.dry_run:
        token_estimate = len(post["prompt"].split(","))
        console.print(f"[dim]~{token_estimate} prompt tokens | --dry-run: no image generated[/]")
        return

    # Save post JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"{character.name.lower()}_{args.content_type}_{ts}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(post, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[dim]Post JSON → {json_path}[/]")

    if args.no_image:
        console.print("[yellow]Image generation skipped (--no-image)[/]")
        return

    # Generate batch
    for i in range(args.batch):
        seed = (args.seed + i) if args.seed is not None else None
        suffix = f"_{i + 1}of{args.batch}" if args.batch > 1 else ""
        img_path = OUTPUT_DIR / f"{character.name.lower()}_{args.content_type}_{ts}{suffix}.png"
        if args.batch > 1:
            console.log(f"[cyan]Generating {i + 1}/{args.batch}...[/]")
        if args.comfyui:
            run_comfyui(post, img_path, args.workflow, seed)
        else:
            run_diffusers(post, img_path, seed)


if __name__ == "__main__":
    main()
