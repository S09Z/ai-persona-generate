"""
prepare_dataset.py — Prepare a LoRA training dataset from scraped images.

What it does:
  1. Copies/validates images from the source folder
  2. Auto-captions every image using BLIP-2 (or Salesforce/blip-image-captioning-large)
  3. Prepends a trigger word to every caption  →  "<trigger>, <blip caption>"
  4. Writes a <filename>.txt alongside each image (kohya_ss format)
  5. Prints a summary table

Usage:
    # Caption with BLIP-2 (default, requires ~8 GB VRAM or Apple Silicon MPS)
    uv run prepare_dataset.py --src datasets/instagram/spectre.a.i --trigger "spectre_ai"

    # Use the lighter BLIP-large instead (faster, ~4 GB)
    uv run prepare_dataset.py --src datasets/instagram/spectre.a.i --trigger "luna_v1" --model blip

    # Skip captioning — just add the trigger word as the only caption
    uv run prepare_dataset.py --src datasets/instagram/spectre.a.i --trigger "luna_v1" --no-caption

    # Custom output directory
    uv run prepare_dataset.py --src datasets/instagram/spectre.a.i --trigger "luna_v1" \\
        --out datasets/lora/luna/10_luna_v1

Kohya folder naming convention:
    <repeats>_<trigger>   e.g.  10_luna_v1
    The number of repeats controls how many times each image is seen per epoch.
    For small datasets (30–100 images) use 10–20 repeats.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

console = Console()

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_blip2(device: str):
    """Load Salesforce BLIP-2 captioner."""
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    console.log("[cyan]Loading BLIP-2 (salesforce/blip2-opt-2.7b)...[/]")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    model.eval()
    return processor, model


def load_blip_large(device: str):
    """Load lighter Salesforce BLIP-large captioner (~4 GB)."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    console.log("[cyan]Loading BLIP-large (Salesforce/blip-image-captioning-large)...[/]")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    return processor, model


def caption_image(img_path: Path, processor, model, device: str, model_type: str) -> str:
    """Return a short descriptive caption for the image."""
    image = Image.open(img_path).convert("RGB")
    with torch.no_grad():
        if model_type == "blip2":
            inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
            out = model.generate(**inputs, max_new_tokens=60)
            return processor.decode(out[0], skip_special_tokens=True).strip()
        else:  # blip-large
            inputs = processor(image, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=60)
            return processor.decode(out[0], skip_special_tokens=True).strip()


def prepare(
    src_dir: Path,
    out_dir: Path,
    trigger: str,
    model_type: str,
    no_caption: bool,
    min_size_px: int,
    overwrite: bool,
) -> None:
    # ── collect source images ────────────────────────────────────────────────
    images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not images:
        console.print(f"[red]No images found in {src_dir}[/]")
        sys.exit(1)

    console.rule(f"[bold magenta]Dataset Prep → {trigger}[/]")
    console.print(f"  source     : [cyan]{src_dir}[/]  ({len(images)} images)")
    console.print(f"  output     : [cyan]{out_dir}[/]")
    console.print(f"  trigger    : [bold yellow]{trigger}[/]")
    console.print(f"  captioner  : {'none (trigger only)' if no_caption else model_type}\n")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load captioner ────────────────────────────────────────────────────────
    processor = model = None
    if not no_caption:
        device = pick_device()
        console.log(f"[dim]device: {device}[/]")
        if model_type == "blip2":
            processor, model = load_blip2(device)
        else:
            processor, model = load_blip_large(device)

    # ── process each image ────────────────────────────────────────────────────
    stats = {"ok": 0, "skip_small": 0, "skip_exists": 0, "error": 0}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}[/]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]{task.completed}/{task.total}[/]"),
        console=console,
    ) as progress:
        task = progress.add_task("Captioning", total=len(images))

        for img_path in images:
            progress.update(task, description=f"[dim]{img_path.name[:50]}[/]")

            # validate minimum resolution
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                if min(w, h) < min_size_px:
                    stats["skip_small"] += 1
                    progress.advance(task)
                    continue
            except Exception:
                stats["error"] += 1
                progress.advance(task)
                continue

            dest_img = out_dir / img_path.name
            dest_txt = out_dir / (img_path.stem + ".txt")

            if dest_txt.exists() and not overwrite:
                stats["skip_exists"] += 1
                progress.advance(task)
                continue

            # copy image
            if not dest_img.exists() or overwrite:
                shutil.copy2(img_path, dest_img)

            # generate caption
            if no_caption:
                caption = trigger
            else:
                try:
                    raw = caption_image(img_path, processor, model, device, model_type)
                    caption = f"{trigger}, {raw}"
                except Exception as exc:
                    console.log(f"[yellow]Caption failed for {img_path.name}:[/] {exc}")
                    caption = trigger

            dest_txt.write_text(caption, encoding="utf-8")
            stats["ok"] += 1
            progress.advance(task)

    # ── summary ───────────────────────────────────────────────────────────────
    console.rule("[bold green]Done[/]")
    t = Table(show_header=False, box=None)
    t.add_row("[green]✓ prepared[/]", str(stats["ok"]))
    t.add_row("[dim]skipped (exists)[/]", str(stats["skip_exists"]))
    t.add_row("[yellow]skipped (too small)[/]", str(stats["skip_small"]))
    t.add_row("[red]errors[/]", str(stats["error"]))
    console.print(t)
    console.print(f"\nDataset ready at: [bold cyan]{out_dir}[/]")
    console.print(
        "\nNext step:\n"
        f"  [bold]uv run train_lora.py --dataset {out_dir.parent} --trigger {trigger}[/]"
    )

    # print a sample caption
    sample_txts = list(out_dir.glob("*.txt"))
    if sample_txts:
        console.print(f"\nSample caption ([dim]{sample_txts[0].name}[/]):")
        console.print(f"  [italic]{sample_txts[0].read_text(encoding='utf-8')}[/]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare LoRA training dataset with auto-captions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Folder of scraped images (e.g. datasets/instagram/spectre.a.i)",
    )
    parser.add_argument(
        "--trigger",
        required=True,
        help="Trigger word/token for the LoRA (e.g. luna_v1, spectre_ai)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output dataset folder. Default: datasets/lora/<trigger>/10_<trigger>",
    )
    parser.add_argument(
        "--model",
        choices=["blip2", "blip"],
        default="blip",
        help="Captioner model: blip2 (~8GB) or blip (~4GB). Default: blip",
    )
    parser.add_argument(
        "--no-caption",
        action="store_true",
        help="Skip captioning — write only the trigger word as caption",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=512,
        help="Minimum image dimension in pixels (default: 512)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Kohya repeat count in folder name (default: 10)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing caption .txt files",
    )
    args = parser.parse_args()

    src_dir = Path(args.src)
    if not src_dir.exists():
        console.print(f"[red]Source directory not found: {src_dir}[/]")
        sys.exit(1)

    out_dir = (
        Path(args.out)
        if args.out
        else Path("datasets") / "lora" / args.trigger / f"{args.repeats}_{args.trigger}"
    )

    prepare(
        src_dir=src_dir,
        out_dir=out_dir,
        trigger=args.trigger,
        model_type=args.model,
        no_caption=args.no_caption,
        min_size_px=args.min_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
