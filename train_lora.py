"""
train_lora.py -- Launch LoRA training via kohya_ss sd-scripts.

What it does:
  1. Checks that kohya_ss is cloned into vendor/kohya_ss
  2. Installs/verifies kohya deps
  3. Validates the dataset folder structure
  4. Builds the accelerate launch command from a TOML config
  5. Streams training output with a Rich progress display

Usage:
    # Quick start with defaults (trains luna_v1 LoRA)
    uv run train_lora.py

    # Custom config and trigger
    uv run train_lora.py --config configs/lora_luna.toml

    # Dry run -- print the launch command without running
    uv run train_lora.py --dry-run

    # Use CUDA instead of auto-detected device
    uv run train_lora.py --device cuda

Requirements (run once):
    git clone https://github.com/kohya-ss/sd-scripts vendor/kohya_ss
    pip install -r vendor/kohya_ss/requirements.txt
    # or let this script do it automatically on first run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

KOHYA_DIR = Path("vendor/kohya_ss")
KOHYA_TRAIN_SCRIPT = KOHYA_DIR / "sdxl_train_network.py"
KOHYA_REQUIREMENTS = KOHYA_DIR / "requirements.txt"


# ── helpers ───────────────────────────────────────────────────────────────────


def check_kohya() -> None:
    """Ensure kohya_ss is present and its deps are installed."""
    if not KOHYA_DIR.exists():
        console.print(
            Panel(
                "[red]vendor/kohya_ss not found.[/]\n\n"
                "Clone it once:\n"
                "  [bold]git clone https://github.com/kohya-ss/sd-scripts vendor/kohya_ss[/]",
                title="Missing kohya_ss",
                border_style="red",
            )
        )
        sys.exit(1)

    if not KOHYA_TRAIN_SCRIPT.exists():
        console.print(f"[red]Expected training script not found: {KOHYA_TRAIN_SCRIPT}[/]")
        sys.exit(1)

    console.log(f"[green]kohya_ss found[/] at {KOHYA_DIR}")


def install_kohya_deps() -> None:
    """Install kohya requirements into the current venv."""
    if not KOHYA_REQUIREMENTS.exists():
        console.log("[yellow]No requirements.txt found in kohya_ss — skipping dep install[/]")
        return
    console.log("[cyan]Installing kohya_ss requirements...[/]")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(KOHYA_REQUIREMENTS), "-q"],
        check=True,
    )
    console.log("[green]kohya_ss deps installed[/]")


def validate_dataset(dataset_dir: Path, trigger: str) -> tuple[int, int]:
    """Check dataset folder and return (image_count, caption_count)."""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in dataset_dir.rglob("*") if p.suffix.lower() in exts]
    captions = list(dataset_dir.rglob("*.txt"))

    console.log(f"[cyan]Dataset:[/] {len(images)} images, {len(captions)} captions")

    missing = [p for p in images if not (p.with_suffix(".txt")).exists()]
    if missing:
        console.log(
            f"[yellow]{len(missing)} images have no caption.[/] "
            "Run: [bold]uv run prepare_dataset.py ...[/]"
        )

    sample_caps = list(dataset_dir.rglob("*.txt"))[:3]
    for cap in sample_caps:
        text = cap.read_text(encoding="utf-8").strip()
        has_trigger = trigger in text
        status = "[green]✓[/]" if has_trigger else "[yellow]⚠ trigger missing[/]"
        console.log(f"  {status} {cap.name}: [dim]{text[:80]}[/]")

    return len(images), len(captions)


def build_command(config_path: Path, device: str, extra_args: list[str]) -> list[str]:
    """Build the accelerate + kohya launch command from the TOML config."""
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    train_cfg = cfg.get("train", {})
    lora_cfg = cfg.get("lora", {})
    model_cfg = cfg.get("model", {})
    output_cfg = cfg.get("output", {})
    logging_cfg = cfg.get("logging", {})
    sample_cfg = cfg.get("sample", {})
    caption_cfg = cfg.get("caption", {})
    general_cfg = cfg.get("general", {})

    # Determine dataset dir from config
    datasets_cfg = cfg.get("datasets", {})
    subsets = datasets_cfg.get("subsets", [])
    dataset_dir = Path(subsets[0]["image_dir"]) if subsets else None

    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--mixed_precision",
        train_cfg.get("mixed_precision", "bf16"),
        "--num_processes",
        "1",
    ]

    # MPS needs special accelerate flags
    if device == "mps":
        cmd += ["--use_mps_device"]
    elif device == "cpu":
        cmd += ["--cpu"]

    cmd += [str(KOHYA_TRAIN_SCRIPT)]

    # ── model ────────────────────────────────────────────────────────────────
    cmd += ["--pretrained_model_name_or_path", model_cfg.get("pretrained_model_name_or_path", "")]
    if model_cfg.get("sdxl"):
        pass  # sdxl_train_network.py is already SDXL-specific

    # ── dataset ──────────────────────────────────────────────────────────────
    if dataset_dir:
        cmd += ["--train_data_dir", str(dataset_dir)]
    cmd += [
        "--caption_extension",
        subsets[0].get("caption_extension", ".txt") if subsets else ".txt",
    ]

    # ── bucketing ────────────────────────────────────────────────────────────
    if general_cfg.get("enable_bucket"):
        cmd += ["--enable_buckets"]
    cmd += [
        "--min_bucket_reso",
        str(general_cfg.get("min_bucket_reso", 512)),
        "--max_bucket_reso",
        str(general_cfg.get("max_bucket_reso", 1024)),
        "--bucket_reso_steps",
        str(general_cfg.get("bucket_reso_steps", 64)),
    ]
    if general_cfg.get("bucket_no_upscale"):
        cmd += ["--bucket_no_upscale"]

    # ── output ───────────────────────────────────────────────────────────────
    Path(output_cfg.get("output_dir", "models/loras")).mkdir(parents=True, exist_ok=True)
    cmd += [
        "--output_dir",
        output_cfg.get("output_dir", "models/loras"),
        "--output_name",
        output_cfg.get("output_name", "lora"),
        "--save_model_as",
        output_cfg.get("save_model_as", "safetensors"),
        "--save_every_n_epochs",
        str(output_cfg.get("save_every_n_epochs", 2)),
    ]

    # ── training ─────────────────────────────────────────────────────────────
    cmd += [
        "--max_train_epochs",
        str(train_cfg.get("max_train_epochs", 15)),
        "--train_batch_size",
        str(train_cfg.get("train_batch_size", 1)),
        "--resolution",
        train_cfg.get("resolution", "1024,1024"),
        "--mixed_precision",
        train_cfg.get("mixed_precision", "bf16"),
        "--optimizer_type",
        train_cfg.get("optimizer_type", "AdamW"),
        "--learning_rate",
        str(train_cfg.get("learning_rate", 1e-4)),
        "--unet_lr",
        str(train_cfg.get("unet_lr", 1e-4)),
        "--text_encoder_lr",
        str(train_cfg.get("text_encoder_lr", 5e-5)),
        "--lr_scheduler",
        train_cfg.get("lr_scheduler", "cosine_with_restarts"),
        "--lr_warmup_steps",
        str(train_cfg.get("lr_warmup_steps", 100)),
        "--lr_scheduler_num_cycles",
        str(train_cfg.get("lr_scheduler_num_cycles", 3)),
        "--gradient_accumulation_steps",
        str(train_cfg.get("gradient_accumulation_steps", 4)),
        "--clip_skip",
        str(train_cfg.get("clip_skip", 2)),
        "--seed",
        str(train_cfg.get("seed", 42)),
    ]

    if train_cfg.get("gradient_checkpointing"):
        cmd += ["--gradient_checkpointing"]

    # ── LoRA network ──────────────────────────────────────────────────────────
    cmd += [
        "--network_module",
        lora_cfg.get("network_module", "networks.lora"),
        "--network_dim",
        str(lora_cfg.get("network_dim", 32)),
        "--network_alpha",
        str(lora_cfg.get("network_alpha", 16)),
    ]
    if lora_cfg.get("network_train_unet_only"):
        cmd += ["--network_train_unet_only"]

    # ── captions ─────────────────────────────────────────────────────────────
    if caption_cfg.get("shuffle_caption"):
        cmd += ["--shuffle_caption"]
    cmd += [
        "--keep_tokens",
        str(caption_cfg.get("keep_tokens", 1)),
        "--caption_dropout_rate",
        str(caption_cfg.get("caption_dropout_rate", 0.05)),
    ]

    # ── logging ──────────────────────────────────────────────────────────────
    if logging_cfg.get("log_with"):
        Path(logging_cfg.get("logging_dir", "logs")).mkdir(parents=True, exist_ok=True)
        cmd += [
            "--log_with",
            logging_cfg["log_with"],
            "--logging_dir",
            logging_cfg.get("logging_dir", "logs"),
        ]

    # ── sample previews ───────────────────────────────────────────────────────
    if sample_cfg.get("sample_every_n_epochs"):
        cmd += [
            "--sample_every_n_epochs",
            str(sample_cfg["sample_every_n_epochs"]),
            "--sample_sampler",
            sample_cfg.get("sample_sampler", "euler_a"),
        ]
        prompts = sample_cfg.get("sample_prompts", [])
        if prompts:
            # kohya expects prompts as a file
            prompt_file = Path("configs") / "_sample_prompts.txt"
            prompt_file.write_text("\n".join(prompts), encoding="utf-8")
            cmd += ["--sample_prompts", str(prompt_file)]

    cmd += extra_args
    return cmd


def run_training(cmd: list[str]) -> None:
    """Run the training command, streaming output."""
    console.print(
        Panel(
            " ".join(cmd[:6]) + " \\\n  " + " \\\n  ".join(cmd[6:]),
            title="[bold cyan]Launch command[/]",
            border_style="cyan",
        )
    )
    console.print("\n[bold green]Starting training...[/] (Ctrl+C to stop)\n")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        for line in iter(process.stdout.readline, ""):
            console.print(line, end="", markup=False)
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        console.print("\n[yellow]Training interrupted.[/]")
        return

    if process.returncode == 0:
        console.rule("[bold green]Training complete![/]")
    else:
        console.print(f"[red]Training exited with code {process.returncode}[/]")
        sys.exit(process.returncode)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch LoRA training via kohya_ss",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/lora_luna.toml",
        help="Path to training config TOML (default: configs/lora_luna.toml)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install kohya_ss requirements before training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launch command without running it",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded directly to kohya train script",
    )
    args = parser.parse_args()

    # ── device detection ──────────────────────────────────────────────────────
    import torch

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # ── pre-flight ────────────────────────────────────────────────────────────
    check_kohya()
    if args.install_deps:
        install_kohya_deps()

    config_path = Path(args.config)
    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}[/]")
        sys.exit(1)

    # Load config for validation display
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    subsets = cfg.get("datasets", {}).get("subsets", [])
    dataset_dir = Path(subsets[0]["image_dir"]) if subsets else None
    trigger = cfg.get("output", {}).get("output_name", "lora")

    # ── summary table ─────────────────────────────────────────────────────────
    console.rule("[bold magenta]LoRA Training Setup[/]")
    t = Table(show_header=False, box=None)
    t.add_row("[cyan]config[/]", str(config_path))
    t.add_row("[cyan]device[/]", f"[bold]{device}[/]")
    t.add_row("[cyan]base model[/]", cfg.get("model", {}).get("pretrained_model_name_or_path", ""))
    t.add_row("[cyan]output name[/]", f"[bold yellow]{trigger}[/]")
    t.add_row("[cyan]output dir[/]", cfg.get("output", {}).get("output_dir", ""))
    t.add_row("[cyan]epochs[/]", str(cfg.get("train", {}).get("max_train_epochs", "?")))
    t.add_row("[cyan]rank (dim)[/]", str(cfg.get("lora", {}).get("network_dim", "?")))
    console.print(t)
    console.print()

    if dataset_dir and dataset_dir.exists():
        validate_dataset(dataset_dir, trigger)
    elif dataset_dir:
        console.print(
            f"[red]Dataset directory not found: {dataset_dir}[/]\n"
            f"Run first: [bold]uv run prepare_dataset.py --src <images> --trigger {trigger}[/]"
        )
        sys.exit(1)

    # ── build and run ─────────────────────────────────────────────────────────
    cmd = build_command(config_path, device, args.extra or [])

    if args.dry_run:
        console.print(
            Panel(
                " \\\n  ".join(cmd),
                title="[bold cyan]Dry run — command[/]",
                border_style="dim",
            )
        )
        return

    run_training(cmd)

    # ── post-training: print where LoRA was saved ─────────────────────────────
    output_dir = Path(cfg.get("output", {}).get("output_dir", "models/loras"))
    output_name = cfg.get("output", {}).get("output_name", "lora")
    lora_path = output_dir / f"{output_name}.safetensors"
    console.print(f"\nLoRA saved to: [bold cyan]{lora_path}[/]")
    console.print(
        f"\nUpdate [bold]characters/luna.json[/] lora.path → [dim]{lora_path}[/]\n"
        f"and    lora.trigger_word → [dim]{trigger}[/]"
    )


if __name__ == "__main__":
    main()
