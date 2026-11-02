"""
scripts/eval_grid.py -- Generate a fixed-seed evaluation grid and contact sheets.

Every image uses identical inference settings (same base model, sampler,
steps, CFG, size, negative prompt); only the LoRA / weight varies. Output:

    <out>/<label>_w<weight>/<prompt>_s<seed>.png
    <out>/settings.json
    <out>/contact_<prompt>.png     rows = seeds, columns = variants

Usage:
    # EXP-006: weight sweep with one LoRA
    uv run scripts/eval_grid.py --out experiments/EXP-006-lora-weight/evaluation \
        --set weight --lora auri=models/loras/auri_v1-000004.safetensors \
        --weights 0.6 0.8 1.0 1.2

    # EXP-007: control vs variant at a fixed weight
    uv run scripts/eval_grid.py --out experiments/EXP-007-network-dropout/evaluation \
        --set dropout --weights 1.0 \
        --lora control=experiments/EXP-007-network-dropout/control/model/EXP-007-network-dropout-control.safetensors \
        --lora dropout=experiments/EXP-007-network-dropout/variant/model/EXP-007-network-dropout-variant.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

load_dotenv(ROOT / ".env")


def contact_sheet(cells: dict, seeds: list[int], variants: list[str], out: Path, thumb: int = 384) -> None:
    pad, head = 6, 24
    w = len(variants) * (thumb + pad) + pad
    h = len(seeds) * (thumb + pad) + pad + head
    sheet = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(sheet)
    for ci, v in enumerate(variants):
        draw.text((pad + ci * (thumb + pad) + 4, 6), v, fill="black")
    for ri, s in enumerate(seeds):
        for ci, v in enumerate(variants):
            im = Image.open(cells[(v, s)]).convert("RGB")
            im.thumbnail((thumb, thumb))
            sheet.paste(im, (pad + ci * (thumb + pad), head + pad + ri * (thumb + pad)))
    sheet.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", dest="pset", required=True, help="prompt set in experiments/eval_prompts.toml")
    ap.add_argument("--lora", action="append", default=[], metavar="label=path",
                    help="repeatable; omit to include only the base model")
    ap.add_argument("--weights", type=float, nargs="+", default=[1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--base", action="store_true", help="also render a no-LoRA column")
    ap.add_argument("--steps", type=int, default=int(os.getenv("STEPS", "25")))
    ap.add_argument("--cfg", type=float, default=float(os.getenv("GUIDANCE_SCALE", "6.0")))
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    spec = tomllib.loads((ROOT / "experiments" / "eval_prompts.toml").read_text(encoding="utf-8"))[args.pset]
    prompts: dict[str, str] = spec["prompts"]
    seeds = args.seeds or spec["seeds"]
    loras = dict(item.split("=", 1) for item in args.lora)

    import torch

    from idol_generate.pipeline import load_pipeline
    from idol_generate.prompt_builder import NEGATIVE_PROMPT

    device = os.getenv("DEVICE", "cuda")
    pipe = load_pipeline(
        os.getenv("MODEL_ID", "SG161222/RealVisXL_V4.0"), os.getenv("MODEL_PATH", ""), device
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, str, float]] = []  # (column label, lora label, weight)
    if args.base or not loras:
        variants.append(("base", "", 0.0))
    for label in loras:
        variants += [(f"{label}_w{w}", label, w) for w in args.weights]

    cells: dict[tuple[str, str, int], Path] = {}
    active = None
    for col, label, weight in variants:
        if label != active:
            if active:
                pipe.delete_adapters("character")
            if label:
                p = Path(loras[label])
                if not p.exists():
                    sys.exit(f"LoRA not found: {p}")
                pipe.load_lora_weights(str(p.parent), weight_name=p.name, adapter_name="character")
            active = label
        if label:
            pipe.set_adapters(["character"], adapter_weights=[weight])
        (out / col).mkdir(exist_ok=True)
        for pname, prompt in prompts.items():
            for seed in seeds:
                path = out / col / f"{pname}_s{seed}.png"
                gen = torch.Generator(device=device).manual_seed(seed)
                img = pipe(
                    prompt=prompt, negative_prompt=NEGATIVE_PROMPT, num_inference_steps=args.steps,
                    guidance_scale=args.cfg, width=args.size, height=args.size, generator=gen,
                ).images[0]
                img.save(path)
                cells[(pname, col, seed)] = path
                print("saved", path, flush=True)

    cols = [c for c, _, _ in variants]
    for pname in prompts:
        contact_sheet({(c, s): cells[(pname, c, s)] for c in cols for s in seeds}, seeds, cols,
                      out / f"contact_{pname}.png")
    (out / "settings.json").write_text(json.dumps({
        "prompt_set": args.pset, "prompts": prompts, "seeds": seeds, "loras": loras,
        "weights": args.weights, "steps": args.steps, "cfg": args.cfg, "size": args.size,
        "sampler": "DPM++ multistep, karras sigmas", "negative_prompt": NEGATIVE_PROMPT,
        "base_model": os.getenv("MODEL_PATH") or os.getenv("MODEL_ID"),
    }, indent=2), encoding="utf-8")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
