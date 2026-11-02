"""
scripts/master_face_candidates.py -- Stage 1 of the MODEL_001 Master Face plan.

Text-only exploration (base RealVisXL, no LoRA, no identity model). Six face-detail
variants x N fixed seeds, all inside the locked choices in characters/model_001.json
(soft oval face, pale milky blue eyes, long dark coffee brown hair, Korean-Thai natural
girl-next-door look, fair skin with light freckles).

Output (default characters/model_001/master_face/candidates/):
    v<k>_s<seed>.png         individual images
    contact_v<k>.png         one sheet per variant
    settings.json            prompts, seeds, sampler settings

Usage:
    uv run scripts/master_face_candidates.py
    uv run scripts/master_face_candidates.py --seeds 1001 1002 --variants 1 2 --steps 8 --size 768   # quick test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

load_dotenv(ROOT / ".env")

# CLIP reads only 77 tokens, so the variant goes first and BASE stays short enough to fit after it.
BASE = (
    "photo of a young Thai woman, 27, korean-thai natural beauty, soft oval face, girl-next-door, "
    "fair skin, light freckles, pale milky blue eyes, long dark coffee brown hair, minimal makeup, "
    "ivory top, headshot, soft daylight, cream backdrop"
)

# Only soft facial details vary; no real-person or celebrity references.
VARIANTS = {
    1: "subtle gentle smile, softly arched brows",
    2: "slightly wide-set eyes, straight soft brows, small nose with a low soft bridge",
    3: "fuller lips, gentle cheekbones, arched brows, calm neutral expression",
    4: "slightly upturned eye corners, delicate nose, soft pointed chin",
    5: "rounder cheeks, soft jawline, warm closed-lip smile",
    6: "defined brows, high cheekbones, calm neutral expression, direct gaze",
}
DEFAULT_SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]


def sheet(paths: list[Path], out: Path, thumb: int = 384, cols: int = 4) -> None:
    rows = (len(paths) + cols - 1) // cols
    pad = 6
    s = Image.new("RGB", (cols * (thumb + pad) + pad, rows * (thumb + pad) + pad), "white")
    d = ImageDraw.Draw(s)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        im.thumbnail((thumb, thumb))
        x, y = pad + (i % cols) * (thumb + pad), pad + (i // cols) * (thumb + pad)
        s.paste(im, (x, y))
        d.text((x + 6, y + 6), p.stem.split("_s")[-1], fill="white")
    s.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "characters/model_001/master_face/candidates"))
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--variants", type=int, nargs="+", default=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    import torch

    from idol_generate.pipeline import load_pipeline
    from idol_generate.prompt_builder import NEGATIVE_PROMPT

    device = os.getenv("DEVICE", "cuda")
    pipe = load_pipeline(os.getenv("MODEL_ID", "SG161222/RealVisXL_V4.0"), os.getenv("MODEL_PATH", ""), device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for v in args.variants:
        prompt = f"{VARIANTS[v]}, {BASE}"
        paths = []
        for seed in args.seeds:
            path = out / f"v{v}_s{seed}.png"
            gen = torch.Generator(device=device).manual_seed(seed)
            pipe(
                prompt=prompt, negative_prompt=NEGATIVE_PROMPT, num_inference_steps=args.steps,
                guidance_scale=args.cfg, width=args.size, height=args.size, generator=gen,
            ).images[0].save(path)
            paths.append(path)
            print("saved", path, flush=True)
        sheet(paths, out / f"contact_v{v}.png")

    (out / "settings.json").write_text(json.dumps({
        "base_prompt": BASE, "variants": {str(k): VARIANTS[k] for k in args.variants},
        "seeds": args.seeds, "steps": args.steps, "cfg": args.cfg, "size": args.size,
        "negative_prompt": NEGATIVE_PROMPT, "sampler": "DPM++ multistep, karras sigmas",
        "base_model": os.getenv("MODEL_PATH") or os.getenv("MODEL_ID"),
    }, indent=2), encoding="utf-8")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
