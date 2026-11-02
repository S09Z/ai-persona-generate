"""
scripts/build_clean_snapshot.py -- Build a deduplicated, cleaned, dated dataset snapshot.

Reads the live kohya folder (never modifies it) and writes:
    <out>/<repeats>_<trigger>/   training images + captions
    <out>/holdout/               reference images never used for training
    <out>/manifest.csv           one row per source image with its fate

Steps: drop near-duplicates, clean BLIP artifacts from captions, caption any
image that has none (BLIP-large), and hold out N images deterministically.

Usage:
    uv run scripts/build_clean_snapshot.py datasets/lora/auri_v1/5_auri_v1 \
        --out datasets/lora/auri_v1_snapshot_20261101 --trigger auri_v1 --repeats 5
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from audit_dataset import EXTS, dhash, hamming  # noqa: E402

# BLIP artifacts and filler that add no signal to a caption
_CLEANUPS = [
    (re.compile(r"\b(arafed|araffe|araffed)\b", re.I), ""),
    (re.compile(r"^(there (is|are) )", re.I), ""),
    (re.compile(r"\b(a picture of|for a picture|for a photo)\b", re.I), ""),
    (re.compile(r"\s+"), " "),
]


def clean_caption(text: str, trigger: str) -> str:
    body = text.strip()
    if body.startswith(trigger):
        body = body[len(trigger):].lstrip(", ")
    for pat, rep in _CLEANUPS:
        body = pat.sub(rep, body)
    body = body.strip(" ,.")
    if body and body.split()[0].lower() in {"in", "on", "with", "sitting", "standing", "wearing"}:
        body = "a woman " + body
    return f"{trigger}, {body}" if body else trigger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trigger", default="auri_v1")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--holdout", type=int, default=10)
    ap.add_argument("--max-dist", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-blip", action="store_true", help="trigger-only caption for uncaptioned images")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if out.exists():
        sys.exit(f"{out} already exists; snapshots are immutable. Pick a new name.")
    train_dir = out / f"{args.repeats}_{args.trigger}"
    hold_dir = out / "holdout"
    train_dir.mkdir(parents=True)
    hold_dir.mkdir()

    images = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    rows, kept = [], []
    for p in images:
        with Image.open(p) as im:
            hv = dhash(im)
        dup = next((k for k in kept if hamming(k[1], hv) <= args.max_dist), None)
        if dup:
            rows.append({"file": p.name, "role": "dropped_duplicate", "dup_of": dup[0].name})
        else:
            kept.append((p, hv))
            rows.append({"file": p.name, "role": "", "dup_of": ""})
    by_name = {r["file"]: r for r in rows}

    rng = random.Random(args.seed)
    holdout = {p.name for p, _ in rng.sample(kept, min(args.holdout, len(kept)))}

    pending_caption: list[Path] = []
    for p, _ in kept:
        row = by_name[p.name]
        if p.name in holdout:
            shutil.copy2(p, hold_dir / p.name)
            row["role"] = "holdout"
            continue
        shutil.copy2(p, train_dir / p.name)
        cap_file = src / f"{p.stem}.txt"
        if cap_file.exists():
            text = cap_file.read_text(encoding="utf-8")
            row["role"] = "train"
            row["caption_source"] = "existing_cleaned"
        else:
            text = ""
            row["role"] = "train"
            row["caption_source"] = "trigger_only" if args.no_blip else "blip_new"
            if not args.no_blip:
                pending_caption.append(p)
        (train_dir / f"{p.stem}.txt").write_text(clean_caption(text, args.trigger), encoding="utf-8")

    if pending_caption:
        import torch
        from prepare_dataset import caption_image, load_blip_large, pick_device

        device = pick_device()
        processor, model = load_blip_large(device)
        for p in pending_caption:
            raw = caption_image(p, processor, model, device, "blip")
            (train_dir / f"{p.stem}.txt").write_text(clean_caption(raw, args.trigger), encoding="utf-8")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    fields = ["file", "role", "dup_of", "caption_source", "view", "notes"]
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})

    n = {k: sum(r["role"] == k for r in rows) for k in ("train", "holdout", "dropped_duplicate")}
    print(f"snapshot: {out}  train={n['train']} holdout={n['holdout']} dup_dropped={n['dropped_duplicate']}"
          f" newly_captioned={len(pending_caption)}")


if __name__ == "__main__":
    main()
