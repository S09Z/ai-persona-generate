"""
scripts/build_model_snapshot.py -- Build a LoRA training snapshot from the MODEL_001 synthetic dataset.

Unlike build_clean_snapshot.py (BLIP captions, random holdout) this honours the manual split in
characters/model_001/dataset/manifest.json (train / holdout / excluded). Captions are the trigger word
plus what varies (angle, expression, outfit, scene): the identity phrases are dropped on purpose so the
trigger word absorbs them.

Usage:
    uv run scripts/build_model_snapshot.py --out datasets/lora/model_001_snapshot_20261102 --trigger m001_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_comfy import BODY_BASE, HEAD_BASE  # noqa: E402

DATASET = ROOT / "characters/model_001/dataset"


def caption(prompt: str, trigger: str) -> str:
    for base in (HEAD_BASE, BODY_BASE):
        prompt = prompt.replace(base, "")
    prompt = re.sub(r"\(([^():]+):[\d.]+\)", r"\1", prompt)  # drop emphasis weights
    parts = [p.strip() for p in prompt.split(",") if p.strip()]
    return ", ".join([trigger, *parts])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trigger", default="m001_v1")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; snapshots are immutable, pick a new name")
    train_dir = out / f"{args.repeats}_{args.trigger}"
    hold_dir = out / "holdout"
    train_dir.mkdir(parents=True)
    hold_dir.mkdir()

    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for name, m in sorted(manifest.items()):
        split = m["split"]
        if split == "train":
            shutil.copy2(DATASET / f"{name}.png", train_dir / f"{name}.png")
            (train_dir / f"{name}.txt").write_text(caption(m["prompt"], args.trigger), encoding="utf-8")
        elif split == "holdout":
            shutil.copy2(DATASET / f"{name}.png", hold_dir / f"{name}.png")
        rows.append([name, m["view"], split, m.get("excluded_reason", "")])
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "view", "split", "reason"])
        w.writerows(rows)
    n_train = sum(r[2] == "train" for r in rows)
    print(f"snapshot {out}: {n_train} train x{args.repeats}, {sum(r[2] == 'holdout' for r in rows)} holdout")


if __name__ == "__main__":
    main()
