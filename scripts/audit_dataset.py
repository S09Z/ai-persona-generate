"""
scripts/audit_dataset.py -- Audit a kohya image folder and write a manifest.

Reports missing captions, orphan captions, tiny images and near-duplicates
(difference hash), and writes manifest.csv next to the output you choose.
Read-only: it never modifies or deletes dataset files.

Usage:
    uv run scripts/audit_dataset.py datasets/lora/auri_v1/5_auri_v1 --out datasets/lora/auri_v1/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image
from rich.console import Console

console = Console()
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def dhash(img: Image.Image, size: int = 8) -> int:
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | (px[row * (size + 1) + col] > px[row * (size + 1) + col + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--max-dist", type=int, default=4, help="max hamming distance for duplicates")
    args = ap.parse_args()

    folder = Path(args.folder)
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    captions = {p.stem for p in folder.glob("*.txt")}

    rows = []
    for p in images:
        with Image.open(p) as im:
            w, h = im.size
            hv = dhash(im)
        cap = ""
        if p.stem in captions:
            cap = (folder / f"{p.stem}.txt").read_text(encoding="utf-8").strip()
        rows.append(
            {"file": p.name, "width": w, "height": h, "dhash": f"{hv:016x}",
             "has_caption": bool(cap), "dup_of": "", "caption": cap, "_h": hv}
        )

    # near-duplicate groups: later file points at the first kept one
    kept: list[dict] = []
    for r in rows:
        match = next((k for k in kept if hamming(k["_h"], r["_h"]) <= args.max_dist), None)
        if match:
            r["dup_of"] = match["file"]
        else:
            kept.append(r)

    image_stems = {p.stem for p in images}
    orphans = sorted(captions - image_stems)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "_h"])
        wr.writeheader()
        for r in rows:
            wr.writerow({k: v for k, v in r.items() if k != "_h"})

    console.print(f"images            : {len(images)}")
    console.print(f"captions          : {len(captions)}")
    console.print(f"images w/o caption: {sum(not r['has_caption'] for r in rows)}")
    console.print(f"orphan captions   : {len(orphans)}")
    console.print(f"near-duplicates   : {sum(bool(r['dup_of']) for r in rows)}")
    console.print(f"smallest side     : {min(min(r['width'], r['height']) for r in rows)} px")
    console.print(f"manifest          : {args.out}")


if __name__ == "__main__":
    main()
