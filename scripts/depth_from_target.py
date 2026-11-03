"""
scripts/depth_from_target.py -- depth map of the body in the Sora target sheet, as a shape guide.

Depth Anything V2 Small (models/depth/depth-anything-v2-small, see README) estimates depth for the
full-body panel of characters/model_001/target/sora_sheet.png; the result is scaled to the generator's
832x1216 canvas and centred, with the sides filled from the panel edge. generate_comfy.py --depth-image
feeds it to the depth ControlNet so the body proportions follow the target while the face stays the Master Face.

Usage:
    uv run scripts/depth_from_target.py                       # writes characters/model_001/target/depth_walk.png
    uv run scripts/depth_from_target.py --box 0,0,522,1536    # other panel of the sheet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/depth/depth-anything-v2-small"
SIZE = (832, 1216)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=str(ROOT / "characters/model_001/target/sora_sheet.png"))
    ap.add_argument("--box", default="0,0,522,1536", help="left,top,right,bottom of the full-body panel")
    ap.add_argument("--out", default=str(ROOT / "characters/model_001/target/depth_walk.png"))
    args = ap.parse_args()

    img = Image.open(args.sheet).convert("RGB").crop(tuple(int(v) for v in args.box.split(",")))
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL).eval()
    with torch.no_grad():
        out = model(**proc(images=img, return_tensors="pt")).predicted_depth
    d = torch.nn.functional.interpolate(out[None], size=img.size[::-1], mode="bicubic")[0, 0].numpy()
    d = (d - d.min()) / (d.max() - d.min())
    depth = Image.fromarray((d * 255).astype("uint8"))

    w, h = SIZE
    depth = depth.resize((round(depth.width * h / depth.height), h), Image.LANCZOS)
    arr = np.asarray(depth)
    pad = w - arr.shape[1]
    left = pad // 2
    arr = np.pad(arr, ((0, 0), (left, pad - left)), mode="edge")
    Image.fromarray(arr).convert("RGB").save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
