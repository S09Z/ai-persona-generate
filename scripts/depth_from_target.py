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

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/depth/depth-anything-v2-small"
SIZE = (832, 1216)


def shrink_head(arr: np.ndarray, scale: float, cx: int, cy: int) -> np.ndarray:
    """Scale the rows above the neck about (cx, cy); the effect fades out between cy and cy + 80."""
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    fade = np.clip(1 - (yy - cy) / 80.0, 0, 1)
    s = 1 + (scale - 1) * fade
    mx = cx + (xx - cx) / s
    my = cy + (yy - cy) / s
    return cv2.remap(arr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=str(ROOT / "characters/model_001/target/sora_sheet.png"))
    ap.add_argument("--box", default="0,0,522,1536", help="left,top,right,bottom of the full-body panel")
    ap.add_argument("--out", default=str(ROOT / "characters/model_001/target/depth_walk.png"))
    ap.add_argument("--no-flat-bg", dest="flat_bg", action="store_false",
                    help="keep the estimated background depth (it gets copied as scenery)")
    ap.add_argument("--person-level", type=float, default=0.5, help="depth above this counts as the person")
    ap.add_argument("--edge-px", type=int, default=6, help="width of the side strips that sample the floor depth")
    ap.add_argument("--floor-margin", type=float, default=0.08, help="how far above the floor depth the person starts")
    ap.add_argument("--flat-bg-value", type=float, default=0.2, help="constant depth for the background")
    ap.add_argument("--head-scale", type=float, default=1.0,
                    help="scale the head and hair in the depth map about the neck (0.9 = 10%% smaller)")
    ap.add_argument("--neck-x", type=int, default=430, help="neck anchor on the 832x1216 canvas")
    ap.add_argument("--neck-y", type=int, default=240)
    args = ap.parse_args()

    img =Image.open(args.sheet).convert("RGB").crop(tuple(int(v) for v in args.box.split(",")))
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL).eval()
    with torch.no_grad():
        out = model(**proc(images=img, return_tensors="pt")).predicted_depth
    d = torch.nn.functional.interpolate(out[None], size=img.size[::-1], mode="bicubic")[0, 0].numpy()
    d = (d - d.min()) / (d.max() - d.min())
    if args.flat_bg:  # the street behind her would otherwise be copied as stairs and pillars
        from scipy.ndimage import gaussian_filter
        # the floor gets nearer (brighter) towards the bottom: take its depth per row from the panel's side
        # edges, where only floor is visible, and call "person" what stands out from it
        edge = np.concatenate([d[:, :args.edge_px], d[:, -args.edge_px:]], axis=1)
        floor = gaussian_filter(np.median(edge, axis=1), 15)
        level = np.maximum(args.person_level, floor + args.floor_margin)[:, None]
        person = gaussian_filter((d > level).astype("float32"), 6)
        d = d * person + args.flat_bg_value * (1 - person)
    depth = Image.fromarray((d * 255).astype("uint8"))

    w, h = SIZE
    depth = depth.resize((round(depth.width * h / depth.height), h), Image.LANCZOS)
    arr = np.asarray(depth)
    pad = w - arr.shape[1]
    left = pad // 2
    arr = np.pad(arr, ((0, 0), (left, pad - left)), mode="edge")
    if args.head_scale != 1.0:
        arr = shrink_head(arr, args.head_scale, args.neck_x, args.neck_y)
    Image.fromarray(arr).convert("RGB").save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
