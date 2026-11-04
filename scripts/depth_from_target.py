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


def widen_shoulders(arr: np.ndarray, scale: float, cx: int = 430, y0: int = 215, y1: int = 330) -> np.ndarray:
    """Stretch the rows y0..y1 (shoulders and upper chest) sideways about cx, fading in and out over 40 rows."""
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    fade = np.clip(np.minimum(yy - (y0 - 40), (y1 + 40) - yy) / 40.0, 0, 1)
    s = 1 + (scale - 1) * fade
    return cv2.remap(arr, cx + (xx - cx) / s, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def boost_bust(arr: np.ndarray, amount: int, centres=((398, 305), (478, 305)), sigma: float = 32.0) -> np.ndarray:
    """Add two soft bumps to the depth on the chest so the bust reads fuller."""
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    bump = sum(np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)) for x, y in centres)
    out = arr.astype("float32") + amount * bump * (arr > 100)
    return np.clip(out, 0, 255).astype(arr.dtype)


ARM_BOXES =((525, 200, 625, 540), (270, 130, 355, 340))  # right arm, raised left arm on the 832x1216 canvas


def grow_arms(arr: np.ndarray, px: int) -> np.ndarray:
    """Fatten the arms by about px pixels: a max filter inside feathered boxes around them."""
    from scipy.ndimage import gaussian_filter
    grown = cv2.dilate(arr, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1)))
    w = np.zeros(arr.shape, "float32")
    for x0, y0, x1, y1 in ARM_BOXES:
        w[y0:y1, x0:x1] = 1
    w = gaussian_filter(w, 8)
    return (arr * (1 - w) + grown * w).astype(arr.dtype)


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
    ap.add_argument("--head-scale", type=float, default=0.8,
                    help="scale the head and hair in the depth map about the neck (0.9 = 10%% smaller)")
    ap.add_argument("--arm-grow", type=int, default=4, help="fatten the arms in the depth map by about this many pixels")
    ap.add_argument("--shoulder-scale", type=float, default=1.08, help="widen the shoulders and upper chest sideways")
    ap.add_argument("--bust-boost", type=int, default=20, help="depth added on the chest (0-255 scale, try 15-25)")
    ap.add_argument("--flip", action="store_true",
                    help="mirror the guide (a second pose facing the other way; use --no-skin-balance with it)")
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
    if args.shoulder_scale != 1.0:
        arr = widen_shoulders(arr, args.shoulder_scale)
    if args.bust_boost:
        arr = boost_bust(arr, args.bust_boost)
    if args.arm_grow:
        arr = grow_arms(arr, args.arm_grow)
    if args.flip:
        arr = arr[:, ::-1].copy()
    Image.fromarray(arr).convert("RGB").save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
