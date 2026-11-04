"""
scripts/body_metrics.py -- check arm size against the head and skin-tone balance in depth-guided full-body images.

For images made with --depth-image (pose fixed, so the right forearm sits in the same place in every image) it
measures, per image and for the Sora target:
  - forearm width / face width (YuNet box on the head crop). Real adults sit around 0.45-0.55.
  - skin tone (Lab) of the cheeks, the neck and the forearm, and how much the forearm varies top to bottom.
The forearm is found as the pixels that differ in chroma from the wall beside it, so it assumes a plain background.
The tone numbers are usable; the arm/face width ratio is noisy (jeans and hands leak into the mask), so judge
arm size by eye from a side-by-side crop and treat arm/face only as a rough hint.

Usage:
    uv run scripts/body_metrics.py outputs/comfy_body/nofloor
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import face_id  # noqa: E402

TARGET_SHEET = ROOT / "characters/model_001/target/sora_sheet.png"
ARM_BOX = (560, 470, 640, 620)   # right forearm on the 832x1216 canvas: x0, y0, x1, y1
HEAD_BOX = (265, 0, 565, 300)    # head_box(0) of generate_comfy.py
# the same forearm and head in the target panel (0,0,522,1536), which maps to the canvas by *0.79 and +209 in x
PANEL_SCALE = 1216 / 1536
PANEL_X0 = (832 - 522 * PANEL_SCALE) / 2
TARGET_ARM_BOX = (420, 595, 530, 785)
TARGET_HEAD_BOX = (100, 40, 420, 360)


def face_box(head: Image.Image) -> tuple[float, np.ndarray] | None:
    """Face-box width and 5 landmarks on a head crop, in the crop's own pixels."""
    scale = 3
    big = head.convert("RGB").resize((head.width * scale, head.height * scale), Image.LANCZOS)
    bgr = cv2.cvtColor(np.asarray(big), cv2.COLOR_RGB2BGR)
    det = cv2.FaceDetectorYN.create(str(face_id.MODELS / "yunet.onnx"), "", (bgr.shape[1], bgr.shape[0]),
                                    score_threshold=0.5)
    _, faces = det.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    f = max(faces, key=lambda r: r[14])
    return float(f[2]) / scale, f[4:14].reshape(5, 2) / scale


def arm_mask(rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Boolean mask over the whole image: pixels in `box` whose a/b chroma differs from the wall at the box's right edge."""
    x0, y0, x1, y1 = box
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ab = lab[..., 1:]
    wall = np.median(ab[y0:y1, x1 - 12:x1].reshape(-1, 2), axis=0)
    dist = np.linalg.norm(ab - wall, axis=-1)
    m = np.zeros(rgb.shape[:2], bool)
    m[y0:y1, x0:x1] = dist[y0:y1, x0:x1] > 5.5
    m = cv2.morphologyEx(m.astype("uint8"), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    return m


def arm_width(m: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = box
    widths = []
    for y in range(y0, y1):
        xs = np.flatnonzero(m[y, x0:x1])
        if xs.size:
            # the widest contiguous run that contains the row's central mass
            runs = np.split(xs, np.flatnonzero(np.diff(xs) > 2) + 1)
            widths.append(max(len(r) for r in runs))
    return float(np.median(widths)) if widths else float("nan")


def lab_median(rgb: np.ndarray, m: np.ndarray) -> np.ndarray:
    v = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[m]
    return np.array([np.median(v[:, 0]) * 100 / 255, np.median(v[:, 1]) - 128, np.median(v[:, 2]) - 128])


def measure(img: Image.Image) -> dict | None:
    found = face_box(img.crop(HEAD_BOX))
    if found is None:
        return None
    fw, p = found
    p = p + np.array(HEAD_BOX[:2])
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    ipd = float(np.linalg.norm(p[1] - p[0]))

    def disc(c, r):
        return (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= r * r

    cheeks = disc(p[0] + (p[3] - p[0]) * 0.45, ipd * 0.2) | disc(p[1] + (p[4] - p[1]) * 0.45, ipd * 0.2)
    neck = disc(p[2] + np.array([0, ipd * 1.35]), ipd * 0.25)
    m = arm_mask(rgb, ARM_BOX)
    aw = arm_width(m, ARM_BOX)
    core = cv2.erode(m.astype("uint8"), np.ones((5, 5), np.uint8)).astype(bool)
    mid = (ARM_BOX[1] + ARM_BOX[3]) // 2
    top, bot = core.copy(), core.copy()
    top[mid:] = False
    bot[:mid] = False
    if top.sum() < 40 or bot.sum() < 40:
        return None
    return {"arm_face": aw / fw, "arm_px": aw, "face_px": fw, "cheek": lab_median(rgb, cheeks),
            "neck": lab_median(rgb, neck), "arm": lab_median(rgb, core),
            "top_bot_L": float(lab_median(rgb, top)[0] - lab_median(rgb, bot)[0])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--suffix", default="face", help="measure *_<suffix>.png (face = before skin balance, final = after)")
    args = ap.parse_args()

    rows = []
    sheet = Image.open(TARGET_SHEET).convert("RGB").crop((0, 0, 522, 1536))
    tf = face_box(sheet.crop(TARGET_HEAD_BOX))
    if tf is not None:
        rgb = np.asarray(sheet)
        aw = arm_width(arm_mask(rgb, TARGET_ARM_BOX), TARGET_ARM_BOX)
        rows.append(("target", {"arm_face": aw / tf[0], "arm_px": aw * PANEL_SCALE, "face_px": tf[0] * PANEL_SCALE}))
    for f in sorted(Path(args.folder).glob(f"*_{args.suffix}.png")):
        m = measure(Image.open(f).convert("RGB"))
        if m is not None:
            rows.append((f.stem.split("_s")[-1].replace(f"_{args.suffix}", ""), m))

    fmt = lambda v: " ".join(f"{x:5.1f}" for x in v) if v is not None else "-"
    print(f"{'image':8} {'arm/face':>8} {'arm px':>7} {'face px':>8}  {'cheek L a b':>17}  {'neck L a b':>17}  {'arm L a b':>17}  top-bot L")
    for name, m in rows:
        print(f"{name:8} {m['arm_face']:8.2f} {m['arm_px']:7.0f} {m['face_px']:8.0f}  {fmt(m.get('cheek')):>17}  "
              f"{fmt(m.get('neck')):>17}  {fmt(m.get('arm')):>17}  {m.get('top_bot_L', float('nan')):5.1f}")


if __name__ == "__main__":
    main()
