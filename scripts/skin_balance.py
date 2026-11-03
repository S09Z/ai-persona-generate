"""
scripts/skin_balance.py -- bring the arms, hands and neck of a depth-guided full-body image to the face's skin tone.

The generator tends to render arms darker and more orange than the face (the face pass fixes only the head).
This finds the skin outside the head -- pixels inside the depth guide's silhouette whose chroma matches the
right forearm, away from the torso and trousers -- and shifts their lightness and yellow-blue towards the
cheek tone, keeping the shading and texture. Plain-background images made with --depth-image only.

Usage:
    uv run scripts/skin_balance.py outputs/comfy_body/skin1            # *_face.png -> *_face_skin.png
    uv run scripts/skin_balance.py outputs/comfy_body/skin1 --debug    # also writes *_mask.png overlays
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
from scipy.ndimage import gaussian_filter  # noqa: E402

import body_metrics as bm  # noqa: E402

GUIDE = ROOT / "characters/model_001/target/depth_walk.png"
# arms hang outside this body zone (x range, below y); trousers and shirt are left alone
BODY_X = (330, 545)
BODY_TOP = 170
SLEEVE_X = 545  # right of this, rows above ARM_TOP are a sleeve or upper arm: left alone
ARM_TOP = 330


def skin_mask(rgb: np.ndarray, guide: np.ndarray, face: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, ab = lab[..., 0] * 100 / 255, lab[..., 1:] - 128
    core = cv2.erode(bm.arm_mask(rgb, bm.ARM_BOX).astype("uint8"), np.ones((5, 5), np.uint8)).astype(bool)
    if core.sum() < 40:
        return np.zeros(core.shape, bool)
    centre = np.median(ab[core], axis=0)
    near = np.linalg.norm(ab - centre, axis=-1) < 7.0
    person = cv2.dilate(guide.astype("uint8"), np.ones((5, 5), np.uint8)).astype(bool)
    skin_L = (L > 45) & (L < 90)
    # Skin and the beige wall have almost the same chroma, so shapes come from the guide, not from colour:
    # right forearm and hand = the guide silhouette right of the body below the sleeve; raised left arm = the
    # guide left of the face; chroma only drops bag, hair and shirt there
    m = np.zeros(guide.shape, bool)
    right = person.copy()
    right[:ARM_TOP] = False
    right[:, :SLEEVE_X] = False
    left = person.copy()
    left[:150] = False
    left[340:] = False
    left[:, 345:] = False
    m |= (right | (left & near)) & skin_L
    nose, ipd = face[2], float(np.linalg.norm(face[1] - face[0]))
    y0 = int(nose[1] + ipd * 1.1)  # below the chin: the neck
    neck = np.zeros(guide.shape, bool)
    neck[y0:int(y0 + ipd * 0.9), int(nose[0] - ipd * 0.9):int(nose[0] + ipd * 0.9)] = True
    m |= neck & near & skin_L
    m = cv2.morphologyEx(m.astype("uint8"), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    return m


def balance(img: Image.Image, guide: np.ndarray, arm_offset=(-3.0, 3.0)) -> tuple[Image.Image, np.ndarray, dict] | None:
    """Shift skin towards the cheek tone; arm_offset is the (L, b) the arms may keep relative to the face."""
    found = bm.face_box(img.crop(bm.HEAD_BOX))
    if found is None:
        return None
    _, p = found
    p = p + np.array(bm.HEAD_BOX[:2])
    rgb = np.asarray(img.convert("RGB"))
    m = skin_mask(rgb, guide, p)
    if m.sum() < 400:
        return None
    h, w = rgb.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    ipd = float(np.linalg.norm(p[1] - p[0]))
    cheeks = (((xx - (p[0] + (p[3] - p[0]) * 0.45)[0]) ** 2 + (yy - (p[0] + (p[3] - p[0]) * 0.45)[1]) ** 2 <= (ipd * 0.2) ** 2)
              | ((xx - (p[1] + (p[4] - p[1]) * 0.45)[0]) ** 2 + (yy - (p[1] + (p[4] - p[1]) * 0.45)[1]) ** 2 <= (ipd * 0.2) ** 2))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    cheek = np.median(lab[cheeks], axis=0)
    arm = np.median(lab[m], axis=0)
    # lightness, a and b towards the cheek (a, b in OpenCV's offset-128 scale, L in 0-255)
    target = cheek + np.array([arm_offset[0] * 255 / 100, 0.0, arm_offset[1]])
    delta = target - arm
    weight = gaussian_filter(m.astype("float32"), 2.5)[..., None]
    lab += weight * delta[None, None, :]
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    info = {"arm_before": (arm * [100 / 255, 1, 1] - [0, 128, 128]).round(1).tolist(),
            "cheek": (cheek * [100 / 255, 1, 1] - [0, 128, 128]).round(1).tolist(), "pixels": int(m.sum())}
    return Image.fromarray(out), m, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--guide", default=str(GUIDE))
    args = ap.parse_args()
    guide = np.asarray(Image.open(args.guide).convert("L")) > 75
    for f in sorted(Path(args.folder).glob("*_face.png")):
        res = balance(Image.open(f).convert("RGB"), guide)
        if res is None:
            print(f"{f.name}: skipped (no face or no skin found)")
            continue
        out, m, info = res
        out.save(f.with_name(f.stem + "_skin.png"))
        if args.debug:
            o = np.asarray(Image.open(f).convert("RGB")).copy()
            o[m] = (0.5 * o[m] + 0.5 * np.array([255, 0, 0])).astype("uint8")
            Image.fromarray(o).save(f.with_name(f.stem + "_mask.png"))
        print(f"{f.name}: {info}")


if __name__ == "__main__":
    main()
