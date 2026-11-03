"""
scripts/body_depth.py -- procedural hourglass body depth map (front/back view) for a depth ControlNet.

OpenPose only gives joint positions, so it cannot say how wide the hips are or how narrow the waist is.
This draws a body-volume map in the same 832x1216 coordinates as openpose_draw.body_pose: a torso
whose half-width follows a (bust, waist, hip) profile and two tapering legs, shaded like a cylinder
(bright centre, dark edges) so a depth ControlNet reads it as a rounded volume. Arms are left out on
purpose; the OpenPose skeleton places them. Front and back share one silhouette.

Usage:
    uv run scripts/body_depth.py            # writes outputs/body_depth_preview.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent

# (y in the 1216-high skeleton frame, torso half-width): neck, shoulder, bust, under-bust, waist, high hip, hip
TORSO = [(292, 26), (318, 78), (400, 80), (460, 68), (520, 54), (600, 80), (690, 96), (740, 80)]
LEG_TOP, LEG_KNEE, LEG_ANKLE = 640, 887, 1101
LEG = {"cx": [(700, 62), (800, 60), (887, 58), (1101, 54)],  # follows the skeleton knee/ankle x
       "hw": [(640, 32), (700, 36), (800, 34), (887, 26), (1000, 20), (1101, 16)]}


def _interp(table: list[tuple[float, float]], y: float) -> float:
    ys, vs = zip(*table)
    return float(np.interp(y, ys, vs))


def hourglass_depth(size: tuple[int, int] = (832, 1216), waist: float = 1.0, hip: float = 1.0, shoulder: float = 1.0) -> Image.Image:
    """`waist`, `hip` and `shoulder` (shoulders and bust, y up to 460) scale the half-width at those heights (1.0 = default)."""
    w, h = size
    s = h / 1216
    cx = w / 2
    depth = np.zeros((h, w), dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    for row in range(h):
        y = row / s
        parts = []
        if TORSO[0][0] <= y <= TORSO[-1][0]:
            k = 1.0
            if y <= 460:
                k = shoulder
            elif 480 <= y <= 560:
                k = waist
            elif y >= 600:
                k = hip
            parts.append((cx, _interp(TORSO, y) * k * s))
        if LEG_TOP <= y <= LEG_ANKLE:
            leg_hw = _interp(LEG["hw"], y) * (hip if y < 800 else 1.0) * s
            for side in (-1, 1):
                parts.append((cx + side * _interp(LEG["cx"], max(y, 700)) * s, leg_hw))
        for centre, half in parts:
            dx = np.abs(xs - centre) / max(half, 1e-3)
            val = np.sqrt(np.clip(1 - dx**2, 0, 1))
            depth[row] = np.maximum(depth[row], val)
    img = Image.fromarray((depth * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(3 * s))
    return img.convert("RGB")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from openpose_draw import body_pose

    d = hourglass_depth()
    p = body_pose(0, "down")
    sheet = Image.new("RGB", (832 * 2, 1216))
    sheet.paste(d, (0, 0))
    sheet.paste(p, (832, 0))
    out = ROOT / "outputs/body_depth_preview.png"
    sheet.save(out)
    print(out)
