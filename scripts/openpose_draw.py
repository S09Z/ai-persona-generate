"""
scripts/openpose_draw.py -- Draw OpenPose (COCO-18) skeleton images from code, so poses can be
specified without a pose-detector model.

Keypoint order: 0 nose, 1 neck, 2 R-shoulder, 3 R-elbow, 4 R-wrist, 5 L-shoulder, 6 L-elbow,
7 L-wrist, 8 R-hip, 9 R-knee, 10 R-ankle, 11 L-hip, 12 L-knee, 13 L-ankle, 14 R-eye, 15 L-eye,
16 R-ear, 17 L-ear. "R"/"L" are the subject's own sides, as in OpenPose.

head_pose(yaw) builds a head-and-shoulders skeleton. yaw is in degrees: 0 = facing the camera,
positive = the face turns toward the viewer's right.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw

LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]
# Standard OpenPose limb colours (RGB), matching what the SDXL OpenPose ControlNet was trained on
COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
    (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
]

Point = tuple[float, float] | None


def draw_pose(points: list[Point], size: int | tuple[int, int] = 1024) -> Image.Image:
    """Render keypoints (pixel coords, None = not visible) as an OpenPose control image.

    size is a square edge or a (width, height) pair.
    """
    w, h = (size, size) if isinstance(size, int) else size
    img = Image.new("RGB", (w, h), "black")
    d = ImageDraw.Draw(img)
    stick = max(2, min(w, h) // 128)
    for (a, b), color in zip(LIMBS, COLORS):
        pa, pb = points[a], points[b]
        if pa is None or pb is None:
            continue
        # Limbs are drawn as thick lines; OpenPose uses filled ellipses, a line is equivalent here
        dim = tuple(int(c * 0.6) for c in color)
        d.line([pa, pb], fill=dim, width=stick * 2)
    for p, color in zip(points, COLORS):
        if p is not None:
            r = stick * 1.6
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)
    return img


def head_pose(yaw_deg: float, size: int = 1024) -> Image.Image:
    """Head-and-shoulders skeleton with the head turned by yaw_deg and the shoulders by half of it."""
    s = size / 1024
    cx, neck_y = 512 * s, 720 * s
    th = math.radians(yaw_deg)
    tb = th * 0.5

    def head(x: float, y: float, z: float) -> tuple[float, float, float]:
        # x: right of the head centre, z: forward from the face; returns screen x, y and depth
        return cx + (x * math.cos(th) + z * math.sin(th)) * s, y * s, -x * math.sin(th) + z * math.cos(th)

    pts: list[Point] = [None] * 18
    pts[1] = (cx, neck_y)
    for idx, side in ((2, -1), (5, 1)):
        sx = cx + side * 200 * s * math.cos(tb)
        pts[idx] = (sx, neck_y + 40 * s)
    nose = head(0, 430, 110)
    pts[0] = (nose[0], nose[1]) if nose[2] > -10 else None
    for idx, side in ((14, -1), (15, 1)):  # R-eye is on the viewer's left when facing the camera
        x, y, z = head(side * 48, 395, 75)
        pts[idx] = (x, y) if z > 15 else None
    for idx, side in ((16, -1), (17, 1)):
        x, y, z = head(side * 135, 425, -10)
        pts[idx] = (x, y) if z > -40 else None
    return draw_pose(pts, size)


ARM_STYLES = ("down", "hand_on_hip")

# Standing figure in a 1216 px tall frame: (x right of the body centre, y from top, z forward).
# Proportions are a slim, petite adult; hips are slightly wider than the shoulders (84 vs 76 half-width) for a full-hip hourglass look.
_BODY_3D = {
    0: (0, 190, 40), 1: (0, 292, 0),
    2: (-84, 312, 0), 5: (84, 312, 0),
    8: (-76, 640, 0), 11: (76, 640, 0),
    9: (-58, 887, 6), 12: (58, 887, 6),
    10: (-54, 1101, 0), 13: (54, 1101, 0),
    14: (-24, 168, 30), 15: (24, 168, 30),
    16: (-62, 186, -8), 17: (62, 186, -8),
}
_ARMS_3D = {
    "down": {3: (-102, 470, 0), 4: (-108, 628, 0), 6: (102, 470, 0), 7: (108, 628, 0)},
    "hand_on_hip": {3: (-150, 480, -4), 4: (-80, 640, 6), 6: (150, 480, -4), 7: (80, 640, 6)},
}


def body_pose(
    yaw_deg: float = 0, arms: str = "down", size: tuple[int, int] = (832, 1216)
) -> Image.Image:
    """Full-body standing skeleton rotated by yaw_deg about the vertical axis.

    yaw 0 faces the camera, +/-90 is a side view, 180 is the back. Positive yaw turns the body
    toward the viewer's right. Face keypoints are dropped when they face away from the camera,
    the way OpenPose omits them for a back view.
    """
    if arms not in _ARMS_3D:
        raise ValueError(f"arms must be one of {ARM_STYLES}")
    w, h = size
    s = h / 1216
    cx = w / 2
    th = math.radians(yaw_deg)
    pts: list[Point] = [None] * 18
    for idx, (x, y, z) in {**_BODY_3D, **_ARMS_3D[arms]}.items():
        sx = cx + (x * math.cos(th) + z * math.sin(th)) * s
        depth = -x * math.sin(th) + z * math.cos(th)
        # The nose still shows in profile (depth ~0); eyes only while facing the camera side
        if (idx == 0 and depth < -15) or (idx in (14, 15) and depth < 5):
            continue
        if idx in (16, 17) and depth < -45:
            continue
        pts[idx] = (sx, y * s)
    return draw_pose(pts, size)


if __name__ == "__main__":
    for yaw in (0, 35, -35, 80, -80):
        head_pose(yaw).save(f"pose_head_{yaw}.png")
    for name, yaw, arms in (("front", 0, "down"), ("side", 90, "down"), ("back", 180, "down"),
                            ("front_hip", 0, "hand_on_hip"), ("three_quarter", 35, "hand_on_hip")):
        body_pose(yaw, arms).save(f"pose_body_{name}.png")
