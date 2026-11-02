import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("openpose_draw", ROOT / "scripts" / "openpose_draw.py")
op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)


def _visible(img) -> set[tuple[int, int, int]]:
    """Distinct non-black colours in the image (joint dots use the full-brightness limb colours)."""
    return {c for _, c in img.getcolors(maxcolors=1 << 20) if c != (0, 0, 0)}


def _has_colour(img, idx: int) -> bool:
    return op.COLORS[idx] in _visible(img)


def test_head_pose_size_and_background():
    img = op.head_pose(0, 512)
    assert img.size == (512, 512)
    assert img.getpixel((0, 0)) == (0, 0, 0)


def test_body_pose_default_and_custom_size():
    assert op.body_pose().size == (832, 1216)
    assert op.body_pose(size=(416, 608)).size == (416, 608)


def test_front_head_shows_both_eyes_and_ears():
    img = op.head_pose(0)
    for idx in (0, 14, 15, 16, 17):
        assert _has_colour(img, idx)


def test_profile_head_hides_the_far_eye():
    img = op.head_pose(80)
    assert not (_has_colour(img, 14) and _has_colour(img, 15))
    assert _has_colour(img, 0)  # the nose still shows in profile


def test_back_view_has_no_face_points_but_keeps_ears():
    img = op.body_pose(180)
    assert not any(_has_colour(img, i) for i in (0, 14, 15))
    assert _has_colour(img, 16) and _has_colour(img, 17)


def test_front_body_has_symmetric_shoulders():
    img = op.body_pose(0)
    xs = [x for x in range(img.width) for y in range(300, 330) if img.getpixel((x, y)) != (0, 0, 0)]
    assert abs((min(xs) + max(xs)) / 2 - img.width / 2) < 5


def test_hand_on_hip_differs_from_arms_down():
    assert op.body_pose(0, "down").tobytes() != op.body_pose(0, "hand_on_hip").tobytes()


def test_yaw_sign_moves_the_nose_toward_the_same_side():
    def nose_x(yaw):
        img = op.head_pose(yaw)
        return sum(x for x in range(img.width) for y in range(img.height)
                   if img.getpixel((x, y)) == op.COLORS[0]) / max(1, sum(
                       1 for x in range(img.width) for y in range(img.height)
                       if img.getpixel((x, y)) == op.COLORS[0]))
    assert nose_x(35) > nose_x(0) > nose_x(-35)


def test_unknown_arm_style_is_rejected():
    with pytest.raises(ValueError):
        op.body_pose(0, "crossed")
