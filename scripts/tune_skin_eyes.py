"""
scripts/tune_skin_eyes.py -- try face-pass prompt variants for skin, freckles and eye colour.

Takes stage-1 images (and JSON sidecars) written by generate_comfy.py, re-runs only the face pass with
each text variant, and measures per head: ArcFace similarity to the Master Face (identity must not
drop), cheek skin colour (Lab), freckle density (dark speckles on cheeks and nose) and iris colour.
The same measurements are taken on head crops of the Sora target sheet so the numbers have a goal.

Usage:
    uv run scripts/tune_skin_eyes.py outputs/comfy_skin
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import face_id  # noqa: E402
from generate_comfy import (POSES, WALL_NEGATIVE, build_face_workflow, head_box, png, upload,  # noqa: E402
                            FACE_CROP)
from idol_generate import comfyui_client as cc  # noqa: E402
from idol_generate.prompt_builder import NEGATIVE_PROMPT, count_tokens  # noqa: E402
from ref_sheet import EXTRA_NEGATIVE, MASTER  # noqa: E402

TARGET = ROOT / "characters/model_001/target/sora_sheet.png"
# frontal head and the big three-quarter head of the 1024x1536 sheet
TARGET_CROPS = {"target_front": (528, 985, 778, 1305), "target_3q": (528, 0, 1024, 560)}

BASE = ("young Thai woman, soft oval face, korean-thai elegant look, girl-next-door freshness, "
        "natural realistic skin texture, minimal makeup")
VARIANTS = {
    "A_base": f"{BASE}, fair skin with a few light freckles, pale milky light blue eyes, detailed eyes",
    "B_freckles": f"{BASE}, (dense light freckles across nose and cheeks:1.3), fair skin, "
                  "pale milky light blue eyes, detailed eyes",
    "C_eyes": f"{BASE}, fair skin, light freckles, (pale grey-blue eyes:1.3), clear light irises, detailed eyes",
    "D_both": f"{BASE}, (porcelain fair skin, rosy cheeks:1.2), (dense light freckles across nose and cheeks:1.3), "
              "(pale grey-blue eyes:1.3), detailed eyes",
}


def landmarks(img: Image.Image):
    rgb = img.convert("RGB")
    scale = max(512 / max(rgb.size), 1.0)
    if scale > 1:
        rgb = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.LANCZOS)
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    pts = face_id._landmarks(bgr)
    return None if pts is None else (np.asarray(rgb), pts)


def measure(img: Image.Image) -> dict | None:
    """Skin Lab, freckle density and iris RGB from landmarks; None when no face is found or it is turned away."""
    found = landmarks(img)
    if found is None:
        return None
    rgb, p = found
    re, le, nose, rm, lm = p
    ipd = float(np.linalg.norm(le - re))
    if ipd < 20:
        return None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = gray.shape

    def disc(c, r):
        yy, xx = np.ogrid[:h, :w]
        return (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= r * r

    # cheek patches: below each eye, towards the mouth corner; nose bridge-to-tip patch
    cheeks = [(re + (rm - re) * 0.45, ipd * 0.22), (le + (lm - le) * 0.45, ipd * 0.22), (nose, ipd * 0.17)]
    mask = np.zeros((h, w), bool)
    for c, r in cheeks:
        mask |= disc(c, r)
    skin = np.median(lab[mask], axis=0)
    deficit = cv2.GaussianBlur(gray, (0, 0), ipd * 0.08) - gray  # dark speckle = local dip below the surroundings
    freckles = float((deficit[mask] > 7).mean())
    # iris: ring between pupil and sclera around each eye landmark
    ring = np.zeros((h, w), bool)
    for c in (re, le):
        ring |= disc(c, ipd * 0.085) & ~disc(c, ipd * 0.035)
    iris = np.median(rgb[ring], axis=0)
    return {"L": skin[0] * 100 / 255, "a": skin[1] - 128, "b": skin[2] - 128, "freckle": freckles,
            "iris_rgb": tuple(int(v) for v in iris), "blue": float(iris[2] - iris[0])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--ip", type=float, default=0.6)
    ap.add_argument("--denoise", type=float, default=0.55)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()

    for name in args.variants:
        n = count_tokens(VARIANTS[name])
        if n > 75:
            sys.exit(f"variant {name} is {n} tokens; shorten it")
    if not cc.is_running():
        sys.exit("ComfyUI is not running: start it with scripts/comfyui.ps1")

    folder = Path(args.folder)
    ref = face_id.embed(Image.open(MASTER))
    face = upload("namphueng_master_face.png", png(Image.open(MASTER).convert("RGB").crop(FACE_CROP)))
    rows: dict[str, list[dict]] = {}

    def record(label: str, crop: Image.Image) -> None:
        m = measure(crop)
        sim = face_id.similarity(ref, face_id.embed(crop))
        if m is not None:
            rows.setdefault(label, []).append(m | {"arc": sim})

    sheet = Image.open(TARGET).convert("RGB")
    for label, box in TARGET_CROPS.items():
        record(label, sheet.crop(box))
    record("master", Image.open(MASTER).convert("RGB"))

    for js in sorted(folder.glob("*.json")):
        post = json.loads(js.read_text(encoding="utf-8"))
        yaw = POSES[post["pose"]][0]
        if yaw == 180:
            continue
        box = head_box(yaw)
        stage1 = folder / post["image"]
        stage1_up = upload(f"namphueng_stage1_{stage1.stem}.png", stage1.read_bytes())
        for name in args.variants:
            out = folder / f"{stage1.stem}_{name}.png"
            if not out.exists():
                wf = build_face_workflow(stage1_up, face, VARIANTS[name], NEGATIVE_PROMPT + EXTRA_NEGATIVE + WALL_NEGATIVE,
                                         box, seed=post["seed"], steps=args.steps, cfg=args.cfg, ip_weight=args.ip,
                                         denoise=args.denoise, prefix=f"namphueng_skin_{uuid.uuid4().hex[:6]}")
                pid = cc._post("prompt", {"prompt": wf})["prompt_id"]
                imgs = cc.wait_for_result(pid, timeout=900)
                if not imgs:
                    sys.exit(f"no image for {out.name}")
                cc.download_image(imgs[0], out)
                print(out.name, flush=True)
            x, y, w, h = box
            record(name, Image.open(out).crop((x, y, x + w, y + h)))

    print(f"\n{'variant':14} {'n':>2} {'arcface':>8} {'skin L':>7} {'a':>6} {'b':>6} {'freckle':>8} {'iris R,G,B':>14} {'B-R':>5}")
    for label, ms in rows.items():
        def mean(k):
            vals = [m[k] for m in ms if m[k] is not None]
            return sum(vals) / len(vals) if vals else float("nan")
        iris = tuple(round(sum(m["iris_rgb"][i] for m in ms) / len(ms)) for i in range(3))
        print(f"{label:14} {len(ms):>2} {mean('arc'):>8.3f} {mean('L'):>7.1f} {mean('a'):>6.1f} {mean('b'):>6.1f} "
              f"{mean('freckle'):>8.3f} {str(iris):>14} {mean('blue'):>5.1f}")


if __name__ == "__main__":
    main()
