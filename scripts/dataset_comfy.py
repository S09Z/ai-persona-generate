"""
scripts/dataset_comfy.py -- MODEL_001 / Namphueng synthetic LoRA dataset through ComfyUI (plan stage 4).

Reuses the locked recipe from ref_sheet_comfy.py (IP-Adapter Plus Face on the cropped master face,
OpenPose ControlNet drawn in code) and varies outfit, scene and angle. Writes images plus a manifest
with a `view` tag per image (frontal / three_quarter / side / back) and a train/holdout split, so
scripts/build_clean_snapshot.py and EXP-005 can use it. Images still need a manual filter.

Usage:
    uv run scripts/dataset_comfy.py --dry-run     # list jobs and check prompt lengths only
    uv run scripts/dataset_comfy.py               # generate into characters/model_001/dataset/
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

from idol_generate import comfyui_client as cc  # noqa: E402
from idol_generate.prompt_builder import NEGATIVE_PROMPT, count_tokens  # noqa: E402
from openpose_draw import body_pose, head_pose  # noqa: E402
from ref_sheet import EXTRA_NEGATIVE, MASTER  # noqa: E402
from ref_sheet_comfy import BODY_NEGATIVE, build_workflow, upload  # noqa: E402

HEAD_BASE = ("photo of a young Thai woman, 27, korean-thai natural beauty, fair skin, light freckles, "
             "pale milky blue eyes, long dark coffee brown hair, minimal makeup")
BODY_BASE = ("full-body photo of a young Thai woman, 27, curvy hourglass figure, narrow waist, wide hips, "
             "prominent round glutes, full bust, toned fit body, long dark coffee brown hair, fair skin")

TOPS = ["white tee", "ivory knit top", "faded denim shirt", "sand ribbed tank top", "mocha cardigan",
        "black fitted tee", "taupe sweatshirt", "ivory linen shirt"]
SCENES = ["plain beige wall, soft daylight", "cafe interior, warm window light", "quiet street, morning sun",
          "park path with trees, daylight", "bright white studio", "warm minimal living room"]
LOOKS = [  # (outfit, kind) used for body shots
    "full-length white tee tucked into high-waist jeans, white sneakers",
    "(baseball cap:1.3), zip jacket, wide trousers, sneakers",
    "sand sports bra, taupe leggings, (zip jacket:1.2)",
    "ivory shirt tucked into high-waist sand trousers, brown loafers",
    "knit top, midi skirt, thin gold necklace, flats",
    "oversized denim jacket, white tee, straight jeans, clean sneakers",
    "simple black dress, small bag",
    "mocha cardigan, high-waist wide jeans, chunky sneakers",
]
HEAD_ANGLES = [(0, "front view"), (25, "three-quarter view"), (-25, "three-quarter view"),
               (40, "three-quarter view"), (-40, "three-quarter view")]
EXPR = ["soft smile", "relaxed half-smile", "calm expression", "laughing"]
BODY_ANGLES = [(0, "frontal", "standing front view, relaxed", "down"),
               (0, "frontal", "standing front view, hand on hip", "hand_on_hip"),
               (35, "three_quarter", "three-quarter view, standing", "down"),
               (-35, "three_quarter", "three-quarter view, hand on hip", "hand_on_hip"),
               (80, "side", "side view, standing", "down"),
               (180, "back", "back view, seen from behind", "down")]


def head_jobs(n: int) -> list[dict]:
    jobs = []
    for i in range(n):
        yaw, text = HEAD_ANGLES[i % len(HEAD_ANGLES)]
        view = "frontal" if yaw == 0 else "three_quarter"
        prompt = f"{text}, {EXPR[i % len(EXPR)]}, {HEAD_BASE}, {TOPS[i % len(TOPS)]}, {SCENES[(i * 5) % len(SCENES)]}"
        jobs.append(dict(kind="head", view=view, prompt=prompt, yaw=yaw, seed=5000 + i, ip=0.6,
                         size=(1024, 1024), arms=None))
    return jobs


def body_jobs(n: int) -> list[dict]:
    jobs = []
    for i in range(n):
        yaw, view, text, arms = BODY_ANGLES[i % len(BODY_ANGLES)]
        outfit = LOOKS[(i * 3 + i // len(BODY_ANGLES)) % len(LOOKS)]
        prompt = f"{text}, {outfit}, {BODY_BASE}, {SCENES[(i * 7 + 1) % len(SCENES)]}"
        jobs.append(dict(kind="body", view=view, prompt=prompt, yaw=yaw, seed=6000 + i,
                         ip=0.2 if view == "back" else 0.6, size=(832, 1216), arms=arms))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "characters/model_001/dataset"))
    ap.add_argument("--heads", type=int, default=18)
    ap.add_argument("--bodies", type=int, default=30)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--face-crop", default="80,20,944,884")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = head_jobs(args.heads) + body_jobs(args.bodies)
    too_long = [(j["prompt"], count_tokens(j["prompt"])) for j in jobs if count_tokens(j["prompt"]) > 75]
    for p, n in too_long:
        print(f"TOO LONG ({n}): {p}")
    if too_long:
        sys.exit("shorten the lists above; CLIP would truncate these prompts")
    print(f"{len(jobs)} jobs, all prompts <= 75 tokens")
    if args.dry_run:
        return
    if not cc.is_running():
        sys.exit("ComfyUI is not running: start it with scripts/comfyui.ps1")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = Image.open(MASTER).convert("RGB").crop(tuple(int(v) for v in args.face_crop.split(",")))
    buf = io.BytesIO()
    ref.save(buf, format="PNG")
    face = upload("namphueng_master_face.png", buf.getvalue())

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for i, j in enumerate(jobs):
        name = f"{j['kind']}_{i:03d}_{j['view']}"
        if (out / f"{name}.png").exists():
            continue  # resume after an interruption
        w, h = j["size"]
        pose_img = head_pose(j["yaw"], w) if j["kind"] == "head" else body_pose(j["yaw"], j["arms"], (w, h))
        pose = upload(f"namphueng_pose_{name}.png", _png(pose_img))
        neg = NEGATIVE_PROMPT + EXTRA_NEGATIVE + (BODY_NEGATIVE if j["kind"] == "body" else "")
        wf = build_workflow(j["prompt"], neg, face, pose, seed=j["seed"], steps=args.steps, cfg=args.cfg,
                            size=(w, h), ip_weight=j["ip"], pose_strength=1.0,
                            prefix=f"namphueng_ds_{uuid.uuid4().hex[:6]}")
        t0 = time.monotonic()
        pid = cc._post("prompt", {"prompt": wf})["prompt_id"]
        images = cc.wait_for_result(pid, timeout=900)
        if not images:
            sys.exit(f"no image returned for {name}")
        cc.download_image(images[0], out / f"{name}.png")
        manifest[name] = {k: j[k] for k in ("kind", "view", "prompt", "seed", "yaw")} | {"ip_weight": j["ip"]}
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{name} ({time.monotonic() - t0:.0f}s)", flush=True)
    print(f"done: {out}")


def _png(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


if __name__ == "__main__":
    main()
