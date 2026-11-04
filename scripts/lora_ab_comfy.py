"""
scripts/lora_ab_comfy.py -- A/B a MODEL_001 LoRA inside the locked ComfyUI recipe (IP-Adapter + OpenPose).

Same prompt, seed and pose per column; rows are LoRA strengths (0 = no LoRA, baseline). Needs ComfyUI running.

Usage:
    uv run scripts/lora_ab_comfy.py --lora model_001_v2.safetensors --strengths 0 0.7 1.0
"""

from __future__ import annotations

import argparse
import io
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

from idol_generate import comfyui_client as cc  # noqa: E402
from idol_generate.prompt_builder import NEGATIVE_PROMPT, count_tokens  # noqa: E402
from openpose_draw import body_pose, head_pose  # noqa: E402
from ref_sheet import EXTRA_NEGATIVE, MASTER, label_sheet  # noqa: E402
from ref_sheet_comfy import BODY_NEGATIVE, build_workflow, upload  # noqa: E402

TRIGGER = "m001_v1"
AB_BASE = ("full-body photo of a young Thai woman, curvy hourglass figure, narrow waist, wide hips, "
           "round glutes, full bust, toned, long brown hair")
SCENES = [  # (name, prompt, yaw, arms, kind, seed)
    ("front", "standing front view, white tee tucked into high-waist jeans, white sneakers, "
              f"{AB_BASE}, beige wall", 0, "down", "body", 8001),
    ("hip_3q", "three-quarter view, hand on hip, sports bra, taupe leggings, "
               f"{AB_BASE}, white studio", 35, "hand_on_hip", "body", 8002),
    ("back", "back view, seen from behind, ivory shirt tucked into high-waist sand trousers, "
             f"{AB_BASE}, beige wall", 180, "down", "body", 8003),
]


def png(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--strengths", type=float, nargs="+", default=[0, 0.7, 1.0])
    ap.add_argument("--out", default=str(ROOT / "outputs/overnight/lora_ab"))
    ap.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()
    if not cc.is_running():
        sys.exit("ComfyUI is not running")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ref = Image.open(MASTER).convert("RGB").crop((80, 20, 944, 884))
    face = upload("namphueng_master_face.png", png(ref))
    items = []
    for strength in args.strengths:
        for name, text, yaw, arms, kind, seed in SCENES:
            prompt = text if strength == 0 else f"{TRIGGER}, {text}"
            if count_tokens(prompt) > 75:
                sys.exit(f"prompt too long ({count_tokens(prompt)}): {name}")
            size = (832, 1216)
            pose = upload(f"namphueng_pose_ab_{name}.png", png(body_pose(yaw, arms, size)))
            wf = build_workflow(prompt, NEGATIVE_PROMPT + EXTRA_NEGATIVE + BODY_NEGATIVE, face, pose,
                                seed=seed, steps=args.steps, cfg=6.0, size=size,
                                ip_weight=0.2 if name == "back" else 0.6, pose_strength=1.0,
                                prefix=f"ab_{uuid.uuid4().hex[:6]}")
            if strength > 0:  # LoRA between the checkpoint and both the IP-Adapter and the text encoders
                wf["20"] = {"class_type": "LoraLoader", "inputs": {
                    "model": ["1", 0], "clip": ["1", 1], "lora_name": args.lora,
                    "strength_model": strength, "strength_clip": strength}}
                wf["5"]["inputs"]["model"] = ["20", 0]
                wf["6"]["inputs"]["clip"] = ["20", 1]
                wf["7"]["inputs"]["clip"] = ["20", 1]
            pid = cc._post("prompt", {"prompt": wf})["prompt_id"]
            img = cc.wait_for_result(pid, timeout=900)[0]
            path = cc.download_image(img, out / f"{name}_s{strength}.png")
            items.append((path, f"{name} lora {strength}"))
            print(f"saved {path.name}", flush=True)
    label_sheet(items, out / "contact_ab.png", thumb=400)
    print("done")


if __name__ == "__main__":
    main()
