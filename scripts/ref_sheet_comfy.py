"""
scripts/ref_sheet_comfy.py -- MODEL_001 / Namphueng head-view reference sheet through ComfyUI.

Same idea as ref_sheet.py (IP-Adapter Plus Face holds the identity from the locked master face,
an OpenPose ControlNet drawn by openpose_draw.py sets the head angle) but run in ComfyUI, whose
memory management copes with 12 GB where the diffusers ControlNet+IP-Adapter pipeline spills
into system RAM. Needs ComfyUI running (scripts/comfyui.ps1) with the ComfyUI_IPAdapter_plus node.

Output (default characters/model_001/reference_sheet_comfy/):
    head_<view>.png, pose_<view>.png, contact_head.png          (--mode head, default)
    body_<view>.png, arche_<name>.png, contact_body.png         (--mode body, full body 832x1216)

Usage:
    uv run scripts/ref_sheet_comfy.py
    uv run scripts/ref_sheet_comfy.py --views side_left --steps 8 --size 768   # quick test
    uv run scripts/ref_sheet_comfy.py --mode body                              # full-body angles + archetypes
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests  # noqa: E402
from PIL import Image  # noqa: E402

from idol_generate import comfyui_client as cc  # noqa: E402
from idol_generate.prompt_builder import NEGATIVE_PROMPT, count_tokens  # noqa: E402
from openpose_draw import body_pose, head_pose  # noqa: E402
from ref_sheet import BASE, EXTRA_NEGATIVE, MASTER, VIEWS, label_sheet  # noqa: E402


def upload(name: str, data: bytes) -> str:
    """Upload an image into ComfyUI's input folder and return the stored name."""
    r = requests.post(
        f"{cc.COMFYUI_URL}/upload/image",
        files={"image": (name, io.BytesIO(data), "image/png")},
        data={"overwrite": "true"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["name"]


def build_workflow(prompt: str, negative: str, face: str, pose: str, *, seed: int, steps: int,
                   cfg: float, size: tuple[int, int], ip_weight: float, pose_strength: float, prefix: str) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "RealVisXL_V4.0.safetensors"}},
        "2": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "model.safetensors"}},
        "3": {"class_type": "IPAdapterModelLoader",
              "inputs": {"ipadapter_file": "ip-adapter-plus-face_sdxl_vit-h.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": face}},
        "5": {"class_type": "IPAdapterAdvanced", "inputs": {
            "model": ["1", 0], "ipadapter": ["3", 0], "image": ["4", 0], "clip_vision": ["2", 0],
            "weight": ip_weight, "weight_type": "linear", "combine_embeds": "concat",
            "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "8": {"class_type": "ControlNetLoader",
              "inputs": {"control_net_name": "openpose-sdxl\\diffusion_pytorch_model.safetensors"}},
        "9": {"class_type": "LoadImage", "inputs": {"image": pose}},
        "10": {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": ["6", 0], "negative": ["7", 0], "control_net": ["8", 0], "image": ["9", 0],
            "strength": pose_strength, "start_percent": 0.0, "end_percent": 1.0, "vae": ["1", 2]}},
        "11": {"class_type": "EmptyLatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}},
        "12": {"class_type": "KSampler", "inputs": {
            "model": ["5", 0], "positive": ["10", 0], "negative": ["10", 1], "latent_image": ["11", 0],
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "dpmpp_2m",
            "scheduler": "karras", "denoise": 1.0}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": prefix}},
    }


# Full-body sheet: (view text, outfit, seed, yaw, arms, IP-Adapter weight). Body descriptors come
# first in BODY_BASE so CLIP keeps them; the face is small in a full-body frame, so the IP-Adapter is
# kept moderate (and low for the back view, where no face is visible).
BODY_BASE = (
    "full-body photo of a young Thai woman, 27, curvy hourglass figure, narrow waist, wide hips, "
    "prominent round glutes, full bust, toned fit body, long dark coffee brown hair, fair skin, "
    "plain beige wall, soft daylight"
)
CASUAL = "full-length white tee tucked into high-waist jeans, white sneakers"
BODY_NEGATIVE = ", cropped top, midriff, bare belly, pink wall, terracotta wall, trash can, bin, plant pot"
BODY_VIEWS = {
    "body_front": ("standing front view, arms relaxed", CASUAL, 3001, 0, "down", 0.6),
    "body_three_quarter": ("three-quarter view, hands in pockets", CASUAL, 3002, 35, "hand_on_hip", 0.6),
    "body_side": ("side view, standing", CASUAL, 3003, 80, "down", 0.5),
    "body_back": ("back view, seen from behind", CASUAL, 3004, 180, "down", 0.2),
    "arche_casual_cool": ("standing front view, hand on hip", CASUAL, 3011, 0, "hand_on_hip", 0.6),
    "arche_tomboy_chic": ("standing front view, relaxed",
                          "(baseball cap:1.3), zip jacket, wide trousers, sneakers", 3012, 0, "down", 0.6),
    "arche_athleisure": ("standing front view, hand on hip",
                         "sand sports bra, taupe leggings, (zip jacket:1.2)", 3013, 0, "hand_on_hip", 0.6),
    "arche_minimalist": ("standing front view, relaxed",
                         "ivory shirt tucked into high-waist sand trousers, brown loafers", 3014, 0, "down", 0.6),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["head", "body"], default="head")
    ap.add_argument("--out", default=str(ROOT / "characters/model_001/reference_sheet_comfy"))
    ap.add_argument("--views", nargs="+", default=None, help="view names for the chosen mode (default: all)")
    ap.add_argument("--scale", type=float, default=0.6, help="IP-Adapter weight (head mode)")
    ap.add_argument("--pose-scale", type=float, default=1.0, help="ControlNet strength")
    ap.add_argument("--face-crop", default="80,20,944,884",
                    help="left,top,right,bottom crop of the master face fed to the IP-Adapter, so clothing "
                         "and backdrop are not copied; use 'none' for the whole image")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--size", type=int, default=1024, help="head mode edge; body mode is 832x1216 scaled by size/1024")
    args = ap.parse_args()

    if not cc.is_running():
        sys.exit("ComfyUI is not running: start it with scripts/comfyui.ps1")
    if not MASTER.exists():
        sys.exit(f"master face not found: {MASTER}")

    # (name, prompt, seed, pose image, ip weight, size)
    jobs = []
    if args.mode == "head":
        names = args.views or list(VIEWS)
        for name in names:
            if name not in VIEWS:
                sys.exit(f"unknown head view {name}; choose from {list(VIEWS)}")
            text, seed, yaw = VIEWS[name]
            jobs.append((f"head_{name}", f"{text}, {BASE}", seed, head_pose(yaw, args.size), args.scale,
                         (args.size, args.size)))
    else:
        names = args.views or list(BODY_VIEWS)
        w, h = round(832 * args.size / 1024), round(1216 * args.size / 1024)
        for name in names:
            if name not in BODY_VIEWS:
                sys.exit(f"unknown body view {name}; choose from {list(BODY_VIEWS)}")
            text, outfit, seed, yaw, arms, ipw = BODY_VIEWS[name]
            jobs.append((name, f"{text}, {outfit}, {BODY_BASE}", seed, body_pose(yaw, arms, (w, h)), ipw, (w, h)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = Image.open(MASTER).convert("RGB")
    if args.face_crop != "none":
        ref = ref.crop(tuple(int(v) for v in args.face_crop.split(",")))
    buf = io.BytesIO()
    ref.save(buf, format="PNG")
    face = upload("namphueng_master_face.png", buf.getvalue())

    items: list[tuple[Path, str]] = []
    for name, prompt, seed, pose_img, ipw, size in jobs:
        if count_tokens(prompt) > 75:
            sys.exit(f"prompt for {name} is {count_tokens(prompt)} tokens; CLIP would truncate it")
        pose_path = out / f"pose_{name}.png"
        pose_img.save(pose_path)
        pose = upload(f"namphueng_pose_{name}.png", pose_path.read_bytes())
        wf = build_workflow(prompt, NEGATIVE_PROMPT + EXTRA_NEGATIVE + (BODY_NEGATIVE if args.mode == "body" else ""), face, pose, seed=seed, steps=args.steps,
                            cfg=args.cfg, size=size, ip_weight=ipw, pose_strength=args.pose_scale,
                            prefix=f"namphueng_{name}_{uuid.uuid4().hex[:6]}")
        t0 = time.monotonic()
        pid = cc._post("prompt", {"prompt": wf})["prompt_id"]
        images = cc.wait_for_result(pid, timeout=900)
        if not images:
            sys.exit(f"no image returned for {name}")
        path = cc.download_image(images[0], out / f"{name}.png")
        items.append((path, name))
        print(f"saved {path}  ({time.monotonic() - t0:.0f}s)", flush=True)

    label_sheet(items, out / f"contact_{args.mode}.png", thumb=384 if args.mode == "head" else 512)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
