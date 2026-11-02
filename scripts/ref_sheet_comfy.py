"""
scripts/ref_sheet_comfy.py -- MODEL_001 / Namphueng head-view reference sheet through ComfyUI.

Same idea as ref_sheet.py (IP-Adapter Plus Face holds the identity from the locked master face,
an OpenPose ControlNet drawn by openpose_draw.py sets the head angle) but run in ComfyUI, whose
memory management copes with 12 GB where the diffusers ControlNet+IP-Adapter pipeline spills
into system RAM. Needs ComfyUI running (scripts/comfyui.ps1) with the ComfyUI_IPAdapter_plus node.

Output (default characters/model_001/reference_sheet_comfy/):
    head_<view>.png, pose_<view>.png, contact_head.png

Usage:
    uv run scripts/ref_sheet_comfy.py
    uv run scripts/ref_sheet_comfy.py --views profile_left --steps 8 --size 768   # quick test
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

from idol_generate import comfyui_client as cc  # noqa: E402
from idol_generate.prompt_builder import NEGATIVE_PROMPT, count_tokens  # noqa: E402
from openpose_draw import head_pose  # noqa: E402
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
                   cfg: float, size: int, ip_weight: float, pose_strength: float, prefix: str) -> dict:
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
        "11": {"class_type": "EmptyLatentImage", "inputs": {"width": size, "height": size, "batch_size": 1}},
        "12": {"class_type": "KSampler", "inputs": {
            "model": ["5", 0], "positive": ["10", 0], "negative": ["10", 1], "latent_image": ["11", 0],
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "dpmpp_2m",
            "scheduler": "karras", "denoise": 1.0}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": prefix}},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "characters/model_001/reference_sheet_comfy"))
    ap.add_argument("--views", nargs="+", default=list(VIEWS), choices=list(VIEWS))
    ap.add_argument("--scale", type=float, default=0.5, help="IP-Adapter weight")
    ap.add_argument("--pose-scale", type=float, default=0.8, help="ControlNet strength")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    if not cc.is_running():
        sys.exit("ComfyUI is not running: start it with scripts/comfyui.ps1")
    if not MASTER.exists():
        sys.exit(f"master face not found: {MASTER}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    face = upload("namphueng_master_face.png", MASTER.read_bytes())

    items: list[tuple[Path, str]] = []
    for name in args.views:
        text, seed, yaw = VIEWS[name]
        prompt = f"{text}, {BASE}"
        if count_tokens(prompt) > 75:
            sys.exit(f"prompt for {name} would be truncated by CLIP")
        pose_img = head_pose(yaw, args.size)
        pose_path = out / f"pose_{name}.png"
        pose_img.save(pose_path)
        pose = upload(f"namphueng_pose_{name}.png", pose_path.read_bytes())

        wf = build_workflow(prompt, NEGATIVE_PROMPT + EXTRA_NEGATIVE, face, pose, seed=seed, steps=args.steps,
                            cfg=args.cfg, size=args.size, ip_weight=args.scale, pose_strength=args.pose_scale,
                            prefix=f"namphueng_head_{name}_{uuid.uuid4().hex[:6]}")
        t0 = time.monotonic()
        pid = cc._post("prompt", {"prompt": wf})["prompt_id"]
        images = cc.wait_for_result(pid, timeout=900)
        if not images:
            sys.exit(f"no image returned for {name}")
        path = cc.download_image(images[0], out / f"head_{name}.png")
        items.append((path, name))
        print(f"saved {path}  ({time.monotonic() - t0:.0f}s)", flush=True)

    label_sheet(items, out / "contact_head.png")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
