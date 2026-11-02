"""
scripts/generate_comfy.py -- MODEL_001 / Namphueng content images through ComfyUI, no LoRA.

Identity comes from IP-Adapter Plus Face fed with the cropped master face, pose and angle from an
OpenPose skeleton drawn in code (openpose_draw.py), and everything else from prompt_builder: the
SDXL dual-CLIP prompts go to a CLIPTextEncodeSDXL node (prompt -> text_l, prompt_2 -> text_g).
Writes the image plus a post JSON (caption, tags, prompts, seed, pose) to outputs/comfy/.
Full-body 832x1216 only for now.

Usage:
    uv run scripts/generate_comfy.py --type casual_cool --batch 3
    uv run scripts/generate_comfy.py --type athleisure --pose three_quarter_hip --seed 42
    uv run scripts/generate_comfy.py --type tomboy_chic --dry-run
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image  # noqa: E402

from idol_generate import comfyui_client as cc  # noqa: E402
from idol_generate.character import Character  # noqa: E402
from idol_generate.prompt_builder import CONTENT_CONFIGS, build_post  # noqa: E402
from body_depth import hourglass_depth  # noqa: E402
from openpose_draw import body_pose  # noqa: E402
from ref_sheet import EXTRA_NEGATIVE, MASTER  # noqa: E402
from ref_sheet_comfy import upload  # noqa: E402

SIZE = (832, 1216)
# Without this the model frames the shot at thigh/shin level and ignores the skeleton's legs
FRAMING = "full-body photo, head to toe, shoes visible"
FACE_CROP = (80, 20, 944, 884)
POSES = {  # name -> (yaw, arms, IP-Adapter weight)
    "front": (0, "down", 0.6),
    "front_hip": (0, "hand_on_hip", 0.6),
    "three_quarter": (35, "down", 0.6),
    "three_quarter_hip": (35, "hand_on_hip", 0.6),
    "three_quarter_hip_r": (-35, "hand_on_hip", 0.6),
    "side": (80, "down", 0.5),
    "back": (180, "down", 0.2),
}
# The skeleton sets the angle, but the text has to agree or the model turns the face to the camera
VIEW_TEXT = {"three_quarter": "three-quarter view", "three_quarter_hip": "three-quarter view",
             "three_quarter_hip_r": "three-quarter view", "side": "side view",
             "back": "back view, seen from behind"}
AUTO_POSES = ["front", "front_hip", "three_quarter", "three_quarter_hip", "three_quarter_hip_r"]


def build_workflow(post: dict, face: str, pose: str, *, seed: int, steps: int, cfg: float,
                   ip_weight: float, prefix: str, depth: str | None = None, depth_strength: float = 0.0,
                   depth_end: float = 0.6) -> dict:
    w, h = SIZE

    def encode(g: str, l: str) -> dict:
        return {"class_type": "CLIPTextEncodeSDXL", "inputs": {
            "clip": ["1", 1], "width": w, "height": h, "crop_w": 0, "crop_h": 0,
            "target_width": w, "target_height": h, "text_g": g, "text_l": l}}

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "RealVisXL_V4.0.safetensors"}},
        "2": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "model.safetensors"}},
        "3": {"class_type": "IPAdapterModelLoader",
              "inputs": {"ipadapter_file": "ip-adapter-plus-face_sdxl_vit-h.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": face}},
        "5": {"class_type": "IPAdapterAdvanced", "inputs": {
            "model": ["1", 0], "ipadapter": ["3", 0], "image": ["4", 0], "clip_vision": ["2", 0],
            "weight": ip_weight, "weight_type": "linear", "combine_embeds": "concat",
            "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only"}},
        "6": encode(post["prompt_2"], post["prompt"]),
        "7": encode(post["negative_prompt"] + EXTRA_NEGATIVE, post["negative_prompt"] + EXTRA_NEGATIVE),
        "8": {"class_type": "ControlNetLoader",
              "inputs": {"control_net_name": "openpose-sdxl\diffusion_pytorch_model.safetensors"}},
        "9": {"class_type": "LoadImage", "inputs": {"image": pose}},
        "10": {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": ["6", 0], "negative": ["7", 0], "control_net": ["8", 0], "image": ["9", 0],
            "strength": 1.0, "start_percent": 0.0, "end_percent": 1.0, "vae": ["1", 2]}},
        "11": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "12": {"class_type": "KSampler", "inputs": {
            "model": ["5", 0], "positive": ["10", 0], "negative": ["10", 1], "latent_image": ["11", 0],
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "dpmpp_2m",
            "scheduler": "karras", "denoise": 1.0}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": prefix}},
    }
    if depth and depth_strength > 0:  # second ControlNet: body volume (hips, waist) that OpenPose cannot carry
        wf["15"] = {"class_type": "ControlNetLoader",
                    "inputs": {"control_net_name": "depth-sdxl\diffusion_pytorch_model.safetensors"}}
        wf["16"] = {"class_type": "LoadImage", "inputs": {"image": depth}}
        wf["17"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": ["10", 0], "negative": ["10", 1], "control_net": ["15", 0], "image": ["16", 0],
            "strength": depth_strength, "start_percent": 0.0, "end_percent": depth_end, "vae": ["1", 2]}}
        wf["12"]["inputs"]["positive"] = ["17", 0]
        wf["12"]["inputs"]["negative"] = ["17", 1]
    return wf


def png(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="model_001")
    ap.add_argument("--type", dest="content_type", choices=list(CONTENT_CONFIGS), default="casual_cool")
    ap.add_argument("--pose", choices=["auto", *POSES], default="auto")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--depth", type=float, default=0.0,
                    help="strength of the hourglass body-volume depth ControlNet (0 = off; try 0.4-0.7); "
                         "front and back poses only, the silhouette is not rotated")
    ap.add_argument("--depth-hip", type=float, default=1.0, help="scale the hip width of the depth body")
    ap.add_argument("--depth-waist", type=float, default=1.0, help="scale the waist width of the depth body")
    ap.add_argument("--depth-end", type=float, default=0.6, help="stop the depth ControlNet at this fraction of steps")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--out", default=str(ROOT / "outputs/comfy"))
    ap.add_argument("--dry-run", action="store_true", help="print the prompts and exit")
    args = ap.parse_args()

    character = Character.load(ROOT / "characters" / f"{args.character}.json")
    base_seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    jobs = []
    for i in range(args.batch):
        pose_name = args.pose if args.pose != "auto" else AUTO_POSES[(base_seed + i) % len(AUTO_POSES)]
        random.seed(base_seed + i)  # prompt_builder draws pose/environment/mood from `random`
        jobs.append((pose_name, base_seed + i, build_post(character, args.content_type, include_pose=False,
                                                           framing=", ".join(filter(None, [FRAMING, VIEW_TEXT.get(pose_name, "")])))))

    if args.dry_run:
        for pose_name, seed, post in jobs:
            print(f"[{pose_name} seed {seed}]\n prompt:   {post['prompt']}\n prompt_2: {post['prompt_2']}\n "
                  f"caption:  {post['caption']}\n")
        return
    if not cc.is_running():
        sys.exit("ComfyUI is not running: start it with scripts/comfyui.ps1")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    face = upload("namphueng_master_face.png", png(Image.open(MASTER).convert("RGB").crop(FACE_CROP)))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for pose_name, seed, post in jobs:
        yaw, arms, ipw = POSES[pose_name]
        pose = upload(f"namphueng_pose_{pose_name}.png", png(body_pose(yaw, arms, SIZE)))
        depth_name = None
        if args.depth > 0 and yaw in (0, 180):
            depth_name = upload("namphueng_body_depth.png", png(hourglass_depth(SIZE, waist=args.depth_waist, hip=args.depth_hip)))
        elif args.depth > 0:
            print(f"note: --depth skipped for pose {pose_name} (front/back only)")
        wf = build_workflow(post, face, pose, seed=seed, steps=args.steps, cfg=args.cfg, ip_weight=ipw,
                            prefix=f"namphueng_{uuid.uuid4().hex[:6]}", depth=depth_name,
                            depth_strength=args.depth, depth_end=args.depth_end)
        t0 = time.monotonic()
        pid = cc._post("prompt", {"prompt": wf})["prompt_id"]
        images = cc.wait_for_result(pid, timeout=900)
        if not images:
            sys.exit("no image returned")
        stem = f"{character.name.lower()}_{args.content_type}_{ts}_s{seed}"
        cc.download_image(images[0], out / f"{stem}.png")
        post.update(seed=seed, pose=pose_name, ip_weight=ipw, image=f"{stem}.png")
        (out / f"{stem}.json").write_text(json.dumps(post, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved {out / (stem + '.png')} ({time.monotonic() - t0:.0f}s) | {post['caption']}", flush=True)


if __name__ == "__main__":
    main()
