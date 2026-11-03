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
import math
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

from PIL import Image, ImageOps  # noqa: E402

from idol_generate import comfyui_client as cc  # noqa: E402
from idol_generate.character import Character  # noqa: E402
from idol_generate.prompt_builder import CONTENT_CONFIGS, NEGATIVE_PROMPT, _pack, _phrases, build_post  # noqa: E402
from body_depth import hourglass_depth  # noqa: E402
from openpose_draw import body_pose  # noqa: E402
from ref_sheet import EXTRA_NEGATIVE, MASTER  # noqa: E402
from ref_sheet_comfy import upload  # noqa: E402

SIZE = (832, 1216)
# Without this the model frames the shot at thigh/shin level and ignores the skeleton's legs
# The model leans toward pink/terracotta walls for "warm"/"sand" palettes; push it to neutral
# also keeps the arms from coming out tanned next to a pale face
WALL_NEGATIVE = (", pink wall, terracotta wall, orange color cast, red brick, "
                 "tanned skin, dark arms, uneven skin tone, holding a phone, object in hand, "
                 "torn sleeve, slit sleeve")
# extra negatives per content type (appended to WALL_NEGATIVE in main)
TYPE_NEGATIVE = {"boho_set": ", pants, trousers, wide-leg pants, shorts, split skirt"}
# Face-pass eye wording (variant C of tune_skin_eyes.py): bluer eyes with ArcFace identity unchanged
FACE_PASS_EYES = "(pale grey-blue eyes:1.3), clear light irises, detailed eyes, natural skin texture"
FRAMING = "full-body photo, head to toe, shoes visible, fair even skin tone on face and arms"
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
                   depth_end: float = 0.6, pose_strength: float = 1.0) -> dict:
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
        "7": encode(post["negative_prompt"] + EXTRA_NEGATIVE + WALL_NEGATIVE,
                post["negative_prompt"] + EXTRA_NEGATIVE + WALL_NEGATIVE),
        "8": {"class_type": "ControlNetLoader",
              "inputs": {"control_net_name": "openpose-sdxl\diffusion_pytorch_model.safetensors"}},
        "9": {"class_type": "LoadImage", "inputs": {"image": pose}},
        "10": {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": ["6", 0], "negative": ["7", 0], "control_net": ["8", 0], "image": ["9", 0],
            "strength": pose_strength, "start_percent": 0.0, "end_percent": 1.0, "vae": ["1", 2]}},
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


def head_box(yaw: float, size: tuple[int, int] = SIZE, side: int = 300) -> tuple[int, int, int, int]:
    """Square crop (x, y, w, h) around the head, from the same nose keypoint openpose_draw uses."""
    w, h = size
    s = h / 1216
    nose_x = w / 2 + (0 * math.cos(math.radians(yaw)) + 40 * math.sin(math.radians(yaw))) * s
    side = round(side * s)
    x = min(max(round(nose_x - side / 2), 0), w - side)
    y = min(max(round(150 * s - side / 2), 0), h - side)
    return x, y, side, side


def build_face_workflow(image: str, face: str, text: str, negative: str, box: tuple[int, int, int, int], *,
                        seed: int, steps: int, cfg: float, ip_weight: float, denoise: float,
                        prefix: str) -> dict:
    """Second pass: re-render only the head at 1024px with a strong IP-Adapter, then feather it back in."""
    x, y, bw, bh = box
    feather = max(bw // 6, 8)
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
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["1", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "20": {"class_type": "LoadImage", "inputs": {"image": image}},
        "21": {"class_type": "ImageCrop", "inputs": {"image": ["20", 0], "width": bw, "height": bh, "x": x, "y": y}},
        "22": {"class_type": "ImageScale", "inputs": {"image": ["21", 0], "upscale_method": "lanczos",
                                                      "width": 1024, "height": 1024, "crop": "disabled"}},
        "23": {"class_type": "VAEEncode", "inputs": {"pixels": ["22", 0], "vae": ["1", 2]}},
        "12": {"class_type": "KSampler", "inputs": {
            "model": ["5", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["23", 0],
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "dpmpp_2m",
            "scheduler": "karras", "denoise": denoise}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}},
        "24": {"class_type": "ImageScale", "inputs": {"image": ["13", 0], "upscale_method": "lanczos",
                                                      "width": bw, "height": bh, "crop": "disabled"}},
        "25": {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": bw, "height": bh}},
        "26": {"class_type": "FeatherMask", "inputs": {"mask": ["25", 0], "left": feather, "top": feather,
                                                       "right": feather, "bottom": feather}},
        "27": {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": ["20", 0], "source": ["24", 0], "x": x, "y": y, "resize_source": False,
            "mask": ["26", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["27", 0], "filename_prefix": prefix}},
    }


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
    ap.add_argument("--depth-image", default=None,
                    help="use this 832x1216 depth map (see depth_from_target.py) instead of the drawn body; "
                         "needs --depth > 0 and turns OpenPose off so the depth map alone sets pose and shape")
    ap.add_argument("--no-skin-balance", action="store_true",
                    help="with --depth-image, skip matching the arms, hands and neck to the face tone "
                         "(skin_balance.py); the balanced result is saved as <stem>_final.png")
    ap.add_argument("--guide-flipped", action="store_true",
                    help="the --depth-image was made with depth_from_target.py --flip: mirror it for the skin balance")
    ap.add_argument("--depth-hip", type=float, default=1.0, help="scale the hip width of the depth body")
    ap.add_argument("--depth-waist", type=float, default=1.0, help="scale the waist width of the depth body")
    ap.add_argument("--face-pass", action="store_true",
                    help="re-render the head at 1024px with a strong IP-Adapter and blend it back (fixes small, "
                         "off-model faces and eye colour); skipped for back views")
    ap.add_argument("--face-ip", type=float, default=None,
                    help="IP-Adapter weight in the face pass (default 0.6; 0.35 for turned poses so the head can turn)")
    ap.add_argument("--face-denoise", type=float, default=None,
                    help="how far the face pass may change the head (default 0.55; 0.4 for turned poses)")
    ap.add_argument("--depth-end", type=float, default=0.6, help="stop the depth ControlNet at this fraction of steps")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--out", default=str(ROOT / "outputs/comfy"))
    ap.add_argument("--dry-run", action="store_true", help="print the prompts and exit")
    args = ap.parse_args()

    global WALL_NEGATIVE
    WALL_NEGATIVE += TYPE_NEGATIVE.get(args.content_type, "")
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
        if args.depth_image:
            if args.depth <= 0:
                sys.exit("--depth-image needs --depth > 0")
            depth_name = upload("namphueng_depth_target.png", Path(args.depth_image).read_bytes())
        elif args.depth > 0 and yaw in (0, 180):
            depth_name = upload("namphueng_body_depth.png", png(hourglass_depth(SIZE, waist=args.depth_waist, hip=args.depth_hip)))
        elif args.depth > 0:
            print(f"note: --depth skipped for pose {pose_name} (front/back only)")
        wf = build_workflow(post, face, pose, seed=seed, steps=args.steps, cfg=args.cfg, ip_weight=ipw,
                            prefix=f"namphueng_{uuid.uuid4().hex[:6]}", depth=depth_name,
                            depth_strength=args.depth, depth_end=args.depth_end,
                            pose_strength=0.0 if args.depth_image else 1.0)
        t0 = time.monotonic()
        pid = cc._post("prompt", {"prompt": wf})["prompt_id"]
        images = cc.wait_for_result(pid, timeout=900)
        if not images:
            sys.exit("no image returned")
        stem = f"{character.name.lower()}_{args.content_type}_{ts}_s{seed}"
        cc.download_image(images[0], out / f"{stem}.png")
        if args.face_pass and yaw != 180:
            a = character.appearance
            text = _pack([_phrases(a.face, 5), FACE_PASS_EYES])
            stage1 = upload(f"namphueng_stage1_{stem}.png", (out / f"{stem}.png").read_bytes())
            turned = yaw not in (0, 180)
            face_ip = args.face_ip if args.face_ip is not None else (0.35 if turned else 0.6)
            face_denoise = args.face_denoise if args.face_denoise is not None else (0.4 if turned else 0.55)
            fwf = build_face_workflow(stage1, face, text, NEGATIVE_PROMPT + EXTRA_NEGATIVE + WALL_NEGATIVE, head_box(yaw),
                                      seed=seed, steps=args.steps, cfg=args.cfg, ip_weight=face_ip,
                                      denoise=face_denoise, prefix=f"namphueng_face_{uuid.uuid4().hex[:6]}")
            fpid = cc._post("prompt", {"prompt": fwf})["prompt_id"]
            fimgs = cc.wait_for_result(fpid, timeout=900)
            if fimgs:
                cc.download_image(fimgs[0], out / f"{stem}_face.png")
                post["image_face_pass"] = f"{stem}_face.png"
        if args.depth_image and not args.no_skin_balance:
            # the depth guide fixes where the arms are, so their skin can be matched to the face
            import numpy as np
            from skin_balance import balance
            src = post.get("image_face_pass", f"{stem}.png")
            guide = np.asarray(Image.open(args.depth_image).convert("L")) > 75
            img = Image.open(out / src).convert("RGB")
            if args.guide_flipped:  # skin_balance's geometry is for the unflipped guide: mirror, balance, mirror back
                img, guide = ImageOps.mirror(img), guide[:, ::-1].copy()
            res = balance(img, guide)
            if res is not None and args.guide_flipped:
                res = (ImageOps.mirror(res[0]), *res[1:])
            if res is None:
                print("note: skin balance skipped (no face or skin found)")
            else:
                res[0].save(out / f"{stem}_final.png")
                post["image_final"] = f"{stem}_final.png"
        post.update(seed=seed, pose=pose_name, ip_weight=ipw, image=f"{stem}.png")
        (out / f"{stem}.json").write_text(json.dumps(post, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved {out / (stem + '.png')} ({time.monotonic() - t0:.0f}s) | {post['caption']}", flush=True)


if __name__ == "__main__":
    main()
