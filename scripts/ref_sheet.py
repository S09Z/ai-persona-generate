"""
scripts/ref_sheet.py -- Stage 3 of the MODEL_001 Master Face plan: character reference sheet.

Head views: IP-Adapter Plus Face conditioned on characters/model_001/master_face/master_face.png
holds the identity, and an OpenPose ControlNet (skeleton drawn by openpose_draw.py) sets the head
angle, because the IP-Adapter alone copies the master face's frontal pose. Fixed seed per view,
base RealVisXL (no LoRA). Each output gets a rough identity score: cosine
similarity between its CLIP image embedding and the master face's (global, not face-specific,
so use it to rank drift, not as a pass/fail threshold).

Output (default characters/model_001/reference_sheet/):
    head_<view>.png          individual images
    contact_head.png         all head views side by side, labelled with the similarity score
    head_settings.json       prompts, seeds, scale, scores

Usage:
    uv run scripts/ref_sheet.py
    uv run scripts/ref_sheet.py --views front three_quarter_left --steps 8 --size 768   # quick test
    uv run scripts/ref_sheet.py --scale 0.4 --pose-scale 0.9                            # looser identity, stricter angle
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openpose_draw import head_pose  # noqa: E402

load_dotenv(ROOT / ".env")

IP_DIR = ROOT / "models" / "ip_adapter"
CONTROLNET_DIR = ROOT / "models" / "controlnet" / "openpose-sdxl"
MASTER = ROOT / "characters" / "model_001" / "master_face" / "master_face.png"

# View text goes first: CLIP drops everything past 77 tokens, so BASE must stay short.
BASE = (
    "photo of a young Thai woman, 27, ivory knit top, korean-thai natural beauty, fair skin, "
    "light freckles, pale milky blue eyes, long dark coffee brown hair, minimal makeup, "
    "soft daylight, cream backdrop"
)
EXTRA_NEGATIVE = ", blue top, light blue shirt, turquoise clothing"
# (view text, seed, yaw in degrees; positive turns the face toward the viewer's right)
VIEWS = {
    "front": ("front view, looking straight at camera, calm neutral expression", 2001, 0),
    "three_quarter_left": ("three-quarter view, head turned to her left, calm expression", 2002, 35),
    "three_quarter_right": ("three-quarter view, head turned to her right, calm expression", 2003, -35),
    "profile_left": ("side profile view facing left, calm expression", 2004, 80),
    "profile_right": ("side profile view facing right, calm expression", 2005, -80),
    "smile": ("front view, natural gentle smile", 2006, 0),
    "calm_looking_down": ("front view, eyes lowered, soft calm expression", 2007, 0),
}


def label_sheet(items: list[tuple[Path, str]], out: Path, thumb: int = 384, cols: int = 4) -> None:
    rows = (len(items) + cols - 1) // cols
    pad, head = 6, 18
    s = Image.new("RGB", (cols * (thumb + pad) + pad, rows * (thumb + pad + head) + pad), "white")
    d = ImageDraw.Draw(s)
    for i, (p, text) in enumerate(items):
        im = Image.open(p).convert("RGB")
        im.thumbnail((thumb, thumb))
        x, y = pad + (i % cols) * (thumb + pad), pad + (i // cols) * (thumb + pad + head)
        d.text((x + 2, y + 2), text, fill="black")
        s.paste(im, (x, y + head))
    s.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "characters/model_001/reference_sheet"))
    ap.add_argument("--views", nargs="+", default=list(VIEWS), choices=list(VIEWS))
    ap.add_argument("--scale", type=float, default=0.5, help="IP-Adapter strength (higher = closer to master face)")
    ap.add_argument("--pose-scale", type=float, default=0.8, help="ControlNet conditioning scale (higher = stricter angle)")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    if not MASTER.exists():
        sys.exit(f"master face not found: {MASTER}")

    import torch
    from transformers import CLIPImageProcessor

    from idol_generate.pipeline import load_pipeline
    from idol_generate.prompt_builder import NEGATIVE_PROMPT, count_tokens

    device = os.getenv("DEVICE", "cuda")
    pipe = load_pipeline(os.getenv("MODEL_ID", "SG161222/RealVisXL_V4.0"), os.getenv("MODEL_PATH", ""), device)
    pipe.load_ip_adapter(
        str(IP_DIR), subfolder="sdxl_models", weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
        image_encoder_folder="models/image_encoder",
    )
    pipe.set_ip_adapter_scale(args.scale)

    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

    controlnet = ControlNetModel.from_pretrained(str(CONTROLNET_DIR), torch_dtype=pipe.unet.dtype)
    cpipe = StableDiffusionXLControlNetPipeline.from_pipe(pipe, controlnet=controlnet)
    # UNet + IP-Adapter + ControlNet + encoders do not fit in 12 GB together; without offload the
    # overflow spills into system RAM and a step takes ~12 minutes instead of ~1 second.
    cpipe.enable_model_cpu_offload()

    face = Image.open(MASTER).convert("RGB")
    proc = CLIPImageProcessor()

    # Similarity scoring runs on the CPU with its own copy of the encoder so it never competes for VRAM
    from transformers import CLIPVisionModelWithProjection

    scorer = CLIPVisionModelWithProjection.from_pretrained(str(IP_DIR / "models" / "image_encoder")).eval()

    def embed(img: Image.Image) -> torch.Tensor:
        px = proc(images=img, return_tensors="pt").pixel_values
        with torch.no_grad():
            e = scorer(px).image_embeds
        return torch.nn.functional.normalize(e.float(), dim=-1)

    ref = embed(face)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items: list[tuple[Path, str]] = []
    records = {}
    for name in args.views:
        text, seed, yaw = VIEWS[name]
        prompt = f"{text}, {BASE}"
        n = count_tokens(prompt)
        if n > 75:
            sys.exit(f"prompt for {name} is {n} tokens; CLIP would truncate it")
        gen = torch.Generator(device=device).manual_seed(seed)
        pose = head_pose(yaw, args.size)
        pose.save(out / f"pose_{name}.png")
        img = cpipe(
            prompt=prompt, negative_prompt=NEGATIVE_PROMPT + EXTRA_NEGATIVE, ip_adapter_image=face,
            image=pose, controlnet_conditioning_scale=args.pose_scale,
            num_inference_steps=args.steps, guidance_scale=args.cfg,
            width=args.size, height=args.size, generator=gen,
        ).images[0]
        path = out / f"head_{name}.png"
        img.save(path)
        sim = float((embed(img) @ ref.T).item())
        records[name] = {"prompt": prompt, "seed": seed, "yaw": yaw, "similarity": round(sim, 4)}
        items.append((path, f"{name}  sim={sim:.3f}"))
        print(f"saved {path}  similarity={sim:.3f}", flush=True)

    label_sheet(items, out / "contact_head.png")
    (out / "head_settings.json").write_text(json.dumps({
        "master_face": str(MASTER.relative_to(ROOT)), "ip_adapter_scale": args.scale,
        "pose_scale": args.pose_scale,
        "steps": args.steps, "cfg": args.cfg, "size": args.size, "negative_prompt": NEGATIVE_PROMPT,
        "sampler": "DPM++ multistep, karras sigmas", "views": records,
        "base_model": os.getenv("MODEL_PATH") or os.getenv("MODEL_ID"),
    }, indent=2), encoding="utf-8")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
