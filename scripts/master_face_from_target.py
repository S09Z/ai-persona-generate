"""
scripts/master_face_from_target.py -- Master Face candidates from a target face image (IP-Adapter Plus Face in ComfyUI).

Takes a face crop from characters/model_001/target/ (an AI-generated target look, not a real person),
renders headshot candidates at a few IP-Adapter weights and seeds, and scores each with ArcFace against
the target crop and against the current locked Master Face. Writes to characters/model_001/master_face/target_candidates/
(gitignored). Choosing a candidate is a manual step.

Usage:
    uv run scripts/master_face_from_target.py --crop 528,0,1024,556 --ip 0.7 0.9 --seeds 4001 4002 4003
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

import face_id  # noqa: E402
from idol_generate import comfyui_client as cc  # noqa: E402
from idol_generate.prompt_builder import NEGATIVE_PROMPT, count_tokens  # noqa: E402
from ref_sheet import MASTER, label_sheet  # noqa: E402
from ref_sheet_comfy import upload  # noqa: E402

PROMPT = ("photo of a young Thai woman, 27, fair skin, light freckles, pale blue eyes, long layered brown hair, "
          "center part, soft smile, minimal makeup, ivory top, headshot, soft daylight, cream backdrop")


def workflow(face: str, *, ip: float, seed: int, prefix: str) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "RealVisXL_V4.0.safetensors"}},
        "2": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "model.safetensors"}},
        "3": {"class_type": "IPAdapterModelLoader",
              "inputs": {"ipadapter_file": "ip-adapter-plus-face_sdxl_vit-h.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": face}},
        "5": {"class_type": "IPAdapterAdvanced", "inputs": {
            "model": ["1", 0], "ipadapter": ["3", 0], "image": ["4", 0], "clip_vision": ["2", 0],
            "weight": ip, "weight_type": "linear", "combine_embeds": "concat",
            "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE_PROMPT, "clip": ["1", 1]}},
        "11": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "12": {"class_type": "KSampler", "inputs": {
            "model": ["5", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["11", 0],
            "seed": seed, "steps": 25, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": prefix}},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=str(ROOT / "characters/model_001/target/sora_sheet.png"))
    ap.add_argument("--crop", required=True, help="left,top,right,bottom of the face panel in the target image")
    ap.add_argument("--ip", type=float, nargs="+", default=[0.7, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[4001, 4002, 4003])
    ap.add_argument("--out", default=str(ROOT / "characters/model_001/master_face/target_candidates"))
    args = ap.parse_args()
    if count_tokens(PROMPT) > 75:
        sys.exit("prompt too long")
    if not cc.is_running():
        sys.exit("ComfyUI is not running")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = Image.open(args.target).convert("RGB").crop(tuple(int(v) for v in args.crop.split(",")))
    target.save(out / "target_crop.png")
    buf = io.BytesIO()
    target.save(buf, format="PNG")
    face = upload("namphueng_target_face.png", buf.getvalue())
    t_emb, m_emb = face_id.embed(target), face_id.embed(Image.open(MASTER))

    items = []
    for ip in args.ip:
        for seed in args.seeds:
            pid = cc._post("prompt", {"prompt": workflow(face, ip=ip, seed=seed, prefix=f"tgt_{uuid.uuid4().hex[:6]}")})["prompt_id"]
            img = cc.wait_for_result(pid, timeout=900)[0]
            path = cc.download_image(img, out / f"cand_ip{ip}_s{seed}.png")
            emb = face_id.embed(Image.open(path))
            st, sm = face_id.similarity(t_emb, emb), face_id.similarity(m_emb, emb)
            print(f"ip {ip} seed {seed}: vs target {st if st is None else round(st, 3)}, vs current master "
                  f"{sm if sm is None else round(sm, 3)}", flush=True)
            items.append((path, f"ip{ip} s{seed}"))
    label_sheet(items, out / "contact.png", thumb=384, cols=3)
    print(f"target vs current master: {face_id.similarity(t_emb, m_emb)}")


if __name__ == "__main__":
    main()
