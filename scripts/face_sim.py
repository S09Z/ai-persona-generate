"""
scripts/face_sim.py -- CLIP similarity of generated heads to the MODEL_001 master face.

Uses the CLIP ViT-H image encoder that IP-Adapter ships with (the same space the identity is injected
in), on CPU so it does not fight ComfyUI for VRAM. Each image is cropped to the head box that
generate_comfy.py derives from the pose in its JSON sidecar. Scores are cosine similarity of
embeddings: global, framing-sensitive, NOT a face-recognition score. Read them relative to the
"other faces" baseline (master face candidates that were rejected) and to each other.

Usage:
    uv run scripts/face_sim.py outputs/comfy_batch                 # *_s<seed>.png and *_face.png pairs
    uv run scripts/face_sim.py outputs/comfy_batch --baseline      # also score rejected candidates
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection  # noqa: E402

from generate_comfy import POSES, head_box  # noqa: E402
from ref_sheet import IP_DIR, MASTER  # noqa: E402

FACE_CROP = (80, 20, 944, 884)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--baseline", action="store_true", help="also score rejected master-face candidates")
    args = ap.parse_args()

    proc = CLIPImageProcessor()
    model = CLIPVisionModelWithProjection.from_pretrained(str(IP_DIR / "models" / "image_encoder")).eval()

    @torch.no_grad()
    def embed(img: Image.Image) -> torch.Tensor:
        e = model(**proc(images=img.convert("RGB"), return_tensors="pt")).image_embeds[0]
        return e / e.norm()

    ref = embed(Image.open(MASTER).crop(FACE_CROP))
    rows = []
    for js in sorted(Path(args.folder).glob("*.json")):
        post = json.loads(js.read_text(encoding="utf-8"))
        yaw = POSES[post["pose"]][0]
        if yaw == 180:
            continue
        box = head_box(yaw)
        crop = (box[0], box[1], box[0] + box[2], box[1] + box[3])
        for key in ("image", "image_face_pass"):
            if key in post:
                img = Image.open(js.parent / post[key])
                rows.append((js.stem.replace("namphueng_", ""), key.replace("image_face_pass", "face-pass")
                             .replace("image", "stage-1"), float(ref @ embed(img.crop(crop)))))
    print(f"{'image':46} {'version':10} similarity")
    for name, version, score in rows:
        print(f"{name:46} {version:10} {score:.3f}")
    for version in ("stage-1", "face-pass"):
        vals = [r[2] for r in rows if r[1] == version]
        if vals:
            print(f"mean {version:10} {sum(vals) / len(vals):.3f}  (n={len(vals)})")

    if args.baseline:
        cands = sorted((MASTER.parent / "candidates").glob("v*_s*.png"))
        vals = [float(ref @ embed(Image.open(c).crop(FACE_CROP))) for c in cands if c.stat().st_size > 0]
        vals = [v for v in vals if v < 0.99]  # drop the master face itself, which is among the candidates
        if vals:
            print(f"baseline other faces (rejected candidates): mean {sum(vals) / len(vals):.3f} "
                  f"min {min(vals):.3f} max {max(vals):.3f} (n={len(vals)})")


if __name__ == "__main__":
    main()
