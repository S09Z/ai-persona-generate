"""
scripts/tune_face_pass.py -- grid-search the face-pass settings against the ArcFace score.

Re-runs only stage 2 (generate_comfy.build_face_workflow) on stage-1 images that already exist, so a
grid is cheap. Scores each result's head against the master face (face_id.py).

Usage:
    uv run scripts/tune_face_pass.py outputs/comfy_batch2 --seeds 31 33 35 --ip 0.85 1.1 1.4 --denoise 0.35 0.5 0.65
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

import face_id  # noqa: E402
from idol_generate import comfyui_client as cc  # noqa: E402
from idol_generate.character import Character  # noqa: E402
from idol_generate.prompt_builder import NEGATIVE_PROMPT, _pack, _phrases  # noqa: E402
from generate_comfy import FACE_CROP, POSES, WALL_NEGATIVE, build_face_workflow, head_box, png, upload  # noqa: E402
from ref_sheet import EXTRA_NEGATIVE, MASTER  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--ip", type=float, nargs="+", default=[0.85, 1.1, 1.4])
    ap.add_argument("--denoise", type=float, nargs="+", default=[0.35, 0.5, 0.65])
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--out", default=str(ROOT / "outputs/face_tune"))
    args = ap.parse_args()
    if not cc.is_running():
        sys.exit("ComfyUI is not running")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    master = Image.open(MASTER).convert("RGB")
    ref = face_id.embed(master)
    face = upload("namphueng_master_face.png", png(master.crop(FACE_CROP)))
    a = Character.load(ROOT / "characters/model_001.json").appearance
    text = _pack([_phrases(a.face, 5), _phrases(a.eye_color, 1), "detailed eyes, natural skin texture"])
    neg = NEGATIVE_PROMPT + EXTRA_NEGATIVE + WALL_NEGATIVE

    jobs = []
    for js in sorted(Path(args.folder).glob("*.json")):
        post = json.loads(js.read_text(encoding="utf-8"))
        if post["seed"] in args.seeds:
            jobs.append((js.parent / post["image"], post))
    results: dict[tuple[float, float], list[float]] = {}
    for (img_path, post), ip, dn in itertools.product(jobs, args.ip, args.denoise):
        yaw = POSES[post["pose"]][0]
        box = head_box(yaw)
        stage1 = upload(f"namphueng_tune_{img_path.stem}.png", img_path.read_bytes())
        wf = build_face_workflow(stage1, face, text, neg, box, seed=post["seed"], steps=args.steps, cfg=6.0,
                                 ip_weight=ip, denoise=dn, prefix=f"tune_{uuid.uuid4().hex[:6]}")
        res = cc.wait_for_result(cc._post("prompt", {"prompt": wf})["prompt_id"], timeout=900)
        dest = cc.download_image(res[0], out / f"{img_path.stem}_ip{ip}_dn{dn}.png")
        x, y, w, h = box
        head = Image.open(dest).crop((x, y, x + w, y + h))
        score = face_id.similarity(ref, face_id.embed(head))
        print(f"{img_path.stem[-12:]} ip {ip} dn {dn}: {score if score is None else round(score, 3)}", flush=True)
        if score is not None:
            results.setdefault((ip, dn), []).append(score)
    print("\nmean ArcFace cos by setting (higher = closer to the master face)")
    for (ip, dn), v in sorted(results.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"ip {ip:<5} denoise {dn:<5} {sum(v) / len(v):.3f}  (n={len(v)})")


if __name__ == "__main__":
    main()
