"""
scripts/face_sim.py -- ArcFace similarity of generated heads to the MODEL_001 master face.

Each image is cropped to the head box generate_comfy.py derives from the pose in its JSON sidecar,
then scored with ArcFace (face_id.py). Back views are skipped. --baseline also scores the rejected
master-face candidates (different people from the same generator) so the scores have a floor.

Usage:
    uv run scripts/face_sim.py outputs/comfy_batch2 --baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

import face_id  # noqa: E402
from generate_comfy import POSES, head_box  # noqa: E402
from ref_sheet import MASTER  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--baseline", action="store_true", help="also score rejected master-face candidates")
    args = ap.parse_args()

    ref = face_id.embed(Image.open(MASTER))
    if ref is None:
        sys.exit("no face found in the master face image")
    rows = []
    for js in sorted(Path(args.folder).glob("*.json")):
        post = json.loads(js.read_text(encoding="utf-8"))
        yaw = POSES[post["pose"]][0]
        if yaw == 180:
            continue
        x, y, w, h = head_box(yaw)
        for key, label in (("image", "stage-1"), ("image_face_pass", "face-pass")):
            if key in post:
                img = Image.open(js.parent / post[key]).crop((x, y, x + w, y + h))
                rows.append((js.stem.replace("namphueng_", ""), label, face_id.similarity(ref, face_id.embed(img))))
    print(f"{'image':46} {'version':10} arcface cos")
    for name, version, score in rows:
        print(f"{name:46} {version:10} {'no face' if score is None else f'{score:.3f}'}")
    for version in ("stage-1", "face-pass"):
        vals = [r[2] for r in rows if r[1] == version and r[2] is not None]
        miss = sum(1 for r in rows if r[1] == version and r[2] is None)
        if vals:
            print(f"mean {version:10} {sum(vals) / len(vals):.3f}  (n={len(vals)}, no face found: {miss})")

    if args.baseline:
        vals = []
        for c in sorted((MASTER.parent / "candidates").glob("v*_s*.png")):
            s = face_id.similarity(ref, face_id.embed(Image.open(c)))
            if s is not None and s < 0.99:  # drop the master face itself, which is among the candidates
                vals.append(s)
        if vals:
            print(f"baseline other faces (rejected candidates): mean {sum(vals) / len(vals):.3f} "
                  f"min {min(vals):.3f} max {max(vals):.3f} (n={len(vals)})")


if __name__ == "__main__":
    main()
