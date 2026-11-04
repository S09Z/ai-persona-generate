"""
scripts/face_id.py -- ArcFace face-recognition helper (YuNet detection + 5-point alignment + ArcFace R50 ONNX).

Models (see README): models/face/yunet.onnx and models/face/arcface_r50.onnx. Runs on CPU through
onnxruntime so it never competes with ComfyUI for VRAM. Cosine similarity of ArcFace embeddings is
a real identity score: roughly >0.4-0.5 is usually the same person, <0.2-0.3 a different one (the
exact cut depends on image quality; compare against known different faces, see face_sim.py).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models" / "face"
# ArcFace 112x112 landmark template: right eye, left eye, nose, right mouth, left mouth (as seen in the image)
_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                      [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

_session: ort.InferenceSession | None = None


def _arcface() -> ort.InferenceSession:
    global _session
    if _session is None:
        _session = ort.InferenceSession(str(MODELS / "arcface_r50.onnx"), providers=["CPUExecutionProvider"])
    return _session


def _landmarks(bgr: np.ndarray) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    det = cv2.FaceDetectorYN.create(str(MODELS / "yunet.onnx"), "", (w, h), score_threshold=0.5)
    _, faces = det.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    face = max(faces, key=lambda f: f[14])  # highest confidence
    return face[4:14].reshape(5, 2).astype(np.float32)


def embed(img: Image.Image, upscale_to: int = 512) -> np.ndarray | None:
    """Unit ArcFace embedding of the most confident face in `img`, or None if no face is found."""
    rgb = img.convert("RGB")
    scale = max(upscale_to / max(rgb.size), 1.0)  # small head crops detect better when enlarged
    if scale > 1:
        rgb = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.LANCZOS)
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    pts = _landmarks(bgr)
    if pts is None:
        return None
    m, _ = cv2.estimateAffinePartial2D(pts, _TEMPLATE, method=cv2.LMEDS)
    aligned = cv2.warpAffine(bgr, m, (112, 112), borderValue=0)
    blob = (cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
    sess = _arcface()
    out = sess.run(None, {sess.get_inputs()[0].name: blob.transpose(2, 0, 1)[None]})[0][0]
    return out / np.linalg.norm(out)


def similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    return None if a is None or b is None else float(a @ b)
