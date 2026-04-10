"""
src/idol_generate/comfyui_client.py
ComfyUI API client - submits workflows and polls for results.

Usage:
    Make sure ComfyUI is running: ./comfyui.sh
    Then call run_workflow() from generate.py or directly.
"""

from __future__ import annotations

import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

COMFYUI_URL = "http://127.0.0.1:8188"

# Node IDs in luna_portrait.json
NODE_POSITIVE_PROMPT = "2"
NODE_NEGATIVE_PROMPT = "3"
NODE_SAMPLER = "5"
NODE_LATENT = "4"
NODE_SAVE = "7"


def _post(endpoint: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _get(endpoint: str) -> dict:
    with urllib.request.urlopen(f"{COMFYUI_URL}/{endpoint}", timeout=10) as resp:
        return json.loads(resp.read())


def is_running() -> bool:
    """Check if ComfyUI server is up."""
    try:
        _get("system_stats")
        return True
    except Exception:
        return False


def queue_workflow(
    workflow_path: str | Path,
    positive_prompt: str,
    negative_prompt: str,
    seed: int | None = None,
    steps: int = 25,
    cfg: float = 6.0,
    width: int = 832,
    height: int = 1216,
) -> str:
    """
    Load a workflow JSON, inject prompts + params, submit to ComfyUI.
    Returns the prompt_id for polling.
    """
    workflow = json.loads(Path(workflow_path).read_text(encoding="utf-8"))

    # Inject prompts
    workflow[NODE_POSITIVE_PROMPT]["inputs"]["text"] = positive_prompt
    workflow[NODE_NEGATIVE_PROMPT]["inputs"]["text"] = negative_prompt

    # Inject sampler settings
    workflow[NODE_SAMPLER]["inputs"]["seed"] = (
        seed if seed is not None else random.randint(0, 2**32)
    )
    workflow[NODE_SAMPLER]["inputs"]["steps"] = steps
    workflow[NODE_SAMPLER]["inputs"]["cfg"] = cfg

    # Inject image size
    workflow[NODE_LATENT]["inputs"]["width"] = width
    workflow[NODE_LATENT]["inputs"]["height"] = height

    result = _post("prompt", {"prompt": workflow})
    return result["prompt_id"]


def wait_for_result(
    prompt_id: str, poll_interval: float = 2.0, timeout: float = 300.0
) -> list[Path]:
    """
    Poll ComfyUI history until the prompt is done.
    Returns list of output image paths from ComfyUI output folder.
    """
    elapsed = 0.0
    while elapsed < timeout:
        history = _get(f"history/{prompt_id}")
        if prompt_id in history:
            outputs = history[prompt_id].get("outputs", {})
            images: list[Path] = []
            for node_output in outputs.values():
                for img in node_output.get("images", []):
                    images.append(Path(img["filename"]))
            return images
        time.sleep(poll_interval)
        elapsed += poll_interval

    msg = f"Timeout waiting for ComfyUI prompt {prompt_id}"
    raise TimeoutError(msg)


def download_image(filename: str, dest_path: Path) -> Path:
    """Download a generated image from ComfyUI to local outputs/."""
    url = f"{COMFYUI_URL}/view?filename={urllib.parse.quote(filename)}&type=output"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    return dest_path
