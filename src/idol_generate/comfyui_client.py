"""
src/idol_generate/comfyui_client.py
ComfyUI API client - WebSocket progress + HTTP polling fallback.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
_WS_URL = COMFYUI_URL.replace("http://", "ws://").replace("https://", "wss://")

# Node IDs matching workflows/luna_portrait.json
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
        body = json.loads(resp.read())
    # Surface ComfyUI structured error payloads clearly
    if "error" in body:
        err = body["error"]
        node_errors = body.get("node_errors", {})
        msg = f"ComfyUI error: {err.get('message', err)}"
        if node_errors:
            msg += f" | node errors: {node_errors}"
        raise RuntimeError(msg)
    return body


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
    Load workflow JSON, inject prompts + params (deep-copied to avoid mutation),
    submit to ComfyUI. Returns the prompt_id.
    """
    raw = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
    # Never mutate the original; drop top-level metadata such as "_meta", which current ComfyUI
    # rejects with "Node has no class_type" because it treats every key as a node.
    workflow = copy.deepcopy({k: v for k, v in raw.items() if isinstance(v, dict) and "class_type" in v})

    workflow[NODE_POSITIVE_PROMPT]["inputs"]["text"] = positive_prompt
    workflow[NODE_NEGATIVE_PROMPT]["inputs"]["text"] = negative_prompt
    workflow[NODE_SAMPLER]["inputs"]["seed"] = (
        seed if seed is not None else random.randint(0, 2**32)
    )
    workflow[NODE_SAMPLER]["inputs"]["steps"] = steps
    workflow[NODE_SAMPLER]["inputs"]["cfg"] = cfg
    workflow[NODE_LATENT]["inputs"]["width"] = width
    workflow[NODE_LATENT]["inputs"]["height"] = height
    # ComfyUI caches by node inputs: a repeated identical prompt (same --seed) is fully cached and
    # returns no output. A unique save prefix makes SaveImage run again and return the image.
    save_inputs = workflow[NODE_SAVE]["inputs"]
    save_inputs["filename_prefix"] = f"{save_inputs.get('filename_prefix', 'idol')}_{int(time.time())}"

    result = _post("prompt", {"prompt": workflow})
    return result["prompt_id"]


def wait_for_result(
    prompt_id: str,
    timeout: float = 300.0,
    use_websocket: bool = True,
) -> list[str]:
    """
    Wait for ComfyUI to finish generating.
    Tries WebSocket first (real-time progress), falls back to HTTP polling.
    Returns list of output image filenames.
    """
    if use_websocket:
        try:
            return _wait_websocket(prompt_id, timeout)
        except Exception:
            pass  # fall through to polling
    return _wait_polling(prompt_id, timeout)


def _wait_websocket(prompt_id: str, timeout: float) -> list[str]:
    """Real-time listener — exits when ComfyUI signals execution complete."""
    import websocket  # websocket-client

    ws = websocket.WebSocket()
    ws.connect(f"{_WS_URL}/ws", timeout=2.0)  # short recv timeout so we can also check history
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                # No status message arrived (some ComfyUI versions only message the submitting
                # client); the history endpoint is the source of truth.
                if _is_done(prompt_id):
                    return _extract_images(prompt_id)
                continue
            if isinstance(raw, bytes):  # binary frames carry preview images, not status
                continue
            msg = json.loads(raw)
            data = msg.get("data", {})
            kind = msg.get("type")
            # Current ComfyUI sends "execution_success"; older versions send "executing" with
            # node=None. Without this the loop never sees the end and burns the whole timeout.
            if kind == "execution_success" and data.get("prompt_id") == prompt_id:
                return _extract_images(prompt_id)
            if kind == "executing" and data.get("node") is None and data.get("prompt_id") == prompt_id:
                return _extract_images(prompt_id)
            if kind == "execution_error" and data.get("prompt_id") == prompt_id:
                raise RuntimeError(f"ComfyUI execution error: {data.get('exception_message', data)}")
    finally:
        ws.close()
    raise TimeoutError(f"Timeout waiting for ComfyUI prompt {prompt_id}")


def _wait_polling(prompt_id: str, timeout: float) -> list[str]:
    """HTTP polling fallback with monotonic timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_done(prompt_id):
            return _extract_images(prompt_id)
        time.sleep(2.0)
    raise TimeoutError(f"Timeout waiting for ComfyUI prompt {prompt_id}")


def _is_done(prompt_id: str) -> bool:
    """True once the history entry shows the run finished (it can appear while still queued)."""
    entry = _get(f"history/{prompt_id}").get(prompt_id)
    if not entry:
        return False
    return bool(entry.get("outputs")) or bool(entry.get("status", {}).get("completed"))


def _extract_images(prompt_id: str) -> list[str]:
    history = _get(f"history/{prompt_id}")
    outputs = history.get(prompt_id, {}).get("outputs", {})
    return [
        img["filename"] for node_output in outputs.values() for img in node_output.get("images", [])
    ]


def download_images(filenames: list[str], output_dir: Path) -> list[Path]:
    """Download all output images from ComfyUI to local output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename in filenames:
        url = f"{COMFYUI_URL}/view?filename={urllib.parse.quote(filename)}&type=output"
        dest = output_dir / filename
        urllib.request.urlretrieve(url, dest)
        paths.append(dest)
    return paths


def download_image(filename: str, dest_path: Path) -> Path:
    """Download a single image (backward compat helper)."""
    url = f"{COMFYUI_URL}/view?filename={urllib.parse.quote(filename)}&type=output"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    return dest_path
