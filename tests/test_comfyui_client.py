import json
from pathlib import Path

from idol_generate import comfyui_client as cc

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / "workflows" / "luna_portrait.json"


def test_queue_workflow_drops_meta_and_uniquifies_save_prefix(monkeypatch):
    sent = {}
    monkeypatch.setattr(cc, "_post", lambda endpoint, payload: sent.update(payload) or {"prompt_id": "x"})
    cc.queue_workflow(str(WORKFLOW), "pos", "neg", seed=1)
    nodes = sent["prompt"]
    assert "_meta" not in nodes
    assert all("class_type" in n for n in nodes.values())
    assert nodes[cc.NODE_POSITIVE_PROMPT]["inputs"]["text"] == "pos"
    assert nodes[cc.NODE_SAVE]["inputs"]["filename_prefix"].startswith("luna_idol_")
    assert json.loads(WORKFLOW.read_text(encoding="utf-8"))["7"]["inputs"]["filename_prefix"] == "luna_idol"


def test_is_done_requires_completion_not_just_presence(monkeypatch):
    history = {}
    monkeypatch.setattr(cc, "_get", lambda endpoint: history)
    assert cc._is_done("p") is False
    history["p"] = {"status": {"completed": False}, "outputs": {}}
    assert cc._is_done("p") is False
    history["p"] = {"status": {"completed": True}, "outputs": {}}
    assert cc._is_done("p") is True
    history["p"] = {"outputs": {"7": {"images": []}}}
    assert cc._is_done("p") is True
