import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from idol_generate.character import Character

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit = _load("audit_dataset")
snapshot = _load("build_clean_snapshot")
experiment = _load("new_experiment")


def test_dhash_identical_and_resized_images_match():
    img = Image.linear_gradient("L").resize((64, 64)).convert("RGB")
    assert audit.hamming(audit.dhash(img), audit.dhash(img.resize((128, 128)))) <= 4


def test_dhash_different_images_are_far_apart():
    # linear_gradient runs top to bottom; rotate so the brightness changes left to right
    a = Image.linear_gradient("L").resize((64, 64)).transpose(Image.Transpose.ROTATE_90).convert("RGB")
    b = a.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    assert audit.hamming(audit.dhash(a), audit.dhash(b)) > 4


def test_hamming():
    assert audit.hamming(0b1010, 0b0110) == 2


def test_clean_caption_strips_blip_artifacts_and_keeps_trigger():
    out = snapshot.clean_caption("auri_v1, araffe woman smiling", "auri_v1")
    assert out == "auri_v1, woman smiling"


def test_clean_caption_strips_leading_there_is():
    assert snapshot.clean_caption("there is a woman smiling", "auri_v1") == "auri_v1, a woman smiling"


def test_clean_caption_empty_body_is_trigger_only():
    assert snapshot.clean_caption("auri_v1", "auri_v1") == "auri_v1"


BASE = '[model]\nname = "x"\n\n[lora]\nnetwork_dim = 32\n# network_dropout = 0.0\n'


def test_set_value_replaces_existing_key():
    out = experiment.set_value(BASE, "lora", "network_dim", "64")
    assert "network_dim = 64" in out and "network_dim = 32" not in out


def test_set_value_uncomments_commented_key():
    out = experiment.set_value(BASE, "lora", "network_dropout", "0.10")
    assert "network_dropout = 0.10" in out and "# network_dropout" not in out


def test_set_value_unknown_section_exits():
    with pytest.raises(SystemExit):
        experiment.set_value(BASE, "nope", "a", "1")


def test_character_loader_ignores_extra_top_level_keys(tmp_path):
    data = json.loads((ROOT / "characters" / "model_001.json").read_text(encoding="utf-8"))
    data["something_new"] = {"a": 1}
    path = tmp_path / "c.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert Character.load(path).name == "Namphueng"


def test_character_loader_rejects_unknown_appearance_field(tmp_path):
    data = json.loads((ROOT / "characters" / "model_001.json").read_text(encoding="utf-8"))
    data["appearance"]["skin_tone"] = "fair"
    path = tmp_path / "c.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(TypeError):
        Character.load(path)
