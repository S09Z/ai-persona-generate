import random

import pytest

from idol_generate.character import Character
from idol_generate.prompt_builder import (
    CLIP_TOKEN_BUDGET,
    CONTENT_CONFIGS,
    build_image_prompt,
    build_post,
    count_tokens,
)

CHARACTERS = ["model_001", "auri", "luna"]


@pytest.mark.parametrize("name", CHARACTERS)
@pytest.mark.parametrize("content_type", list(CONTENT_CONFIGS))
def test_both_prompts_fit_clip_window(name, content_type):
    character = Character.load(f"characters/{name}.json")
    for seed in range(10):
        random.seed(seed)
        prompt, prompt_2, _ = build_image_prompt(character, content_type)
        assert count_tokens(prompt) <= CLIP_TOKEN_BUDGET
        assert count_tokens(prompt_2) <= CLIP_TOKEN_BUDGET


def test_identity_survives_packing():
    character = Character.load("characters/model_001.json")
    prompt, _, _ = build_image_prompt(character, "athleisure")
    assert "freckles" in prompt
    assert "hourglass" in prompt


def test_outfit_reaches_the_prompt():
    character = Character.load("characters/model_001.json")
    _, prompt_2, _ = build_image_prompt(character, "athleisure")
    assert "sports bra" in prompt_2


def test_mood_is_drawn_once():
    character = Character.load("characters/model_001.json")
    for seed in range(20):
        random.seed(seed)
        post = build_post(character, "minimalist")
        assert f"{post['mood']} mood" in post["prompt_2"]
