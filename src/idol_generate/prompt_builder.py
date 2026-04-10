"""
idol_generate/prompt_builder.py
Builds Stable Diffusion prompts and captions from a Character + content type.
"""

from __future__ import annotations

import random
from typing import Literal

from .character import Character

ContentType = Literal["daily_life", "fashion", "selfie", "story"]

CONTENT_CONFIGS: dict[str, dict] = {
    "daily_life": {
        "poses": [
            "sitting cross-legged on a windowsill",
            "leaning against a kitchen counter holding a mug",
            "reading on a couch with legs tucked under",
            "stretching arms up by a sunlit balcony railing",
        ],
        "environments": [
            "cozy apartment living room, warm morning light",
            "sunlit cafe corner, pastel walls",
            "rooftop terrace at golden hour, city skyline blurred",
            "bedroom with sheer curtains blowing in breeze",
        ],
        "mood_tags": ["calm", "dreamy", "introspective", "warm"],
        "caption_hooks": [
            "some days the light just hits different... {hook}",
            "mornings like this make everything feel possible ☁️ {hook}",
            "caught between doing everything and doing nothing 🌿 {hook}",
        ],
    },
    "fashion": {
        "poses": [
            "standing with one hand on hip, slight side angle",
            "walking mid-stride on a cobblestone street",
            "sitting on steps, legs crossed, looking off-camera",
            "leaning against a white wall, arms loosely folded",
        ],
        "environments": [
            "clean white studio backdrop, soft diffused light",
            "outdoor street with dappled tree shadows",
            "minimalist boutique interior, marble floor",
            "golden-hour park path, bokeh leaves",
        ],
        "mood_tags": ["editorial", "confident", "chic", "effortless"],
        "caption_hooks": [
            "outfit does the talking today 🤫 {hook}",
            "dressed for the version of myself I'm becoming ✨ {hook}",
            "if the fit is right, nothing else matters 🖤 {hook}",
        ],
    },
    "selfie": {
        "poses": [
            "close-up face shot, slight tilt, soft smile",
            "holding phone up, mirror selfie, natural expression",
            "chin resting on folded hands, direct eye contact",
            "looking just off lens, soft laugh caught mid-moment",
        ],
        "environments": [
            "bathroom vanity, warm bulb lighting",
            "car window, overcast soft light",
            "golden afternoon bedroom light",
            "cafe window seat, bokeh street outside",
        ],
        "mood_tags": ["candid", "intimate", "playful", "warm"],
        "caption_hooks": [
            "hi 👋 missed you {hook}",
            "not edited, just me today 🌸 {hook}",
            "felt cute, might delete later... probably won't 😚 {hook}",
        ],
    },
    "story": {
        "poses": [
            "gazing out a window at rain",
            "sitting alone at a table with a letter",
            "walking away down an empty night street, glancing back",
            "standing in front of a mirror, hands on glass",
        ],
        "environments": [
            "moody late-night bedroom, single lamp",
            "rainy cafe window, steamed glass",
            "empty rooftop at dusk, fairy lights",
            "quiet library corner, amber light",
        ],
        "mood_tags": ["mysterious", "melancholic", "nostalgic", "cinematic"],
        "caption_hooks": [
            "some things are better left unfinished... 🌙 {hook}",
            "she remembered everything they said she'd forget 🕯️ {hook}",
            "the city felt smaller that night {hook}",
        ],
    },
}

ENGAGEMENT_HOOKS = [
    "what would you do?",
    "tell me your version 👇",
    "tag someone who gets it",
    "how are you spending today?",
    "what's your slow morning ritual?",
    "",
]

NEGATIVE_PROMPT = (
    "deformed, blurry, bad anatomy, extra limbs, cloned face, disfigured, "
    "low quality, lowres, text, watermark, signature, out of frame, ugly, "
    "overexposed, grainy, unrealistic skin"
)


def build_image_prompt(character: Character, content_type: ContentType) -> tuple[str, str]:
    """Returns (positive_prompt, negative_prompt)."""
    cfg = CONTENT_CONFIGS[content_type]
    a = character.appearance
    v = character.visual_style

    pose = random.choice(cfg["poses"])
    env = random.choice(cfg["environments"])
    mood = random.choice(cfg["mood_tags"])
    palette = ", ".join(v.color_palette[:2])

    positive = (
        f"{character.name}, {a.face}, {a.hair}, {a.signature_features}, "
        f"{a.body_type} figure, {pose}, {env}, "
        f"{v.lighting}, {v.texture}, {palette} tones, "
        f"{mood} mood, {character.style}, "
        f"high detail, photorealistic, sharp focus"
    )
    return positive, NEGATIVE_PROMPT


def build_caption(character: Character, content_type: ContentType) -> str:
    cfg = CONTENT_CONFIGS[content_type]
    template = random.choice(cfg["caption_hooks"])
    hook = random.choice(ENGAGEMENT_HOOKS)
    return template.format(hook=hook).strip()


def build_tags(character: Character, content_type: ContentType) -> list[str]:
    cfg = CONTENT_CONFIGS[content_type]
    base = [
        character.name,
        "AIIdol",
        "VirtualIdol",
        content_type.replace("_", ""),
    ]
    mood_tags = [t.replace(" ", "") for t in cfg["mood_tags"][:2]]
    style_tags = [t.replace(" ", "") for t in character.visual_style.mood_keywords[:2]]
    return base + mood_tags + style_tags


def build_post(character: Character, content_type: ContentType) -> dict:
    positive_prompt, negative_prompt = build_image_prompt(character, content_type)
    caption = build_caption(character, content_type)
    tags = build_tags(character, content_type)
    mood = random.choice(CONTENT_CONFIGS[content_type]["mood_tags"])
    return {
        "prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "caption": caption,
        "tags": tags,
        "mood": mood,
        "style": character.style,
        "content_type": content_type,
    }
