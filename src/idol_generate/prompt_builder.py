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
            "mornings like this make everything feel possible {hook}",
            "caught between doing everything and doing nothing {hook}",
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
            "outfit does the talking today {hook}",
            "dressed for the version of myself I'm becoming {hook}",
            "if the fit is right, nothing else matters {hook}",
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
            "hi, missed you {hook}",
            "not edited, just me today {hook}",
            "felt cute, might delete later... probably won't {hook}",
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
            "some things are better left unfinished... {hook}",
            "she remembered everything they said she'd forget {hook}",
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
]

# Full SDXL-tuned negative prompt for RealVisXL portrait work
NEGATIVE_PROMPT = (
    "(worst quality, low quality:1.4), (bad anatomy:1.3), "
    "(deformed, distorted, disfigured:1.3), "
    "bad hands, bad fingers, missing fingers, extra fingers, fused fingers, "
    "too many fingers, mutated hands, poorly drawn hands, "
    "long neck, mutation, deformed iris, deformed pupils, "
    "cartoon, painting, illustration, anime, 3d render, cgi, fake, plastic, "
    "blurry, out of focus, overexposed, underexposed, "
    "text, watermark, signature, username, logo, frame, border, "
    "ugly, duplicate, morbid, cloned face, extra limbs, "
    "nsfw, nude, explicit"
)

# CLIP-L prompt_1: scene/subject description — keep lean to fit 77 tokens
# CLIP-G prompt_2: quality/booster tags — SDXL's bigger encoder handles these
QUALITY_TAGS_CLIP_G = (
    "masterpiece, best quality, 8k uhd, ultra-detailed, "
    "RAW photo, professional photograph, award-winning photography, "
    "DSLR, photorealistic, photon mapping, volumetric lighting, "
    "sharp focus, intricate details"
)

# Content-type to clean hashtag mapping
_CONTENT_TAG_MAP: dict[str, str] = {
    "daily_life": "DailyLife",
    "fashion": "Fashion",
    "selfie": "Selfie",
    "story": "Story",
}


def build_image_prompt(character: Character, content_type: ContentType) -> tuple[str, str, str]:
    """Returns (prompt, prompt_2, negative_prompt) — SDXL dual-CLIP split."""
    cfg = CONTENT_CONFIGS[content_type]
    a = character.appearance
    v = character.visual_style

    pose = random.choice(cfg["poses"])
    env = random.choice(cfg["environments"])
    mood = random.choice(cfg["mood_tags"])
    palette = ", ".join(v.color_palette[:2])

    # LoRA trigger word prefix (empty if no LoRA configured)
    trigger = f"{character.lora.trigger_word}, " if character.lora.trigger_word else ""

    # CLIP-L (prompt): subject + scene — kept lean to stay under 77 tokens
    prompt_1 = (
        f"{trigger}"
        f"{a.face}, {a.eye_color}, {a.hair}, {a.signature_features}, "
        f"{a.body_type} figure, {pose}, "
        f"{env}, {v.lighting}, "
        f"{v.camera_settings}, "
        f"{palette} color palette, {mood} mood"
    )
    # CLIP-G (prompt_2): quality/booster tags — SDXL's 2nd encoder handles long tags well
    prompt_2 = f"{QUALITY_TAGS_CLIP_G}, {character.style}"

    return prompt_1, prompt_2, NEGATIVE_PROMPT


def build_caption(character: Character, content_type: ContentType) -> str:
    cfg = CONTENT_CONFIGS[content_type]
    template = random.choice(cfg["caption_hooks"])
    hook = random.choice(ENGAGEMENT_HOOKS)
    emoji = random.choice(character.emoji_pool)
    return f"{template.format(hook=hook)} {emoji}".strip()


def build_tags(character: Character, content_type: ContentType) -> list[str]:
    cfg = CONTENT_CONFIGS[content_type]
    base = [
        character.name,
        "AIIdol",
        "VirtualIdol",
        _CONTENT_TAG_MAP.get(content_type, content_type),
    ]
    mood_tags = [t.title().replace(" ", "") for t in cfg["mood_tags"][:2]]
    style_tags = [t.title().replace(" ", "") for t in character.visual_style.mood_keywords[:2]]
    # Deduplicate while preserving insertion order
    seen: set[str] = set()
    result: list[str] = []
    for tag in base + mood_tags + style_tags:
        if tag not in seen:
            seen.add(tag)
            result.append(tag)
    return result


def build_post(character: Character, content_type: ContentType) -> dict:
    prompt, prompt_2, negative_prompt = build_image_prompt(character, content_type)
    caption = build_caption(character, content_type)
    tags = build_tags(character, content_type)
    # Single draw for mood so prompt and metadata stay consistent
    mood = random.choice(CONTENT_CONFIGS[content_type]["mood_tags"])
    return {
        "prompt": prompt,
        "prompt_2": prompt_2,
        "negative_prompt": negative_prompt,
        "caption": caption,
        "tags": tags,
        "mood": mood,
        "style": character.style,
        "content_type": content_type,
    }
