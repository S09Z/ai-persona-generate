"""
idol_generate/prompt_builder.py
Builds Stable Diffusion prompts and captions from a Character + content type.
"""

from __future__ import annotations

import random
import re
from functools import lru_cache
from typing import Literal

from .character import Character

ContentType = Literal[
    "daily_life", "fashion", "selfie", "story",
    "minimalist", "modern_executive", "soft_elegant", "evening_muse", "athleisure",
    "casual_cool", "tomboy_chic",
]

# Each SDXL text encoder reads at most 77 tokens (2 are BOS/EOS); anything past that is dropped.
CLIP_TOKEN_BUDGET = 75

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
    # Quiet-luxury archetypes (MODEL_001); palette and backdrop rules come from the character
    "minimalist": {
        "outfit": "ivory shirt, tailored sand trousers, loafers",
        "poses": [
            "standing relaxed, one hand in pocket, slight side angle",
            "walking slowly, head turned a little toward camera",
            "leaning on a plain cream wall, arms loosely crossed",
        ],
        "environments": [
            "warm minimalist interior, cream walls",
            "beige studio backdrop, soft diffused light",
        ],
        "mood_tags": ["understated", "calm", "refined", "effortless"],
        "caption_hooks": [
            "less, but better {hook}",
            "quiet outfit, quiet confidence {hook}",
        ],
    },
    "modern_executive": {
        "outfit": "taupe tailored blazer, straight espresso trousers, structured bag",
        "poses": [
            "standing tall, one hand holding the bag, calm gaze",
            "walking past a glass door, relaxed stride",
            "sitting on a low bench, legs crossed, bag on lap",
        ],
        "environments": [
            "modern beige office lobby, soft daylight",
            "cream studio backdrop, soft diffused light",
        ],
        "mood_tags": ["polished", "confident", "refined", "composed"],
        "caption_hooks": [
            "workwear that works as hard as I do {hook}",
            "tailored and calm {hook}",
        ],
    },
    "soft_elegant": {
        "outfit": "soft knit top, mocha midi skirt, thin gold jewelry",
        "poses": [
            "standing with a gentle smile, hands relaxed",
            "turning softly, skirt moving slightly",
            "sitting by a window, hands in lap",
        ],
        "environments": [
            "warm minimalist living room, sand tones",
            "sunlit beige interior, sheer curtains",
        ],
        "mood_tags": ["soft", "warm", "graceful", "approachable"],
        "caption_hooks": [
            "soft textures, warm tones {hook}",
            "gold in the details {hook}",
        ],
    },
    "evening_muse": {
        "outfit": "simple black dress, small gold earrings",
        "poses": [
            "standing straight, chin slightly lifted, calm gaze",
            "glancing over one shoulder",
            "sitting on a velvet chair, ankles crossed",
        ],
        "environments": [
            "dim warm lounge, soft amber light",
            "cream backdrop with soft side light",
        ],
        "mood_tags": ["elegant", "poised", "quiet", "luminous"],
        "caption_hooks": [
            "one black dress, nothing more {hook}",
            "evening, simplified {hook}",
        ],
    },
    "casual_cool": {
        "outfit": "oversized white tee, relaxed straight jeans, clean white sneakers",
        "poses": [
            "standing relaxed, hands in jeans pockets, easy half-smile",
            "leaning on a wall, one knee bent, looking off-camera",
            "walking mid-stride, glancing back over one shoulder",
        ],
        "environments": [
            "sunlit cafe terrace, warm tones",
            "quiet city street, soft daylight",
        ],
        "mood_tags": ["chill", "cool", "easygoing", "natural"],
        "caption_hooks": [
            "tee, jeans, done {hook}",
            "no plan, just good light {hook}",
        ],
    },
    "tomboy_chic": {
        "outfit": "cap, loose zip jacket, wide trousers, chunky sneakers, thin gold necklace",
        "poses": [
            "standing with hands in jacket pockets, cool relaxed look",
            "sitting on steps, elbows on knees, direct calm gaze",
            "adjusting her cap, easy half-smile",
        ],
        "environments": [
            "quiet street with warm concrete walls",
            "park path in soft daylight",
        ],
        "mood_tags": ["cool", "relaxed", "effortless", "confident"],
        "caption_hooks": [
            "cap on, thinking off {hook}",
            "comfy first, pretty second, somehow both {hook}",
        ],
    },
    "athleisure": {
        "outfit": "fitted sand sports bra, high-waist taupe leggings",
        "poses": [
            "standing relaxed, hands at hips, slight side angle",
            "stretching arms overhead, calm expression",
            "walking, towel over one shoulder",
        ],
        "environments": [
            "clean beige studio, soft daylight",
            "warm minimalist gym, large window light",
        ],
        "mood_tags": ["fresh", "healthy", "confident", "natural"],
        "caption_hooks": [
            "moving, then matcha {hook}",
            "strong and soft at once {hook}",
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

# The tags that matter most; kept short so they still fit after the scene and style
QUALITY_CORE = "RAW photo, photorealistic, sharp focus"

# Content-type to clean hashtag mapping
_CONTENT_TAG_MAP: dict[str, str] = {
    "daily_life": "DailyLife",
    "fashion": "Fashion",
    "selfie": "Selfie",
    "story": "Story",
}


@lru_cache(maxsize=1)
def _clip_tokenizer():
    """CLIP-L tokenizer from the local HF cache, or None when it is not available offline."""
    try:
        from transformers import CLIPTokenizer

        return CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """CLIP token count without BOS/EOS; falls back to a rough estimate if no tokenizer is cached."""
    tok = _clip_tokenizer()
    if tok is not None:
        return len(tok(text).input_ids) - 2
    return round(len(re.findall(r"\w+|[^\w\s]", text)) * 1.3)


def _phrases(text: str, n: int) -> str:
    """First n comma-separated phrases of a free-text field."""
    return ", ".join(p.strip() for p in text.split(",")[:n] if p.strip())


def _pack(parts: list[str], budget: int = CLIP_TOKEN_BUDGET) -> str:
    """Join parts in priority order, skipping any that would push past the token budget."""
    kept: list[str] = []
    used = 0
    for part in parts:
        if not part:
            continue
        cost = count_tokens(part) + 1  # +1 for the joining comma
        if used + cost <= budget:
            kept.append(part)
            used += cost
    return ", ".join(kept)


def build_image_prompt(
    character: Character, content_type: ContentType, mood: str | None = None, include_pose: bool = True
) -> tuple[str, str, str]:
    """Returns (prompt, prompt_2, negative_prompt) — SDXL dual-CLIP split.

    Each encoder reads only 77 tokens, so both prompts are packed in priority order and
    lower-priority details are dropped rather than silently truncated mid-phrase.
    Pass `mood` to reuse a mood already drawn for the caption/metadata.
    `include_pose=False` leaves the pose phrase out, for backends where a skeleton sets the pose.
    """
    cfg = CONTENT_CONFIGS[content_type]
    a = character.appearance
    v = character.visual_style

    pose = random.choice(cfg["poses"])
    env = random.choice(cfg["environments"])
    mood = mood or random.choice(cfg["mood_tags"])
    palette = ", ".join(v.color_palette[:2])
    trigger = character.lora.trigger_word  # empty if no LoRA configured

    # CLIP-L (prompt): who she is; the outfit only if there is room left
    prompt_1 = _pack([
        trigger,
        _phrases(a.face, 5),
        _phrases(a.eye_color, 1),
        _phrases(a.hair, 1),
        _phrases(a.body_type, 3),
        cfg.get("outfit", ""),
    ])
    # CLIP-G (prompt_2): what she wears and does, where, light and mood first; quality tags last
    prompt_2 = _pack([
        cfg.get("outfit", ""),
        pose if include_pose else "",
        env,
        v.lighting,
        f"{mood} mood",
        f"{palette} color palette",
        _phrases(v.camera_settings, 2),
        QUALITY_CORE,
        character.style,
        QUALITY_TAGS_CLIP_G,
    ])

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


def build_post(character: Character, content_type: ContentType, include_pose: bool = True) -> dict:
    # Single draw for mood so prompt and metadata stay consistent
    mood = random.choice(CONTENT_CONFIGS[content_type]["mood_tags"])
    prompt, prompt_2, negative_prompt = build_image_prompt(character, content_type, mood, include_pose)
    caption = build_caption(character, content_type)
    tags = build_tags(character, content_type)
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
