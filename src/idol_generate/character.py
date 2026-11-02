"""
idol_generate/character.py
Loads and validates a character JSON from the characters/ directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Appearance:
    face: str
    hair: str
    body_type: str
    signature_features: str
    eye_color: str = ""


@dataclass
class VisualStyle:
    lighting: str
    color_palette: list[str]
    texture: str
    mood_keywords: list[str]
    camera_settings: str = ""


@dataclass
class LoraConfig:
    path: str = ""
    trigger_word: str = ""
    weight: float = 0.8


@dataclass
class Character:
    name: str
    age_virtual: int
    personality_traits: list[str]
    tone_of_voice: str
    style: str
    backstory: str
    content_themes: list[str]
    appearance: Appearance
    visual_style: VisualStyle
    lora: LoraConfig = field(default_factory=LoraConfig)
    ip_adapter_reference_images: list[str] = field(default_factory=list)
    emoji_pool: list[str] = field(default_factory=lambda: ["✨", "🌸", "☁️"])
    past_events: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> Character:
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        lora_data = data.get("lora", {})
        return cls(
            name=data["name"],
            age_virtual=data["age_virtual"],
            personality_traits=data["personality_traits"],
            tone_of_voice=data["tone_of_voice"],
            style=data["style"],
            backstory=data["backstory"],
            content_themes=data["content_themes"],
            appearance=Appearance(**data["appearance"]),
            visual_style=VisualStyle(**data["visual_style"]),
            lora=LoraConfig(**lora_data) if lora_data else LoraConfig(),
            ip_adapter_reference_images=data.get("ip_adapter_reference_images", []),
            emoji_pool=data.get("emoji_pool", ["✨", "🌸", "☁️"]),
            past_events=data.get("past_events", []),
        )

    def save(self, path: str | Path) -> None:
        import dataclasses

        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=2), encoding="utf-8")
