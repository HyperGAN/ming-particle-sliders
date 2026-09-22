"""Bare UNI prompt rows for Ming layer decomposition.

Attributes are bookkeeping, not a caption prefix. The negative line is a
canary, not a teacher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ming.defaults import CONTROL_PROMPT, GUIDANCE


@dataclass
class SliderPrompt:
    target: str
    positive: str
    neutral: str
    negative: str = ""
    attributes: list[str] = field(default_factory=list)
    action: str = "enhance"
    guidance_scale: float = GUIDANCE
    resolution: int = 1024
    num_layers: int = 6
    unconditional: str = ""


@dataclass
class PromptsMeta:
    plus_label: str = ""
    minus_label: str = ""
    recommended_range: list[float] = field(default_factory=lambda: [0.0, 1.0])
    concept_words: str = ""
    control_prompt: str = CONTROL_PROMPT
    bare_captions: bool = False


def word_tokens(text: str) -> list[str]:
    cleaned = str(text).replace(",", " ").replace(".", " ")
    return [part.lower() for part in cleaned.split() if part.strip()]


def unused_words_for(prompt: SliderPrompt) -> list[str]:
    words: list[str] = []
    for attr in prompt.attributes:
        words.extend(word_tokens(attr))
    return words


def load_prompts(path: Path) -> tuple[list[SliderPrompt], PromptsMeta]:
    with Path(path).open(encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)
    meta = PromptsMeta()
    if isinstance(raw, dict):
        meta.plus_label = str(raw.get("plus_label") or "")
        meta.minus_label = str(raw.get("minus_label") or "")
        meta.concept_words = str(raw.get("concept_words") or "")
        meta.control_prompt = str(raw.get("control_prompt") or CONTROL_PROMPT)
        # Attributes stay off the caption. A card may say so with bare_captions
        # or with prefix_attributes: false.
        prefix_attributes = bool(raw.get("prefix_attributes", False))
        if raw.get("bare_captions"):
            prefix_attributes = False
        meta.bare_captions = not prefix_attributes
        if prefix_attributes:
            raise ValueError(
                "Ming prompt cards keep attributes off the caption. "
                "Set bare_captions: true."
            )
        rng = raw.get("recommended_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            meta.recommended_range = [float(rng[0]), float(rng[1])]
        raw = raw.get("rows")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"prompts file is empty: {path}")
    prompts: list[SliderPrompt] = []
    for item in raw:
        if not isinstance(item, dict) or "positive" not in item:
            raise ValueError(f"each prompt must be a mapping with positive: {item!r}")
        target = str(item.get("target") or item.get("neutral") or item["positive"])
        prompts.append(
            SliderPrompt(
                target=target,
                positive=str(item["positive"]),
                neutral=str(item.get("neutral") or target),
                negative=str(item.get("negative") or ""),
                attributes=[str(a) for a in (item.get("attributes") or [])],
                action=str(item.get("action") or "enhance"),
                guidance_scale=float(item.get("guidance_scale", GUIDANCE)),
                resolution=int(item.get("resolution", 1024)),
                num_layers=int(item.get("num_layers", 6)),
                unconditional=str(item.get("unconditional") or ""),
            )
        )
    return prompts, meta
