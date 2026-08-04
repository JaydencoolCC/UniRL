"""Small pipeline helpers shared by WAN-family variants."""

from __future__ import annotations

from typing import Any, Optional

from unirl.types.primitives import Texts

from .conditions import WANConditions


def build_wan_text_conditions(
    *,
    text_embed: Any,
    texts: Texts,
    negatives: Optional[Texts] = None,
    guidance_scale: float = 1.0,
    owner: str,
) -> WANConditions:
    """Encode the positive and optional CFG-negative WAN text branches."""
    if negatives is not None and len(negatives.texts) != len(texts.texts):
        raise ValueError(
            f"{owner}.build_conditions: negative_text length {len(negatives.texts)} != text length {len(texts.texts)}"
        )
    text_cond = text_embed.embed(texts)
    if negatives is None and float(guidance_scale) > 1.0:
        negatives = Texts(texts=[""] * len(texts.texts))
    negative_text_cond = text_embed.embed(negatives) if negatives is not None else None
    return WANConditions(text=text_cond, negative_text=negative_text_cond)


__all__ = ["build_wan_text_conditions"]
