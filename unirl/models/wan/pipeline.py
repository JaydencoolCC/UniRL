"""Pipeline behavior shared by WAN 2.1 and WAN 2.2."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

from .clip_vision_encode import WANCLIPVisionEncodeStage
from .conditions import WANConditions
from .image_encode import WANImageLatentEncodeStage


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


def generate_wan_t2v_or_i2v(
    *,
    sample: Sample,
    owner: str,
    bundle: Any,
    build_conditions: Callable[..., WANConditions],
    diffusion: Any,
    vae_decode: Any,
    use_secondary_guidance: bool,
) -> Sample:
    """Run the common WAN text/image-to-video pipeline and fill its frontier."""
    frontier = sample.parts[-1]
    params = frontier.sampling_params
    if not isinstance(params, DiffusionSamplingParams):
        raise TypeError(
            f"{owner}.generate: frontier gen Part must carry DiffusionSamplingParams, "
            f"got {type(params).__name__ if params is not None else 'None'}"
        )
    if params.sigmas is None:
        raise ValueError(
            f"{owner}.generate: gen part sampling_params.sigmas is None. The hosting "
            "engine must pin σ before invoking pipeline.generate; see the σ ownership note "
            "in unirl.models.types.pipeline."
        )

    conditioning = sample.conditioning()
    texts = conditioning[0] if conditioning else None
    if not isinstance(texts, Texts):
        raise TypeError(
            f"{owner}.generate: expected a Texts prompt from sample.conditioning()[0], "
            f"got {type(texts).__name__ if texts is not None else 'None'}"
        )
    images_prim = next((value for value in conditioning[1:] if isinstance(value, Images)), None)

    guidance_scale = float(params.guidance_scale)
    if use_secondary_guidance:
        secondary = float(params.guidance_scale_2) if params.guidance_scale_2 is not None else guidance_scale
        guidance_scale = max(guidance_scale, secondary)
    wan_conds = build_conditions(texts, guidance_scale=guidance_scale)

    if images_prim is not None:
        if images_prim.pixels is None or int(images_prim.pixels.shape[0]) != len(texts.texts):
            raise ValueError(
                f"{owner}.generate: image count "
                f"{None if images_prim.pixels is None else int(images_prim.pixels.shape[0])} "
                f"!= text count {len(texts.texts)}"
            )
        image_latent = WANImageLatentEncodeStage(
            bundle,
            num_frames=int(params.num_frames),
            height=int(params.height),
            width=int(params.width),
        ).encode(images_prim)
        image_embed = (
            WANCLIPVisionEncodeStage(bundle).encode(images_prim) if getattr(bundle, "uses_clip_vision", False) else None
        )
        wan_conds = dataclasses.replace(
            wan_conds,
            image_latent=image_latent,
            image_embed=image_embed,
        )

    schedule = params.sigmas.to(bundle.device)
    initial_latents = NoiseRecipe.from_sample(sample).resolve()
    latent_seg = diffusion.diffuse(
        wan_conds,
        schedule=schedule,
        params=params,
        initial_latents=initial_latents,
    )
    videos = vae_decode.decode(latent_seg)

    filled = frontier.fill(segment=latent_seg, primitives={"video": videos}, conditions=wan_conds.to_dict())
    return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)


__all__ = ["build_wan_text_conditions", "generate_wan_t2v_or_i2v"]
