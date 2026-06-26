"""Request-side ``Sample`` extraction helpers the adapters' ``build_inputs`` call.

Pure and family-agnostic. A ``vllm_omni`` request ``Sample`` is the input
``Part`` (``parts[0]``: the prompts) followed by the pre-forked generation
shells — one per stage, identified by the *type* of their ``sampling_params``
(``ARSamplingParams`` → the ``"ar"`` stage; ``DiffusionSamplingParams`` → the
diffusion stage). These helpers recover the prompts and locate those shells;
the HI3 prompt construction (the task presets + the per-prompt entry builder)
lives with the HI3 sub-adapters in ``adapters/hi3.py``.
"""

from __future__ import annotations

from typing import List, Optional

import PIL.Image

from unirl.types.primitives import Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams


def texts_from_sample(sample: Sample) -> Texts:
    """The input ``Part``'s prompt ``Texts``, tiled to the gen-sample count.

    ``vllm_omni`` runs ``num_outputs_per_prompt=1`` — one engine output per gen
    sample — so the prompts must align 1:1 with the frontier gen shell. The
    request keeps the input ``Part`` un-fanned (one prompt per group); the gen
    shell fans out ``samples_per_prompt`` via :meth:`Sample.fork`, group-by-parent
    contiguous. So tile each prompt across its gen siblings here. A request that
    is already 1:1 (no fan-out) is returned unchanged.
    """
    texts = sample.parts[0].primitive
    if not isinstance(texts, Texts):
        raise TypeError(
            f"input Part.primitive must be Texts; got "
            f"{type(texts).__name__ if texts is not None else 'None'}"
        )
    n_prompt = len(texts.texts)
    n_gen = len(sample.parts[-1].sample_ids)
    if n_prompt == n_gen:
        return texts
    if n_prompt == 0 or n_gen % n_prompt != 0:
        raise ValueError(
            f"prompt count {n_prompt} does not evenly tile to gen sample count {n_gen}"
        )
    branch = n_gen // n_prompt
    return Texts(texts=[t for t in texts.texts for _ in range(branch)])


def diffusion_gen_part(sample: Sample) -> Optional[Part]:
    """The gen ``Part`` carrying ``DiffusionSamplingParams`` (image/video), or None."""
    for part in sample.parts[1:]:
        if isinstance(part.sampling_params, DiffusionSamplingParams):
            return part
    return None


def ar_gen_part(sample: Sample) -> Optional[Part]:
    """The gen ``Part`` carrying ``ARSamplingParams`` (the AR prelude), or None."""
    for part in sample.parts[1:]:
        if isinstance(part.sampling_params, ARSamplingParams):
            return part
    return None


def image_input_part(sample: Sample) -> Optional[Part]:
    """The image *input* ``Part`` (``Images`` primitive) chained before the gen
    shells, or None. Multi-input requests chain a second modality off the prompt
    head via :meth:`Part.input_child`; this locates it by primitive type."""
    for part in sample.parts[:-1]:
        if isinstance(part.primitive, Images):
            return part
    return None


def cot_text_from_sample(sample: Sample) -> Texts:
    """The chained cot_text input ``Part``'s ``Texts`` (the AR-generated recaption).

    The two-engine DiT recaption producer takes a SECOND text input — the
    recaption — chained after the prompt head via :meth:`Part.input_child`.
    Scans the non-head input Parts (``parts[1:-1]``) for a ``Texts`` primitive
    and asserts a 1:1 count with the prompts. The original read this off
    ``req.primitives['cot_text']``.
    """
    prompts = texts_from_sample(sample)
    cot_part = next((p for p in sample.parts[1:-1] if isinstance(p.primitive, Texts)), None)
    if cot_part is None:
        raise ValueError(
            "cot_text_from_sample: no chained cot_text input Part (Texts primitive) found; "
            "dit_recaption requires the recaption chained off the prompt (Part.input_child)."
        )
    cot = cot_part.primitive
    if len(cot.texts) != len(prompts.texts):
        raise ValueError(f"cot_text count {len(cot.texts)} != prompt count {len(prompts.texts)}")
    return cot


def pil_images_from_sample(sample: Sample, n: int) -> List[PIL.Image.Image]:
    """Extract an image input ``Part`` (``Images``) as a list of PIL images.

    Returns an empty list when there's no image input Part. Asserts batch
    alignment when present; the conversion itself is :meth:`Images.to_pils`.
    """
    part = image_input_part(sample)
    if part is None:
        return []
    images = part.primitive
    if len(images) != n:
        raise ValueError(f"image batch {len(images)} != prompt count {n}")
    return images.to_pils()


__all__ = [
    "ar_gen_part",
    "cot_text_from_sample",
    "diffusion_gen_part",
    "image_input_part",
    "pil_images_from_sample",
    "texts_from_sample",
]
