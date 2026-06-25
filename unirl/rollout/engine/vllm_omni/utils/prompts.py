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
    """The input ``Part``'s prompt ``Texts``, asserted to match the gen count.

    ``vllm_omni`` runs ``num_outputs_per_prompt=1`` (the caller pre-expands),
    so the frontier gen shell holds one sample per prompt — the same count as
    the input primitive.
    """
    texts = sample.parts[0].primitive
    if not isinstance(texts, Texts):
        raise TypeError(
            f"input Part.primitive must be Texts; got "
            f"{type(texts).__name__ if texts is not None else 'None'}"
        )
    n_gen = len(sample.parts[-1].sample_ids)
    if len(texts.texts) != n_gen:
        raise ValueError(f"prompt count {len(texts.texts)} != gen sample count {n_gen}")
    return texts


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
    """An image *input* ``Part`` (``Images`` primitive) chained before the gen
    shells, or None. Multi-input text+image request Samples are not wired yet
    (docs/rollout-sample-refactor.md §2 non-goals); this returns None until the
    caller that builds chained input Parts lands."""
    for part in sample.parts[:-1]:
        if isinstance(part.primitive, Images):
            return part
    return None


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
    "diffusion_gen_part",
    "image_input_part",
    "pil_images_from_sample",
    "texts_from_sample",
]
