"""Request-side ``Sample`` extraction helpers.

Pure, family-agnostic readers over a request :class:`~unirl.types.sample.Sample`:
recover the prompt(s) and locate the pre-forked generation shells (by the *type*
of their ``sampling_params``) or chained input Parts (by primitive type). Shared
by the model pipelines' ``generate`` (``models/<model>/pipeline.py``) and the
rollout-engine adapters' input construction, so both read the request the same
way — they live in :mod:`unirl.types` rather than under a rollout engine so a
model pipeline can import them without a ``models → rollout`` layer inversion.

A request ``Sample`` is the input ``Part`` (``parts[0]``: the prompts), optionally
followed by chained input Parts (extra modalities via :meth:`Part.input_child`),
then the pre-forked generation shells — one per stage, identified by the type of
their ``sampling_params`` (``ARSamplingParams`` → the AR stage;
``DiffusionSamplingParams`` → the diffusion stage).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from unirl.types.primitives import Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams

if TYPE_CHECKING:
    import PIL.Image


def texts_from_sample(sample: Sample) -> Texts:
    """The input ``Part``'s prompt ``Texts``, tiled to the gen-sample count.

    A 1:1 engine path (``num_outputs_per_prompt=1`` — one engine output per gen
    sample) needs the prompts aligned 1:1 with the frontier gen shell. The request
    keeps the input ``Part`` un-fanned (one prompt per group); the gen shell fans
    out ``samples_per_prompt`` via :meth:`Sample.fork`, group-by-parent contiguous.
    So tile each prompt across its gen siblings here. A request that is already 1:1
    (no fan-out) is returned unchanged.
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
    and asserts a 1:1 count with the prompts.
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


def pil_images_from_sample(sample: Sample, n: int) -> List["PIL.Image.Image"]:
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
