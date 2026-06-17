"""BagelDiffusionConditions — per-sample conditioning TEXT for the Bagel stage.

Bagel conditions image generation on three prebuilt KV-cache contexts (gen /
cfg_text / cfg_img) produced by running the prompt through the und path. But those
caches are large, GPU-resident, **opaque** objects (the vendored ``NaiveCache``,
holding per-layer K/V tensors) that the framework's tensor transport cannot
"dehydrate" into worker-local refs — only ``torch.Tensor`` leaves reachable through
``Batch`` / ``dict`` / ``list`` / ``tuple`` get kept on the worker. Carrying the
caches on the track therefore makes every Ray round-trip (DP collect → reward →
train dispatch) Ray-pickle them and deserialize them onto the **driver's cuda:0**
— the head GPU0 that also hosts rank-0 — a large, persistent load imbalance.

So this container stores only the **conditioning text** (``{prompt}`` for plain
T2I; ``{prompt}\\n{thinking}`` for the unified reasoning→image path) plus the image
shape — both tiny and dehydration-trivial, exactly like ``BagelARConditions.prompts``
and the HunyuanImage3 / PE conditions (which carry light tensors and re-derive their
KV cache in-forward). :class:`~unirl.models.bagel.diffusion.BagelDiffusionStage`
rebuilds the three KV contexts from this text ON THE WORKER (``_build_contexts``),
once per ``diffuse`` / ``replay`` / velocity-MSE pass. The build is deterministic
under fixed weights, so rollout (``diffuse``) and replay reproduce byte-identical
contexts → on-policy ratio ≈ 1.

``texts`` / ``image_shapes`` are ``concat_field`` lists so :meth:`RolloutTrack.slice`
/ ``concat`` / ``select`` (which the train stack drives per micro-batch) re-index
them per sample. ``Condition`` subclass so it is a valid ``RolloutTrack.conditions``
dict value; ``to_dict`` emits it under a single ``"bagel"`` key, ``from_dict`` reads
it back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Tuple

from unirl.config.require import require
from unirl.distributed.tensor.batch import concat_field
from unirl.types.conditions.base import Condition, Modality


@dataclass
class BagelDiffusionConditions(Condition):
    """Per-sample lightweight conditioning (text + image shape) for Bagel."""

    modality: ClassVar[Modality] = Modality.IMAGE

    texts: List[str] = concat_field(default_factory=list)
    image_shapes: List[Tuple[int, int]] = concat_field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return len(self.texts)

    @classmethod
    def for_sample(cls, *, text: str, image_shape: Tuple[int, int]) -> "BagelDiffusionConditions":
        """Build a single-sample conditions (1-element lists).

        ``text`` is the full conditioning string the image is rendered from
        (``{prompt}`` for plain T2I, ``{prompt}\\n{thinking}`` for UniGRPO); the
        stage rebuilds the gen / cfg_text / cfg_img KV contexts from it. The pipeline
        calls this per sample, then concatenates into the batched track conditions.
        """
        if text is None:
            raise ValueError("BagelDiffusionConditions.for_sample: text is required.")
        if image_shape is None or len(image_shape) != 2:
            raise ValueError(
                f"BagelDiffusionConditions.for_sample: image_shape must be a (H, W) pair; got {image_shape!r}."
            )
        return cls(texts=[text], image_shapes=[tuple(image_shape)])

    def single(self) -> Tuple[str, Tuple[int, int]]:
        """Return ``(text, image_shape)`` for a 1-sample batch.

        The diffusion stage runs one prompt per ``_forward_flow`` call (navit
        ``bs=1``), so it consumes conditions one sample at a time (the train stack
        slices to ``micro_batch_size=1``) and rebuilds the KV contexts from ``text``.
        Raises if the batch isn't exactly one sample.
        """
        require(
            self.batch_size == 1,
            f"BagelDiffusionConditions.single: expected exactly 1 sample (navit bs=1; "
            f"set micro_batch_size=1), got {self.batch_size}.",
        )
        return self.texts[0], tuple(self.image_shapes[0])

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BagelDiffusionConditions":
        """Read the conditions back from a ``RolloutTrack.conditions`` dict.

        Accepts the canonical ``{"bagel": <BagelDiffusionConditions>}`` shape that
        :meth:`to_dict` emits (already an instance, possibly sliced by the train
        stack). Raises otherwise.
        """
        bagel = d.get("bagel")
        if isinstance(bagel, cls):
            return bagel
        raise ValueError(
            "BagelDiffusionConditions.from_dict: expected a 'bagel' key holding a "
            f"BagelDiffusionConditions instance; got keys {sorted(d.keys())}."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Emit as a single ``"bagel"`` entry for ``RolloutTrack.conditions``.

        The whole container is one ``Condition`` dict value; the train stack's
        ``slice`` / ``concat`` re-index its per-sample lists alongside the segment.
        """
        return {"bagel": self}


@dataclass
class BagelARConditions(Condition):
    """Per-sample thinking-prompt text for the Bagel AR (reasoning) stage.

    BAGEL generates the reasoning ("thinking") text autoregressively from a
    prompt context built off this text, then replays it teacher-forced for the
    GRPO log-prob. Both paths tokenize ``prompts[i]`` identically (the bundle
    tokenizer + ``prepare_prompts`` bos/eos wrapping), so storing the text —
    not a pre-packed id tensor — keeps autoregress and replay byte-aligned.

    Held as a per-sample ``concat_field`` list so ``RolloutTrack.slice`` /
    ``concat`` / ``select`` re-index it exactly like the diffusion conditions
    (navit ``bs=1`` per sample; the list-field machinery handles arbitrary
    objects, here plain strings).
    """

    modality: ClassVar[Modality] = Modality.TEXT

    prompts: List[str] = concat_field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return len(self.prompts)

    @classmethod
    def for_sample(cls, *, prompt: str) -> "BagelARConditions":
        """Build a single-sample conditions (1-element list)."""
        if prompt is None:
            raise ValueError("BagelARConditions.for_sample: prompt is required.")
        return cls(prompts=[prompt])

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BagelARConditions":
        """Read back the canonical ``{"bagel_ar": <BagelARConditions>}`` shape."""
        bagel_ar = d.get("bagel_ar")
        if isinstance(bagel_ar, cls):
            return bagel_ar
        raise ValueError(
            "BagelARConditions.from_dict: expected a 'bagel_ar' key holding a "
            f"BagelARConditions instance; got keys {sorted(d.keys())}."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Emit as a single ``"bagel_ar"`` entry for ``RolloutTrack.conditions``."""
        return {"bagel_ar": self}


__all__ = ["BagelARConditions", "BagelDiffusionConditions"]
