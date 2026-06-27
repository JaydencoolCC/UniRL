from __future__ import annotations

from typing import Any, Dict, Optional

from unirl.models.types.ar import ARSamplingParams
from unirl.models.types.pipeline import Pipeline
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Sample

from .ar import QwenVLARParams, QwenVLARStage
from .bundle import QwenVLBundle
from .chat_template import QwenVLChatTemplateStage
from .conditions import QwenVLARConditions
from .config import QwenVLPipelineConfig


class QwenVLPipeline(Pipeline):
    """Qwen-VL AR (understanding) generate pipeline: ``Sample → Sample``.

    Consumes a request ``Sample`` whose frontier (last) Part is a pre-forked AR gen
    shell carrying ``ARSamplingParams``. Reads the prompt — and, for image-input
    tasks, the chained image — via ``sample.conditioning()`` (an optional
    ``{"system_instruction": str}`` override rides on the input Part's
    ``control["chat"]``) and fills the frontier Part:

    - ``segment: TextSegment`` — the generated tokens + full-softmax log-probs.
    - ``primitive: Texts`` — detokenized response strings.

    ``Part.conditions`` carries the encoded conditions for trainer-side replay (the train stack re-types them via ``conditions_cls.from_dict``).conditioning()`` via :meth:`_conditions_for`, so rollout and
    replay build the prompt conditions through one shared path.
    """

    def __init__(
        self,
        *,
        bundle: QwenVLBundle,
        chat_template: QwenVLChatTemplateStage,
        ar: QwenVLARStage,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.chat_template = chat_template
        self.ar = ar

    @classmethod
    def from_bundle(
        cls,
        bundle: QwenVLBundle,
        *,
        max_prompt_length: int = 4096,
        pad_to_max_length: bool = False,
    ) -> "QwenVLPipeline":
        """Wire chat-template + AR stages around an already-loaded bundle.

        The v2 trainer loads the bundle once and injects it
        (``remote_hydra(pipeline_cfg, bundle=...)``); routing pipeline
        construction through ``from_config`` instead would load a second copy
        of the model. This factory shares the single bundle.

        ``pad_to_max_length`` fixes the prompt sequence length to
        ``max_prompt_length`` so DP rollout shards stay concat-compatible at
        merge time (see :class:`QwenVLChatTemplateStage`).
        """
        chat_template = QwenVLChatTemplateStage(
            bundle,
            max_prompt_length=max_prompt_length,
            pad_to_max_length=pad_to_max_length,
        )
        ar = QwenVLARStage(model=bundle)
        return cls(bundle=bundle, chat_template=chat_template, ar=ar)

    @classmethod
    def from_config(cls, config) -> "QwenVLPipeline":
        if isinstance(config, dict):
            config = QwenVLPipelineConfig(**{k: v for k, v in config.items() if k != "_target_"})
        bundle = QwenVLBundle.from_config(config)
        return cls.from_bundle(bundle, max_prompt_length=config.max_prompt_length)

    def _conditions_for(
        self,
        texts: Texts,
        images: Optional[Images] = None,
        control: Optional[Dict[str, Any]] = None,
    ) -> QwenVLARConditions:
        """Chat-template + tokenize prompts (+ optional images) → :class:`QwenVLARConditions`.
        Shared by rollout-``generate`` and trainer-side replay (re-tokenize), so both
        build the prompt conditions through one path — the re-tokenization must be
        byte-identical to rollout, which routing through this single path guarantees.

        An optional per-request ``system_instruction`` override rides on the input
        Part's ``control["chat"]``.
        """
        chat_overrides: Dict[str, Any] = dict((control or {}).get("chat") or {})
        if "system_instruction" in chat_overrides:
            chat_stage = QwenVLChatTemplateStage(
                self.bundle,
                system_instruction=chat_overrides["system_instruction"],
                max_prompt_length=self.chat_template.max_prompt_length,
            )
        else:
            chat_stage = self.chat_template
        pil_images = images.to_pils() if isinstance(images, Images) else None
        return chat_stage.embed(texts, images=pil_images)

    def generate(self, sample: Sample) -> Sample:
        """Run Qwen-VL AR generation end-to-end, filling the frontier (pre-forked) gen Part."""
        frontier = sample.parts[-1]
        ar = frontier.sampling_params
        if not isinstance(ar, ARSamplingParams):
            raise TypeError(
                f"QwenVLPipeline.generate: frontier gen Part must carry ARSamplingParams, "
                f"got {type(ar).__name__ if ar is not None else 'None'}"
            )

        # conditioning() surfaces [text, image?] in turn order — an image-input task
        # chains the image off the prompt head via Part.input_child.
        conditioning = sample.conditioning()
        texts = conditioning[0] if conditioning else None
        if not isinstance(texts, Texts):
            raise TypeError(
                f"QwenVLPipeline.generate: expected a Texts prompt from sample.conditioning()[0], "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )
        images = next((c for c in conditioning[1:] if isinstance(c, Images)), None)

        conds = self._conditions_for(texts, images, sample.parts[0].control)

        # Normalize the gen shell's ARSamplingParams through QwenVLARParams (parity
        # with the prior req-sourced path: stop_token_id reset, types coerced).
        params = QwenVLARParams(
            max_tokens=ar.max_new_tokens,
            temperature=ar.temperature,
            top_p=ar.top_p,
            top_k=ar.top_k,
        )
        sampling_params = ARSamplingParams(
            max_new_tokens=int(params.max_tokens),
            temperature=float(params.temperature),
            top_p=float(params.top_p),
            top_k=int(params.top_k),
            stop_token_id=None,
        )

        segment = self.ar.autoregress(conds, sampling_params=sampling_params, params=params)
        decoded = self._detokenize(segment)

        # Fill the frontier shell, carrying the encoded conditions for trainer-side
        # replay: Part.conditions is the train stack's source (GRPO re-types them via
        # conditions_cls.from_dict in compute_loss_and_backward).
        filled = frontier.fill(segment=segment, primitive=decoded, conditions=conds.to_dict())
        return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)

    def _detokenize(self, segment) -> Texts:
        if segment.tokens is None or segment.cu_seqlens is None:
            return Texts(texts=[])
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        tokenizer = self.bundle.tokenizer
        out: list = []
        n = len(cu) - 1
        for i in range(n):
            chunk = segment.tokens[cu[i] : cu[i + 1]]
            ids = chunk.tolist() if chunk.numel() > 0 else []
            out.append(tokenizer.decode(ids, skip_special_tokens=True))
        return Texts(texts=out)


__all__ = ["QwenVLPipeline"]
