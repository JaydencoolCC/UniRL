"""WAN21Pipeline — ``Sample → Sample`` end-to-end for WAN 2.1 T2V/I2V.

Implements the new four-tier flow::

    Texts ──text_embed──▶ WANConditions ──diffuse──▶ LatentSegment ──vae_decode──▶ Videos

Hydra constructs a pipeline via
``WAN21Pipeline.from_config(WAN21PipelineConfig)`` (see ``config.py``);
``from_config`` loads the ``WAN21Bundle`` then constructs the four
stages with the precision policy from the config.

Default SDE strategy is :class:`DanceSDEStrategy` (legacy WAN default in
``samplers/fsdp/wan_sampler.py::FSDPWanSampler.__init__``). Callers
running other strategies (Flow / CPS / DPM2) should pass an explicit
``strategy=`` built from ``cfg.sampling.sde_strategy``.

Schedule policy: WAN does NOT have a diffusers-side scheduler that
ships with the checkpoint (the bundle may set ``scheduler=None``); the
pipeline always uses :func:`unirl.sde.runtime.get_sigma_schedule`
with the configured ``shift``. This mirrors legacy
``samplers/fsdp/wan_sampler.py::sample()``.
"""

from __future__ import annotations

from typing import Any, Optional

from unirl.models.types.pipeline import Pipeline
from unirl.models.wan.conditions import WANConditions
from unirl.models.wan.geometry import wan_latent_shape
from unirl.models.wan.pipeline import build_wan_text_conditions, generate_wan_t2v_or_i2v
from unirl.models.wan.text_embed import WANTextEmbedStage
from unirl.models.wan.vae import WANVAEDecodeStage
from unirl.sde.kernels import DanceSDEStrategy, StepStrategy
from unirl.types.primitives import Texts
from unirl.types.sample import Sample

from .bundle import WAN21Bundle
from .config import WAN21PipelineConfig
from .diffusion import WAN21DiffusionStage, WAN21DiffusionStep


class WAN21Pipeline(Pipeline):
    """WAN 2.1 T2V/I2V generate pipeline: ``Sample → Sample``.

    Consumes a request ``Sample`` whose frontier Part is a pre-forked diffusion gen
    shell carrying ``DiffusionSamplingParams`` (with ``sigmas`` pinned by the
    hosting engine). Reads the prompt — and, for I2V, the chained first-frame image
    — via ``sample.conditioning()`` and fills the frontier Part:

    - ``segment: LatentSegment`` — the denoising trajectory.
    - ``primitives["video"]: Videos`` — the decoded videos.

    ``Part.conditions`` carries the encoded conditions for trainer-side replay (the train stack re-types them via ``conditions_cls.from_dict``). User-supplied text negatives are
    deferred; CFG uses a synthesized empty negative.
    """

    def __init__(
        self,
        *,
        bundle: WAN21Bundle,
        text_embed: Optional[WANTextEmbedStage] = None,
        diffusion: Optional[WAN21DiffusionStage] = None,
        vae_decode: Optional[WANVAEDecodeStage] = None,
        strategy: Optional[StepStrategy] = None,
        shift: float = 5.0,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        max_sequence_length: int = 512,
    ) -> None:
        # Stages default to None and are built from the (trainer-injected)
        # bundle — mirrors SD3Pipeline so the v2 trainer can construct the
        # pipeline via ``remote_hydra(pipeline_cfg, bundle=self.bundle)`` without
        # reloading weights. ``from_config`` still passes pre-built stages.
        super().__init__()
        self.bundle = bundle
        self.text_embed = (
            text_embed
            if text_embed is not None
            else WANTextEmbedStage(bundle, max_sequence_length=int(max_sequence_length))
        )
        if diffusion is None:
            diffusion = WAN21DiffusionStage(
                model=bundle,
                step=WAN21DiffusionStep(),
                strategy=strategy if strategy is not None else DanceSDEStrategy(),
                autocast_precision=autocast_precision,
                trajectory_precision=trajectory_precision,
                logprob_precision=logprob_precision,
            )
        self.diffusion = diffusion
        self.vae_decode = vae_decode if vae_decode is not None else WANVAEDecodeStage(bundle)
        self.shift = shift

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample 5D latent shape ``(C, T_lat, H_lat, W_lat)`` for
        driver-side noise pre-computation. Matches
        ``WAN21DiffusionStage._latent_shape``.

        WAN 2.1: ``AutoencoderKLWan`` is 16-channel, /8 spatial, /4
        temporal. ``T_lat = (num_frames - 1) // 4 + 1``.
        """
        return wan_latent_shape(
            num_frames=int(sampling_spec.num_frames),
            height=int(sampling_spec.height),
            width=int(sampling_spec.width),
        )

    @classmethod
    def from_config(
        cls,
        config: WAN21PipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "WAN21Pipeline":
        """Build the full pipeline from a config.

        ``strategy`` is the SDE step strategy. Defaults to
        :class:`DanceSDEStrategy` (legacy WAN default in
        ``samplers/fsdp/wan_sampler.py``); callers running GRPO with
        Flow / CPS / DPM2 should pass an explicit strategy built from
        ``cfg.sampling.sde_strategy``.
        """
        bundle = WAN21Bundle.from_config(config)
        text_embed = WANTextEmbedStage(bundle, max_sequence_length=int(config.max_sequence_length))
        step = WAN21DiffusionStep()
        diffusion = WAN21DiffusionStage(
            model=bundle,
            step=step,
            strategy=strategy if strategy is not None else DanceSDEStrategy(),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
        )
        vae_decode = WANVAEDecodeStage(bundle)
        return cls(
            bundle=bundle,
            text_embed=text_embed,
            diffusion=diffusion,
            vae_decode=vae_decode,
            shift=float(config.shift),
        )

    def build_conditions(
        self,
        texts: Texts,
        *,
        negatives: Optional[Texts] = None,
        guidance_scale: float = 1.0,
    ) -> WANConditions:
        """Encode prompts (+ optional CFG negatives) into ``WAN21Conditions``.

        Builds only the text-conditioning slots (``text`` / ``negative_text``);
        the optional I2V ``image_latent`` / ``image_embed`` slots are left
        ``None`` and attached by :meth:`generate` when an input image is
        supplied (their encode path needs ``req`` / ``params``, outside this
        text-only contract).

        CFG negative encoding: WAN's training-time convention encodes
        an empty-string negative when none is provided (legacy
        ``models/wan21.py::encode_inputs`` does ``[""] * len(prompts)``
        — and so does diffusers' upstream WAN pipeline). Without this,
        falling back to ``torch.zeros_like(prompt_embeds)`` in
        ``WAN21DiffusionStep.predict_noise`` would silently use a
        different unconditional embedding than what the model was
        trained against, shifting the distribution and making the
        rollout / replay log-prob ratio drift away from 1.0 in GRPO.
        Encoding ``[""] * B`` explicitly here keeps both sides aligned.
        """
        return build_wan_text_conditions(
            text_embed=self.text_embed,
            texts=texts,
            negatives=negatives,
            guidance_scale=guidance_scale,
            owner=type(self).__name__,
        )

    def generate(self, sample: Sample) -> Sample:
        """Run WAN 2.1 T2V (or I2V) end-to-end, filling the frontier (pre-forked) gen Part.

        Requires σ to be pinned onto the gen part's ``DiffusionSamplingParams.sigmas``
        by the hosting engine before the call; see the σ ownership note in
        ``unirl.models.types.pipeline``.
        """
        return generate_wan_t2v_or_i2v(
            sample=sample,
            owner=type(self).__name__,
            bundle=self.bundle,
            build_conditions=self.build_conditions,
            diffusion=self.diffusion,
            vae_decode=self.vae_decode,
            use_secondary_guidance=False,
        )


__all__ = ["WAN21Pipeline"]
