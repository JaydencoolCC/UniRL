"""WAN22Pipeline — ``Sample → Sample`` end-to-end for WAN 2.2 T2V/I2V.

Implements the new four-tier flow::

    Texts ──text_embed (wan)──▶ WANConditions ──diffuse (wan22)──▶ LatentSegment ──vae_decode (wan)──▶ Videos

Hydra constructs a pipeline via
``WAN22Pipeline.from_config(WAN22PipelineConfig)`` (see ``config.py``);
``from_config`` loads the :class:`WAN22Bundle` (dual transformer + WAN
2.1 VAE/text encoder) then constructs the four stages with the
precision policy from the config.

WAN 2.2 and WAN 2.1 compose the same family-owned text embedding, VAE,
conditions, geometry, and T2V/I2V runtime. Only the bundle and diffusion
step/stage differ for dual-transformer routing.
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

from .bundle import WAN22Bundle
from .config import WAN22PipelineConfig
from .diffusion import WAN22DiffusionStage, WAN22DiffusionStep


class WAN22Pipeline(Pipeline):
    """WAN 2.2 T2V/I2V generate pipeline: ``Sample → Sample``.

    Consumes a request ``Sample`` whose frontier Part is a pre-forked diffusion gen
    shell carrying ``DiffusionSamplingParams`` (with ``sigmas`` pinned by the
    hosting engine). Reads the prompt — and, for I2V, the chained first-frame image
    — via ``sample.conditioning()`` and fills the frontier Part:

    - ``segment: LatentSegment`` — the denoising trajectory.
    - ``primitives["video"]: Videos`` — the decoded videos.

    ``Part.conditions`` carries the encoded conditions for trainer-side replay (the train stack re-types them via ``conditions_cls.from_dict``). User-supplied text negatives are
    deferred; CFG uses a synthesized empty negative. ``DiffusionSamplingParams``
    carries the optional ``guidance_scale_2`` WAN22 routes CFG by.
    """

    def __init__(
        self,
        *,
        bundle: WAN22Bundle,
        text_embed: Optional[WANTextEmbedStage] = None,
        diffusion: Optional[WAN22DiffusionStage] = None,
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
        # reloading the dual transformer. ``from_config`` still passes pre-built stages.
        super().__init__()
        self.bundle = bundle
        self.text_embed = (
            text_embed
            if text_embed is not None
            else WANTextEmbedStage(bundle, max_sequence_length=int(max_sequence_length))
        )
        if diffusion is None:
            diffusion = WAN22DiffusionStage(
                model=bundle,
                step=WAN22DiffusionStep(),
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
        driver-side noise pre-computation. Same VAE family as WAN 2.1
        (``AutoencoderKLWan``: 16-channel, /8 spatial, /4 temporal); the
        dual-transformer routing in WAN 2.2 does not change latent
        geometry."""
        return wan_latent_shape(
            num_frames=int(sampling_spec.num_frames),
            height=int(sampling_spec.height),
            width=int(sampling_spec.width),
        )

    @classmethod
    def from_config(
        cls,
        config: WAN22PipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "WAN22Pipeline":
        """Build the full pipeline from a config.

        ``strategy`` is the SDE step strategy. Defaults to
        :class:`DanceSDEStrategy` (legacy WAN family default). Callers
        running other strategies (Flow / CPS / DPM2) should pass an
        explicit strategy built from ``cfg.sampling.sde_strategy``.
        """
        bundle = WAN22Bundle.from_config(config)

        # WAN 2.1's text embed stage expects a ``WAN21Bundle``-compatible
        # object — we satisfy that contract with ``WAN22Bundle`` (it
        # exposes the same ``text_encoder`` / ``tokenizer`` /
        # ``max_sequence_length`` / ``device`` fields). The stage uses
        # duck-typing so no isinstance check fires.
        text_embed = WANTextEmbedStage(bundle, max_sequence_length=int(config.max_sequence_length))
        step = WAN22DiffusionStep()
        diffusion = WAN22DiffusionStage(
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
        the optional ``image_latent`` / ``image_embed`` slots are left ``None``
        and attached by :meth:`generate` when an input image is supplied.

        CFG empty negative: same rationale as WAN21Pipeline — WAN training
        encodes an empty-string negative when none is supplied. WAN22 routes
        CFG by sigma / ``guidance_scale_2``, so :meth:`generate` passes the
        **effective** guidance (``max(guidance_scale, guidance_scale_2)``) here;
        gating on ``> 1.0`` then reproduces WAN22's two-branch ``cfg_active``
        trigger exactly.
        """
        return build_wan_text_conditions(
            text_embed=self.text_embed,
            texts=texts,
            negatives=negatives,
            guidance_scale=guidance_scale,
            owner=type(self).__name__,
        )

    def generate(self, sample: Sample) -> Sample:
        """Run WAN 2.2 T2V (or I2V) end-to-end, filling the frontier (pre-forked) gen Part.

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
            use_secondary_guidance=True,
        )


__all__ = ["WAN22Pipeline"]
