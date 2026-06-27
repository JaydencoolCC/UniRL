"""WAN22Pipeline — ``Sample → Sample`` end-to-end for WAN 2.2 T2V/I2V.

Implements the new four-tier flow::

    Texts ──text_embed (wan21)──▶ WAN21Conditions ──diffuse (wan22)──▶ LatentSegment ──vae_decode (wan21)──▶ Videos

Hydra constructs a pipeline via
``WAN22Pipeline.from_config(WAN22PipelineConfig)`` (see ``config.py``);
``from_config`` loads the :class:`WAN22Bundle` (dual transformer + WAN
2.1 VAE/text encoder) then constructs the four stages with the
precision policy from the config.

WAN 2.2 reuses WAN 2.1's text embedding and VAE stages verbatim (same
UMT5 with zero-padding, same 3D VAE with per-channel norm) — only the
diffusion stage swaps in for dual-transformer routing. We do **not**
inherit ``WAN21Pipeline``: the reuse is by composition (import the
sibling stages), matching the SD3 convention of one-package-per-model.
"""

from __future__ import annotations

from typing import Any, Optional

from unirl.models.types.pipeline import Pipeline
from unirl.models.wan21.clip_vision_encode import WAN21CLIPVisionEncodeStage
from unirl.models.wan21.conditions import WAN21Conditions
from unirl.models.wan21.image_encode import WAN21ImageLatentEncodeStage
from unirl.models.wan21.text_embed import WAN21TextEmbedStage
from unirl.models.wan21.vae import WAN21VAEDecodeStage
from unirl.sde.kernels import DanceSDEStrategy, StepStrategy
from unirl.types.conditions import ImageEmbedCondition, ImageLatentCondition
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

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
    - ``primitive: Videos`` — the decoded videos.

    ``Part.conditions`` carries the encoded conditions for trainer-side replay (the train stack re-types them via ``conditions_cls.from_dict``). User-supplied text negatives are
    deferred; CFG uses a synthesized empty negative. ``DiffusionSamplingParams``
    carries the optional ``guidance_scale_2`` WAN22 routes CFG by.
    """

    def __init__(
        self,
        *,
        bundle: WAN22Bundle,
        text_embed: Optional[WAN21TextEmbedStage] = None,
        diffusion: Optional[WAN22DiffusionStage] = None,
        vae_decode: Optional[WAN21VAEDecodeStage] = None,
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
            else WAN21TextEmbedStage(bundle, max_sequence_length=int(max_sequence_length))
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
        self.vae_decode = vae_decode if vae_decode is not None else WAN21VAEDecodeStage(bundle)
        self.shift = shift

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample 5D latent shape ``(C, T_lat, H_lat, W_lat)`` for
        driver-side noise pre-computation. Same VAE family as WAN 2.1
        (``AutoencoderKLWan``: 16-channel, /8 spatial, /4 temporal); the
        dual-transformer routing in WAN 2.2 does not change latent
        geometry."""
        height = int(sampling_spec.height)
        width = int(sampling_spec.width)
        num_frames = int(sampling_spec.num_frames)
        if (num_frames - 1) % 4 != 0:
            raise ValueError(
                f"WAN VAE temporal_downsample=4 requires "
                f"(num_frames - 1) % 4 == 0, got num_frames={num_frames}; "
                f"valid choices: 1, 5, 9, 13, 17, 21, ..."
            )
        latent_t = (num_frames - 1) // 4 + 1
        return (16, latent_t, height // 8, width // 8)

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
        text_embed = WAN21TextEmbedStage(bundle, max_sequence_length=int(config.max_sequence_length))
        step = WAN22DiffusionStep()
        diffusion = WAN22DiffusionStage(
            model=bundle,
            step=step,
            strategy=strategy if strategy is not None else DanceSDEStrategy(),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
        )
        vae_decode = WAN21VAEDecodeStage(bundle)
        return cls(
            bundle=bundle,
            text_embed=text_embed,
            diffusion=diffusion,
            vae_decode=vae_decode,
            shift=float(config.shift),
        )

    def _conditions_for(
        self, texts: Texts, params: DiffusionSamplingParams, images_prim: Optional[Images] = None
    ) -> WAN21Conditions:
        """Encode prompts (+ optional i2v first-frame image) → :class:`WAN21Conditions`.
        Shared by rollout-``generate`` and trainer-side replay (re-encode).

        CFG empty negative: same rationale as WAN21Pipeline (see that
        method's comment) — WAN training encodes an empty-string
        negative when none is supplied. WAN22 routes CFG by sigma /
        ``guidance_scale_2`` so we trigger the empty-negative encoding
        whenever either branch's effective guidance is > 1.
        """
        text_cond = self.text_embed.embed(texts)
        primary_g = float(params.guidance_scale)
        low_g = float(params.guidance_scale_2) if params.guidance_scale_2 is not None else primary_g
        cfg_active = max(primary_g, low_g) > 1.0
        negatives = Texts(texts=[""] * len(texts.texts)) if cfg_active else None
        negative_text_cond = self.text_embed.embed(negatives) if negatives is not None else None

        image_latent_cond: Optional[ImageLatentCondition] = None
        image_embed_cond: Optional[ImageEmbedCondition] = None
        if images_prim is not None:
            if not isinstance(images_prim, Images):
                raise TypeError(
                    f"WAN22Pipeline.generate: i2v image must be Images, got {type(images_prim).__name__}"
                )
            if int(images_prim.pixels.shape[0]) != len(texts.texts):
                raise ValueError(
                    f"WAN22Pipeline.generate: image count {images_prim.pixels.shape[0]} "
                    f"!= text count {len(texts.texts)}"
                )
            image_latent_cond = WAN21ImageLatentEncodeStage(
                self.bundle,
                num_frames=int(params.num_frames),
                height=int(params.height),
                width=int(params.width),
            ).encode(images_prim)
            # CLIP-vision branch fires only on bundles that actually
            # loaded a vision tower. WAN 2.2's mainstream checkpoints
            # set ``image_dim == 0`` and skip this; left in place so a
            # future 2.2 variant with ``image_dim > 0`` (if it ever
            # ships) wires up automatically without a pipeline change.
            if getattr(self.bundle, "uses_clip_vision", False):
                image_embed_cond = WAN21CLIPVisionEncodeStage(self.bundle).encode(images_prim)

        return WAN21Conditions(
            text=text_cond,
            negative_text=negative_text_cond,
            image_latent=image_latent_cond,
            image_embed=image_embed_cond,
        )

    def generate(self, sample: Sample) -> Sample:
        """Run WAN 2.2 T2V (or I2V) end-to-end, filling the frontier (pre-forked) gen Part.

        Requires σ to be pinned onto the gen part's ``DiffusionSamplingParams.sigmas``
        by the hosting engine before the call; see the σ ownership note in
        ``unirl.models.types.pipeline``.
        """
        frontier = sample.parts[-1]
        params = frontier.sampling_params
        if not isinstance(params, DiffusionSamplingParams):
            raise TypeError(
                f"WAN22Pipeline.generate: frontier gen Part must carry DiffusionSamplingParams, "
                f"got {type(params).__name__ if params is not None else 'None'}"
            )
        if params.sigmas is None:
            raise ValueError(
                "WAN22Pipeline.generate: gen part sampling_params.sigmas is None. The hosting "
                "engine must pin σ before invoking pipeline.generate; see the σ ownership note "
                "in unirl.models.types.pipeline."
            )

        # conditioning() surfaces [text, image?] in turn order — the i2v first-frame
        # rides as a chained input Part (Part.input_child) on the request.
        conditioning = sample.conditioning()
        texts = conditioning[0] if conditioning else None
        if not isinstance(texts, Texts):
            raise TypeError(
                f"WAN22Pipeline.generate: expected a Texts prompt from sample.conditioning()[0], "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )
        images_prim = next((c for c in conditioning[1:] if isinstance(c, Images)), None)

        wan_conds = self._conditions_for(texts, params, images_prim)
        schedule = params.sigmas.to(self.bundle.device)

        # Driver-authoritative x_T via the model-aware recipe (NoiseRecipe); a
        # pre-shipped initial_latents tensor (img2img / i2v first-frame) still wins.
        initial_latents = NoiseRecipe.from_sample(sample).resolve()

        latent_seg = self.diffusion.diffuse(
            wan_conds, schedule=schedule, params=params, initial_latents=initial_latents
        )
        videos = self.vae_decode.decode(latent_seg)

        # Fill the frontier shell, carrying the encoded conditions for trainer-side
        # replay (FlowGRPO re-types Part.conditions via conditions_cls.from_dict).
        filled = frontier.fill(segment=latent_seg, primitive=videos, conditions=wan_conds.to_dict())
        return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)


__all__ = ["WAN22Pipeline"]
