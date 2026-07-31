from types import SimpleNamespace

import pytest
import torch

from unirl.models.wan import (
    WAN21CLIPVisionEncodeStage,
    WAN21Conditions,
    WAN21ImageLatentEncodeStage,
    WAN21TextEmbedStage,
    WAN21VAEDecodeStage,
    WANCLIPVisionEncodeStage,
    WANConditions,
    WANImageLatentEncodeStage,
    WANTextEmbedStage,
    WANVAEDecodeStage,
    wan_latent_shape,
)
from unirl.models.wan21.clip_vision_encode import WAN21CLIPVisionEncodeStage as LegacyCLIPVisionEncodeStage
from unirl.models.wan21.conditions import WAN21Conditions as LegacyConditions
from unirl.models.wan21.image_encode import WAN21ImageLatentEncodeStage as LegacyImageLatentEncodeStage
from unirl.models.wan21.pipeline import WAN21Pipeline
from unirl.models.wan21.text_embed import WAN21TextEmbedStage as LegacyTextEmbedStage
from unirl.models.wan21.vae import WAN21VAEDecodeStage as LegacyVAEDecodeStage
from unirl.models.wan22.pipeline import WAN22Pipeline
from unirl.models.wan22_v2v.pipeline import WAN22V2VPipeline
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts, Video, Videos
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams


class _FakeTextEmbed:
    def __init__(self) -> None:
        self.calls = []

    def embed(self, texts: Texts) -> TextEmbedCondition:
        self.calls.append(list(texts.texts))
        return TextEmbedCondition(embeds=torch.ones(len(texts.texts), 1, 2))


class _FakeDiffusion:
    def __init__(self) -> None:
        self.conditions = None

    def diffuse(self, conditions, **kwargs):
        self.conditions = conditions
        return object()


class _FakeDecode:
    def decode(self, segment) -> Videos:
        return Videos.from_list([Video(frames=torch.zeros(1, 3, 2, 2))])


def _request(*, guidance_scale: float, guidance_scale_2: float | None) -> Sample:
    params = DiffusionSamplingParams(
        num_inference_steps=1,
        guidance_scale=guidance_scale,
        guidance_scale_2=guidance_scale_2,
        sigmas=torch.tensor([1.0, 0.0]),
    )
    prompt = Part.input(["prompt"], primitives={"text": Texts(texts=["a prompt"])})
    return Sample.request(prompt).fork(1, sampling_params=params)


def test_wan21_compatibility_exports_preserve_identity() -> None:
    assert WANConditions is WAN21Conditions is LegacyConditions
    assert WANTextEmbedStage is WAN21TextEmbedStage is LegacyTextEmbedStage
    assert WANVAEDecodeStage is WAN21VAEDecodeStage is LegacyVAEDecodeStage
    assert WANImageLatentEncodeStage is WAN21ImageLatentEncodeStage is LegacyImageLatentEncodeStage
    assert WANCLIPVisionEncodeStage is WAN21CLIPVisionEncodeStage is LegacyCLIPVisionEncodeStage


@pytest.mark.parametrize("pipeline_cls", [WAN21Pipeline, WAN22Pipeline, WAN22V2VPipeline])
def test_wan_pipelines_share_driver_latent_geometry(pipeline_cls) -> None:
    sampling = SimpleNamespace(num_frames=81, height=480, width=832)
    assert pipeline_cls.latent_shape(model_config=None, sampling_spec=sampling) == (16, 21, 60, 104)


def test_wan_latent_geometry_rejects_invalid_frame_count() -> None:
    with pytest.raises(ValueError, match="temporal_downsample=4"):
        wan_latent_shape(num_frames=80, height=480, width=832)


@pytest.mark.parametrize(
    ("pipeline_cls", "expected_embed_calls"),
    [
        (WAN21Pipeline, 1),
        (WAN22Pipeline, 2),
    ],
)
def test_wan_t2v_runtime_preserves_variant_cfg_behavior(pipeline_cls, expected_embed_calls) -> None:
    text_embed = _FakeTextEmbed()
    diffusion = _FakeDiffusion()
    pipeline = pipeline_cls(
        bundle=SimpleNamespace(device=torch.device("cpu"), uses_clip_vision=False),
        text_embed=text_embed,
        diffusion=diffusion,
        vae_decode=_FakeDecode(),
    )

    result = pipeline.generate(_request(guidance_scale=1.0, guidance_scale_2=5.0))

    assert len(text_embed.calls) == expected_embed_calls
    assert ("negative_text" in result.parts[-1].conditions) is (pipeline_cls is WAN22Pipeline)
    assert result.parts[-1].primitives["video"].frames.shape == (1, 3, 2, 2)
    assert diffusion.conditions is not None
