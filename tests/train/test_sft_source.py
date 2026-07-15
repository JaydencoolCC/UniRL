"""SupervisedSource track building against fake pipelines (CPU, no weights)."""

from __future__ import annotations

import pytest
import torch

from unirl.train.sft.source import ARSupervisedSource, DiffusionSupervisedSource
from unirl.types.conditions.image import ImageLatentCondition
from unirl.types.primitives import Images, Texts


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "<empty>":
            return {"input_ids": []}  # test hook: a response that tokenizes to nothing
        return {"input_ids": [ord(c) % 50 for c in text]}


class FakeConditions:
    def __init__(self, n: int) -> None:
        self.n = n

    def to_dict(self):
        return {"prompt": None}


class FakeChatStage:
    def __init__(self, takes_images: bool = False) -> None:
        self.last_texts = None
        self.last_images = None
        if takes_images:
            self.embed = self._embed_with_images
        else:
            self.embed = self._embed_text_only

    def _embed_text_only(self, texts: Texts) -> FakeConditions:
        self.last_texts = texts
        return FakeConditions(len(texts.texts))

    def _embed_with_images(self, texts: Texts, images=None) -> FakeConditions:
        self.last_texts = texts
        self.last_images = images
        return FakeConditions(len(texts.texts))


class FakeBundle:
    tokenizer = FakeTokenizer()
    device = torch.device("cpu")


class FakeARPipeline:
    def __init__(self, takes_images: bool = False) -> None:
        self.bundle = FakeBundle()
        self.chat_template = FakeChatStage(takes_images)


def _ar_source(**kwargs) -> ARSupervisedSource:
    return ARSupervisedSource(pipeline=FakeARPipeline(kwargs.pop("takes_images", False)), **kwargs)


def test_ar_build_packs_response_tokens_with_eos():
    source = _ar_source()
    records = [
        {"sample_id": "a", "prompt": "p1", "response": "hi"},
        {"sample_id": "b", "prompt": "p2", "response": "yes!"},
    ]
    track = source.build(records)
    assert track.sample_ids == ["a", "b"]
    assert track.parent_ids is None  # root track: no GRPO grouping
    lengths = track.segment.lengths.tolist()
    assert lengths == [3, 5]  # len(text)+eos
    assert int(track.segment.tokens[2]) == 99 and int(track.segment.tokens[-1]) == 99
    assert torch.all(track.segment.loss_mask == 1.0)
    assert track.segment.log_probs is None  # SFT has no behavior policy


def test_ar_eval_pad_rows_carry_zero_mask():
    source = _ar_source()
    records = [
        {"sample_id": "a", "prompt": "p", "response": "hi"},
        {"sample_id": "a/pad1", "prompt": "p", "response": "hi", "_eval_pad": True},
    ]
    track = source.build(records)
    cu = track.segment.cu_seqlens.tolist()
    mask = track.segment.loss_mask
    assert torch.all(mask[cu[0] : cu[1]] == 1.0)
    assert torch.all(mask[cu[1] : cu[2]] == 0.0)


def test_ar_missing_or_empty_response_raises():
    source = _ar_source()
    with pytest.raises(ValueError, match="response"):
        source.build([{"sample_id": "a", "prompt": "p"}])
    with pytest.raises(ValueError, match="response"):
        source.build([{"sample_id": "a", "prompt": "p", "response": ""}])
    with pytest.raises(ValueError, match="zero"):
        source.build([{"sample_id": "a", "prompt": "p", "response": "<empty>"}])


def test_ar_truncation_keeps_eos():
    source = _ar_source(max_response_length=4)
    track = source.build([{"sample_id": "a", "prompt": "p", "response": "abcdefgh"}])
    assert track.segment.lengths.tolist() == [4]
    assert int(track.segment.tokens[-1]) == 99  # EOS survives truncation


def test_ar_multiturn_messages_rejected():
    source = _ar_source()
    with pytest.raises(NotImplementedError, match="messages"):
        source.build([{"sample_id": "a", "prompt": "p", "response": "r", "messages": []}])


def test_ar_vlm_images_ride_the_embed_call(tmp_path):
    from PIL import Image as PILImage

    img_path = tmp_path / "x.png"
    PILImage.new("RGB", (8, 8), (255, 0, 0)).save(img_path)
    source = _ar_source(takes_images=True)
    records = [
        {
            "sample_id": "a",
            "prompt": "what is this?",
            "response": "red",
            "media_refs": [{"modality": "image", "role": "condition", "uri": str(img_path)}],
        },
        {"sample_id": "b", "prompt": "text only", "response": "ok"},
    ]
    track = source.build(records)
    stage = source.pipeline.chat_template
    assert stage.last_images is not None and stage.last_images[0] is not None and stage.last_images[1] is None
    assert int(track.batch_size) == 2


# ---------------------------------------------------------------------------
# Diffusion source
# ---------------------------------------------------------------------------


class FakeEncodeStage:
    def encode(self, images: Images) -> ImageLatentCondition:
        b = images.pixels.shape[0]
        assert images.pixels.min() >= 0.0 and images.pixels.max() <= 1.0
        return ImageLatentCondition(latents=torch.randn(b, 16, 4, 4))


class FakeDiffusionPipeline:
    def __init__(self) -> None:
        self.bundle = FakeBundle()
        self.vae_encode = FakeEncodeStage()
        self.conditions_kwargs = None

    def build_conditions(self, texts: Texts, *, guidance_scale: float = 1.0):
        self.conditions_kwargs = {"guidance_scale": guidance_scale, "n": len(texts.texts)}
        return FakeConditions(len(texts.texts))


def _diffusion_records(tmp_path, n=2, with_target=True):
    from PIL import Image as PILImage

    records = []
    for i in range(n):
        row = {"sample_id": f"d{i}", "prompt": f"a cat {i}"}
        if with_target:
            path = tmp_path / f"t{i}.png"
            PILImage.new("RGB", (32, 48), (i * 40, 10, 10)).save(path)
            row["media_refs"] = [{"modality": "image", "role": "target", "uri": str(path)}]
        records.append(row)
    return records


def test_diffusion_build_encodes_x0_only_segment(tmp_path):
    source = DiffusionSupervisedSource(pipeline=FakeDiffusionPipeline(), height=64, width=64)
    track = source.build(_diffusion_records(tmp_path))
    assert tuple(track.segment.latents.shape) == (2, 1, 16, 4, 4)  # [B, K=1, ...] — x0 at [:, -1]
    assert torch.all(track.segment.loss_mask == 1.0)
    assert source.pipeline.conditions_kwargs == {"guidance_scale": 1.0, "n": 2}


def test_diffusion_missing_target_raises(tmp_path):
    source = DiffusionSupervisedSource(pipeline=FakeDiffusionPipeline(), height=64, width=64)
    with pytest.raises(ValueError, match="target"):
        source.build(_diffusion_records(tmp_path, with_target=False))


def test_diffusion_resolution_alignment_enforced():
    with pytest.raises(ValueError, match="divisible"):
        DiffusionSupervisedSource(pipeline=FakeDiffusionPipeline(), height=100, width=64)


def test_diffusion_eval_pad_rows_zero_weight(tmp_path):
    source = DiffusionSupervisedSource(pipeline=FakeDiffusionPipeline(), height=64, width=64)
    records = _diffusion_records(tmp_path)
    records.append({**records[-1], "sample_id": "d1/pad", "_eval_pad": True})
    track = source.build(records)
    assert track.segment.loss_mask.tolist() == [1.0, 1.0, 0.0]
