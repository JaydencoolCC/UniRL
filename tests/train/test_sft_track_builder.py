from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from unirl.train.sft.track_builder import (
    ARSupervisedTrackBuilder,
    DiffusionSupervisedTrackBuilder,
    _TensorDiskCache,
    _load_pil_images,
    _prefetch_batch_key,
)
from unirl.models.sd3.conditions import SD3Conditions
from unirl.types.conditions import TextEmbedCondition


class _BatchTokenizer:
    eos_token_id = 99

    def __init__(self):
        self.calls = []

    def __call__(self, values, **kwargs):
        self.calls.append((values, kwargs))
        return {
            "input_ids": [
                list(range(1, len(value) + 1))
                for value in values
            ]
        }


def test_response_tokenization_uses_one_batch_call_and_preserves_masks():
    tokenizer = _BatchTokenizer()
    builder = ARSupervisedTrackBuilder.__new__(ARSupervisedTrackBuilder)
    builder.pipeline = SimpleNamespace(bundle=SimpleNamespace(device=torch.device("cpu")))
    builder._tokenizer = tokenizer
    builder.max_response_length = 3
    builder.append_eos = True
    builder._warned_truncation = False
    records = [
        {"sample_id": "a", "response": "abcd"},
        {"sample_id": "pad", "response": "xy", "_eval_pad": True},
    ]

    tokens, masks = builder._tokenize_responses(records)

    assert len(tokenizer.calls) == 1
    assert tokenizer.calls[0][0] == ["abcd", "xy"]
    assert tokenizer.calls[0][1] == {
        "add_special_tokens": False,
        "padding": False,
        "truncation": False,
    }
    assert [tensor.tolist() for tensor in tokens] == [[1, 2, 99], [1, 2, 99]]
    assert [tensor.tolist() for tensor in masks] == [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]


def test_response_tokenization_rejects_empty_target_before_tokenizer():
    tokenizer = _BatchTokenizer()
    builder = ARSupervisedTrackBuilder.__new__(ARSupervisedTrackBuilder)
    builder.pipeline = SimpleNamespace(bundle=SimpleNamespace(device=torch.device("cpu")))
    builder._tokenizer = tokenizer
    builder.max_response_length = 3
    builder.append_eos = True
    builder._warned_truncation = False

    with pytest.raises(ValueError, match="no non-empty 'response'"):
        builder._tokenize_responses([{"sample_id": "bad", "response": ""}])

    assert tokenizer.calls == []


def test_prefetched_token_ids_are_consumed_without_second_tokenizer_call():
    tokenizer = _BatchTokenizer()
    builder = ARSupervisedTrackBuilder.__new__(ARSupervisedTrackBuilder)
    builder.pipeline = SimpleNamespace(bundle=SimpleNamespace(device=torch.device("cpu")))
    builder._tokenizer = tokenizer
    builder.max_response_length = 8
    builder.append_eos = True
    builder._warned_truncation = False
    builder._embed_takes_images = False
    builder.image_load_workers = 1
    builder._prefetch_executor = ThreadPoolExecutor(max_workers=1)
    records = [
        {"sample_id": "a", "response": "abc"},
        {"sample_id": "b", "response": "xy"},
    ]
    builder._prefetch_key = _prefetch_batch_key(records)
    builder._prefetch_future = builder._prefetch_executor.submit(
        builder._prepare_cpu_inputs,
        tuple(records),
    )

    _, batch_ids = builder._take_prefetched(records)
    tokens, _ = builder._tokenize_responses(records, batch_ids=batch_ids)
    builder._prefetch_executor.shutdown()

    assert len(tokenizer.calls) == 1
    assert [tensor.tolist() for tensor in tokens] == [[1, 2, 3, 99], [1, 2, 99]]


def test_eval_build_does_not_consume_pending_train_prefetch():
    builder = ARSupervisedTrackBuilder.__new__(ARSupervisedTrackBuilder)
    executor = ThreadPoolExecutor(max_workers=1)
    builder._prefetch_executor = executor
    builder._prefetch_key = _prefetch_batch_key([{"sample_id": "train"}])
    builder._prefetch_future = executor.submit(lambda: ([None], [[1]]))

    assert builder._take_prefetched([{"sample_id": "eval"}]) is None
    assert builder._prefetch_future is not None
    assert builder._take_prefetched([{"sample_id": "train"}]) == ([None], [[1]])
    executor.shutdown()


def test_parallel_image_loading_preserves_order_and_none_rows(tmp_path):
    paths = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (3, 2), color).save(path)
        paths.append(str(path))

    images = _load_pil_images([paths[0], None, paths[1], paths[2]], max_workers=2)

    assert images[1] is None
    assert [images[index].getpixel((0, 0)) for index in (0, 2, 3)] == [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    ]


def test_diffusion_target_loader_parallel_path_matches_expected_pixels(tmp_path):
    red = tmp_path / "red.png"
    green = tmp_path / "green.png"
    Image.new("RGB", (4, 3), (255, 0, 0)).save(red)
    Image.new("RGB", (4, 3), (0, 255, 0)).save(green)
    builder = DiffusionSupervisedTrackBuilder.__new__(DiffusionSupervisedTrackBuilder)
    builder.width = 2
    builder.height = 2
    builder.image_load_workers = 2
    records = [
        {"media_refs": [{"role": "target", "uri": str(red)}]},
        {"media_refs": [{"role": "target", "uri": str(green)}]},
    ]

    pixels = builder._load_target_pixels(records)

    assert pixels.shape == (2, 3, 2, 2)
    assert torch.allclose(pixels[0, :, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(pixels[1, :, 0, 0], torch.tensor([0.0, 1.0, 0.0]))


def test_tensor_disk_cache_is_atomic_and_fingerprint_namespaced(tmp_path):
    first = _TensorDiskCache(str(tmp_path), fingerprint="model-a", kind="vae", max_entries=8)
    second = _TensorDiskCache(str(tmp_path), fingerprint="model-b", kind="vae", max_entries=8)
    value = torch.tensor([1.0, 2.0])

    first.put("sample", value)

    assert torch.equal(first.get("sample"), value)
    assert second.get("sample") is None

    bounded = _TensorDiskCache(str(tmp_path), fingerprint="bounded", kind="vae", max_entries=2)
    for index in range(3):
        bounded.put(str(index), torch.tensor(index))
    assert len(list(bounded.directory.glob("*.pt"))) == 2


class _FrozenModule(torch.nn.Module):
    def __init__(self, trainable=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=trainable)


def test_cache_guard_disables_trainable_encoder(tmp_path):
    builder = DiffusionSupervisedTrackBuilder.__new__(DiffusionSupervisedTrackBuilder)
    builder.pipeline = SimpleNamespace(
        bundle=SimpleNamespace(
            text_encoder=_FrozenModule(trainable=True),
            vae=_FrozenModule(trainable=False),
        )
    )
    builder._text_cache = _TensorDiskCache(
        str(tmp_path),
        fingerprint="model",
        kind="text",
        max_entries=8,
    )
    builder._vae_cache = _TensorDiskCache(
        str(tmp_path),
        fingerprint="model",
        kind="vae",
        max_entries=8,
    )
    builder._cache_guards_checked = False

    builder._ensure_cache_guards()

    assert builder._text_cache is None
    assert builder._vae_cache is not None


class _FakeConditionPipeline:
    def __init__(self):
        self.bundle = SimpleNamespace(device=torch.device("cpu"))
        self.calls = 0

    def build_conditions(self, texts, **_kwargs):
        self.calls += 1
        values = torch.tensor([[float(len(text))] for text in texts.texts])
        return SD3Conditions(
            text=TextEmbedCondition(
                embeds=values[:, None, :],
                pooled=values,
            )
        )


def test_text_condition_cache_hits_without_changing_batch(tmp_path):
    builder = DiffusionSupervisedTrackBuilder.__new__(DiffusionSupervisedTrackBuilder)
    builder.pipeline = _FakeConditionPipeline()
    builder._conditions_kwargs = {"guidance_scale": 1.0}
    builder._text_cache = _TensorDiskCache(
        str(tmp_path),
        fingerprint="model",
        kind="text",
        max_entries=8,
    )
    builder._cache_stats = {"text_hits": 0, "text_misses": 0, "vae_hits": 0, "vae_misses": 0}
    records = [
        {"sample_id": "a", "prompt": "short", "_manifest_fingerprint": "m"},
        {"sample_id": "b", "prompt": "longer", "_manifest_fingerprint": "m"},
    ]

    first = builder._build_conditions(records)
    second = builder._build_conditions(records)

    assert builder.pipeline.calls == 1
    assert builder._cache_stats["text_misses"] == 2
    assert builder._cache_stats["text_hits"] == 2
    assert torch.equal(first.text.embeds, second.text.embeds)


class _FakeVAEEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, images):
        self.calls += 1
        return SimpleNamespace(latents=images.pixels.mean(dim=1, keepdim=True))


def test_vae_cache_hits_and_invalidates_changed_media(tmp_path):
    image_path = tmp_path / "target.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(image_path)
    builder = DiffusionSupervisedTrackBuilder.__new__(DiffusionSupervisedTrackBuilder)
    builder.pipeline = SimpleNamespace(bundle=SimpleNamespace(device=torch.device("cpu")))
    builder._encode = _FakeVAEEncoder()
    builder._vae_cache = _TensorDiskCache(
        str(tmp_path / "cache"),
        fingerprint="model",
        kind="vae",
        max_entries=8,
    )
    builder._cache_stats = {"text_hits": 0, "text_misses": 0, "vae_hits": 0, "vae_misses": 0}
    builder.width = 2
    builder.height = 2
    builder.image_load_workers = 1
    record = {
        "sample_id": "a",
        "_manifest_fingerprint": "m",
        "media_refs": [{"role": "target", "uri": str(image_path)}],
    }

    first_key = builder._vae_cache_key(record)
    first = builder._encode_latents([record])
    second = builder._encode_latents([record])
    Image.new("RGB", (2, 2), (0, 255, 0)).save(image_path)
    changed_key = builder._vae_cache_key(record)

    assert builder._encode.calls == 1
    assert torch.equal(first, second)
    assert builder._cache_stats["vae_misses"] == 1
    assert builder._cache_stats["vae_hits"] == 1
    assert changed_key != first_key
