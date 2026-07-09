"""BagelAgenticEngineFullyAsync integration test with the fake Bagel pipeline (CPU).

Reuses the fake pipeline from ``test_engine`` and exercises the fully-async
engine: resident-pool harvest, the 2-track resp with version metadata, lineage,
and the ``bump_version`` hook.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

from unirl.agentic.engine_fully_async import BagelAgenticEngineFullyAsync
from unirl.models.bagel.diffusion import BagelDiffusionParams
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments.latent import make_image_segment
from unirl.types.segments.text import TextSegment

SEQ, CHAN, KSTORE, NSDE = 4, 2, 3, 2


class _FakeTokenizer:
    def encode(self, text):
        return [(abs(hash(text)) % 100) + 1, len(text) % 50 + 1]

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids]

    def decode(self, ids):
        return f"text<{len(ids)}>"


class _FakeInferencer:
    def init_gen_context(self):
        return {"texts": []}

    def update_context_text(self, text, ctx):
        return {"texts": ctx["texts"] + [text]}


class _FakeBundle:
    def __init__(self):
        self.new_token_ids = {"bos_token_id": 1, "eos_token_id": 2}
        self.tokenizer = _FakeTokenizer()
        self.inferencer = _FakeInferencer()


class _FakeARStage:
    def __init__(self):
        self._m = nn.Linear(2, 2)

    def trainable_module(self):
        return self._m

    def autoregress(self, conditions, *, sampling_params, **_):
        toks = torch.tensor([5, 6, 7], dtype=torch.long)
        lps = torch.tensor([-0.1, -0.2, -0.3], dtype=torch.float32)
        return TextSegment.pack(tokens=[toks], log_probs=[lps])


class _FakeDiffusionStage:
    def __init__(self):
        self._m = nn.Linear(2, 2)

    def trainable_module(self):
        return self._m

    def diffuse(self, conditions, *, schedule, params, initial_latents=None):
        return make_image_segment(
            latents=torch.randn(1, KSTORE, SEQ, CHAN),
            sigmas=torch.linspace(1.0, 0.0, KSTORE),
            indices=torch.arange(KSTORE),
            sde_logp=torch.randn(1, NSDE),
            sde_indices=torch.arange(NSDE),
        )


class _FakeVAE:
    def decode(self, segment, *, image_shape=None):
        return Images(pixels=torch.rand(1, 3, 8, 8))


class _FakePipeline:
    def __init__(self):
        self.bundle = _FakeBundle()
        self.ar = _FakeARStage()
        self.diffusion = _FakeDiffusionStage()
        self.vae_decode = _FakeVAE()

    def build_schedule_policy(self):
        return FlowMatchSchedulePolicy.static_only(3.0)

    def _autocast_ctx(self):
        return nullcontext()


def _make_engine(max_turns=2, n=2, turns_per_window=2, max_staleness=0):
    return BagelAgenticEngineFullyAsync(
        pipeline=_FakePipeline(),
        max_turns=max_turns,
        sessions_per_prompt=n,
        turns_per_window=turns_per_window,
        max_staleness=max_staleness,
        system_prompt="SYSTEM",
        env_instructions=["refine-0", "refine-1", "refine-2"],
    )


def _make_req(group_prompts):
    """Build a fully-async request: one row per (group_id, prompt) group window."""
    gids = [g for g, _ in group_prompts]
    prompts = [p for _, p in group_prompts]
    return RolloutReq(
        sample_ids=list(gids),
        group_ids=list(gids),
        primitives={"text": Texts(texts=prompts)},
        sampling_params={
            "ar": ARSamplingParams(samples_per_prompt=1, temperature=1.0, max_new_tokens=8),
            "diffusion": BagelDiffusionParams(samples_per_prompt=1, num_inference_steps=4, height=8, width=8),
        },
    )


def test_fully_async_generate_two_versioned_tracks():
    engine = _make_engine(max_turns=2, n=2, turns_per_window=2)
    resp = engine.generate(_make_req([("g0", "a cat"), ("g1", "a dog")]))

    assert set(resp.tracks.keys()) == {"ar", "image"}
    think = resp.tracks["ar"]
    image = resp.tracks["image"]
    # 2 groups x N=2 sessions x T=2 turns = 8 rows per track.
    assert len(think.sample_ids) == 8
    assert len(image.sample_ids) == 8
    assert think.parent_track is None and image.parent_track == "ar"
    assert image.parent_ids == think.sample_ids  # 1:1 lineage
    # group-by-parent: 4 think rows per group.
    assert think.parent_ids.count("g0") == 4 and think.parent_ids.count("g1") == 4

    # version metadata present + aligned with tokens / sde steps (all v=0 here).
    assert think.segment.token_versions is not None
    assert think.segment.token_versions.numel() == think.segment.tokens.numel()
    assert int(think.segment.token_versions.max()) == 0
    assert image.segment.sde_versions is not None
    assert image.segment.sde_versions.shape == image.segment.sde_logp.shape
    # behavior-logprob anchor preserved (the PPO old_logp).
    assert think.segment.log_probs is not None and image.segment.sde_logp is not None


def test_fully_async_bump_version_then_generate_stamps_new_version():
    engine = _make_engine(max_turns=1, n=2, turns_per_window=1, max_staleness=0)
    engine.generate(_make_req([("g0", "p0")]))  # consumes g0 @ v0
    assert engine.bump_version() == 1
    resp = engine.generate(_make_req([("g1", "p1")]))  # fresh @ v1
    assert int(resp.tracks["ar"].segment.token_versions.min()) == 1
    assert int(resp.tracks["image"].segment.sde_versions.min()) == 1


def test_fully_async_resp_lineage_valid():
    engine = _make_engine(max_turns=3, n=2, turns_per_window=3)
    resp = engine.generate(_make_req([("g0", "p0"), ("g1", "p1")]))
    # 2 groups x 2 sessions x 3 turns = 12 rows.
    assert len(resp.tracks["ar"].sample_ids) == 12
    assert set(resp.tracks["image"].parent_ids) <= set(resp.tracks["ar"].sample_ids)
