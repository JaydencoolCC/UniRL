"""BagelAgenticEngine integration test with a fake Bagel pipeline (CPU).

Exercises the full multi-turn engine — workflow loop, per-turn AR/diffusion
calls, and 2-track response assembly — without GPU or real Bagel weights. The
``@distributed`` decorator runs the method body locally when the engine is a
plain instance (not wrapped in a Ray Handle), so ``engine.generate(req)`` is
directly callable here.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

from unirl.agentic.engine import BagelAgenticEngine
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
        return [f"t{i}" for i in ids]  # all mappable (no None)

    def decode(self, ids):
        return f"text<{len(ids)}>"  # no <|im_end|> -> kept whole


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
        # one sample, 3 response tokens
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


def _make_engine(max_turns=2):
    return BagelAgenticEngine(
        pipeline=_FakePipeline(),
        max_turns=max_turns,
        system_prompt="SYSTEM",  # avoids the vendored GEN_THINK_SYSTEM_PROMPT import
        env_instructions=["refine-0", "refine-1", "refine-2"],
    )


def _make_req(n_sessions=2):
    return RolloutReq(
        sample_ids=[f"p{i // 1}/s{i}" for i in range(n_sessions)],
        group_ids=["g0"] * n_sessions,  # all sessions share one prompt group
        primitives={"text": Texts(texts=[f"prompt {i}" for i in range(n_sessions)])},
        sampling_params={
            "ar": ARSamplingParams(samples_per_prompt=1, temperature=1.0, max_new_tokens=8),
            "diffusion": BagelDiffusionParams(samples_per_prompt=1, num_inference_steps=4, height=8, width=8),
        },
    )


def test_generate_builds_two_linked_tracks():
    engine = _make_engine(max_turns=2)
    req = _make_req(n_sessions=2)
    resp = engine.generate(req)

    assert set(resp.tracks.keys()) == {"ar", "image"}
    think = resp.tracks["ar"]
    image = resp.tracks["image"]

    # 2 sessions x 2 turns = 4 rows in each track.
    assert len(think.sample_ids) == 4
    assert len(image.sample_ids) == 4

    # think is root; image is its 1:1 child.
    assert think.parent_track is None
    assert image.parent_track == "ar"
    assert image.parent_ids == think.sample_ids  # 1:1 lineage
    # think grouped by the prompt group (all 4 rows share g0).
    assert think.parent_ids == ["g0"] * 4

    # segments assembled: think packed (4 rows x 3 tokens), image stacked [4,K,seq,C].
    assert int(think.segment.lengths.numel()) == 4
    assert think.segment.tokens.numel() == 12
    assert image.segment.latents.shape == (4, KSTORE, SEQ, CHAN)
    assert image.segment.sde_logp.shape == (4, NSDE)

    # decoded payloads present.
    assert isinstance(think.decoded, Texts) and len(think.decoded.texts) == 4
    assert isinstance(image.decoded, Images) and image.decoded.pixels.shape[0] == 4

    # conditions carry the Bagel typed payloads for replay.
    assert "bagel_ar" in think.conditions
    assert "bagel" in image.conditions
    assert len(think.conditions["bagel_ar"].prompt_splits) == 4
    assert len(image.conditions["bagel"].gen_contexts) == 4


def test_think_context_grows_with_turns():
    """Turn 1's AR prompt splits include turn 0's think + obs (obs as prefix only)."""
    engine = _make_engine(max_turns=2)
    req = _make_req(n_sessions=1)
    resp = engine.generate(req)
    splits = resp.tracks["ar"].conditions["bagel_ar"].prompt_splits
    # turn 0 context: [system, prompt]; turn 1 context: [system, prompt, think_0, obs_0]
    assert len(splits[0]) == 2
    assert len(splits[1]) == 4


def test_resp_post_init_lineage_valid():
    """RolloutResp.__post_init__ foreign-key checks pass (image.parent_ids ⊆ think ids)."""
    engine = _make_engine(max_turns=3)
    resp = engine.generate(_make_req(n_sessions=2))
    # 2 sessions x 3 turns = 6 rows.
    assert len(resp.tracks["ar"].sample_ids) == 6
    assert len(resp.tracks["image"].sample_ids) == 6
    assert set(resp.tracks["image"].parent_ids) <= set(resp.tracks["ar"].sample_ids)
