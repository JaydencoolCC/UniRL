"""CPU unit tests for the SFT losses (unirl/algorithms/sft.py) — fake stages, no weights."""

from __future__ import annotations

import math

import pytest
import torch

from unirl.algorithms.sft import SFT, FlowMatchSFT, flow_shift_sigma
from unirl.sde.runtime import get_sigma_schedule
from unirl.types.segments.latent import make_image_segment
from unirl.types.segments.text import TextSegment


class FakeARStage:
    """Replay returns pre-seeded per-token logp; asserts SFT pins temperature=1.0."""

    def __init__(self, logp: torch.Tensor) -> None:
        self.logp = logp
        self.calls = 0

    def replay(self, conditions, *, segment, temperature):
        assert temperature == 1.0, "SFT must never inherit a sampling temperature"
        self.calls += 1
        total = int(segment.tokens.shape[0])
        return self.logp[:total].clone().requires_grad_(True)


def _text_segment(lengths, logp_values, mask_values=None):
    tokens = [torch.zeros(n, dtype=torch.long) for n in lengths]
    kwargs = {"tokens": tokens}
    if mask_values is not None:
        kwargs["loss_mask"] = [torch.tensor(m, dtype=torch.float32) for m in mask_values]
    seg = TextSegment.pack(**kwargs)
    return seg, torch.tensor(logp_values, dtype=torch.float32)


def test_sft_token_mean_is_masked_nll_mean():
    seg, logp = _text_segment([2, 3], [-1.0, -2.0, -3.0, -4.0, -5.0])
    algo = SFT(stage=FakeARStage(logp), loss_agg_mode="token-mean")
    assert algo.loss_weighting == "token"
    result = algo.compute_loss_and_backward(
        conditions={}, segment=seg, advantages=None, training_progress=0.0, loss_scale=1.0
    )
    assert result.has_backward
    assert result.num_steps_or_tokens == 5
    assert result.loss == pytest.approx(3.0)  # mean of 1..5


def test_sft_loss_mask_zeroes_tokens_and_denominator():
    seg, logp = _text_segment([2, 2], [-1.0, -100.0, -3.0, -100.0], mask_values=[[1, 0], [1, 0]])
    algo = SFT(stage=FakeARStage(logp), loss_agg_mode="token-mean")
    result = algo.compute_loss_and_backward(
        conditions={}, segment=seg, advantages=None, training_progress=0.0, loss_scale=1.0
    )
    assert result.loss == pytest.approx(2.0)  # (1 + 3) / 2 — masked tokens excluded from sum AND denom


def test_sft_seq_mean_token_mean_weighs_sequences_equally():
    seg, logp = _text_segment([1, 3], [-4.0, -1.0, -1.0, -1.0])
    algo = SFT(stage=FakeARStage(logp), loss_agg_mode="seq-mean-token-mean")
    assert algo.loss_weighting == "sample"
    result = algo.compute_loss_and_backward(
        conditions={}, segment=seg, advantages=None, training_progress=0.0, loss_scale=1.0
    )
    assert result.loss == pytest.approx((4.0 + 1.0) / 2.0)


def test_sft_evaluate_loss_returns_raw_sums():
    seg, logp = _text_segment([2, 3], [-1.0, -2.0, -3.0, -4.0, -5.0])
    algo = SFT(stage=FakeARStage(logp))
    ce_sum, tokens = algo.evaluate_loss(conditions={}, segment=seg)
    assert ce_sum == pytest.approx(15.0)
    assert tokens == pytest.approx(5.0)


def test_sft_empty_segment_short_circuits():
    algo = SFT(stage=FakeARStage(torch.zeros(0)))
    result = algo.compute_loss_and_backward(
        conditions={}, segment=None, advantages=None, training_progress=0.0, loss_scale=1.0
    )
    assert not result.has_backward and result.loss == 0.0


def test_sft_rejects_unknown_agg_mode():
    with pytest.raises(ValueError, match="loss_agg_mode"):
        SFT(stage=FakeARStage(torch.zeros(1)), loss_agg_mode="mean")


# ---------------------------------------------------------------------------
# FlowMatchSFT
# ---------------------------------------------------------------------------


class PerfectVelocityStage:
    """Given x_t = (1-σ)x0 + σε, returns (x_t - x0)/σ == ε - x0 — the exact target."""

    def __init__(self, x0: torch.Tensor) -> None:
        self.x0 = x0

    def predict_noise_at_step(self, conditions, *, sample, sigma, params):
        s = sigma.reshape(-1, *([1] * (sample.ndim - 1))) if sigma.ndim else sigma
        return ((sample - self.x0) / s).requires_grad_(True)


class ZeroVelocityStage:
    def __init__(self):
        self.seen_sigma = None

    def predict_noise_at_step(self, conditions, *, sample, sigma, params):
        self.seen_sigma = sigma
        return torch.zeros_like(sample).requires_grad_(True)


def _latent_segment(x0: torch.Tensor, loss_mask=None):
    kwargs = {"latents": x0.unsqueeze(1)}
    if loss_mask is not None:
        kwargs["loss_mask"] = loss_mask
    return make_image_segment(**kwargs)


def test_flowmatch_perfect_prediction_zero_loss():
    x0 = torch.randn(4, 3, 8, 8)
    algo = FlowMatchSFT(params=None, stage=PerfectVelocityStage(x0), timestep_shift=3.0)
    result = algo.compute_loss_and_backward(
        conditions={}, segment=_latent_segment(x0), advantages=None, training_progress=0.0, loss_scale=1.0
    )
    assert result.loss == pytest.approx(0.0, abs=1e-8)
    assert result.num_steps_or_tokens == 4


def test_flowmatch_zero_prediction_matches_target_norm():
    torch.manual_seed(0)
    x0 = torch.randn(2, 3, 4, 4)
    stage = ZeroVelocityStage()
    algo = FlowMatchSFT(params=None, stage=stage, timestep_sampling="uniform", timestep_shift=1.0)
    gen = torch.Generator().manual_seed(algo.eval_seed)
    # Reproduce evaluate_loss's deterministic draw to compute the expected MSE.
    loss_sum, weight = algo.evaluate_loss(conditions={}, segment=_latent_segment(x0))
    u = torch.rand(2, dtype=torch.float32, generator=gen)
    # The σ draw advances the generator BEFORE the noise draw — replicate it so
    # the reproduced ε matches _velocity_mse's; the value itself is target-inert
    # here (zero prediction ⇒ MSE = ‖ε - x0‖² regardless of σ).
    sigma = flow_shift_sigma(u, 1.0).clamp(min=algo.sigma_min, max=1 - algo.sigma_min)
    assert bool((sigma > 0).all() and (sigma < 1).all())
    noise = torch.randn(x0.shape, dtype=torch.float32, generator=gen)
    target = noise - x0
    expected = target.pow(2).mean(dim=(1, 2, 3))
    assert weight == pytest.approx(2.0)
    assert loss_sum == pytest.approx(float(expected.sum()), rel=1e-5)
    assert stage.seen_sigma is not None


def test_flowmatch_eval_is_deterministic_and_mask_aware():
    x0 = torch.randn(3, 3, 4, 4)
    mask = torch.tensor([1.0, 1.0, 0.0])
    seg = _latent_segment(x0, loss_mask=mask)
    algo = FlowMatchSFT(params=None, stage=ZeroVelocityStage(), timestep_shift=3.0)
    s1, w1 = algo.evaluate_loss(conditions={}, segment=seg)
    s2, w2 = algo.evaluate_loss(conditions={}, segment=seg)
    assert (s1, w1) == (s2, w2)  # seeded draw — comparable across evals
    assert w1 == pytest.approx(2.0)  # padded row carries zero weight


def test_flowmatch_shift_matches_inference_schedule():
    # σ(u) under shift=3 must match the static get_sigma_schedule warp at the
    # same time points (u here plays 1-t; the schedule is t=linspace(1,0)).
    shift = 3.0
    t = torch.linspace(1, 0, 11)
    expected = get_sigma_schedule(10, shift=shift)
    assert torch.allclose(flow_shift_sigma(t, shift), expected, atol=1e-6)


def test_flowmatch_rejects_bad_config():
    with pytest.raises(ValueError, match="timestep_sampling"):
        FlowMatchSFT(params=None, stage=ZeroVelocityStage(), timestep_sampling="cosine")
    with pytest.raises(ValueError, match="timestep_shift"):
        FlowMatchSFT(params=None, stage=ZeroVelocityStage(), timestep_shift=0.0)


def test_flowmatch_requires_x0_segment():
    algo = FlowMatchSFT(params=None, stage=ZeroVelocityStage())
    with pytest.raises(ValueError, match="latents"):
        algo.compute_loss_and_backward(
            conditions={}, segment=make_image_segment(), advantages=None, training_progress=0.0, loss_scale=1.0
        )


def test_sft_ppl_metric_is_bounded():
    seg, logp = _text_segment([1], [-50.0])
    algo = SFT(stage=FakeARStage(logp))
    result = algo.compute_loss_and_backward(
        conditions={}, segment=seg, advantages=None, training_progress=0.0, loss_scale=1.0
    )
    assert math.isfinite(result.metrics["sft_ppl"])
