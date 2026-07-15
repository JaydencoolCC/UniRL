"""TrainStack SFT extensions: requires_advantages, token weighting invariance, eval_track."""

from __future__ import annotations

import pytest
import torch

from unirl.algorithms.base import AlgorithmStepResult
from unirl.train.stack.base import TrainStack
from unirl.types.rollout_resp import RolloutTrack
from unirl.types.segments.text import TextSegment


class FakeBackend:
    """The minimal optimizer surface TrainStack drives; grads accumulate on `weight`."""

    grad_sync_deferred = False
    optimizer = None
    scheduler = None

    def __init__(self) -> None:
        self.module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.module.weight.fill_(1.0)  # p=1 ⇒ fake loss == the weighted mean itself
        self.eval_calls = 0

    def trainable_module(self) -> torch.nn.Module:
        return self.module

    def zero_grad(self) -> None:
        self.module.zero_grad(set_to_none=True)

    def set_grad_sync(self, enable: bool) -> None:
        del enable

    def optimizer_step(self, *, max_grad_norm: float) -> float:
        del max_grad_norm
        return 0.0  # no-op: the test reads the accumulated grad afterwards

    def on_rollout_end(self) -> None:
        pass


class TokenMeanAlgo:
    """loss = p · masked-token-mean(values) per micro — the SFT CE shape.

    Per-token values ride the segment's ``log_probs`` slot; after one update
    the accumulated grad of ``p`` equals the composite weighting the stack
    applied — exactly what the invariance test measures.
    """

    requires_advantages = False
    supports_multi_update = True
    loss_weighting = "token"

    def __init__(self, param: torch.nn.Parameter) -> None:
        self.param = param

    def recomputes_anchor(self) -> bool:
        return False

    def prepare_segment(self, *, conditions, segment) -> None:
        pass

    def compute_loss_and_backward(self, *, conditions, segment, advantages, training_progress, loss_scale):
        assert advantages is None  # the stack must forward None for anchor-free SFT
        values = segment.log_probs.float()
        mask = segment.loss_mask.float()
        denom = mask.sum().clamp(min=1.0)
        loss = self.param.squeeze() * (values * mask).sum() / denom
        (loss * loss_scale).backward()
        return AlgorithmStepResult(
            loss=float(loss.detach()), metrics={}, num_steps_or_tokens=int(denom), has_backward=True
        )

    @torch.no_grad()
    def evaluate_loss(self, *, conditions, segment):
        values = segment.log_probs.float()
        mask = segment.loss_mask.float()
        return float((values * mask).sum()), float(mask.sum())


def _track(lengths, values, masks) -> RolloutTrack:
    segment = TextSegment.pack(
        tokens=[torch.zeros(n, dtype=torch.long) for n in lengths],
        log_probs=[torch.tensor(v, dtype=torch.float32) for v in values],
        loss_mask=[torch.tensor(m, dtype=torch.float32) for m in masks],
    )
    return RolloutTrack(sample_ids=[f"s{i}" for i in range(len(lengths))], segment=segment)


LENGTHS = [1, 3, 2, 4]
VALUES = [[2.0], [1.0, 1.0, 4.0], [3.0, 5.0], [0.0, 2.0, 2.0, 0.0]]
MASKS = [[1.0], [1.0, 0.0, 1.0], [1.0, 1.0], [1.0, 1.0, 1.0, 0.0]]
# masked token values: 2 | 1,4 | 3,5 | 0,2,2 → sum 19 over 8 tokens
GLOBAL_TOKEN_MEAN = 19.0 / 8.0


@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
def test_token_weighting_gradient_invariant_to_micro_split(micro_batch_size):
    backend = FakeBackend()
    param = backend.module.weight
    stack = TrainStack(
        fsdp_backend=backend,
        algorithm=TokenMeanAlgo(param),
        micro_batch_size=micro_batch_size,
        max_grad_norm=1.0,
    )
    result = stack.train_track(_track(LENGTHS, VALUES, MASKS), training_progress=0.0)
    # d/dp [global token-mean of masked values] is the same number regardless
    # of how micros carved the batch — the whole point of the token contract.
    assert param.grad is not None
    assert float(param.grad.squeeze()) == pytest.approx(GLOBAL_TOKEN_MEAN, rel=1e-6)
    # And the REPORTED loss is the exact global token-mean, not a sum of micro means.
    assert result.loss == pytest.approx(GLOBAL_TOKEN_MEAN, rel=1e-6)
    assert result.metrics["global_loss_weight"] == pytest.approx(8.0)


def test_sample_weighting_unchanged_for_rl_algorithms():
    backend = FakeBackend()
    param = backend.module.weight

    class SampleAlgo(TokenMeanAlgo):
        loss_weighting = "sample"

    stack = TrainStack(fsdp_backend=backend, algorithm=SampleAlgo(param), micro_batch_size=2, max_grad_norm=1.0)
    stack.train_track(_track(LENGTHS, VALUES, MASKS), training_progress=0.0)
    # Sample-share weighting: mean over micros of their token-means, weighted by
    # sample share — micro1 = (2+1+4)/3 tokens... = 7/3, micro2 = (3+5+0+2+2)/5 = 12/5.
    expected = 0.5 * (7.0 / 3.0) + 0.5 * (12.0 / 5.0)
    assert float(param.grad.squeeze()) == pytest.approx(expected, rel=1e-6)


def test_requires_advantages_gate():
    backend = FakeBackend()

    class RLAlgo(TokenMeanAlgo):
        requires_advantages = True

    stack = TrainStack(fsdp_backend=backend, algorithm=RLAlgo(backend.module.weight), max_grad_norm=1.0)
    with pytest.raises(ValueError, match="advantages"):
        stack.train_track(_track(LENGTHS, VALUES, MASKS), training_progress=0.0)


def test_zero_valid_tokens_fails_loudly():
    backend = FakeBackend()
    stack = TrainStack(fsdp_backend=backend, algorithm=TokenMeanAlgo(backend.module.weight), max_grad_norm=1.0)
    all_masked = _track([2, 2], [[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="zero valid tokens"):
        stack.train_track(all_masked, training_progress=0.0)


def test_eval_track_weighted_mean_and_eval_mode():
    backend = FakeBackend()
    stack = TrainStack(
        fsdp_backend=backend, algorithm=TokenMeanAlgo(backend.module.weight), micro_batch_size=2, max_grad_norm=1.0
    )
    backend.module.train()
    metrics = stack.eval_track(_track(LENGTHS, VALUES, MASKS))
    assert metrics["loss"] == pytest.approx(GLOBAL_TOKEN_MEAN, rel=1e-6)
    assert metrics["weight"] == pytest.approx(8.0)
    assert backend.module.training  # mode restored after forward-only eval


def test_eval_track_requires_evaluate_loss():
    backend = FakeBackend()

    class NoEval:
        requires_advantages = False
        loss_weighting = "sample"

        def recomputes_anchor(self):
            return False

        def prepare_segment(self, **kw):
            pass

        def compute_loss_and_backward(self, **kw):
            raise AssertionError

    stack = TrainStack(fsdp_backend=backend, algorithm=NoEval(), max_grad_norm=1.0)
    with pytest.raises(TypeError, match="evaluate_loss"):
        stack.eval_track(_track(LENGTHS, VALUES, MASKS))
