"""finish_reason → SegmentStatus mapping (LIN-531).

The agentic engine relies on a per-candidate terminal status to tell a finished
turn apart from an interrupted/truncated one; the sglang text adapter derives it
from the seam's ``finish_reason``. Pure/CPU — no engine, no GPU.
"""

import pytest

pytest.importorskip("torch")  # the unirl types import torch at module load

from tests.rollout.engine._fakes import FakeRaw  # noqa: E402
from unirl.rollout.engine.sglang.adapters.text import TextLMAdapter  # noqa: E402
from unirl.types.segments.base import SegmentStatus  # noqa: E402


def _raw(finish_reason: str) -> FakeRaw:
    return FakeRaw(text="", token_ids=[1, 2], logprobs=[-0.1, -0.2], finish_reason=finish_reason)


def test_finish_reason_maps_to_segment_status():
    """stop→COMPLETED, length→TRUNCATED, abort→ABORTED; anything else→PENDING (per row)."""
    raws = [_raw("stop"), _raw("length"), _raw("abort"), _raw("something-else")]
    status = TextLMAdapter.build_status(raws)

    assert status.dtype.is_floating_point is False  # integer status codes
    assert status.tolist() == [
        int(SegmentStatus.COMPLETED),
        int(SegmentStatus.TRUNCATED),
        int(SegmentStatus.ABORTED),
        int(SegmentStatus.PENDING),
    ]


def test_status_is_one_per_candidate():
    """The status tensor is row-aligned with the candidates (one per raw result)."""
    raws = [_raw("stop")] * 5
    assert TextLMAdapter.build_status(raws).shape[0] == 5
