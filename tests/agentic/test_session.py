"""Session / MsgNode unit tests (verification plan #1).

Validates the linear-chain trajectory: think-token packing, image-latent
stacking, and the "obs not in segment" invariant — obs text is a context node,
never a training-segment row.
"""

from __future__ import annotations

import pytest
import torch

from unirl.agentic.session import MsgNode, Session
from unirl.types.segments.latent import make_image_segment


def _think(tok_ids, lp=None):
    t = torch.tensor(tok_ids, dtype=torch.long)
    return MsgNode(kind="think", tokens=t, logprobs=lp, text="plan " + "".join(map(str, tok_ids)))


def _gen(k=3, seq=4, channels=2, n_sde=2, image="img"):
    return MsgNode(
        kind="gen",
        latent=make_image_segment(
            latents=torch.randn(1, k, seq, channels),
            sigmas=torch.linspace(1.0, 0.0, k),
            indices=torch.arange(k),
            sde_logp=torch.randn(1, n_sde),
            sde_indices=torch.arange(n_sde),
        ),
        image=image,
    )


def test_msgnode_kind_validation():
    with pytest.raises(ValueError):
        MsgNode(kind="bogus", tokens=torch.tensor([1]))
    with pytest.raises(ValueError):
        MsgNode(kind="think")  # missing tokens
    with pytest.raises(ValueError):
        MsgNode(kind="gen")  # missing latent
    with pytest.raises(ValueError):
        MsgNode(kind="obs")  # missing text


def test_build_think_segment_packs_per_turn_rows():
    s = Session(prompt="a cat")
    s.append(_think([10, 11, 12]))
    s.append(_gen())
    s.append(MsgNode(kind="obs", text="refine it"))
    s.append(_think([20, 21]))
    s.append(_gen())

    seg = s.build_think_segment()
    assert seg is not None
    # Two think rows of length 3 and 2.
    assert [int(x) for x in seg.lengths.tolist()] == [3, 2]
    cu = [int(c) for c in seg.cu_seqlens.tolist()]
    assert seg.tokens[cu[0] : cu[1]].tolist() == [10, 11, 12]
    assert seg.tokens[cu[1] : cu[2]].tolist() == [20, 21]
    # log_probs default to zeros when not provided, aligned with tokens.
    assert seg.log_probs.numel() == seg.tokens.numel()


def test_build_image_segment_stacks_turns():
    s = Session(prompt="a dog")
    s.append(_think([1]))
    s.append(_gen(k=3, seq=4, channels=2, n_sde=2))
    s.append(MsgNode(kind="obs", text="again"))
    s.append(_think([2]))
    s.append(_gen(k=3, seq=4, channels=2, n_sde=2))

    seg = s.build_image_segment()
    assert seg is not None
    assert seg.latents.shape == (2, 3, 4, 2)  # [T=2, K, seq, C]
    assert seg.sde_logp.shape == (2, 2)  # [T, S]
    # shared trajectory metadata preserved from turn 0
    assert seg.indices.tolist() == [0, 1, 2]


def test_obs_not_in_segment_only_in_context():
    """obs text never becomes a think row; it only appears in prior-context views."""
    s = Session(prompt="p")
    s.append(_think([1, 2]))  # idx 0
    s.append(_gen())  # idx 1
    s.append(MsgNode(kind="obs", text="feedback-0"))  # idx 2
    s.append(_think([3, 4]))  # idx 3
    s.append(_gen())  # idx 4

    # think segment has exactly the two think rows, no obs tokens.
    seg = s.build_think_segment()
    assert int(seg.lengths.numel()) == 2
    assert seg.tokens.tolist() == [1, 2, 3, 4]

    # turn-1 think (index 3) sees turn-0 think + obs as its prefix context.
    ctx = s.context_nodes_before(3)
    kinds = [n.kind for n in ctx]
    assert kinds == ["think", "obs"]  # gen (idx 1) is skipped — images aren't AR text
    assert ctx[1].text == "feedback-0"


def test_turn_views_and_counts():
    s = Session(prompt="p")
    for _ in range(3):
        s.append(_think([1]))
        s.append(_gen())
    assert s.num_turns() == 3
    assert len(s.think_turns()) == 3
    assert len(s.gen_turns()) == 3
    assert s.think_texts() == ["plan 1", "plan 1", "plan 1"]
    assert s.images() == ["img", "img", "img"]


def test_empty_session_builds_none():
    s = Session(prompt="p")
    assert s.build_think_segment() is None
    assert s.build_image_segment() is None
    assert s.num_turns() == 0
