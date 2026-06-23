"""ThinkGenWorkflow integration test with a fake backend (verification plan #3).

Proves the backend-agnostic loop produces a correctly-structured session
(think→gen→obs interleaving, fixed turns, no trailing obs) and that the backend
can rebuild each turn's context from the transcript so far.
"""

from __future__ import annotations

import torch

from unirl.agentic.env import RuleEnv
from unirl.agentic.session import MsgNode, Session
from unirl.agentic.workflow import AgenticBackend, ThinkGenWorkflow
from unirl.types.segments.latent import make_image_segment


class FakeBackend:
    """Deterministic backend: records the prefix it saw each turn."""

    def __init__(self):
        self.think_contexts: list[list[str]] = []
        self.gen_contexts: list[list[str]] = []

    def think(self, *, session: Session, turn: int, sampling) -> MsgNode:
        # The context this think conditions on = prior think/obs text, in order.
        ctx = [n.text or "" for n in session.context_nodes_before(len(session.nodes))]
        self.think_contexts.append(ctx)
        toks = torch.tensor([turn, turn + 1], dtype=torch.long)
        return MsgNode(
            kind="think",
            tokens=toks,
            logprobs=torch.zeros(2),
            text=f"think-{turn}",
            payload={"prompt_splits": ctx},
        )

    def gen(self, *, session: Session, think: MsgNode, turn: int, sampling) -> MsgNode:
        ctx = [n.text or "" for n in session.context_nodes_before(len(session.nodes))]
        self.gen_contexts.append(ctx)
        seg = make_image_segment(
            latents=torch.randn(1, 2, 3, 2),
            sigmas=torch.linspace(1, 0, 2),
            indices=torch.arange(2),
            sde_logp=torch.randn(1, 1),
            sde_indices=torch.arange(1),
        )
        return MsgNode(kind="gen", latent=seg, image=f"img-{turn}", payload={"contexts": ctx})


def test_backend_is_protocol_instance():
    assert isinstance(FakeBackend(), AgenticBackend)


def test_two_turn_session_structure():
    backend = FakeBackend()
    env = RuleEnv(instructions=["refine-0", "refine-1"])
    session = Session(prompt="a red car")
    out = ThinkGenWorkflow().run(session=session, backend=backend, env=env, max_turns=2, sampling=None)

    # Node order: think,gen,obs, think,gen  (no trailing obs after the last turn).
    kinds = [n.kind for n in out.nodes]
    assert kinds == ["think", "gen", "obs", "think", "gen"]
    assert out.status == "completed"
    assert out.num_turns() == 2

    # The obs from turn 0 fed turn 1's think context.
    obs_node = out.nodes[2]
    assert obs_node.text == "refine-0"
    # Turn 0 think saw an empty prefix; turn 1 think saw [think-0, refine-0].
    assert backend.think_contexts[0] == []
    assert backend.think_contexts[1] == ["think-0", "refine-0"]


def test_single_turn_has_no_obs():
    backend = FakeBackend()
    env = RuleEnv(instructions=["refine-0"])
    session = Session(prompt="p")
    out = ThinkGenWorkflow().run(session=session, backend=backend, env=env, max_turns=1, sampling=None)
    assert [n.kind for n in out.nodes] == ["think", "gen"]


def test_segments_assemble_from_workflow_output():
    backend = FakeBackend()
    env = RuleEnv(instructions=["r0", "r1", "r2"])
    session = Session(prompt="p")
    out = ThinkGenWorkflow().run(session=session, backend=backend, env=env, max_turns=3, sampling=None)

    think_seg = out.build_think_segment()
    image_seg = out.build_image_segment()
    assert int(think_seg.lengths.numel()) == 3  # 3 think rows
    assert image_seg.latents.shape[0] == 3  # 3 image rows


def test_honor_done_terminates_early():
    backend = FakeBackend()
    env = RuleEnv(instructions=["x"], max_turns=1)  # env reports done after turn 0
    session = Session(prompt="p")
    out = ThinkGenWorkflow(honor_done=True).run(session=session, backend=backend, env=env, max_turns=5, sampling=None)
    assert out.num_turns() == 1  # stopped early despite max_turns=5
