"""AgenticWorkflow — the pluggable multi-turn turn loop.

A workflow drives ONE sample's session to termination over an
:class:`AgenticBackend` (which actually runs think / gen) and an
:class:`~unirl.agentic.env.AgenticEnv` (which turns each turn into feedback).
The backend seam keeps the loop model-agnostic: :class:`ThinkGenWorkflow` is
tested on CPU with a fake backend and runs on GPU with ``BagelAgenticEngine``
unchanged.

The loop is deliberately the only place that knows the think→gen→obs ordering;
different models with different turn structures supply a different workflow, not
a different engine.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from unirl.agentic.env import AgenticEnv
from unirl.agentic.session import MsgNode, Session


@runtime_checkable
class AgenticBackend(Protocol):
    """The model seam a workflow drives — produces think / gen nodes.

    Implementations own all model specifics (tokenization, KV contexts, σ
    schedule, decode). They receive the running :class:`Session` so they can
    rebuild the turn's context from the transcript so far.
    """

    def think(self, *, session: Session, turn: int, sampling: Any) -> MsgNode:
        """Generate turn ``turn``'s think node from the session's context."""
        ...

    def gen(self, *, session: Session, think: MsgNode, turn: int, sampling: Any) -> MsgNode:
        """Generate turn ``turn``'s image node, conditioned on the context + think."""
        ...


@runtime_checkable
class AgenticWorkflow(Protocol):
    """Drive one sample's multi-turn agentic loop until termination."""

    def run(
        self,
        *,
        session: Session,
        backend: AgenticBackend,
        env: AgenticEnv,
        max_turns: int,
        sampling: Any,
    ) -> Session: ...


class ThinkGenWorkflow:
    """Bagel-style think→gen→obs loop (fixed ``max_turns`` in P0).

    Per turn: the backend plans (``think``), renders (``gen``), then the env
    scores the turn into an ``obs`` node that becomes the next turn's prefix.
    The ``obs`` node is appended only when it carries feedback AND another turn
    will run, so the final turn never trails a dangling obs. ``done`` from the
    env terminates early only when ``honor_done`` is set (P1); P0 keeps it
    ``False`` so every session runs exactly ``max_turns`` turns and each prompt's
    GRPO group stays uniform.
    """

    def __init__(self, *, honor_done: bool = False) -> None:
        self.honor_done = bool(honor_done)

    def run(
        self,
        *,
        session: Session,
        backend: AgenticBackend,
        env: AgenticEnv,
        max_turns: int,
        sampling: Any,
    ) -> Session:
        if max_turns < 1:
            raise ValueError(f"ThinkGenWorkflow.run: max_turns must be >= 1, got {max_turns}")
        for turn in range(max_turns):
            think = backend.think(session=session, turn=turn, sampling=sampling)
            session.append(think)
            gen = backend.gen(session=session, think=think, turn=turn, sampling=sampling)
            session.append(gen)

            obs = env.step(image=gen.image, think_text=think.text or "", turn=turn)
            is_last = turn + 1 >= max_turns
            if obs.feedback_text is not None and not is_last:
                session.append(MsgNode(kind="obs", text=obs.feedback_text, weight_version=session.weight_version))
            if self.honor_done and obs.done:
                break
        session.status = "completed"
        return session


__all__ = ["AgenticBackend", "AgenticWorkflow", "ThinkGenWorkflow"]
