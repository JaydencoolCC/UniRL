"""FullyAsyncPool — the fully-async (v5) carry-over session pool.

Where the half-async :class:`~unirl.agentic.engine.BagelAgenticEngine` runs every
session of a request to completion *inside one* ``generate()`` (so every turn is
on-policy and a session never outlives the call), the fully-async resident pool
keeps sessions **resident across ``generate()`` calls**. A session that has not
finished its ``max_turns`` when the harvest target is met *carries over*: its
remaining turns run in a later ``generate()`` window, under a *newer* policy
version (the shared live model changed at the optimizer step in between). That is
what makes the layer "fully async" within UniRL's blocking ``generate()``
contract — bounded off-policy carry-over instead of gate-and-restart.

Design (faithful to ``bagel_agentic_design_v5.html`` §Pillar 1/2, P0 conservative
coordinator — synchronous, no asyncio: on colocate the GPU cannot overlap a train
step, so a single-threaded coordinator is correct and the channel-parallel
asyncio variant is a disaggregated P1 upgrade):

- **Carry-over** — running sessions persist in ``self._running`` between harvests.
- **Per-turn version stamping** — each turn stamps the think/gen nodes with the
  *current* policy version (``version_ref()``), so a carried-over session's nodes
  span versions; :meth:`Session.version_spread` reads them.
- **Bounded staleness + abort** — before advancing a session, if its version
  spread (folding in the current version) would exceed ``max_staleness`` it is
  aborted and its group re-admitted fresh (replenish), bounding off-policyness.
- **Group-keyed harvest** — a group (one prompt's ``N`` sessions) is harvested
  only when all ``N`` complete; harvest returns whole groups so GRPO group
  structure stays uniform. Carry-over groups keep their real ``group_id`` /
  ``prompt`` (the trainer maps ``group_id → prompt`` for reward), so no
  sample-id reassignment hack is needed.

The pool is **backend-agnostic** (drives an :class:`~unirl.agentic.workflow.AgenticBackend`
+ :class:`~unirl.agentic.env.AgenticEnv`), so it is CPU-testable with a fake
backend exactly like ``ThinkGenWorkflow``.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

from unirl.agentic.env import AgenticEnv
from unirl.agentic.session import MsgNode, Session
from unirl.agentic.version import should_admit

logger = logging.getLogger(__name__)


class FullyAsyncPool:
    """Resident, version-aware, carry-over session pool (synchronous P0 coordinator)."""

    def __init__(
        self,
        *,
        backend,
        env: AgenticEnv,
        version_ref: Callable[[], int],
        max_turns: int,
        sessions_per_prompt: int,
        max_staleness: int = 0,
        turns_per_window: int = 1,
        honor_done: bool = False,
        max_advance_passes: int = 10_000,
    ) -> None:
        """:param version_ref: callable returning the engine's current policy version.
        :param turns_per_window: max turns each session advances per harvest pass.
            ``1`` (default) maximizes carry-over (a ``T``-turn session needs ``T``
            windows, so spans ``T`` versions if the version bumps between windows);
            ``>= max_turns`` collapses to half-async (sessions finish in-window).
        :param max_staleness: max allowed trainable version spread before a session
            is aborted + its group replenished (``0`` = strictly on-policy).
        :param max_advance_passes: safety cap on advance passes per harvest (guards
            against a misconfiguration that can never reach the harvest target).
        """
        self._backend = backend
        self._env = env
        self._version_ref = version_ref
        self._max_turns = int(max_turns)
        self._n = int(sessions_per_prompt)
        self._max_staleness = int(max_staleness)
        self._turns_per_window = max(1, int(turns_per_window))
        self._honor_done = bool(honor_done)
        self._max_advance_passes = int(max_advance_passes)

        # Resident state (persists across harvest calls).
        self._running: List[Session] = []
        self._completed: List[Session] = []
        # Group bookkeeping: real group_id -> prompt (for replenishing aborts and
        # for the trainer's reward prompt lookup). Insertion-ordered for stable,
        # deterministic harvest order (freshness ~ admission order).
        self._group_prompt: "OrderedDict[str, str]" = OrderedDict()

        # Cumulative metrics (exposed via :meth:`drain_metrics`).
        self._stat_admitted = 0
        self._stat_aborted = 0
        self._stat_carried = 0  # sessions that crossed >=1 version boundary at harvest

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def ensure_groups(self, groups: List[Tuple[str, str]]) -> None:
        """Register ``(group_id, prompt)`` pairs and admit their ``N`` sessions if
        the group is new. Idempotent — a carry-over group already resident is left
        untouched (its in-flight sessions keep running)."""
        version = int(self._version_ref())
        for group_id, prompt in groups:
            if group_id in self._group_prompt:
                continue
            self._group_prompt[group_id] = prompt
            for _ in range(self._n):
                self._admit_one(group_id, prompt, version)

    def _admit_one(self, group_id: str, prompt: str, version: int) -> None:
        self._running.append(Session(prompt=prompt, group_id=group_id, weight_version=version, status="running"))
        self._stat_admitted += 1

    def _live_count(self, group_id: str) -> int:
        """running + completed sessions of a group (excludes aborted, which are dropped)."""
        r = sum(1 for s in self._running if s.group_id == group_id)
        c = sum(1 for s in self._completed if s.group_id == group_id)
        return r + c

    def _replenish(self) -> None:
        """Re-admit fresh sessions for any non-harvested group short of ``N`` live
        sessions (because some were aborted), subject to the staleness capacity."""
        version = int(self._version_ref())
        max_spread = self._pool_max_spread(version)
        for group_id, prompt in self._group_prompt.items():
            deficit = self._n - self._live_count(group_id)
            for _ in range(max(0, deficit)):
                if not should_admit(
                    accepted_count=0,
                    running_count=len(self._running),
                    max_staleness=self._max_staleness,
                    max_spread_in_pool=max_spread,
                    batch_size=self._n,
                ):
                    # Backpressure: capacity exhausted; replenish later.
                    return
                self._admit_one(group_id, prompt, version)

    def _pool_max_spread(self, current_version: int) -> int:
        spreads = [s.version_spread(current_version) for s in self._running]
        return max(spreads) if spreads else 0

    # ------------------------------------------------------------------
    # Advancement (one turn per running session)
    # ------------------------------------------------------------------

    def _advance_session_one_turn(self, session: Session, version: int) -> None:
        """Run one think→gen→obs turn, stamping nodes with ``version``.

        Mirrors :meth:`ThinkGenWorkflow.run`'s loop body but for a single turn so
        the pool can interleave / pause sessions. Setting ``session.weight_version``
        first makes the backend stamp this turn's think/gen nodes with ``version``
        (the backend reads ``session.weight_version``)."""
        session.weight_version = int(version)
        turn = session.num_turns()
        think = self._backend.think(session=session, turn=turn, sampling=None)
        session.append(think)
        gen = self._backend.gen(session=session, think=think, turn=turn, sampling=None)
        session.append(gen)
        obs = self._env.step(image=gen.image, think_text=think.text or "", turn=turn)
        is_last = (turn + 1) >= self._max_turns
        if obs.feedback_text is not None and not is_last:
            session.append(MsgNode(kind="obs", text=obs.feedback_text, weight_version=int(version)))
        if (turn + 1) >= self._max_turns or (self._honor_done and obs.done):
            session.status = "completed"

    def _advance_one_pass(self, version: int) -> int:
        """Advance every running session by up to ``turns_per_window`` turns,
        aborting any that would exceed ``max_staleness``. Returns turns executed."""
        executed = 0
        still: List[Session] = []
        for session in self._running:
            # Staleness gate: if continuing would push spread past the bound, abort.
            if session.version_spread(version) > self._max_staleness:
                session.status = "aborted"
                self._stat_aborted += 1
                continue
            for _ in range(self._turns_per_window):
                if session.status != "running":
                    break
                self._advance_session_one_turn(session, version)
                executed += 1
            if session.status == "completed":
                self._completed.append(session)
            elif session.status == "running":
                still.append(session)
            # aborted sessions are dropped
        self._running = still
        return executed

    # ------------------------------------------------------------------
    # Harvest
    # ------------------------------------------------------------------

    def _complete_groups(self) -> List[str]:
        """group_ids that currently have ``N`` completed sessions, in admission order."""
        counts: Dict[str, int] = {}
        for s in self._completed:
            counts[s.group_id] = counts.get(s.group_id, 0) + 1
        return [gid for gid in self._group_prompt if counts.get(gid, 0) >= self._n]

    def harvest(self, target_groups: int) -> List[Session]:
        """Advance the pool until ``target_groups`` groups are complete, then pop and
        return their sessions (``target_groups * N`` sessions, group-contiguous in
        admission order). Carry-over running sessions and surplus completed sessions
        remain resident for the next harvest.

        Returns sessions ordered group-by-group (so the caller can build uniform
        GRPO groups). Raises if it cannot reach the target within the safety cap.
        """
        passes = 0
        while len(self._complete_groups()) < target_groups:
            version = int(self._version_ref())
            self._replenish()
            executed = self._advance_one_pass(version)
            passes += 1
            if executed == 0:
                # No running session could advance (all aborted / none admitted) and
                # we still lack enough complete groups — replenish then retry once.
                self._replenish()
                if self._advance_one_pass(int(self._version_ref())) == 0:
                    break
            if passes > self._max_advance_passes:
                break

        complete = self._complete_groups()
        if len(complete) < target_groups:
            raise RuntimeError(
                f"FullyAsyncPool.harvest: only {len(complete)} complete group(s) "
                f"after {passes} pass(es); need {target_groups}. Check max_turns / "
                f"staleness / admission, or that ensure_groups() was called."
            )

        chosen = complete[:target_groups]
        chosen_set = set(chosen)
        picked: List[Session] = []
        current_version = int(self._version_ref())
        for gid in chosen:
            group_sessions = [s for s in self._completed if s.group_id == gid][: self._n]
            for s in group_sessions:
                if s.version_spread() > 0 or (s.trainable_versions() and min(s.trainable_versions()) < current_version):
                    self._stat_carried += 1
            picked.extend(group_sessions)
        # Drop harvested groups from resident state (completed + any leftover).
        self._completed = [s for s in self._completed if s.group_id not in chosen_set]
        self._running = [s for s in self._running if s.group_id not in chosen_set]
        for gid in chosen:
            self._group_prompt.pop(gid, None)
        return picked

    # ------------------------------------------------------------------
    # Version change hook + metrics
    # ------------------------------------------------------------------

    def on_policy_version_changed(self, new_version: int) -> int:
        """Called after a weight update (optimizer step / sync). Abort any resident
        session whose spread now exceeds ``max_staleness`` and drop it (its group is
        replenished on the next harvest). Returns the number aborted."""
        aborted = 0
        keep: List[Session] = []
        for s in self._running:
            if s.version_spread(int(new_version)) > self._max_staleness:
                s.status = "aborted"
                aborted += 1
            else:
                keep.append(s)
        self._running = keep
        self._stat_aborted += aborted
        return aborted

    def drain_metrics(self) -> Dict[str, float]:
        """Return + reset cumulative pool metrics (admitted / aborted / carried) plus
        the current resident depth and worst version spread."""
        version = int(self._version_ref())
        m = {
            "pool/admitted": float(self._stat_admitted),
            "pool/aborted": float(self._stat_aborted),
            "pool/carried_over": float(self._stat_carried),
            "pool/running": float(len(self._running)),
            "pool/completed_resident": float(len(self._completed)),
            "pool/max_version_spread": float(self._pool_max_spread(version)),
        }
        self._stat_admitted = self._stat_aborted = self._stat_carried = 0
        return m

    def group_prompts(self) -> Dict[str, str]:
        """Current ``group_id -> prompt`` map (for the trainer's reward lookup)."""
        return dict(self._group_prompt)


__all__ = ["FullyAsyncPool"]
