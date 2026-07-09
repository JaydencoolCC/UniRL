"""Session + MsgNode — one sample's multi-turn agentic trajectory.

A :class:`Session` is a **linear chain** of :class:`MsgNode` (the P0 degenerate
tree: every node has exactly one child). Three node kinds interleave:

- ``think`` — the planner's reasoning/plan tokens for a turn. These tokens ARE
  the AR training segment (``loss_mask`` all-1).
- ``gen`` — the image generated for a turn (a one-row ``LatentSegment`` plus the
  decoded pixels). These latents ARE the diffusion training segment.
- ``obs`` — environment feedback text. **Not in any training segment** (the v4
  "obs not in segment" insight): obs text only extends the *next* turn's AR
  context (a prefix split), so there is no response-mask to manage — the AR
  segment of every turn is pure trained think tokens.

The chain therefore encodes, per turn ``t``:

    [system, prompt, think_0, obs_0, ..., think_{t-1}, obs_{t-1}]  <- AR context (prefix)
    think_t                                                        <- AR segment (trained)
    image_t conditioned on [..., think_t]                          <- diffusion segment (trained)

Each ``think`` / ``gen`` node carries an opaque ``payload`` dict the backend
fills with whatever its replay needs (Bagel stores ``prompt_splits`` for the AR
stage and the prefilled KV ``contexts`` for the diffusion stage). The core keeps
``payload`` opaque so :mod:`unirl.agentic.session` stays backend-agnostic and
CPU-testable; the engine reads it back when assembling the typed conditions.

Per-turn flattening (not per-session packing): each turn's think is a distinct
policy action with its own context, so it becomes one AR training row; each
turn's image becomes one diffusion training row. :meth:`Session.think_turns` /
:meth:`Session.gen_turns` expose those rows; the engine concatenates them across
sessions into the two tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import torch

from unirl.types.segments.latent import LatentSegment
from unirl.types.segments.text import TextSegment

NodeKind = Literal["think", "gen", "obs"]


@dataclass
class MsgNode:
    """One message in a session's linear chain.

    Fields are populated by kind (the others stay ``None``):

    - ``think``: ``tokens``, ``logprobs``, ``text``, ``payload`` (AR replay data).
    - ``gen``:   ``latent`` (one-row ``LatentSegment``), ``image``, ``payload``
      (diffusion replay data).
    - ``obs``:   ``text`` (feedback). Never enters a training segment.
    """

    kind: NodeKind
    # think nodes
    tokens: Optional[torch.Tensor] = None  # [n_tok] generated token ids
    logprobs: Optional[torch.Tensor] = None  # [n_tok] behavior log-probs
    # gen nodes
    latent: Optional[LatentSegment] = None  # one-row [1, K, ...] trajectory
    image: Any = None  # one decoded image (for reward / preview)
    # think + obs nodes
    text: Optional[str] = None  # think text, or obs feedback text
    # backend replay payload (opaque to the core): e.g. {"prompt_splits": ...}
    # for think nodes, {"contexts": ...} for gen nodes.
    payload: Dict[str, Any] = field(default_factory=dict)
    # the policy version this node was generated under (staleness accounting).
    weight_version: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("think", "gen", "obs"):
            raise ValueError(f"MsgNode.kind must be 'think'/'gen'/'obs'; got {self.kind!r}")
        if self.kind == "think" and self.tokens is None:
            raise ValueError("MsgNode(kind='think') requires tokens.")
        if self.kind == "gen" and self.latent is None:
            raise ValueError("MsgNode(kind='gen') requires a latent segment.")
        if self.kind == "obs" and self.text is None:
            raise ValueError("MsgNode(kind='obs') requires feedback text.")


@dataclass
class Session:
    """One sample's multi-turn trajectory (a linear chain of nodes).

    ``prompt`` is the original user request; ``status`` mirrors the rollout
    lifecycle. Nodes are appended in generation order
    (think, gen, [obs], think, gen, [obs], ...).
    """

    prompt: str
    nodes: List[MsgNode] = field(default_factory=list)
    status: Literal["running", "completed", "aborted"] = "running"
    weight_version: int = 0
    # Prompt-group identity, used by the fully-async resident pool (v5) to harvest
    # carry-over sessions by group (GRPO groups by ``group_id``, not ``sample_id``,
    # so a carried-over session may be re-assigned a new sample_id at harvest as
    # long as its group matches the output window). ``None`` for the half-async
    # path, which binds one session to one pre-expanded request sample.
    group_id: Optional[str] = None

    def append(self, node: MsgNode) -> "Session":
        self.nodes.append(node)
        return self

    # ---- version / staleness (fully-async resident pool) ---------------------

    def trainable_versions(self) -> List[int]:
        """Policy versions of this session's trainable (think + gen) nodes."""
        return [int(n.weight_version) for n in self.nodes if n.kind in ("think", "gen")]

    def version_spread(self, current_version: Optional[int] = None) -> int:
        """``max - min`` policy version over trainable nodes (folding in
        ``current_version`` when given, to bound a still-running session)."""
        vs = self.trainable_versions()
        if current_version is not None:
            vs = vs + [int(current_version)]
        return (max(vs) - min(vs)) if vs else 0

    def think_versions(self) -> List[List[int]]:
        """Per-turn think token versions (one list per think node, broadcast to
        the node's token count) — packs alongside ``TextSegment.token_versions``."""
        out: List[List[int]] = []
        for n in self.think_turns():
            ntok = int(n.tokens.numel()) if n.tokens is not None else 0
            out.append([int(n.weight_version)] * ntok)
        return out

    # ---- structural views --------------------------------------------------

    def think_turns(self) -> List[MsgNode]:
        """The think node of each turn, in turn order (one AR training row each)."""
        return [n for n in self.nodes if n.kind == "think"]

    def gen_turns(self) -> List[MsgNode]:
        """The gen node of each turn, in turn order (one diffusion training row each)."""
        return [n for n in self.nodes if n.kind == "gen"]

    def num_turns(self) -> int:
        """Number of completed think->gen turns (gen nodes drive the count)."""
        return len(self.gen_turns())

    def context_nodes_before(self, index: int) -> List[MsgNode]:
        """The think + obs nodes that precede ``self.nodes[index]`` (its AR prefix).

        Used by a backend to rebuild turn ``t``'s AR context: every prior
        ``think`` and ``obs`` text, in order. ``gen`` nodes are skipped — images
        are not part of the AR text prefix.
        """
        return [n for n in self.nodes[:index] if n.kind in ("think", "obs")]

    # ---- training-segment assembly (generic, CPU-testable) -----------------

    def build_think_segment(self) -> Optional[TextSegment]:
        """Pack every turn's think tokens into one varlen ``TextSegment``.

        One packed row per turn (``think_turns`` order). ``loss_mask`` is left
        unset (all-1 by convention) because every token in a think segment is a
        trained think token — obs tokens never enter the segment. Returns
        ``None`` for a session with no think nodes.
        """
        thinks = self.think_turns()
        if not thinks:
            return None
        tokens = [n.tokens for n in thinks]
        log_probs = [
            n.logprobs if n.logprobs is not None else torch.zeros(n.tokens.numel(), dtype=torch.float32) for n in thinks
        ]
        return TextSegment.pack(tokens=tokens, log_probs=log_probs)

    def build_image_segment(self) -> Optional[LatentSegment]:
        """Concatenate every turn's one-row ``LatentSegment`` into one ``[T, ...]``.

        The shared trajectory metadata (``sigmas`` / ``indices`` / ``sde_indices``)
        is taken from the first gen node — every turn runs the same σ schedule by
        construction — and the per-sample ``latents`` / ``sde_logp`` stack along
        the batch axis. Returns ``None`` for a session with no gen nodes.
        """
        gens = self.gen_turns()
        if not gens:
            return None
        segs = [n.latent for n in gens]
        if len(segs) == 1:
            return segs[0]
        latents = torch.cat([s.latents for s in segs], dim=0)
        sde_logp = torch.cat([s.sde_logp for s in segs], dim=0) if segs[0].sde_logp is not None else None
        return LatentSegment(
            modality=segs[0].modality,
            latents=latents,
            sigmas=segs[0].sigmas,
            indices=segs[0].indices,
            sde_logp=sde_logp,
            sde_indices=segs[0].sde_indices,
        )

    def think_texts(self) -> List[str]:
        return [n.text or "" for n in self.think_turns()]

    def images(self) -> List[Any]:
        """The decoded image of each turn (for reward / preview), in turn order."""
        return [n.image for n in self.gen_turns()]


__all__ = ["MsgNode", "NodeKind", "Session"]
