"""Sample + Part — the rollout endomorphism types (LIN-446).

See ``docs/rollout-sample-refactor.md``. One recursive type for the rollout
boundary (``step: Sample -> Sample``), replacing the ``RolloutReq → RolloutResp``
pair. A ``Sample`` holds an ordered ``parts: List[Part]`` whose position *is* the
lineage chain — a part's parent is the entry before it (index ``i-1``), so parts
carry no name/key. Within a part, each sample's parent is recovered from its id
path (``sample_ids``; see ``docs/sample-id-design.md`` and
:mod:`unirl.types.sample_id`), not a stored index. A *request* is a ``Sample`` with
only input Part(s); ``fork``
appends a generation shell and the fill step populates it. Conditioning is collected
from the ancestor prefix as primitives (:meth:`Sample.conditioning`), not stored.
Reward/advantage/split machinery is ported from ``rollout_resp.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import torch

from unirl.distributed.tensor.batch import (
    Batch,
    FieldKind,
    concat_field,
    field,
    max_field,
    shared_field,
)
from unirl.distributed.tensor.ref import hydrate
from unirl.types.conditions import Condition
from unirl.types.media_preview import MediaPreview
from unirl.types.primitives import Audios, Images, Texts, Videos
from unirl.types.sample_id import child_id, parent_id
from unirl.types.sampling import BaseSamplingParams
from unirl.types.segments import Segment
from unirl.utils.shard_balance import lpt_shard_permutation, shard_token_spread

logger = logging.getLogger(__name__)

# A Part's content in raw/primitive form (text / image / …) — the counterpart of
# the encoded ``segment``: given content on an input Part, decoded output on a
# generation Part. One value (a Part is single-modality).
Primitive = Union[Texts, Images, Videos, Audios]


@dataclass
class Part(Batch):
    """One rollout slice — a single modality.

    A node in a :class:`Sample`'s positional chain: its parent is the preceding
    entry in ``Sample.parts`` (index ``i-1``). Each sample's parent *within* that
    preceding part is recovered from its id path (``parent_id(sample_ids[i])``,
    located by id; see :mod:`unirl.types.sample_id`) — there is no stored parent
    index; :attr:`is_root` marks the chain head. ``segment`` holds the encoded
    payload, ``primitive`` the same content in raw form. Conditioning is collected
    from the prefix (:meth:`Sample.conditioning`), not stored.
    """

    sample_ids: List[str] = concat_field(default_factory=list)

    segment: Optional[Segment] = field(kind=FieldKind.CONCAT, default=None)
    primitive: Optional[Primitive] = field(kind=FieldKind.CONCAT, default=None)
    # Encoded conditioning produced for this part, kept for trainer-side replay —
    # the carrier for what the old ``RolloutTrack.conditions`` held. Per-sample
    # (CONCAT); defaults to ``{}`` so an unpopulated part is an empty dict, not None.
    conditions: Dict[str, Condition] = field(kind=FieldKind.CONCAT, default_factory=dict)
    media_preview: Optional[MediaPreview] = concat_field(default=None)

    rewards: Optional[torch.Tensor] = concat_field(default=None)
    component_rewards: Optional[Dict[str, torch.Tensor]] = concat_field(default=None)
    advantages: Optional[torch.Tensor] = concat_field(default=None)
    # Per-sample grouping labels the GRPO advantages were computed under (recorded
    # by :meth:`compute_advantages`). May be COARSER than ``group_ids`` (e.g. a
    # root-prompt scope passes ``Sample.root_group_ids``), so metrics that bucket
    # by the advantage baseline must read this, not the immediate-parent ``group_ids``.
    advantage_group_ids: Optional[List[str]] = concat_field(default=None)
    status: Optional[torch.Tensor] = concat_field(default=None)

    metadata: List[Dict[str, Any]] = concat_field(default_factory=list)
    # Request-side routing / override metadata (task / bot_task / chat / ar);
    # renamed from the old ``RolloutReq.stage_config``. Shared across a part's samples.
    control: Dict[str, Any] = shared_field(default_factory=dict)
    # The sampling params this part was generated under (provenance; set at fork).
    sampling_params: Optional[BaseSamplingParams] = shared_field(default=None)

    @classmethod
    def input(
        cls,
        sample_ids: List[str],
        segment: Optional[Segment] = None,
        *,
        primitive: Optional[Primitive] = None,
        control: Optional[Dict[str, Any]] = None,
        metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> "Part":
        """Build a turn-0 input Part (the prompt, a root; untrained).

        ``segment`` is the encoded prompt; ``primitive`` optionally carries the same
        content in raw form (what :meth:`Sample.conditioning` surfaces). Always a
        root — its ids carry no lineage segment, so they must not contain the ``/``
        path delimiter (this is the boundary where driver-supplied ids enter).
        """
        bad = [s for s in sample_ids if "/" in s]
        if bad:
            raise ValueError(
                f"Part.input: root sample_ids must not contain '/' (the lineage delimiter); offending ids: {bad[:3]!r}"
            )
        return cls(
            sample_ids=list(sample_ids),
            segment=segment,
            primitive=primitive,
            control=dict(control) if control else {},
            metadata=list(metadata) if metadata else [],
        )

    def input_child(self, primitive: Primitive) -> "Part":
        """A branch-1 *input* child carrying an extra conditioning modality.

        Multi-input multimodal (e.g. image+text → image): a Part is
        single-modality, so a second input rides as a chained input Part — one
        child per parent sample (ids extended by ``/0``), ``sampling_params``
        left None (it generates nothing). Chaining keeps only the head a root,
        so the request stays a valid :class:`Sample` and
        :meth:`Sample.conditioning` surfaces every input primitive in turn order
        (root → …). See ``docs/rollout-sample-refactor.md`` §3.
        """
        if not self.sample_ids:
            raise ValueError("Part.input_child: parent has no sample_ids")
        return Part(
            sample_ids=[child_id(pid, 0) for pid in self.sample_ids],
            primitive=primitive,
        )

    @property
    def batch_size(self) -> int:
        if self.sample_ids:
            return len(self.sample_ids)
        return super().batch_size

    @property
    def is_root(self) -> bool:
        """Whether this is a chain head (input/root) — its sample ids carry no
        lineage segment. An empty part is not a root (no samples to root)."""
        return bool(self.sample_ids) and not any("/" in sid for sid in self.sample_ids)

    @property
    def group_ids(self) -> List[str]:
        """Grouping labels: each sample's parent id (siblings = one group), or its
        own id for a root (each sample its own group)."""
        return [p if (p := parent_id(sid)) is not None else sid for sid in self.sample_ids]

    def split(self) -> List["Part"]:
        """Split into one Part per :attr:`group_ids` equivalence class."""
        gids = self.group_ids
        if not gids:
            return [self]
        groups: Dict[str, List[int]] = {}
        for i, gid in enumerate(gids):
            groups.setdefault(gid, []).append(i)
        results: List["Part"] = []
        for gid in dict.fromkeys(gids):
            indices = torch.tensor(groups[gid], dtype=torch.long)
            results.append(self.select(indices))
        return results

    def balance_shards(self, num_shards: int, *, min_spread: float = 0.05) -> "Part":
        """Reorder samples so ``num_shards`` equal contiguous shards carry ~equal
        tokens (verl ``balance_batch`` parity, greedy LPT). No-op when balancing
        can't apply or shards are already within ``min_spread``."""
        if self.segment is None or self.segment.lengths is None or num_shards <= 1:
            return self
        total = self.batch_size
        if total % num_shards != 0:
            return self
        lengths = [int(x) for x in self.segment.lengths.tolist()]
        if len(lengths) != total:
            logger.warning("balance_shards: lengths (%d) != batch_size (%d); skipping.", len(lengths), total)
            return self

        before = shard_token_spread(lengths, num_shards)
        if before < min_spread:
            return self

        perm = lpt_shard_permutation(lengths, num_shards)
        after = shard_token_spread([lengths[i] for i in perm], num_shards)
        logger.info("balance_shards: token spread %.1f%% -> %.1f%%", 100 * before, 100 * after)
        return self.select(perm)

    def fork(
        self,
        branch: int,
        *,
        sampling_params: Optional[BaseSamplingParams] = None,
        new_segment: Optional[Segment] = None,
    ) -> "Part":
        """Create a child generation shell — ``N`` self-samples → ``N*branch``.

        The mechanic behind :meth:`Sample.fork` (which appends the child so its
        parent is positionally ``self``). Child ids extend each parent id with one
        ``/{branch}`` segment, group-by-parent contiguous (siblings adjacent) — so the
        id *is* the lineage and ``parent_id`` recovers the parent. Control rides the
        shell; ``segment`` is left for the fill step. No conditioning assembled
        (collected via :meth:`Sample.conditioning`).
        """
        if not self.sample_ids:
            raise ValueError("Part.fork: parent has no sample_ids")
        if branch < 1:
            raise ValueError(f"Part.fork: branch must be >= 1, got {branch}")

        child_sample_ids = [child_id(pid, j) for pid in self.sample_ids for j in range(branch)]

        return Part(
            sample_ids=child_sample_ids,
            segment=new_segment,
            sampling_params=sampling_params,
        )

    def fill(
        self,
        *,
        segment: Optional[Segment] = None,
        primitive: Optional[Primitive] = None,
        conditions: Optional[Dict[str, Condition]] = None,
        media_preview: Optional[MediaPreview] = None,
        status: Optional[torch.Tensor] = None,
    ) -> "Part":
        """Return a copy of this gen-shell part with generation outputs written.

        The producer-side counterpart of :meth:`fork`: ``fork`` makes the empty
        shell (ids + ``sampling_params``), the engine generates, then ``fill``
        writes the results. Only non-``None`` arguments are written; ids,
        ``sampling_params`` and everything else are preserved.
        """
        kwargs: Dict[str, Any] = {f.name: getattr(self, f.name) for f in dc_fields(self)}
        for name, value in (
            ("segment", segment),
            ("primitive", primitive),
            ("conditions", conditions),
            ("media_preview", media_preview),
            ("status", status),
        ):
            if value is not None:
                kwargs[name] = value
        return type(self)(**kwargs)

    def compute_advantages(
        self,
        normalize: bool = True,
        eps: float = 1e-8,
        scope: str = "group",
        use_global_std: bool = False,
        group_ids: Optional[List[str]] = None,
    ) -> "Part":
        """GRPO per-group advantage ``(reward - group_mean) / (group_std + eps)``.

        Groups are :attr:`group_ids` (each sample's parent id, or per-sample for a
        root) and must be group-by-parent contiguous, so the reduce is one ``view``.
        ``scope="global"`` normalizes over the whole batch; ``use_global_std`` keeps
        per-group means but one batch-wide std; ``group_ids`` overrides the grouping
        (e.g. to group at a coarser lineage level). Population std
        (``unbiased=False``) makes ``branch=1`` degenerate to advantage 0.
        """
        if self.rewards is None:
            raise ValueError("Part.compute_advantages: part has no rewards")
        n = len(self.sample_ids)
        if n == 0:
            return self

        # rewards may arrive as a TensorRef proxy from the reward workers; hydrate.
        rewards_local = hydrate(self.rewards)

        if scope == "global":
            rewards_g = rewards_local.to(torch.float32)
            if normalize:
                adv_g = (rewards_g - rewards_g.mean()) / (rewards_g.std() + eps)
            else:
                adv_g = rewards_g - rewards_g.mean()
            return _part_with_field(self, "advantages", adv_g)

        if group_ids is not None and len(group_ids) != n:
            raise ValueError(
                f"compute_advantages: group_ids length {len(group_ids)} != sample count {n}; "
                f"the override must be one label per sample."
            )
        group_labels = list(group_ids) if group_ids is not None else (None if self.is_root else self.group_ids)

        # Root (no grouping labels): each sample is its own group → advantage 0.
        if group_labels is None:
            return _part_with_field(self, "advantages", torch.zeros_like(rewards_local, dtype=torch.float32))

        unique_pids = list(dict.fromkeys(group_labels))
        n_groups = len(unique_pids)
        if n_groups == 0 or n % n_groups != 0:
            raise ValueError(
                f"compute_advantages: non-uniform group sizes (n={n}, n_groups={n_groups}). "
                f"Expected uniform branching with group-by-parent ordering — use fork to build the Part."
            )
        branch = n // n_groups
        expected = [pid for pid in unique_pids for _ in range(branch)]
        if list(group_labels) != expected:
            raise ValueError(
                "compute_advantages: grouping labels not in group-by-parent contiguous order. "
                "Siblings must be consecutive (use fork), got interleaved ordering."
            )

        rewards = rewards_local.to(torch.float32)
        reshaped = rewards.view(n_groups, branch)
        mean = reshaped.mean(dim=1, keepdim=True)
        if normalize:
            if use_global_std:
                std = rewards.std() + eps
            else:
                std = (reshaped.var(dim=1, unbiased=False, keepdim=True) + eps).sqrt()
            adv = (reshaped - mean) / std
        else:
            adv = reshaped - mean
        # Record the grouping the advantages were computed under so reward/zero-std
        # metrics bucket by the actual GRPO baseline (which may be coarser than
        # ``group_ids`` when a ``group_ids`` override is passed, e.g. root scope).
        result = _part_with_field(self, "advantages", adv.flatten())
        return _part_with_field(result, "advantage_group_ids", list(group_labels))


def _part_with_field(part: Part, field_name: str, value: Any) -> Part:
    """Copy of ``part`` with one field replaced."""
    kwargs: Dict[str, Any] = {f.name: getattr(part, f.name) for f in dc_fields(part)}
    kwargs[field_name] = value
    return type(part)(**kwargs)


@dataclass
class Sample(Batch):
    """Rollout container — an ordered ``parts: List[Part]`` (the merged
    ``RolloutReq`` + ``RolloutResp``). Position is lineage (parent = the preceding
    part); per-Part invariants (the chain foreign-keys) are checked in
    :meth:`__post_init__`."""

    parts: List[Part] = field(kind=FieldKind.CONCAT, default_factory=list)
    reward_compute_s: float = max_field(default=0.0)

    def __post_init__(self) -> None:
        for i, p in enumerate(self.parts):
            if len(p.sample_ids) == 0:
                continue  # empty part: no lineage to validate
            if p.is_root:
                # Only the head (index 0) may be a root — position is lineage.
                if i != 0:
                    raise ValueError(
                        f"Sample.parts[{i}] is a root (ids carry no lineage segment) but is not the head; "
                        f"only the first part may be a root."
                    )
                continue
            if i == 0:
                raise ValueError("Sample.parts[0] is non-root; the head has no parent.")
            prev_ids = set(self.parts[i - 1].sample_ids)
            bad = [sid for sid in p.sample_ids if parent_id(sid) not in prev_ids]
            if bad:
                raise ValueError(
                    f"Sample.parts[{i}]: {len(bad)} ids whose parent id is not in the preceding part "
                    f"(index {i - 1}); first bad: {bad[:3]!r}"
                )

    @classmethod
    def concat(cls, items: Sequence["Sample"]) -> "Sample":
        """Concat Samples (e.g. DP gather): concat each part position-wise across
        shards. All shards carry the same parts in the same order (shards of one
        Sample). ``reward_compute_s`` reduces by max."""
        if not items:
            raise ValueError("Sample.concat: cannot concat an empty sequence")
        if len(items) == 1:
            return items[0]
        n_parts = len(items[0].parts)
        merged = [Part.concat([it.parts[i] for it in items]) for i in range(n_parts)]
        return cls(parts=merged, reward_compute_s=max(it.reward_compute_s for it in items))

    @classmethod
    def request(cls, *input_parts: Part) -> "Sample":
        """A *request* — a ``Sample`` of only input Part(s), e.g.
        ``Sample.request(Part.input(ids, seg))``.

        Multi-input multimodal chains the extra inputs off the head via
        :meth:`Part.input_child` so only the head is a root, e.g.::

            text = Part.input(ids, primitive=Texts(...))
            Sample.request(text, text.input_child(Images(...)))  # image+text

        :meth:`Sample.conditioning` then surfaces both primitives (text, image)
        in turn order for the gen step. See ``docs/rollout-sample-refactor.md`` §3.
        """
        return cls(parts=list(input_parts))

    @property
    def batch_size(self) -> int:
        """Size of the root Part (one prompt + its fan-out = "one sample"); max
        across parts when the root isn't unique."""
        if not self.parts:
            return 0
        roots = [p for p in self.parts if p.is_root]
        if len(roots) == 1:
            return roots[0].batch_size
        return max(p.batch_size for p in self.parts)

    def split(self) -> List["Sample"]:
        """Split into one ``Sample`` per root-group, tree-complete: each shard holds
        one prompt's whole subtree across all parts. Requires a unique root."""
        if not self.parts:
            return [self]

        root_gids = self.parts[0].group_ids
        if not root_gids:
            return [self]

        per_part_root_groups = self._root_groups_per_part()
        # One pass per part: bucket sample indices by root-group label (was an
        # O(num_groups x batch) rescan of each part's labels for every group).
        per_part_buckets: List[Dict[str, List[int]]] = []
        for labels in per_part_root_groups:
            buckets: Dict[str, List[int]] = {}
            for k, rg in enumerate(labels):
                buckets.setdefault(rg, []).append(k)
            per_part_buckets.append(buckets)

        results: List["Sample"] = []
        for rgid in dict.fromkeys(root_gids):
            shard_parts: List[Part] = []
            for i, part in enumerate(self.parts):
                indices = per_part_buckets[i].get(rgid)
                if not indices:
                    raise RuntimeError(
                        f"Sample.split: part {i} has no samples in root group {rgid!r}; lineage tree is malformed."
                    )
                shard_parts.append(part.select(torch.tensor(indices, dtype=torch.long)))
            results.append(type(self)(parts=shard_parts))
        return results

    def _root_groups_per_part(self) -> List[List[str]]:
        """Root-prompt group label for every sample of every part.

        Propagated forward along the chain (parent = preceding part): the head is
        its own group; each child inherits its parent's label, located by parent id
        (position-independent). Assumes a unique root head — the chain invariant
        :meth:`__post_init__` enforces."""
        per_part: List[List[str]] = []
        for i, part in enumerate(self.parts):
            if part.is_root:
                per_part.append(list(part.group_ids))
            else:
                prev_part = self.parts[i - 1]
                sid_to_grp = dict(zip(prev_part.sample_ids, per_part[-1]))
                per_part.append([sid_to_grp[parent_id(sid)] for sid in part.sample_ids])
        return per_part

    def root_group_ids(self, part_index: int) -> List[str]:
        """Root-prompt group label per sample of ``parts[part_index]`` — its lineage
        climbed to the root part.

        Groups a descendant Part by the prompt it descends from (coarser than its
        immediate parent) for GRPO — the replacement for the old
        ``RolloutResp.compute_track_advantages(group_key="root")``. The labels stay
        group-by-parent contiguous (the lineage keeps a prompt's samples
        consecutive), as :meth:`Part.compute_advantages` ``group_ids`` requires."""
        if not self.parts:
            return []
        return self._root_groups_per_part()[part_index]

    def gen_parts(self) -> List[Part]:
        """The generated (non-input) Parts — those carrying ``sampling_params``.
        Input Parts (the prompt head and any :meth:`Part.input_child`) have
        ``sampling_params is None`` and are skipped."""
        return [p for p in self.parts if p.sampling_params is not None]

    def gen_part(self, params_type: type) -> Part:
        """The first gen Part whose ``sampling_params`` is an instance of
        ``params_type`` (e.g. ``ARSamplingParams`` / ``DiffusionSamplingParams``)
        — locating a stage by TYPE, not position (the migration's convention;
        mirrors the engine-side ``ar_gen_part`` / ``diffusion_gen_part`` readers)."""
        for p in self.parts:
            if isinstance(p.sampling_params, params_type):
                return p
        raise ValueError(f"Sample.gen_part: no Part with sampling_params of type {params_type.__name__}")

    def gen_part_index(self, params_type: type) -> int:
        """Index of :meth:`gen_part` — for write-back (e.g. replacing a Part with
        its advantage-filled version via :meth:`with_parts`)."""
        for i, p in enumerate(self.parts):
            if isinstance(p.sampling_params, params_type):
                return i
        raise ValueError(f"Sample.gen_part_index: no Part with sampling_params of type {params_type.__name__}")

    def with_parts(self, parts: List[Part]) -> "Sample":
        """A copy carrying replacement ``parts`` but the same ``reward_compute_s``
        — the idiom for swapping in advantage-filled Parts without dropping the
        accumulated reward-compute time."""
        return type(self)(parts=list(parts), reward_compute_s=self.reward_compute_s)

    def slice(self, start: int, end: int) -> "Sample":
        """Shard ``[start, end)`` along the batch dim (the P root prompts) by whole
        prompt-TREE, not by the ``parts`` list.

        A ``Sample``'s batch dim is its P root prompts, but its only CONCAT field
        is ``parts`` (length = #stages, 2-3) — so the inherited ``Batch.slice``
        would wrongly slice the parts list and hand every shard the full Sample.
        Route through :meth:`split`/:meth:`concat` so each shard holds whole prompt
        subtrees across all parts. This is the hook ``@distributed(DP_SCATTER)``
        (via ``Batch.chunk`` -> ``slice``) and trainside micro-batching rely on."""
        picked = self.split()[start:end]
        parts = Sample.concat(picked).parts if picked else []
        return type(self)(parts=parts, reward_compute_s=self.reward_compute_s)

    def select(self, indices: "torch.Tensor") -> "Sample":
        """Gather whole root prompt-trees by index (shuffle / subsample), mirroring
        :meth:`slice`'s tree-sharding rather than the inherited parts-list select."""
        groups = self.split()
        idx = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        picked = [groups[int(i)] for i in idx]
        parts = Sample.concat(picked).parts if picked else []
        return type(self)(parts=parts, reward_compute_s=self.reward_compute_s)

    def fork(
        self,
        branch: int,
        *,
        sampling_params: Optional[BaseSamplingParams] = None,
        new_segment: Optional[Segment] = None,
    ) -> "Sample":
        """Append a generation shell forked from the frontier (the last part) — the
        sole fan-out edge (§5), the "fork" half of ``step: fork → fill``."""
        if not self.parts:
            raise ValueError("Sample.fork: no parts to fork from (empty Sample)")
        child = self.parts[-1].fork(
            branch,
            sampling_params=sampling_params,
            new_segment=new_segment,
        )
        return type(self)(parts=[*self.parts, child], reward_compute_s=self.reward_compute_s)

    def propagate_rewards(self, op: Literal["mean", "max", "sum"] = "mean") -> "Sample":
        """Aggregate child rewards up the chain (leaf → root) into unscored parents.
        Walks ``parts`` in reverse; per parent, reduces the successor's rewards
        ``view(n_parent, branch).reduce(dim=1)``. Direct rewards win. Single-child
        only (the chain guarantees it; §7)."""
        new_parts = list(self.parts)
        for i in range(len(new_parts) - 1, -1, -1):
            part = self.parts[i]
            if part.rewards is not None:
                continue
            if i + 1 >= len(new_parts):
                continue
            child = new_parts[i + 1]
            if child.is_root:  # successor isn't a child of this part
                continue
            if child.rewards is None:
                raise ValueError(
                    f"propagate_rewards: cannot aggregate from part {i + 1} to {i} — child.rewards is None. "
                    f"Score the leaf parts first."
                )
            n_parent = len(part.sample_ids)
            n_child = len(child.sample_ids)
            if n_parent == 0 or n_child % n_parent != 0:
                raise ValueError(
                    f"propagate_rewards: non-uniform branching from part {i + 1} ({n_child} samples) "
                    f"to {i} ({n_parent} samples). Group-by-parent ordering requires n_child % n_parent == 0."
                )
            branch = n_child // n_parent
            reshaped = child.rewards.view(n_parent, branch)
            if op == "mean":
                aggregated = reshaped.mean(dim=1)
            elif op == "max":
                aggregated = reshaped.amax(dim=1)
            elif op == "sum":
                aggregated = reshaped.sum(dim=1)
            else:
                raise ValueError(f"propagate_rewards: unknown op {op!r}; expected 'mean', 'max', or 'sum'.")
            new_parts[i] = _part_with_field(part, "rewards", aggregated)

        return type(self)(parts=new_parts, reward_compute_s=self.reward_compute_s)

    def conditioning(self) -> List[Primitive]:
        """Conditioning inputs for generating the frontier (last) part: each
        ancestor's ``primitive`` (raw text/image), row-aligned to the frontier's
        samples, in chronological order (root → frontier-parent). The model encodes
        these into its own conditions, mapping position → turn via the fixed
        schedule. Undecoded ancestors (``primitive is None``) are skipped; to get a
        non-frontier part's conditioning (replay), call this on ``parts[:i+1]``.

        Ancestors are walked by id: ``active_ids`` are the ancestor ids aligned to
        the frontier's samples, climbed one level per step via ``parent_id``; each
        level's rows are looked up by id (position-independent)."""
        if not self.parts:
            return []
        frontier = self.parts[-1]
        if frontier.is_root:
            return []
        out: List[Primitive] = []
        active_ids = [parent_id(sid) for sid in frontier.sample_ids]
        anc = len(self.parts) - 2
        while anc >= 0:
            part = self.parts[anc]
            sid_to_row = {sid: r for r, sid in enumerate(part.sample_ids)}
            try:
                rows = [sid_to_row[aid] for aid in active_ids]
            except KeyError as e:
                raise ValueError(
                    f"Sample.conditioning: ancestor id {e.args[0]!r} not found in part {anc}; "
                    f"lineage chain is malformed."
                ) from None
            if part.primitive is not None:
                out.append(part.primitive.select(torch.tensor(rows, dtype=torch.long)))
            if part.is_root:
                break
            active_ids = [parent_id(aid) for aid in active_ids]
            anc -= 1
        out.reverse()  # chronological: root → frontier-parent
        return out


__all__ = [
    "Sample",
    "Part",
    "Primitive",
]
