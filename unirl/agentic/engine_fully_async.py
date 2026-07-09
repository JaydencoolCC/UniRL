"""BagelAgenticEngineFullyAsync — fully-async (v5) fully-async resident pool engine.

Subclasses the half-async :class:`~unirl.agentic.engine.BagelAgenticEngine`,
reusing *all* of its Bagel specifics (AR/diffusion context building,
``_BagelTurnBackend``, detokenize) and adding only the v5 fully-async layer:

- a resident :class:`~unirl.agentic.pool.FullyAsyncPool` that survives
  across ``generate()`` calls (carry-over),
- a policy-version counter bumped by :meth:`bump_version` after each optimizer
  step / weight sync (so carried-over sessions span versions),
- ``generate()`` = *harvest* ``B`` complete groups from the pool (admitting the
  request's groups, advancing + aborting stale sessions), and
- behavior-logprob anchor preserved + per-token / per-SDE-step **version
  metadata** packed into the standard segments (``TextSegment.token_versions`` /
  ``LatentSegment.sde_versions``) — metadata only; GRPO/FlowGRPO are unchanged.

Request contract is identical to the half-async engine (``B*N`` pre-expanded
session rows, group-by-parent), so DP_SCATTER divisibility and the downstream
2-track lineage are unchanged. The difference is purely that completed groups may
be *carried over* from an earlier ``generate()`` (and thus span versions) rather
than always finishing in the call that admitted them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from unirl.agentic.config import IMAGE_TRACK, THINK_TRACK
from unirl.agentic.engine import BagelAgenticEngine, _BagelTurnBackend
from unirl.agentic.pool import FullyAsyncPool
from unirl.agentic.session import Session
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.sde.runtime import ensure_req_sigmas
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.segments.latent import LatentSegment
from unirl.types.segments.text import TextSegment

logger = logging.getLogger(__name__)


class BagelAgenticEngineFullyAsync(BagelAgenticEngine):
    """fully-async resident pool engine for fully-async Bagel agentic RL."""

    _component_name = "bagel_agentic_fully_async"

    def __init__(
        self,
        *,
        pipeline: Any,
        max_turns: int = 2,
        sessions_per_prompt: int = 2,
        max_staleness: int = 0,
        turns_per_window: int = 1,
        system_prompt: Optional[str] = None,
        env_instructions: Optional[List[str]] = None,
        honor_done: bool = False,
        think_track: str = THINK_TRACK,
        image_track: str = IMAGE_TRACK,
    ) -> None:
        super().__init__(
            pipeline=pipeline,
            max_turns=max_turns,
            system_prompt=system_prompt,
            env_instructions=env_instructions,
            honor_done=honor_done,
            think_track=think_track,
            image_track=image_track,
        )
        self.sessions_per_prompt = int(sessions_per_prompt)
        self._version = 0
        self._pool = FullyAsyncPool(
            backend=None,  # set per-generate (holds per-request sampling params)
            env=self.env,
            version_ref=lambda: self._version,
            max_turns=self.max_turns,
            sessions_per_prompt=self.sessions_per_prompt,
            max_staleness=int(max_staleness),
            turns_per_window=int(turns_per_window),
            honor_done=honor_done,
        )

    # ------------------------------------------------------------------
    # Version control (driver calls these after the optimizer step)
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def bump_version(self) -> int:
        """Increment the policy version after a weight update; abort now-too-stale
        resident sessions. Returns the new version."""
        self._version += 1
        self._pool.on_policy_version_changed(self._version)
        return self._version

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def pool_metrics(self) -> Dict[str, float]:
        """Drain + return the resident pool's cumulative metrics (rank 0's view)."""
        return self._pool.drain_metrics()

    # ------------------------------------------------------------------
    # Generate = harvest B complete groups from the resident pool
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, req: RolloutReq) -> RolloutResp:
        ensure_req_sigmas(req, self.schedule_policy)
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError("BagelAgenticEngineFullyAsync.generate: req.primitives['text'] must be Texts.")
        ar_params = req.sampling_params.get("ar")
        diff_params = req.sampling_params.get("diffusion")
        if ar_params is None or diff_params is None:
            raise TypeError("BagelAgenticEngineFullyAsync.generate: sampling_params must carry 'ar' and 'diffusion'.")

        prompts = list(texts.texts)
        group_ids = (
            list(req.group_ids) if req.group_ids else list(req.sample_ids or [f"g{i}" for i in range(len(prompts))])
        )
        if len(group_ids) != len(prompts):
            raise ValueError("BagelAgenticEngineFullyAsync.generate: group_ids and prompts length mismatch.")

        # Unique (group_id, prompt) pairs for this request window, in order. The
        # request is pre-expanded to N rows/group (half-async parity); collapse to
        # the N distinct groups — the pool fans each out to N sessions internally.
        groups: List[Tuple[str, str]] = []
        seen = set()
        for gid, prompt in zip(group_ids, prompts):
            if gid not in seen:
                seen.add(gid)
                groups.append((gid, prompt))
        target_groups = len(groups)

        prev_modes = [m.training for m in self._models]
        for m in self._models:
            m.eval()
        try:
            with torch.no_grad():
                backend = _BagelTurnBackend(self, ar_params=ar_params, diff_params=diff_params, sigmas=req.sigmas)
                self._pool._backend = backend  # per-request params (stable across rollouts)
                self._pool.ensure_groups(groups)
                picked = self._pool.harvest(target_groups)
                return self._build_resp_fully_async(picked)
        finally:
            for m, mode in zip(self._models, prev_modes):
                m.train(mode)

    # ------------------------------------------------------------------
    # Response assembly (flatten carry-over sessions → 2 versioned tracks)
    # ------------------------------------------------------------------

    def _build_resp_fully_async(self, sessions: List[Session]) -> RolloutResp:
        """Flatten harvested sessions (each carrying its real ``group_id``) into the
        ``{think, image}`` 2-track resp, packing per-token / per-SDE-step version
        metadata alongside the standard behavior-logprob anchors.

        Group-contiguous by construction (``harvest`` returns group-by-group); a
        session's sample id is ``f"{group_id}/s{k}"`` where ``k`` is its index
        within the group, so think rows of a group stay contiguous (the GRPO group)."""
        from unirl.models.bagel.conditions import BagelARConditions, BagelDiffusionConditions

        think_sids: List[str] = []
        think_pids: List[str] = []
        think_tokens: List[torch.Tensor] = []
        think_logps: List[torch.Tensor] = []
        think_versions: List[torch.Tensor] = []
        think_splits: List[List[Dict[str, Any]]] = []
        think_texts: List[str] = []

        img_sids: List[str] = []
        img_pids: List[str] = []
        img_segs: List[LatentSegment] = []
        img_versions: List[torch.Tensor] = []
        img_gen_ctx: List[Any] = []
        img_cfg_text_ctx: List[Any] = []
        img_cfg_img_ctx: List[Any] = []
        img_shapes: List[Tuple[int, int]] = []
        img_pixels: List[torch.Tensor] = []

        # Index sessions within their group for stable sample ids.
        per_group_idx: Dict[str, int] = {}
        for session in sessions:
            gid = session.group_id or "g"
            k = per_group_idx.get(gid, 0)
            per_group_idx[gid] = k + 1
            sid = f"{gid}/s{k}"
            thinks = session.think_turns()
            gens = session.gen_turns()
            tv = session.think_versions()
            for t, (th, gn) in enumerate(zip(thinks, gens)):
                tsid = f"{sid}/t{t}#think"
                think_sids.append(tsid)
                think_pids.append(gid)
                think_tokens.append(th.tokens)
                think_logps.append(
                    th.logprobs if th.logprobs is not None else torch.zeros(th.tokens.numel(), dtype=torch.float32)
                )
                think_versions.append(torch.tensor(tv[t], dtype=torch.long))
                think_splits.append(th.payload["prompt_splits"])
                think_texts.append(th.text or "")

                isid = f"{sid}/t{t}"
                img_sids.append(isid)
                img_pids.append(tsid)
                seg = gn.latent
                img_segs.append(seg)
                img_versions.append(self._sde_versions_for(seg, int(gn.weight_version)))
                g, ct, ci = gn.payload["contexts"]
                img_gen_ctx.append(g)
                img_cfg_text_ctx.append(ct)
                img_cfg_img_ctx.append(ci)
                img_shapes.append(gn.payload["image_shape"])
                img_pixels.append(gn.image.pixels)

        think_segment = TextSegment.pack(tokens=think_tokens, log_probs=think_logps, token_versions=think_versions)
        think_conditions = BagelARConditions(prompt_splits=think_splits)
        think_rt = RolloutTrack(
            sample_ids=think_sids,
            parent_ids=think_pids,
            parent_track=None,
            conditions=think_conditions.to_dict(),
            segment=think_segment,
            decoded=Texts(texts=think_texts),
        )

        image_segment = self._stack_latents_versioned(img_segs, img_versions)
        image_conditions = BagelDiffusionConditions(
            gen_contexts=img_gen_ctx,
            cfg_text_contexts=img_cfg_text_ctx,
            cfg_img_contexts=img_cfg_img_ctx,
            image_shapes=img_shapes,
        )
        image_rt = RolloutTrack(
            sample_ids=img_sids,
            parent_track=self.think_track,
            parent_ids=img_pids,
            conditions=image_conditions.to_dict(),
            segment=image_segment,
            decoded=Images(pixels=torch.cat(img_pixels, dim=0)),
        )
        return RolloutResp(tracks={self.think_track: think_rt, self.image_track: image_rt})

    @staticmethod
    def _sde_versions_for(seg: LatentSegment, version: int) -> torch.Tensor:
        """Per-SDE-step version row for one gen node (``[1, S]``, broadcast version)."""
        if seg.sde_logp is not None:
            return torch.full_like(seg.sde_logp, float(version)).to(torch.long)
        s = int(seg.sde_indices.numel()) if seg.sde_indices is not None else 0
        return torch.full((1, s), int(version), dtype=torch.long)

    @staticmethod
    def _stack_latents_versioned(segs: List[LatentSegment], versions: List[torch.Tensor]) -> LatentSegment:
        """Stack per-turn one-row latent segments into ``[M, ...]`` and attach the
        stacked per-step ``sde_versions`` (metadata; FlowGRPO ignores it)."""
        sde_versions = torch.cat(versions, dim=0) if versions else None
        if len(segs) == 1:
            base = segs[0]
            return LatentSegment(
                modality=base.modality,
                latents=base.latents,
                sigmas=base.sigmas,
                indices=base.indices,
                sde_logp=base.sde_logp,
                sde_versions=sde_versions,
                sde_indices=base.sde_indices,
            )
        latents = torch.cat([s.latents for s in segs], dim=0)
        sde_logp = torch.cat([s.sde_logp for s in segs], dim=0) if segs[0].sde_logp is not None else None
        return LatentSegment(
            modality=segs[0].modality,
            latents=latents,
            sigmas=segs[0].sigmas,
            indices=segs[0].indices,
            sde_logp=sde_logp,
            sde_versions=sde_versions,
            sde_indices=segs[0].sde_indices,
        )


__all__ = ["BagelAgenticEngineFullyAsync"]
