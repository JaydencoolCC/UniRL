"""BagelAgenticEngine — trainside multi-turn think→gen→obs rollout for Bagel.

A :class:`~unirl.rollout.engine.base.BaseRolloutEngine` that generalizes Bagel's
single-turn ``t2ti`` (``BagelPipeline._generate_t2ti``) to a multi-turn agentic
loop. It is **trainside / colocate** (like
:class:`~unirl.rollout.engine.trainside.engine.TrainsideRolloutEngine`): the live
FSDP-wrapped Bagel pipeline IS the sampler, so there is no separate inference
engine and no weight sync — exactly the constraint that makes Bagel half-async
*colocate* rather than disaggregated. The single-thread non-blocking launch/reap
machinery (the disaggregated upgrade) is the trainer's job and a P3 concern.

Per request sample (one **session**), :meth:`generate` drives
:class:`~unirl.agentic.workflow.ThinkGenWorkflow` over an internal
:class:`AgenticBackend` that calls the existing Bagel stages:

- ``think`` — build the AR context from the transcript so far (system + prompt +
  prior think/obs text), run ``pipeline.ar.autoregress`` (the und path), detokenize.
- ``gen``   — build the (gen, cfg_text, cfg_img) KV contexts from
  ``[system, prompt, transcript, think]``, run ``pipeline.diffusion.diffuse`` (the
  gen experts), VAE-decode to pixels.
- ``obs``   — :class:`~unirl.agentic.env.RuleEnv` feedback (P0), woven into the
  next turn's AR context only (the v4 "obs not in segment" invariant).

It returns a 2-track ``RolloutResp`` ``{"ar": think, "image": gen}`` whose lineage
(``image.parent_track="ar"``) lets the trainer credit-assign the image reward up
to the think track and run the dual-algorithm step on the shared backbone.

The request is **pre-expanded** by the trainer to ``P*N`` session rows (``N`` =
sessions per prompt = GRPO group size), so this engine runs exactly one session
per request sample. Think rows of a prompt's ``N`` sessions × ``T`` turns stay
contiguous (group-by-parent), which is what ``compute_advantages`` requires.

Zero-invasion: nothing under ``unirl/models/bagel/`` is modified — the engine
reaches the bundle's inferencer + stages through the injected pipeline.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch

from unirl.agentic.config import IMAGE_TRACK, THINK_TRACK
from unirl.agentic.env import RuleEnv
from unirl.agentic.session import MsgNode, Session
from unirl.agentic.workflow import ThinkGenWorkflow
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.sde.runtime import ensure_req_sigmas
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.segments.latent import LatentSegment
from unirl.types.segments.text import TextSegment

logger = logging.getLogger(__name__)


class BagelAgenticEngine(BaseRolloutEngine):
    """Trainside multi-turn think→gen agentic engine for Bagel."""

    _component_name = "bagel_agentic"

    def __init__(
        self,
        *,
        pipeline: Any,
        max_turns: int = 2,
        system_prompt: Optional[str] = None,
        env_instructions: Optional[List[str]] = None,
        honor_done: bool = False,
        think_track: str = THINK_TRACK,
        image_track: str = IMAGE_TRACK,
    ) -> None:
        self.pipeline = pipeline
        self.max_turns = int(max_turns)
        if self.max_turns < 1:
            raise ValueError(f"BagelAgenticEngine.max_turns must be >= 1, got {max_turns}")
        self.think_track = think_track
        self.image_track = image_track

        # Default system prompt = Bagel's native gen-think planner role (lazy import:
        # the vendored inferencer hard-imports flash_attn, available on the worker).
        if system_prompt is None:
            from unirl.models.bagel.vendor.inferencer import GEN_THINK_SYSTEM_PROMPT

            system_prompt = GEN_THINK_SYSTEM_PROMPT
        self.system_prompt = system_prompt

        # P0 env: deterministic per-turn refinement instructions (model-free).
        self.env = RuleEnv(instructions=env_instructions, max_turns=self.max_turns if honor_done else None)
        self.workflow = ThinkGenWorkflow(honor_done=honor_done)

        # Trainable modules to eval-scope around generate (both Bagel stages share
        # ``bundle.transformer``; dedupe by identity).
        models: List[torch.nn.Module] = []
        for stage in (pipeline.ar, pipeline.diffusion):
            m = stage.trainable_module()
            if all(m is not seen for seen in models):
                models.append(m)
        self._models = models

        # σ-schedule policy for the diffusion turns (pins req.sigmas, like the
        # trainside engine does for SD3 / Bagel t2i).
        self.schedule_policy = pipeline.build_schedule_policy()

    # ------------------------------------------------------------------
    # BaseRolloutEngine plumbing
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        pass

    def health_check(self) -> bool:
        return self.pipeline is not None and all(m is not None for m in self._models)

    # ------------------------------------------------------------------
    # Generate (one session per pre-expanded request sample)
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, req: RolloutReq) -> RolloutResp:
        ensure_req_sigmas(req, self.schedule_policy)
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"BagelAgenticEngine.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )
        ar_params = req.sampling_params.get("ar")
        diff_params = req.sampling_params.get("diffusion")
        if ar_params is None or diff_params is None:
            raise TypeError(
                "BagelAgenticEngine.generate: sampling_params must carry both 'ar' and "
                f"'diffusion' entries; got keys {sorted(req.sampling_params)}."
            )

        prompts = list(texts.texts)
        n = len(prompts)
        sample_ids = list(req.sample_ids) if req.sample_ids else [f"s{i}" for i in range(n)]
        group_ids = list(req.group_ids) if req.group_ids else list(sample_ids)

        prev_modes = [m.training for m in self._models]
        for m in self._models:
            m.eval()
        try:
            with torch.no_grad():
                backend = _BagelTurnBackend(self, ar_params=ar_params, diff_params=diff_params, sigmas=req.sigmas)
                sessions: List[Session] = []
                for prompt in prompts:
                    session = Session(prompt=prompt)
                    self.workflow.run(
                        session=session,
                        backend=backend,
                        env=self.env,
                        max_turns=self.max_turns,
                        sampling=None,
                    )
                    sessions.append(session)
                return self._build_resp(sample_ids, group_ids, sessions)
        finally:
            for m, mode in zip(self._models, prev_modes):
                m.train(mode)

    # ------------------------------------------------------------------
    # Context building (generalizes BagelPipeline._build_think_contexts to
    # an arbitrary transcript prefix — multi-turn).
    # ------------------------------------------------------------------

    def _text_split(self, text: str) -> Dict[str, Any]:
        """One AR prompt split: ``[bos] + encode(text) + [eos]`` (vendor wrap)."""
        ntk = self.pipeline.bundle.new_token_ids
        tok = self.pipeline.bundle.tokenizer
        ids = [ntk["bos_token_id"]] + tok.encode(text) + [ntk["eos_token_id"]]
        return {"kind": "text", "ids": torch.tensor(ids, dtype=torch.long)}

    def _think_prompt_splits(self, session: Session) -> List[Dict[str, Any]]:
        """AR context for the *current* turn: system + prompt + prior think/obs text.

        ``context_nodes_before(len)`` returns every think + obs node generated so
        far (gen nodes excluded — images are not AR text), which is exactly the
        prefix the next think conditions on. obs text enters here (prefix only),
        never the think segment — the "obs not in segment" invariant.
        """
        texts = [self.system_prompt, session.prompt]
        texts += [n.text or "" for n in session.context_nodes_before(len(session.nodes))]
        return [self._text_split(t) for t in texts]

    def _gen_contexts(self, session: Session, think_text: str) -> Tuple[Any, Any, Any]:
        """Build (gen, cfg_text, cfg_img) KV contexts for the current turn's image.

        Generalizes ``BagelPipeline._build_think_contexts`` to a transcript:

            gen      = init + system + prompt + transcript + think   (full)
            cfg_text = init + system                                 (drop all but system)
            cfg_img  = init + system + prompt + transcript           (drop only this think)

        ``transcript`` = prior think + obs text (the node just appended for this
        turn — ``think_text`` — is added last to ``gen`` only). cfg branches are
        inert when ``cfg_text_scale == 1`` but built for parity with the t2ti path.
        """
        inf = self.pipeline.bundle.inferencer
        # The think node is already appended (index len-1); its prefix is everything before it.
        prefix_nodes = session.context_nodes_before(len(session.nodes) - 1)
        prefix_texts = [session.prompt] + [n.text or "" for n in prefix_nodes]

        gen = inf.init_gen_context()
        cfg_img = deepcopy(gen)
        with torch.no_grad(), self.pipeline._autocast_ctx():
            gen = inf.update_context_text(self.system_prompt, gen)
            cfg_img = inf.update_context_text(self.system_prompt, cfg_img)
            cfg_text = deepcopy(gen)  # init + system
            for txt in prefix_texts:
                gen = inf.update_context_text(txt, gen)
                cfg_img = inf.update_context_text(txt, cfg_img)
            gen = inf.update_context_text(think_text, gen)  # this turn's think → gen only
        return gen, cfg_text, cfg_img

    def _detokenize(self, tokens: torch.Tensor) -> str:
        """Decode response tokens to text, stripped at ``<|im_end|>`` (t2ti parity)."""
        return self.pipeline.bundle.tokenizer.decode([int(t) for t in tokens.tolist()]).split("<|im_end|>")[0]

    # ------------------------------------------------------------------
    # Response assembly (flatten sessions → 2 tracks)
    # ------------------------------------------------------------------

    @staticmethod
    def _stack_latents(segs: List[LatentSegment]) -> LatentSegment:
        """Stack one-row per-turn latent segments into ``[M, K, seq, C]`` (t2ti parity)."""
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

    def _build_resp(self, sample_ids: List[str], group_ids: List[str], sessions: List[Session]) -> RolloutResp:
        """Flatten ``(session, turn)`` rows into the {think, image} 2-track resp.

        think track (root): one row per (session, turn), grouped by the session's
        prompt group id (N*T rows per prompt → meaningful GRPO baseline). image
        track: 1:1 child of think (``parent_track=think``) so ``propagate_rewards``
        credits each image's reward to its think.
        """
        from unirl.models.bagel.conditions import BagelARConditions, BagelDiffusionConditions

        think_sids: List[str] = []
        think_pids: List[str] = []
        think_tokens: List[torch.Tensor] = []
        think_logps: List[torch.Tensor] = []
        think_splits: List[List[Dict[str, Any]]] = []
        think_texts: List[str] = []

        img_sids: List[str] = []
        img_pids: List[str] = []
        img_segs: List[LatentSegment] = []
        img_gen_ctx: List[Any] = []
        img_cfg_text_ctx: List[Any] = []
        img_cfg_img_ctx: List[Any] = []
        img_shapes: List[Tuple[int, int]] = []
        img_pixels: List[torch.Tensor] = []

        for sid, gid, session in zip(sample_ids, group_ids, sessions):
            thinks = session.think_turns()
            gens = session.gen_turns()
            for t, (th, gn) in enumerate(zip(thinks, gens)):
                tsid = f"{sid}/t{t}#think"
                think_sids.append(tsid)
                think_pids.append(gid)
                think_tokens.append(th.tokens)
                think_logps.append(
                    th.logprobs if th.logprobs is not None else torch.zeros(th.tokens.numel(), dtype=torch.float32)
                )
                think_splits.append(th.payload["prompt_splits"])
                think_texts.append(th.text or "")

                isid = f"{sid}/t{t}"
                img_sids.append(isid)
                img_pids.append(tsid)  # 1:1 child of this turn's think
                img_segs.append(gn.latent)
                g, ct, ci = gn.payload["contexts"]
                img_gen_ctx.append(g)
                img_cfg_text_ctx.append(ct)
                img_cfg_img_ctx.append(ci)
                img_shapes.append(gn.payload["image_shape"])
                img_pixels.append(gn.image.pixels)

        think_segment = TextSegment.pack(tokens=think_tokens, log_probs=think_logps)
        think_conditions = BagelARConditions(prompt_splits=think_splits)
        think_rt = RolloutTrack(
            sample_ids=think_sids,
            parent_ids=think_pids,
            parent_track=None,
            conditions=think_conditions.to_dict(),
            segment=think_segment,
            decoded=Texts(texts=think_texts),
        )

        image_segment = self._stack_latents(img_segs)
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


class _BagelTurnBackend:
    """Per-request :class:`~unirl.agentic.workflow.AgenticBackend` over Bagel stages.

    Holds the resolved per-request params (AR + diffusion sampling, σ schedule)
    and delegates the heavy lifting to the engine's context helpers + the Bagel
    stages, so the workflow stays model-agnostic.
    """

    def __init__(self, engine: BagelAgenticEngine, *, ar_params: Any, diff_params: Any, sigmas: torch.Tensor) -> None:
        self.engine = engine
        self.ar_params = ar_params
        self.diff_params = diff_params
        self.sigmas = sigmas

    def think(self, *, session: Session, turn: int, sampling: Any) -> MsgNode:
        splits = self.engine._think_prompt_splits(session)
        from unirl.models.bagel.conditions import BagelARConditions

        conditions = BagelARConditions(prompt_splits=[splits])
        segment = self.engine.pipeline.ar.autoregress(conditions, sampling_params=self.ar_params)
        n = int(segment.lengths[0].item()) if segment.lengths is not None else int(segment.tokens.numel())
        tokens = segment.tokens[:n].detach().clone()
        logps = segment.log_probs[:n].detach().clone() if segment.log_probs is not None else None
        text = self.engine._detokenize(tokens)
        return MsgNode(
            kind="think",
            tokens=tokens,
            logprobs=logps,
            text=text,
            payload={"prompt_splits": splits},
            weight_version=session.weight_version,
        )

    def gen(self, *, session: Session, think: MsgNode, turn: int, sampling: Any) -> MsgNode:
        from unirl.models.bagel.conditions import BagelDiffusionConditions

        gen_ctx, cfg_text, cfg_img = self.engine._gen_contexts(session, think.text or "")
        image_shape = (int(self.diff_params.height), int(self.diff_params.width))
        conditions = BagelDiffusionConditions.for_sample(
            gen_context=gen_ctx,
            cfg_text_context=cfg_text,
            cfg_img_context=cfg_img,
            image_shape=image_shape,
        )
        segment = self.engine.pipeline.diffusion.diffuse(
            conditions, schedule=self.sigmas, params=self.diff_params, initial_latents=None
        )
        images = self.engine.pipeline.vae_decode.decode(segment, image_shape=image_shape)
        return MsgNode(
            kind="gen",
            latent=segment,
            image=images,
            payload={"contexts": (gen_ctx, cfg_text, cfg_img), "image_shape": image_shape},
            weight_version=session.weight_version,
        )


__all__ = ["BagelAgenticEngine"]
