"""``fastvideo`` engine core — in-process FastVideo ``VideoGenerator`` rollout.

Mirrors the ``TrainsideRolloutEngine`` / ``SGLangDiffusionRolloutEngine`` shells:
``generate`` is ``@distributed(DP_SCATTER)``, pins σ via ``ensure_req_sigmas``,
optionally chunks by ``forward_batch_size``, and packs one ``RolloutResp`` track
with a ``LatentSegment`` (trajectory + native per-step log-probs).

The FastVideo-driving logic (VideoGenerator boot, PR #1222 ``ForwardBatch.RLData``
native-logprob path, transformer hot-swap, sleep/wake) is ported from the proven
DiffusionRL FastVideo engine; only the typed boundary (RolloutReq/RolloutResp/
LatentSegment, σ SSOT) is new.

NOTE (draft / WIP — pending GPU smoke):
  * The SDE-step alignment between FastVideo's per-step log-probs and UniRL's
    ``sde_indices`` is selected in :meth:`_build_segment`; verify on first smoke
    that ``sde_logp`` columns line up with ``params.sde_indices`` (eta gating in
    FastVideo's denoise must match the SDE window).
  * x_T SSOT: FastVideo currently regenerates its own initial noise from
    ``sp.seed`` rather than consuming the driver's NoiseRecipe x_T; wiring the
    shared x_T into FastVideo (for on-policy ratio parity) is a follow-up.
  * Local-mode colocate, single model_family (wan2.1) only for now.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.rollout.engine.fastvideo.config import FastVideoEngineConfig, FastVideoPorts
from unirl.sde.runtime import FlowMatchSchedulePolicy, ensure_req_sigmas
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts, Video, Videos
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.segments.latent import make_video_segment

logger = logging.getLogger(__name__)


class FastVideoRolloutEngine(BaseRolloutEngine):
    """Rollout engine backed by FastVideo ``VideoGenerator`` (RL fork, PR #1222)."""

    _component_name = "fastvideo"

    def __init__(
        self,
        config: FastVideoEngineConfig,
        *,
        device: Optional[torch.device] = None,
        strategy: Any = None,
        rank: Optional[int] = None,
        model_config: Optional[Any] = None,
        ports: Optional[FastVideoPorts] = None,
    ) -> None:
        require(
            isinstance(config, FastVideoEngineConfig),
            f"FastVideoRolloutEngine requires FastVideoEngineConfig; got {type(config).__name__}",
        )
        require(
            model_config is not None and bool(model_config.pretrained_model_ckpt_path),
            "FastVideoRolloutEngine requires model_config.pretrained_model_ckpt_path",
        )
        self.cfg = config
        self.model_config = model_config
        self.strategy = strategy
        self.rank = rank
        self._device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._is_offloaded = False
        self._generator: Any = None
        self._fastvideo_args: Any = None

        if ports is None:
            ports = FastVideoPorts.reserve()
        self._ports = ports

        self._ensure_fastvideo_importable()
        self._build_generator()

        # σ SSOT: same schedule policy the trainer/replay uses, so the engine can
        # pin req.sigmas and FastVideo consumes that exact schedule.
        self.schedule_policy = FlowMatchSchedulePolicy.from_pretrained(
            model_config.pretrained_model_ckpt_path,
            shift=float(model_config.shift),
            require_dynamic=bool(getattr(model_config, "use_dynamic_shifting", False)),
            dynamic_overrides=getattr(model_config, "dynamic_shift_overrides", None),
        )
        logger.info(
            "Initialized fastvideo engine (rank=%s, native_logprob=%s, master_port=%s)",
            rank, config.native_logprob, ports.master_port,
        )

    # ------------------------------------------------------------------ #
    # FastVideo import + VideoGenerator boot (ported from DiffusionRL)
    # ------------------------------------------------------------------ #
    def _ensure_fastvideo_importable(self) -> None:
        try:
            importlib.import_module("fastvideo")
            return
        except ModuleNotFoundError:
            pass
        path = self.cfg.fastvideo_path or os.getenv("FASTVIDEO_PATH", "")
        require(bool(path), "fastvideo not importable; set cfg.fastvideo_path or $FASTVIDEO_PATH")
        if path not in sys.path:
            sys.path.insert(0, str(Path(path).expanduser()))
        importlib.import_module("fastvideo")

    def _build_generator(self) -> None:
        from fastvideo import VideoGenerator
        from fastvideo.fastvideo_args import FastVideoArgs

        ekw = dict(self.cfg.engine_kwargs or {})
        fv_kwargs: Dict[str, Any] = {
            "model_path": self.model_config.pretrained_model_ckpt_path,
            "num_gpus": int(self.cfg.num_gpus),
            "tp_size": int(self.cfg.tp_size),
            "sp_size": int(self.cfg.sp_size),
            "inference_mode": True,
            # Force decoded pixels as a [B, C, T, H, W] tensor (not PIL/latent)
            # so execute_forward populates batch.output for the reward path.
            "output_type": "pt",
            "dit_cpu_offload": False,
            "dit_layerwise_offload": False,
            "text_encoder_cpu_offload": False,
            "vae_cpu_offload": False,
            "master_port": int(self._ports.master_port),
        }
        fv_kwargs.update(ekw)
        self._fastvideo_args = FastVideoArgs.from_kwargs(**fv_kwargs)
        self._generator = VideoGenerator.from_fastvideo_args(self._fastvideo_args)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, req: RolloutReq) -> RolloutResp:
        require(
            int(req.batch_size) > 0,
            "FastVideoRolloutEngine.generate requires a non-empty req (batch_size > 0)",
        )
        # σ SSOT: pin once on the full batch (shared field, survives req.slice).
        ensure_req_sigmas(req, self.schedule_policy)

        fbs = self.cfg.forward_batch_size
        bs = int(req.batch_size)
        if fbs is None or bs <= fbs:
            return self._generate_batch(req)

        outputs: List[RolloutResp] = []
        for start in range(0, bs, fbs):
            end = min(start + fbs, bs)
            outputs.append(self._generate_batch(req.slice(start, end)))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return RolloutResp.concat(outputs)

    def _generate_batch(self, req: RolloutReq) -> RolloutResp:
        text_primitive = req.primitives.get("text")
        require(
            text_primitive is not None and isinstance(text_primitive, Texts),
            f"fastvideo engine requires req.primitives['text']: Texts; "
            f"got {type(text_primitive).__name__ if text_primitive is not None else 'None'}",
        )
        prompts = list(text_primitive.texts)
        require(
            len(prompts) == int(req.batch_size),
            f"fastvideo engine expects req.primitives['text'] of len batch_size; "
            f"got {len(prompts)} vs {int(req.batch_size)}",
        )
        params = req.sampling_params.get("diffusion")
        require(
            params is not None,
            "fastvideo engine requires req.sampling_params['diffusion']",
        )
        raw = self._drive_fastvideo(prompts, params, req.sigmas)
        return self._build_resp(req, params, raw)

    def _drive_fastvideo(
        self, prompts: List[str], params: Any, sigmas: torch.Tensor,
    ) -> Dict[str, Any]:
        """PR #1222 native-logprob path via executor.execute_forward + RLData.

        Returns dict(trajectory=[B,T+1,...], log_probs=[B,T], samples=[B,...]).
        """
        from copy import deepcopy

        from fastvideo.configs.sample.base import SamplingParam
        from fastvideo.pipelines import ForwardBatch
        from fastvideo.utils import shallow_asdict

        sp = SamplingParam()
        sp.height = int(params.height)
        sp.width = int(params.width)
        sp.num_frames = int(params.num_frames)
        sp.num_inference_steps = int(params.num_inference_steps)
        sp.guidance_scale = float(params.guidance_scale)
        sp.seed = int(params.seed)
        sp.num_videos_per_prompt = 1
        sp.save_video = False
        sp.return_frames = False
        sp.return_trajectory_latents = True
        sp.return_trajectory_decoded = False
        # σ SSOT: hand FastVideo the trainer-pinned schedule verbatim (drop the
        # terminal 0 — FastVideo's scheduler appends its own endpoint).
        sp.sigmas = [float(x) for x in sigmas.detach().cpu().tolist()[:-1]]

        all_log_probs: List[torch.Tensor] = []
        all_traj: List[torch.Tensor] = []
        all_samples: List[torch.Tensor] = []
        all_decoded: List[torch.Tensor] = []
        all_text_embeds: List[torch.Tensor] = []
        all_text_masks: List[Optional[torch.Tensor]] = []
        all_neg_embeds: List[torch.Tensor] = []
        all_neg_masks: List[Optional[torch.Tensor]] = []

        for prompt in prompts:
            one = deepcopy(sp)
            one.prompt = prompt
            latents_size = [(one.num_frames - 1) // 4 + 1, one.height // 8, one.width // 8]
            n_tokens = latents_size[0] * latents_size[1] * latents_size[2]
            sp_dict = shallow_asdict(one)
            sp_dict.pop("eta", None)
            batch = ForwardBatch(
                **sp_dict,
                eta=float(params.eta),
                n_tokens=n_tokens,
                VSA_sparsity=self._fastvideo_args.VSA_sparsity,
                rl_data=ForwardBatch.RLData(
                    enabled=True,
                    collect_log_probs=bool(self.cfg.native_logprob),
                    store_trajectory=True,
                    keep_trajectory_on_cpu=True,
                ),
            )
            out = self._generator.executor.execute_forward(batch, self._fastvideo_args)
            rl = out.rl_data
            traj = rl.trajectory_latents if rl is not None else None
            if traj is None:
                traj = out.trajectory_latents
            require(torch.is_tensor(traj), "FastVideo returned no trajectory tensor")
            if traj.dim() == 5:
                traj = traj.unsqueeze(0)
            all_traj.append(traj.detach().cpu())
            samples = out.latents.cpu() if out.latents is not None else traj[:, -1].cpu()
            all_samples.append(samples.detach().cpu())
            # Decoded pixels: the FastVideo pipeline's DecodingStage writes the
            # final video to batch.output as [B, C, T, H, W] in [0, 1] (float32,
            # CPU). The reward path needs this as track.decoded (Videos).
            dec = getattr(out, "output", None)
            require(torch.is_tensor(dec), "FastVideo returned no decoded output (batch.output)")
            if dec.dim() == 4:
                dec = dec.unsqueeze(0)
            all_decoded.append(dec.detach().cpu().float())

            # Text conditioning: reuse the *exact* prompt embeddings FastVideo fed
            # its transformer this rollout, so the trainer's replay forward yields
            # an on-policy importance ratio (no re-encode drift). prompt_embeds is
            # a per-encoder list; WAN uses a single UMT5 encoder -> index 0.
            pe = out.prompt_embeds
            require(
                isinstance(pe, (list, tuple)) and len(pe) > 0 and torch.is_tensor(pe[0]),
                "FastVideo returned no prompt_embeds for text conditioning",
            )
            te = pe[0]
            if te.dim() == 2:
                te = te.unsqueeze(0)
            all_text_embeds.append(te.detach().cpu().float())
            pm = out.prompt_attention_mask
            tm = pm[0] if isinstance(pm, (list, tuple)) and len(pm) > 0 and torch.is_tensor(pm[0]) else None
            all_text_masks.append(tm.detach().cpu() if tm is not None else None)

            ne = out.negative_prompt_embeds
            if isinstance(ne, (list, tuple)) and len(ne) > 0 and torch.is_tensor(ne[0]):
                nte = ne[0]
                if nte.dim() == 2:
                    nte = nte.unsqueeze(0)
                all_neg_embeds.append(nte.detach().cpu().float())
                nm = out.negative_attention_mask
                ntm = nm[0] if isinstance(nm, (list, tuple)) and len(nm) > 0 and torch.is_tensor(nm[0]) else None
                all_neg_masks.append(ntm.detach().cpu() if ntm is not None else None)

            if self.cfg.native_logprob:
                lp = rl.log_probs if rl is not None else None
                require(torch.is_tensor(lp), "FastVideo native rollout returned no log_probs")
                all_log_probs.append(lp.detach().cpu())

        return {
            "trajectory": torch.cat(all_traj, dim=0),
            "samples": torch.cat(all_samples, dim=0),
            "decoded": torch.cat(all_decoded, dim=0),
            "log_probs": torch.cat(all_log_probs, dim=0) if all_log_probs else None,
            "text_embeds": all_text_embeds,
            "text_masks": all_text_masks,
            "neg_embeds": all_neg_embeds,
            "neg_masks": all_neg_masks,
        }

    def _build_resp(self, req: RolloutReq, params: Any, raw: Dict[str, Any]) -> RolloutResp:
        segment = self._build_segment(req, params, raw)
        decoded = self._build_decoded(raw)
        conditions = self._build_conditions(raw)
        return RolloutResp(
            tracks={
                "video": RolloutTrack(
                    sample_ids=list(req.sample_ids),
                    parent_ids=list(req.group_ids),
                    conditions=conditions,
                    segment=segment,
                    decoded=decoded,
                ),
            }
        )

    def _build_conditions(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble the WAN21 ``conditions`` dict the trainer replays against.

        Packs the captured FastVideo prompt embeddings into ``TextEmbedCondition``s
        (``text`` + optional CFG ``negative_text``), padding variable token lengths
        with zeros via ``TextEmbedCondition.concat`` (WAN's zeroed-pad convention).
        """
        text_embeds: List[torch.Tensor] = raw.get("text_embeds") or []
        require(len(text_embeds) > 0, "fastvideo engine produced no text embeddings")
        text_masks: List[Optional[torch.Tensor]] = raw.get("text_masks") or [None] * len(text_embeds)

        text = TextEmbedCondition.concat(
            [
                TextEmbedCondition(embeds=text_embeds[i], pooled=None, attn_mask=text_masks[i])
                for i in range(len(text_embeds))
            ]
        )
        conditions: Dict[str, Any] = {"text": text}

        neg_embeds: List[torch.Tensor] = raw.get("neg_embeds") or []
        if len(neg_embeds) == len(text_embeds) and len(neg_embeds) > 0:
            neg_masks: List[Optional[torch.Tensor]] = raw.get("neg_masks") or [None] * len(neg_embeds)
            conditions["negative_text"] = TextEmbedCondition.concat(
                [
                    TextEmbedCondition(embeds=neg_embeds[i], pooled=None, attn_mask=neg_masks[i])
                    for i in range(len(neg_embeds))
                ]
            )
        return conditions

    def _build_decoded(self, raw: Dict[str, Any]) -> Videos:
        """Pack FastVideo's decoded output [B, C, T, H, W] into a ``Videos``.

        Mirrors WAN21VAEDecodeStage: permute each sample (C, T, H, W) →
        (T, C, H, W) so Video.frames matches the canonical [T, C, H, W]
        contract the reward path (video_pickscore) consumes.
        """
        frames = raw["decoded"]
        require(torch.is_tensor(frames) and frames.dim() == 5,
                f"fastvideo decoded must be [B, C, T, H, W]; got "
                f"{tuple(frames.shape) if torch.is_tensor(frames) else type(frames).__name__}")
        videos = [
            Video(frames=frames[i].permute(1, 0, 2, 3).contiguous())
            for i in range(int(frames.shape[0]))
        ]
        return Videos.from_list(videos)

    def _build_segment(self, req: RolloutReq, params: Any, raw: Dict[str, Any]):
        traj = raw["trajectory"]  # [B, T+1, C, T_lat, H, W]
        device = traj.device
        T = int(traj.shape[1]) - 1
        indices = torch.arange(traj.shape[1], dtype=torch.long, device=device)

        # sde_indices is ALWAYS populated (the trainer needs to know which steps
        # to replay). Mirror the SGLang reference: when the caller didn't pin a
        # subset, every transition is an SDE step (arange(num_steps)).
        sde_set = sorted(int(i) for i in (params.sde_indices or []))
        if not sde_set:
            sde_set = list(range(T))
        sde_indices = torch.tensor(sde_set, dtype=torch.long, device=device)

        # sde_logp: native per-step log-prob [B, T] from FastVideo's RLData. Slice
        # to the SDE columns when a strict subset was requested; otherwise the
        # full [B, T] already matches the all-steps schedule.
        sde_logp = None
        lp = raw.get("log_probs")
        if lp is not None:
            if lp.shape[1] == T and len(sde_set) < T:
                cols = [s for s in sde_set if 0 <= s < lp.shape[1]]
                sde_logp = lp[:, cols].contiguous()
            else:
                sde_logp = lp.contiguous()

        return make_video_segment(
            latents=traj,
            sigmas=req.sigmas,
            indices=indices,
            sde_logp=sde_logp,
            sde_indices=sde_indices,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        if self._is_offloaded:
            return
        if self._generator is not None:
            try:
                self._generator.shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.warning("fastvideo sleep/shutdown warning: %s", exc)
            self._generator = None
        self._is_offloaded = True

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        if not self._is_offloaded:
            return
        from fastvideo import VideoGenerator

        self._generator = VideoGenerator.from_fastvideo_args(self._fastvideo_args)
        self._is_offloaded = False

    @property
    def is_offloaded(self) -> bool:
        return self._is_offloaded

    def onload_weights(self, *, track_prefix: str = "") -> None:
        del track_prefix
        self.wake_up()

    def shutdown(self) -> None:
        if self._generator is not None:
            self._generator.shutdown()
            self._generator = None

    # ------------------------------------------------------------------ #
    # Weight sync — checkpoint_path (full-param hot-swap). Reached per worker
    # via the local sibling call from CheckpointWeightSync (not @distributed).
    # ------------------------------------------------------------------ #
    def update_weights_from_path(self, checkpoint_path: str, *, track_prefix: str = "") -> None:
        del track_prefix
        require(bool(checkpoint_path), "update_weights_from_path requires a non-empty path")
        require(self._generator is not None, "fastvideo engine is offloaded/not initialized")
        self._generator.update_transformer_weights_from_path(checkpoint_path)
        logger.info("fastvideo transformer weights updated from %s", checkpoint_path)


__all__ = ["FastVideoRolloutEngine"]
