"""Bagel AR (reasoning / "thinking") stage.

BAGEL is a unified MoT model: the SAME transformer generates the reasoning text
(und experts) that then conditions image synthesis (gen experts). This module
adds the text half UniGRPO needs, mirroring :mod:`unirl.models.qwen3.ar` but for
BAGEL's navit (packed, ``bs=1``) generation:

- ``BagelARParams`` — per-request reasoning knobs.
- ``BagelARStage`` — implements ``ARStage[BagelARConditions]``:
  - ``autoregress`` samples the thinking text per prompt via the vendored
    KV-cache decode (``Bagel.generate_text``), then scores it with a
    teacher-forced packed und forward to store a real per-token π_old, packing
    a varlen ``TextSegment``.
  - ``replay`` recomputes per-token log-probs by the SAME grad-capable
    teacher-forced packed und forward (:func:`unirl.models.bagel.rl_ops.und_forward`).

``GRPO`` (the AR algorithm) is rollout-anchored: it reads ``segment.log_probs``
as π_old directly (no replay-anchor option). Sampling uses the vendored
incremental ``generate_text`` (forward_inference, KV cache), but π_old here is
computed by the SAME packed ``forward_train`` path ``replay`` uses (just under
``no_grad``), so the step-0 PPO ratio is 1 by construction — both sides share
:meth:`_token_logp`.

Like the diffusion side, the vendored modeling (+ flash_attn) is reached through
the bundle at call time, so this module stays CPU-importable.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch
import torch.nn.functional as F

from unirl.models.types.ar import ARSamplingParams, ARStage
from unirl.types.segments import TextSegment
from unirl.utils.dtypes import parse_torch_dtype

from . import rl_ops
from .conditions import BagelARConditions

if TYPE_CHECKING:
    from .bundle import BagelBundle


@dataclass
class BagelARParams:
    """Per-request reasoning knobs for BAGEL thinking generation.

    BAGEL's vendored ``generate_text`` samples via plain
    ``multinomial(softmax(logits / T))`` — there is no top-p / top-k — so only
    ``max_new_tokens`` / ``temperature`` are honored (temperature <= 0 => greedy).
    """

    max_new_tokens: int = 1024
    temperature: float = 1.0


class BagelARStage(ARStage[BagelARConditions]):
    """BAGEL reasoning (thinking) stage — autoregress + teacher-forced replay."""

    def __init__(
        self,
        *,
        model: "BagelBundle",
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        self.model = model
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="BagelARStage.autocast_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="BagelARStage.logprob_precision")
        # Navit pack size for the N thinking chains (forward_batch_size pack-B), set by
        # the pipeline via set_forward_batch_size. None/1 = legacy per-chain bs=1 decode.
        self.forward_batch_size: Optional[int] = None

    def trainable_module(self) -> "torch.nn.Module":
        """The MoT transformer (== ``bundle.transformer``) — FSDP/LoRA wrap target.

        Same module the diffusion stage trains; the und experts (text path) and
        gen experts (image path) both live in its ``Qwen2MoTDecoderLayer`` blocks.
        """
        return self.model.transformer

    def _autocast_ctx(self, device: torch.device):
        if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", self.autocast_dtype)
        return nullcontext()

    def _prompt_ids(self, prompt: str, device: torch.device) -> torch.Tensor:
        """``[bos] + encode(prompt) + [eos]`` — identical to the vendored
        ``Bagel.prepare_prompts`` wrapping, so the teacher-forced score's prompt
        matches the KV context ``autoregress`` conditioned on."""
        nti = self.model.new_token_ids
        ids = [nti["bos_token_id"]] + self.model.tokenizer.encode(prompt) + [nti["eos_token_id"]]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _token_logp(self, prompt: str, response: torch.Tensor, *, temperature: float) -> torch.Tensor:
        """Per-token log-prob of ``response`` given ``prompt`` via a packed und forward.

        Builds ``prompt + gen_bos + response[:-1]`` and scores each response token
        at its predicting position (positions ``[P, P+n)``). Used by BOTH
        ``autoregress`` (under ``no_grad`` -> stored π_old) and ``replay`` (under
        grad -> π_new); the caller owns the grad / ``.train()`` scope, this manages
        only autocast. Returns ``[n]`` in ``logprob_precision``.
        """
        device = next(self.model.transformer.parameters()).device
        response = response.to(device=device, dtype=torch.long)
        n = int(response.shape[0])
        if n == 0:
            return torch.zeros(0, dtype=self.logprob_dtype, device=device)
        gen_bos = int(self.model.new_token_ids["bos_token_id"])
        prompt_ids = self._prompt_ids(prompt, device)
        p = int(prompt_ids.shape[0])
        gen_bos_t = torch.tensor([gen_bos], dtype=torch.long, device=device)
        packed = torch.cat([prompt_ids, gen_bos_t, response[:-1]], dim=0)  # [P+n]
        predict = torch.arange(p, p + n, device=device, dtype=torch.long)
        temp = float(temperature) if float(temperature) > 0.0 else 1.0
        with self._autocast_ctx(device):
            hidden = rl_ops.und_forward(self.model.model, packed)  # [P+n, H]
            logits = self.model.transformer.lm_head(hidden[predict])  # [n, vocab]
        logp_full = F.log_softmax(logits.float() / temp, dim=-1)  # fp32
        return logp_full.gather(-1, response.unsqueeze(-1)).squeeze(-1).to(dtype=self.logprob_dtype)

    def autoregress(
        self,
        conditions: BagelARConditions,
        *,
        sampling_params: ARSamplingParams,
        params: Any = None,
        **_kwargs: Any,
    ) -> TextSegment:
        """Sample thinking text per prompt (navit ``bs=1``); pack a ``TextSegment``.

        Stores a real per-token π_old (teacher-forced ``_token_logp`` under
        ``no_grad``) so GRPO's rollout-anchored ratio is 1 at step 0.
        """
        bagel = self.model.model
        nti = self.model.new_token_ids
        inferencer = self.model.inferencer
        device = next(bagel.parameters()).device

        temperature = float(sampling_params.temperature)
        max_new = int(sampling_params.max_new_tokens)
        do_sample = temperature > 0.0
        logp_temp = temperature if do_sample else 1.0

        tokens_list: List[torch.Tensor] = []
        logp_list: List[torch.Tensor] = []
        prompts = list(conditions.prompts)
        fbs = self.forward_batch_size
        if fbs is None or fbs <= 1:
            for prompt in prompts:  # legacy navit bs=1 (untouched)
                ctx = inferencer.init_gen_context()
                ctx = inferencer.update_context_text(prompt, ctx)  # build prompt KV (no_grad)
                start = bagel.prepare_start_tokens(ctx["kv_lens"], ctx["ropes"], nti)
                # Pin the CPU index tensors to the model device (insurance: the
                # vendored decode reads ``key_values_lens.device`` for its arange).
                start = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in start.items()}
                with torch.no_grad():
                    unpacked = bagel.generate_text(
                        past_key_values=ctx["past_key_values"],
                        max_length=max_new,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else 1.0,
                        end_token_id=nti["eos_token_id"],
                        **start,
                    )
                    # ``unpacked`` = [bos, t1, ..., t_k] (batch=1); drop the leading bos.
                    resp = unpacked[1:, 0].to(device=device, dtype=torch.long)
                    old_logp = self._token_logp(prompt, resp, temperature=logp_temp)
                tokens_list.append(resp)
                logp_list.append(old_logp)
        else:
            # forward_batch_size pack-B: navit-pack up to fbs chains per generate_text.
            for s in range(0, len(prompts), fbs):
                chunk = prompts[s : s + fbs]
                resps = self._decode_batched(
                    chunk, max_new=max_new, do_sample=do_sample, temperature=temperature, device=device
                )
                for prompt, resp in zip(chunk, resps):
                    with torch.no_grad():
                        old_logp = self._token_logp(prompt, resp, temperature=logp_temp)
                    tokens_list.append(resp)
                    logp_list.append(old_logp)
        return TextSegment.pack(tokens=tokens_list, log_probs=logp_list)

    def _decode_batched(
        self,
        prompts: List[str],
        *,
        max_new: int,
        do_sample: bool,
        temperature: float,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Batched navit decode of B thinking chains (forward_batch_size pack-B).

        Builds B contexts in ONE packed prefill (``prepare_prompts`` zips the B prompts
        → block-diagonal KV), runs ONE packed ``generate_text`` with ``end_token_id=None``
        (the vendored loop's early-stop only checks seq 0, so we disable it and trim per
        sequence), then trims each at its FIRST eos — matching the per-chain path, which
        excludes the eos. ``generate_text`` already does the packed multi-sequence forward
        (per-seq ``key_values_lens`` + packed indexes). Finished sequences decode on to
        ``max_new`` (wasted compute, not padding); same-prompt chains finish at similar
        lengths. Returns B variable-length response token tensors.
        """
        bagel = self.model.model
        inferencer = self.model.inferencer
        nti = self.model.new_token_ids
        eos = int(nti["eos_token_id"])
        n = len(prompts)
        lm = self.model.transformer
        was_training = lm.training
        if was_training:
            lm.eval()
        try:
            with torch.no_grad():
                gi, kv_lens, ropes = bagel.prepare_prompts(
                    curr_kvlens=[0] * n,
                    curr_rope=[0] * n,
                    prompts=list(prompts),
                    tokenizer=inferencer.tokenizer,
                    new_token_ids=nti,
                )
                pkv = bagel.forward_cache_update_text(inferencer.init_gen_context()["past_key_values"], **gi)
                start = bagel.prepare_start_tokens(kv_lens, ropes, nti)
                start = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in start.items()}
                unpacked = bagel.generate_text(
                    past_key_values=pkv,
                    max_length=max_new,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else 1.0,
                    end_token_id=None,  # per-seq EOS handled by the trim below
                    **start,
                )
        finally:
            if was_training:
                lm.train()
        gen = unpacked[1:].to(device=device, dtype=torch.long)  # [steps-1, B]; drop the start row
        resps: List[torch.Tensor] = []
        for i in range(n):
            col = gen[:, i]
            hit = (col == eos).nonzero(as_tuple=True)[0]
            end = int(hit[0]) if hit.numel() > 0 else int(col.shape[0])
            resps.append(col[:end].contiguous())
        return resps

    def replay(
        self,
        conditions: BagelARConditions,
        *,
        segment: TextSegment,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Per-token log-prob replay over a stored thinking segment.

        Recomputes π_new for every response token via the same teacher-forced
        packed und forward as ``autoregress`` (:meth:`_token_logp`). Returns a
        packed-varlen ``[total_tokens]`` tensor aligned with ``segment.log_probs``.
        Caller owns grad / ``.train()`` scope.
        """
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError(
                "BagelARStage.replay: segment requires tokens with framework-managed "
                "cu_seqlens (construct via TextSegment.pack)"
            )
        prompts = list(conditions.prompts)
        device = next(self.model.transformer.parameters()).device
        lengths = [int(n) for n in segment.lengths.tolist()]
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        if len(prompts) != len(lengths):
            raise ValueError(
                f"BagelARStage.replay: {len(prompts)} prompt(s) but {len(lengths)} segment sample(s)."
            )

        out: List[torch.Tensor] = []
        for b, prompt in enumerate(prompts):
            n = lengths[b]
            if n == 0:
                continue
            response = segment.tokens[cu[b] : cu[b] + n].to(device=device, dtype=torch.long)
            out.append(self._token_logp(prompt, response, temperature=temperature))
        if not out:
            return torch.zeros(0, dtype=self.logprob_dtype, device=device)
        return torch.cat(out, dim=0)


__all__ = ["BagelARParams", "BagelARStage"]
