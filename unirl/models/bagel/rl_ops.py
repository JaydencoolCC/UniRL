"""Navit-forward adapter over the PRISTINE official Bagel modeling.

The official ``ByteDance-Seed/Bagel`` ``_forward_flow`` is the velocity predictor
the RL path needs, but it (a) consumes a *packed* (navit) sequence + three KV-cache
contexts rather than a dense ``predict_noise(sample, sigma)`` and (b) carries an
upstream ``@torch.no_grad``. This module is the **thin adapter** that bridges those
two facts to UniRL's shared diffusion runtime — and nothing more:

- :func:`forward_flow`           grad-capable velocity via the pristine
                                 ``Bagel._forward_flow`` (bypasses ``@torch.no_grad``
                                 through ``functools.wraps``' ``__wrapped__``).
- :func:`disable_inference_cache` turns off TaylorSeer (per-step determinism for replay).

Everything else the RL loop needs is UniRL's, NOT a flow_grpo port:

- the SDE transition + log-prob  → :class:`unirl.sde.kernels.FlowSDEStrategy`
- which steps run SDE            → :meth:`DiffusionSamplingParams.resolve_sde_indices`
                                   (``unirl.utils.scheduler_utils.AllSDEScheduler``)
- the σ / timestep schedule      → :class:`unirl.sde.runtime.FlowMatchSchedulePolicy`
- the initial noise x_T          → :class:`unirl.types.noise_recipe.NoiseRecipe`

so :class:`unirl.models.bagel.diffusion.BagelDiffusionStage` reads exactly like
``SD3DiffusionStage`` (central schedule + sde_indices + kernel + noise), with this
adapter supplying only the model-specific velocity call. ``vendor/`` stays
byte-pristine; an upstream bump is a re-vendor + import-rewrite with this file
untouched.

Gradients
---------
``Bagel._forward_flow`` carries ``@torch.no_grad`` upstream. :func:`forward_flow`
reaches the undecorated function via ``functools.wraps``' ``__wrapped__`` so replay
can backprop while the vendored file stays unedited (verified on torch 2.11: the
decorated form blocks grad even under ``enable_grad``; ``__wrapped__`` restores it).
Under an outer ``torch.no_grad()`` (e.g. rollout) it stays grad-free, so the same
function serves rollout, the ratio test, and training.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "disable_inference_cache",
    "forward_flow",
    "und_forward",
]


def disable_inference_cache(model: Any) -> None:
    """Turn off the TaylorSeer cache for the RL path (per-step determinism).

    The pristine ``_forward_flow`` reads ``self.language_model.model.enable_taylorseer``;
    the official ``generate_image`` sets it, but the RL loop calls ``_forward_flow``
    directly so we set the flag here (the cache would break per-step determinism →
    replay would not be bit-exact). Best-effort; ignored if the attribute path is
    absent (e.g. a fake model in unit tests).
    """
    try:
        model.language_model.model.enable_taylorseer = False
    except AttributeError:
        pass


def _raw_forward_flow(model: Any):
    """The undecorated ``Bagel._forward_flow`` (bypasses upstream ``@torch.no_grad``)."""
    fn = type(model)._forward_flow
    return getattr(fn, "__wrapped__", fn)


def forward_flow(model: Any, **kwargs: Any) -> Any:
    """Velocity prediction via the pristine vendored ``Bagel._forward_flow``.

    Bypasses upstream's ``@torch.no_grad`` (via ``__wrapped__``) so gradients flow
    during replay; under an outer ``torch.no_grad()`` it is still grad-free. The
    TaylorSeer cache kwargs (``model_pred_*``) are left at their ``None`` defaults —
    the RL path disables that cache (see :func:`disable_inference_cache`).

    ``model._forward_flow`` already does the CFG combine internally (gen / cfg_text /
    cfg_img contexts + ``cfg_text_scale`` / ``cfg_img_scale`` / ``cfg_renorm_*``), so
    the returned velocity is the CFG-combined ``v_t`` the SDE kernel consumes.

    Training-mode contract (mirror of :func:`und_forward`, opposite polarity): the
    vendored decoder layer dispatches train vs inference on ``self.training``, and
    ``_forward_flow`` goes through the ``forward_inference`` (packed-query) signature,
    so the language model MUST be in ``eval()`` here. The two stages share one MoT
    instance within a single optimizer step (AR teacher-force sets train(); this
    diffusion replay needs eval()), so we cannot rely on the inherited mode. Force
    eval; under grad (replay) KEEP it eval so activation-checkpointing's recompute in
    the LATER ``.backward()`` still takes ``forward_inference`` (reverting to a stray
    train() would dispatch the packed-query kwargs into ``forward_train`` →
    "unexpected keyword argument 'packed_query_sequence'"). Restore only when no
    backward follows (rollout / no_grad).
    """
    lm = model.language_model
    was_training = lm.training
    grad_enabled = torch.is_grad_enabled()
    if was_training:
        lm.eval()
    try:
        return _raw_forward_flow(model)(model, **kwargs)
    finally:
        if was_training and not grad_enabled:
            lm.train()


def und_forward(model: Any, packed_text_ids: torch.Tensor) -> torch.Tensor:
    """Grad-capable packed text-only (und) forward -> last_hidden_state ``[L, H]``.

    Teacher-forced scoring path for the BAGEL reasoning policy: a single navit
    sample (``bs=1``), fully causal, run through the und (text) experts of the
    MoT transformer. Used by :meth:`BagelARStage.replay` (and the ARGRPO
    ``old_logp_source="replay"`` anchor) to recompute per-token log-probs with
    gradient flow through BOTH the prompt and the response positions.

    Drives the per-sample **List-mask** branch of the navit attention (a dense
    additive causal mask -> ``scaled_dot_product_attention``), so it never hits
    the module-level ``torch.compile``-d ``flex_attention`` block-mask path.

    The vendored modules dispatch ``forward_train`` vs ``forward_inference`` on
    ``module.training`` recursively, so the transformer must be in ``train()``
    mode for the packed training forward to run (the steady state inside the
    train stack). A defensive train()/restore guard keeps a stray eval-mode call
    from silently taking the wrong-signature inference path.
    """
    lm = model.language_model
    device = next(lm.parameters()).device
    ids = packed_text_ids.to(device=device, dtype=torch.long)
    seq_len = int(ids.shape[0])

    embed = lm.model.embed_tokens(ids)  # [L, H]
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
    # Dense additive causal mask [L, L]: 0 on/below the diagonal, -inf above.
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
    und_idx = torch.arange(seq_len, device=device, dtype=torch.long)

    was_training = lm.training
    grad_enabled = torch.is_grad_enabled()
    if not was_training:
        lm.train()
    try:
        hidden = lm(
            packed_sequence=embed,
            sample_lens=[seq_len],
            attention_mask=[causal_mask],
            packed_position_ids=position_ids,
            packed_und_token_indexes=und_idx,
            packed_gen_token_indexes=None,
        )
    finally:
        # Restore eval ONLY when no backward will follow (rollout / no_grad
        # pi_old scoring). Under grad (training replay) the model MUST stay in
        # train() because activation-checkpointing recomputes this forward during
        # the LATER .backward() — outside this scope — and the decoder layer
        # dispatches train vs inference on ``self.training``. Reverting to eval
        # here makes that recompute take ``forward_inference`` with the packed
        # ``forward_train`` kwargs → "unexpected keyword argument 'packed_sequence'".
        # The next rollout's engine.generate() resets eval() before sampling, and
        # dropout=0 / RMSNorm (no running stats) make train vs eval numerically
        # identical here — only the dispatch path differs.
        if not was_training and not grad_enabled:
            lm.eval()
    return hidden
