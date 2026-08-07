"""Eval-time diffusion sampling: the overlay resolver shared by every diffusion-capable trainer."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Optional, Set

from omegaconf import OmegaConf

from unirl.types.sampling import BaseSamplingParams


def cfg_scale_of(params: Any) -> float:
    """The CFG scale a diffusion params object will actually be sampled with.

    BAGEL-family params carry ``cfg_text_scale``; every other family carries
    ``guidance_scale``. Reading the scale through here keeps log lines and eval
    overrides on the same field the pipeline consumes.
    """
    scale = getattr(params, "cfg_text_scale", None)
    return float(params.guidance_scale if scale is None else scale)


def build_eval_sampling(
    sampling_params: Dict[str, BaseSamplingParams],
    *,
    cfg_text_scale: Optional[float] = None,
    eta: float = 0.0,
    samples_per_prompt: Optional[int] = None,
    overrides: Any = None,
) -> Dict[str, BaseSamplingParams]:
    """Return ``sampling_params`` with its ``diffusion`` entry rebuilt for evaluation.

    Eval INHERITS the training ``sampling:`` block and overlays only what the
    recipe asks for, later winning over earlier:

    1. ``eta`` — recipe ``eval_eta`` (default ``0.0``: deterministic ODE eval).
    2. ``samples_per_prompt`` when given — recipe ``eval_samples_per_prompt``.
    3. the CFG scale when ``cfg_text_scale`` is not None — recipe
       ``eval_cfg_text_scale``. ``None`` inherits the training guidance, so a
       CFG-off run cannot silently evaluate with CFG on.
    4. ``overrides`` — the recipe's ``eval_sampling:`` block: any
       :class:`~unirl.types.sampling.DiffusionSamplingParams` field
       (``num_inference_steps``, ``height`` / ``width``, ``seed``, ...). Unknown
       keys raise rather than being silently dropped.

    A resolved ``eta <= 0`` then clears the SDE gate (``sde_indices=[]``,
    ``scheduler=None``): eta=0 with gated steps is a contradictory request — the
    central kernel degrades such steps to ODE, and worker-resident schedulers
    (BAGEL) refuse the pair outright. Overlaying ``eta`` back above 0 keeps the
    training gate, so an SDE eval stays expressible.
    """
    base = sampling_params.get("diffusion")
    if base is None:
        raise ValueError("build_eval_sampling: sampling params carry no `diffusion` entry to override.")
    field_names = {f.name for f in dataclasses.fields(base)}

    updates: Dict[str, Any] = {"eta": float(eta)}
    if samples_per_prompt is not None:
        updates["samples_per_prompt"] = int(samples_per_prompt)
    if cfg_text_scale is not None:
        updates["cfg_text_scale" if "cfg_text_scale" in field_names else "guidance_scale"] = float(cfg_text_scale)
    updates.update(_resolve_overrides(overrides, field_names))
    if float(updates["eta"]) <= 0.0:
        updates["sde_indices"] = []
        updates["scheduler"] = None
    return {**sampling_params, "diffusion": dataclasses.replace(base, **updates)}


def _resolve_overrides(overrides: Any, field_names: Set[str]) -> Dict[str, Any]:
    """Validate a recipe ``eval_sampling:`` block into plain ``dataclasses.replace`` kwargs."""
    if overrides is None:
        return {}
    if OmegaConf.is_config(overrides):
        overrides = OmegaConf.to_container(overrides, resolve=True)
    if not isinstance(overrides, Mapping):
        raise TypeError(
            "eval_sampling must be a mapping of diffusion sampling fields, "
            f"got {type(overrides).__name__}. It overlays `sampling:`, so it takes no `_target_`."
        )
    unknown = sorted(set(overrides) - field_names)
    if unknown:
        raise ValueError(f"eval_sampling has unknown field(s) {unknown}; valid fields are {sorted(field_names)}.")
    return dict(overrides)
