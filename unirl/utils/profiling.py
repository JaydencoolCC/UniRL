"""Opt-in torch.profiler harness for the worker-side training step.

UniRL is an RL framework (rollout → reward → advantage → optimizer step). To
profile *only* the training compute (forward / loss / backward / optimizer) of a
backend — without the SGLang/vLLM rollout — wrap the per-rollout train call on
the worker with :class:`TrainStepProfiler`. The rollout runs in a separate engine
phase, so the profiled region here is the pure train step.

Entirely env-gated; a no-op unless ``UNIRL_PROFILE`` is set, so it can ship in the
hot path. ONE switch selects a fully-configured preset:

* ``UNIRL_PROFILE=overlap`` — one optimizer update only (backward + FSDP comm +
  optimizer), CUDA on, memory/shapes off, rank0. Best for compute/comm OVERLAP
  analysis; small trace. Output goes to ``outputs/profiler`` (or ``UNIRL_PROFILE_DIR``).
* ``UNIRL_PROFILE=1`` (or ``step``) — whole train step (anchor forward + updates).

Everything below is auto-set by the preset but can still be overridden:

* ``UNIRL_PROFILE``         — ``overlap`` / ``1`` / ``step`` to enable (default off).
* ``UNIRL_PROFILE_DIR``     — trace output dir (default ``outputs/profiler``).
* ``UNIRL_PROFILE_RANKS``   — ``0`` (default), ``all``, or a comma list ``0,8``.
* ``UNIRL_PROFILE_WAIT``    — schedule wait steps   (default ``2``).
* ``UNIRL_PROFILE_WARMUP``  — schedule warmup steps (default ``2``).
* ``UNIRL_PROFILE_ACTIVE``  — schedule active steps (default ``3``).
* ``UNIRL_PROFILE_REPEAT``  — schedule repeat cycles (default ``1``).
* ``UNIRL_PROFILE_MEMORY``  — record CUDA mem (default ``1``).
* ``UNIRL_PROFILE_SHAPES``  — record op input shapes (default ``1``).
* ``UNIRL_PROFILE_STACK``   — record python stacks (default ``0``; heavy).

One profiler ``step()`` = one rollout's train call. After ``(wait+warmup+active)*
repeat`` steps the trace is exported (Chrome/TensorBoard JSON, one file per rank
worker) and the profiler stops itself — later steps are cheap no-ops.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator, Optional

import torch

logger = logging.getLogger(__name__)


def _truthy(value: Optional[str], *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("profiling: %s=%r is not an int; using default %d", name, raw, default)
        return default


def _rank_enabled(rank: int) -> bool:
    spec = os.environ.get("UNIRL_PROFILE_RANKS", "0").strip().lower()
    if spec in ("all", "*"):
        return True
    try:
        return rank in {int(p) for p in spec.split(",") if p.strip()}
    except ValueError:
        logger.warning("profiling: UNIRL_PROFILE_RANKS=%r unparseable; defaulting to rank 0 only", spec)
        return rank == 0


def profile_mode() -> str:
    """Resolve the single-switch mode from ``UNIRL_PROFILE``.

    A single env var is all a caller needs: ``UNIRL_PROFILE=overlap`` auto-configures
    the whole compute/comm-overlap preset (one optimizer update, CUDA on, memory/shapes
    off, rank0, local output dir); ``UNIRL_PROFILE=1``/``step``/``true`` is the whole-step
    preset; anything falsy/unset is off. Individual ``UNIRL_PROFILE_*`` vars still override.
    """
    v = os.environ.get("UNIRL_PROFILE", "").strip().lower()
    if v in ("", "0", "false", "no", "off"):
        return "off"
    return "overlap" if v == "overlap" else "step"


def profile_enabled() -> bool:
    return profile_mode() != "off"


def profile_scope() -> str:
    """``step`` (whole train_track) or ``update`` (one optimizer update, for overlap).

    Explicit ``UNIRL_PROFILE_SCOPE`` wins; otherwise ``overlap`` mode implies ``update``.
    """
    s = os.environ.get("UNIRL_PROFILE_SCOPE", "").strip().lower()
    if s in ("step", "update"):
        return s
    return "update" if profile_mode() == "overlap" else "step"


def _out_dir() -> str:
    """Trace output dir. Defaults to ``outputs/profiler`` (relative to cwd, created if
    missing); set ``UNIRL_PROFILE_DIR`` to write elsewhere. NOTE: writing a very large
    (multi-GB) trace to a network FS can stall the export — point ``UNIRL_PROFILE_DIR``
    at a node-local path for whole-step captures."""
    return os.environ.get("UNIRL_PROFILE_DIR", "").strip() or "outputs/profiler"


class TrainStepProfiler:
    """Thin wrapper: a torch profiler stepped once per rollout train call."""

    def __init__(self, prof: "torch.profiler.profile", total_steps: int, out_dir: str) -> None:
        self._prof = prof
        self._total = total_steps
        self._out_dir = out_dir
        self._n = 0
        self._stopped = False
        prof.start()

    def step(self) -> None:
        """Advance the schedule by one rollout; auto-stop + export after the window."""
        if self._stopped:
            return
        self._prof.step()
        self._n += 1
        if self._n >= self._total:
            self._prof.stop()
            self._stopped = True
            logger.info("TrainStepProfiler: %d steps profiled; trace written to %s", self._n, self._out_dir)

    @contextmanager
    def record(self, name: str) -> Iterator[None]:
        with torch.profiler.record_function(name):
            yield


def maybe_build_train_profiler(rank: int) -> Optional[TrainStepProfiler]:
    """Build a :class:`TrainStepProfiler` from env, or ``None`` if disabled.

    Called lazily on the worker the first time a train step runs (so the device
    is bound and the profiler attaches to the right CUDA context).
    """
    if not profile_enabled():
        return None
    # The caller passes a backend-specific rank that is 0 on every worker for some
    # backends (e.g. FSDP colocate lacks `_rank`). Prefer the true global rank from
    # the process group so UNIRL_PROFILE_RANKS actually restricts to one worker —
    # profiling every rank makes 8 CUPTI trace-flushes contend and stall the export.
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    if not _rank_enabled(int(rank)):
        return None

    wait = _int_env("UNIRL_PROFILE_WAIT", 2)
    warmup = _int_env("UNIRL_PROFILE_WARMUP", 2)
    active = _int_env("UNIRL_PROFILE_ACTIVE", 3)
    repeat = _int_env("UNIRL_PROFILE_REPEAT", 1)
    out_dir = _out_dir()
    os.makedirs(out_dir, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    # CUDA (CUPTI) activity can be disabled with UNIRL_PROFILE_CUDA=0. On some
    # torch/driver/CUPTI combos the CUDA kineto trace-finalize (stop_trace) hangs
    # the export; a CPU-only trace still opens in Perfetto and shows the step
    # structure + cudaLaunchKernel timeline.
    if _truthy(os.environ.get("UNIRL_PROFILE_CUDA"), default=True) and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    sched = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    prof = torch.profiler.profile(
        activities=activities,
        schedule=sched,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(out_dir, worker_name=f"rank{int(rank)}", use_gzip=True),
        record_shapes=_truthy(os.environ.get("UNIRL_PROFILE_SHAPES"), default=True),
        profile_memory=_truthy(os.environ.get("UNIRL_PROFILE_MEMORY"), default=True),
        with_stack=_truthy(os.environ.get("UNIRL_PROFILE_STACK"), default=False),
    )
    total = max(1, (wait + warmup + active) * max(1, repeat))
    logger.info(
        "TrainStepProfiler[rank%d]: enabled (wait=%d warmup=%d active=%d repeat=%d) -> %s",
        int(rank),
        wait,
        warmup,
        active,
        repeat,
        out_dir,
    )
    return TrainStepProfiler(prof, total_steps=total, out_dir=out_dir)


@contextmanager
def maybe_profile_update(owner, rank: int) -> Iterator[None]:
    """One-shot profiler around a SINGLE ``_run_update`` (``UNIRL_PROFILE_SCOPE=update``).

    torch.profiler records continuously while active, so the schedule-based
    :class:`TrainStepProfiler` (which spans a whole rollout) always sweeps in the big
    SDE-replay ``prepare_segment`` too. For compute/comm OVERLAP analysis we want just
    one optimizer update — backward + FSDP reduce-scatter/all-gather + optimizer_step.
    This wraps exactly that region in its own profiler and exports immediately, so the
    trace is small and contains only the overlap-relevant window.

    Fires once, on rank0 (true global rank), after skipping ``UNIRL_PROFILE_WARMUP``
    updates (default 2) so the profiled step is past first-iter compile/allocation.
    A no-op context otherwise.
    """
    enabled = profile_enabled()
    if enabled:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        enabled = _rank_enabled(int(rank))

    n = getattr(owner, "_prof_update_seen", 0)
    owner._prof_update_seen = n + 1
    skip = _int_env("UNIRL_PROFILE_WARMUP", 2)
    if not enabled or getattr(owner, "_prof_update_done", False) or n != skip:
        yield
        return

    out_dir = _out_dir()
    os.makedirs(out_dir, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if _truthy(os.environ.get("UNIRL_PROFILE_CUDA"), default=True) and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    prof = torch.profiler.profile(
        activities=activities,
        record_shapes=_truthy(os.environ.get("UNIRL_PROFILE_SHAPES"), default=False),
        profile_memory=_truthy(os.environ.get("UNIRL_PROFILE_MEMORY"), default=False),
        with_stack=_truthy(os.environ.get("UNIRL_PROFILE_STACK"), default=False),
    )
    logger.info("maybe_profile_update[rank%d]: profiling one optimizer update -> %s", int(rank), out_dir)
    prof.start()
    try:
        yield
    finally:
        prof.stop()
        owner._prof_update_done = True
        raw = os.path.join(out_dir, f"update_rank{int(rank)}.pt.trace.json")
        prof.export_chrome_trace(raw)
        # gzip in place -> opens directly in Perfetto and is small enough to download
        import gzip
        import shutil

        out = raw + ".gz"
        with open(raw, "rb") as fin, gzip.open(out, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        os.remove(raw)
        logger.info("maybe_profile_update[rank%d]: trace written to %s", int(rank), out)


__all__ = [
    "TrainStepProfiler",
    "maybe_build_train_profiler",
    "maybe_profile_update",
    "profile_mode",
    "profile_enabled",
    "profile_scope",
]
