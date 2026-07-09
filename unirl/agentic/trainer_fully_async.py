"""FullyAsyncAgenticTrainer — fully-async (v5) fully-async resident pool trainer.

A thin subclass of the half-async :class:`~unirl.agentic.trainer.AsyncAgenticTrainer`
that pairs with :class:`~unirl.agentic.engine_fully_async.BagelAgenticEngineFullyAsync`. It reuses
the half-async collect → score → credit-assign → dual-algorithm step verbatim, and
adds only the two fully-async hooks the resident pool needs:

1. **Version bump after the optimizer step.** On colocate the rollout engine shares
   the live FSDP model, so the optimizer step *is* a weight update — the policy
   version must advance. After ``stack.train_track`` the trainer calls
   ``engine.bump_version()`` (broadcast), which increments the engine's version and
   aborts any resident session now too stale. Carry-over sessions advanced in the
   next ``generate()`` are then stamped with the new version → real version spread.

2. **Version / pool observability.** Logs the resident pool's metrics
   (admitted / aborted / carried-over / max version spread) and the trained
   batch's per-token version spread each rollout, so staleness can be tuned
   (design v5 §正确性论证: versions drive backpressure + metrics, never the loss).

Divisibility note: the fully-async engine harvests **whole groups** per DP shard, so a
group's ``N`` session rows must not be split across rollout workers. Keep
``batch_size`` (the number of prompt groups) divisible by the rollout DP size.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch

from unirl.agentic.trainer import AsyncAgenticTrainer
from unirl.distributed.tensor import hydrate
from unirl.train.stack import TrainStepResult

logger = logging.getLogger(__name__)


class FullyAsyncAgenticTrainer(AsyncAgenticTrainer):
    """Colocate fully-async agentic trainer (fully-async resident pool)."""

    def train_step(self, rollout_id: int, training_progress: float) -> Tuple[Dict[str, TrainStepResult], float]:
        # Reuse the proven half-async step (collect → score → advantages → dual
        # backward → one optimizer step); it drives the resident-pool engine
        # through the same generate() contract.
        results, mean_reward = super().train_step(rollout_id, training_progress)

        # Fully-async hooks: the shared live model just changed, so bump the policy
        # version and surface staleness observability. Wrapped so a metrics/RPC
        # hiccup can never kill the training loop.
        try:
            self.rollout.bump_version()
            self._weight_version += 1
            metrics = self._pool_metrics()
            metrics["version/policy_version"] = float(self._weight_version)
            logger.info(
                "[fully-async] rollout=%d version=%d pool{admitted=%.0f aborted=%.0f carried=%.0f "
                "running=%.0f max_spread=%.0f}",
                rollout_id,
                self._weight_version,
                metrics.get("pool/admitted", 0.0),
                metrics.get("pool/aborted", 0.0),
                metrics.get("pool/carried_over", 0.0),
                metrics.get("pool/running", 0.0),
                metrics.get("pool/max_version_spread", 0.0),
            )
            self._log_metrics_to_wandb(rollout_id, metrics)
        except Exception as exc:  # pragma: no cover - observability must never crash training
            logger.warning("[fully-async] version bump / metrics failed (non-fatal): %r", exc)

        return results, mean_reward

    def _pool_metrics(self) -> Dict[str, float]:
        """Drain the rollout engine's resident-pool metrics (broadcast → rank 0)."""
        pm = self.rollout.pool_metrics()
        if isinstance(pm, list):  # broadcast may return one dict per worker
            agg: Dict[str, float] = {}
            for d in pm:
                for k, v in (d or {}).items():
                    agg[k] = agg.get(k, 0.0) + float(hydrate(v) if torch.is_tensor(v) else v)
            return agg
        return {k: float(v) for k, v in (pm or {}).items()}

    def _log_metrics_to_wandb(self, rollout_id: int, metrics: Dict[str, float]) -> None:
        wl = getattr(self, "wandb_logger", None)
        if wl is None:
            return
        for meth in ("log_metrics", "log_scalars", "log"):
            fn = getattr(wl, meth, None)
            if callable(fn):
                try:
                    fn(metrics, step=rollout_id)
                    return
                except TypeError:
                    try:
                        fn(metrics)
                        return
                    except Exception:
                        pass


__all__ = ["FullyAsyncAgenticTrainer"]
