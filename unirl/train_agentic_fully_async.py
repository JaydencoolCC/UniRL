#!/usr/bin/env python
"""UniRL fully-async (v5 fully-async resident pool) agentic training entry point.

Drives :class:`unirl.agentic.trainer_fully_async.FullyAsyncAgenticTrainer` — the colocate,
multi-turn think→gen→obs agentic RL trainer with a resident, version-aware
carry-over session pool (GRPO over think + FlowGRPO over image → one optimizer
step), bumping the policy version after each step so carried-over sessions span
versions (bounded by ``max_staleness``).

Launch (single node, 4xH20):
  DATA_PATH=/path/to/prompts.txt BAGEL_PATH=/path/to/BAGEL-7B-MoT \
  python -m unirl.train_agentic_fully_async --config-name=agentic/bagel_thinkgen_fully_async num_devices=4

Divisibility (DP_SCATTER): the resident pool harvests WHOLE groups per shard, so
keep ``batch_size`` divisible by ``num_devices`` (and
``batch_size*sessions_per_prompt*max_turns`` divisible by ``num_devices``).
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.agentic.trainer_fully_async import FullyAsyncAgenticTrainer


@hydra.main(version_base=None, config_path="../examples", config_name="agentic/bagel_thinkgen_fully_async")
def main(cfg: DictConfig) -> None:
    trainer = FullyAsyncAgenticTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        bundle_cfg=cfg.bundle,
        pipeline_cfg=cfg.pipeline,
        backend_cfg=cfg.backend,
        rollout_cfg=cfg.rollout,
        reward_cfg=cfg.reward,
        ar_algorithm_cfg=cfg.ar_algorithm,
        image_algorithm_cfg=cfg.image_algorithm,
        stack_cfg=cfg.stack,
        data_source_cfg=cfg.data_source,
        sampling_cfg=cfg.sampling,
        logging_cfg=cfg.get("logging"),
        sessions_per_prompt=int(cfg.get("sessions_per_prompt", 2)),
        max_turns=int(cfg.get("max_turns", 2)),
        over_sample_ratio=float(cfg.get("over_sample_ratio", 1.0)),
        reward_std_eps=float(cfg.get("reward_std_eps", 1e-6)),
        buffer_max_staleness=int(cfg.get("buffer_max_staleness", 0)),
        adv_normalization_scope=cfg.get("adv_normalization_scope", "group"),
        normalize_adv_by_std=bool(cfg.get("normalize_adv_by_std", True)),
        dump_dir=cfg.get("dump_dir"),
    )
    trainer.train(
        num_rollouts=int(cfg.get("num_rollouts", 100)),
        save_interval=int(cfg.get("save_interval", 0)),
        save_dir=cfg.get("save_dir"),
        load_dir=cfg.get("load_dir"),
        save_mode=str(cfg.get("save_mode", "auto")),
    )


if __name__ == "__main__":
    main()
