"""Half-async agentic RL infrastructure for UniRL.

A general-purpose, multi-turn agentic RL layer built on the proven
:class:`~unirl.trainer.async_ar.AsyncARTrainer` half-async model and the
multi-track ``RolloutResp`` lineage. P0 target: Bagel think→gen→obs training
(plan jolly-sprouting-music.md).

Only the **backend-agnostic core** is re-exported here so ``import
unirl.agentic`` stays CPU-clean (no flash_attn / vendored modeling). The Bagel
engine and the trainer pull heavy model deps and are imported directly:

    from unirl.agentic.engine import BagelAgenticEngine   # needs the Bagel bundle
    from unirl.agentic.trainer import AsyncAgenticTrainer
"""

from unirl.agentic.buffer import AgenticBuffer
from unirl.agentic.config import (
    IMAGE_TRACK,
    THINK_TRACK,
    AgenticAsyncConfig,
    AgenticWorkflowConfig,
)
from unirl.agentic.env import AgenticEnv, Observation, RuleEnv
from unirl.agentic.pool import FullyAsyncPool
from unirl.agentic.session import MsgNode, NodeKind, Session
from unirl.agentic.version import should_admit, staleness_capacity, version_spread
from unirl.agentic.workflow import AgenticBackend, AgenticWorkflow, ThinkGenWorkflow

__all__ = [
    "AgenticAsyncConfig",
    "AgenticBackend",
    "AgenticBuffer",
    "AgenticEnv",
    "AgenticWorkflow",
    "AgenticWorkflowConfig",
    "IMAGE_TRACK",
    "MsgNode",
    "NodeKind",
    "Observation",
    "RuleEnv",
    "Session",
    "THINK_TRACK",
    "ThinkGenWorkflow",
    "FullyAsyncPool",
    "should_admit",
    "staleness_capacity",
    "version_spread",
]
