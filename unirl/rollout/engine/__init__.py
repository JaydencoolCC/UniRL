"""Rollout engines over the canonical ``Sample`` request type.

Two halves of one design: ``synchronous.py`` records the worker-side sync contracts
(``BaseRolloutEngine`` — the broad ABC including coordinator engines — and
``SyncRolloutEngine``, the ``Sample`` → ``Sample`` refinement the per-backend
subpackages implement); ``asynchronous.py`` records the driver side (the
``AsyncRolloutEngine`` protocol, its batch/agentic engines, and their
mechanisms).

Re-exports are lazy so importing the driver-side module stays ray/torch-free.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # worker-side sync contracts
    "BaseRolloutEngine": ("unirl.rollout.engine.synchronous", "BaseRolloutEngine"),
    "SyncRolloutEngine": ("unirl.rollout.engine.synchronous", "SyncRolloutEngine"),
    "chunked_engine_generate": ("unirl.rollout.engine.synchronous", "chunked_engine_generate"),
    # driver-side async engines
    "AsyncRolloutEngine": ("unirl.rollout.engine.asynchronous", "AsyncRolloutEngine"),
    "AsyncBatchRolloutEngine": ("unirl.rollout.engine.asynchronous", "AsyncBatchRolloutEngine"),
    "AsyncAgenticRolloutEngine": ("unirl.rollout.engine.asynchronous", "AsyncAgenticRolloutEngine"),
}

__all__ = list(_LAZY_ATTRS.keys())


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
