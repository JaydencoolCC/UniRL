"""SFT domain package — worker-side supervised data producers.

The losses live in ``unirl/algorithms/sft.py`` (peers of GRPO/FlowGRPO); the
driver is ``unirl/trainer/sft.py``; this package holds only the piece that is
genuinely new to supervision: turning dataset records into stage-ready tracks.
"""

from unirl.train.sft.source import ARSupervisedSource, DiffusionSupervisedSource

__all__ = ["ARSupervisedSource", "DiffusionSupervisedSource"]
