"""Tools a :class:`~unirl.rollout.loop.tool_environment.ToolEnvironment` dispatches to (LIN-492).

One class per module: :class:`~unirl.rollout.loop.tools.tool.Tool` (the interface) and
:class:`~unirl.rollout.loop.tools.calculator.CalculatorTool` (the first concrete tool).
"""

from unirl.rollout.loop.tools.calculator import CalculatorTool
from unirl.rollout.loop.tools.tool import Tool

__all__ = ["Tool", "CalculatorTool"]
