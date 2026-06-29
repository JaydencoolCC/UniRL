"""Tool — the abstract interface a ``ToolEnvironment`` dispatches to (LIN-492).

A ``Tool`` is the two things the environment needs: a JSON schema (so the rollout prompt can
advertise the tool to the model via ``tokenizer.apply_chat_template(tools=...)``) and an executor
(run the parsed call, return a text result). One concrete tool per module; see
:class:`~unirl.rollout.loop.tools.calculator.CalculatorTool`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class Tool(ABC):
    """A single callable tool: a name, a JSON function-schema, and an executor."""

    #: The tool name the model emits in ``<tool_call>{"name": ...}</tool_call>``.
    name: str

    @abstractmethod
    def json_schema(self) -> Dict[str, Any]:
        """The OpenAI function-tool schema — the shape ``apply_chat_template(tools=[...])`` consumes:
        ``{"type": "function", "function": {"name", "description", "parameters": {JSON Schema}}}``."""
        ...

    @abstractmethod
    def execute(self, arguments: Dict[str, Any]) -> str:
        """Run the tool on parsed ``arguments`` and return the result as text.

        May raise on bad input — :class:`~unirl.rollout.loop.tool_environment.ToolEnvironment`
        catches and surfaces the error to the model as the observation, so the policy can recover.
        """
        ...


__all__ = ["Tool"]
