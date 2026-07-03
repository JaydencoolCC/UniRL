"""VisitTool — read webpage(s) and summarize toward a goal (LIN-519).

A concrete :class:`~unirl.rollout.loop.tools.tool.Tool` for the deep-research
agent: fetch a URL's content with the Jina reader (needs ``$JINA_API_KEYS``) and
summarize the parts relevant to a stated goal with an OpenAI-compatible LLM
(hosted out-of-band; ``$SUMMARY_URL`` / ``$SUMMARY_MODEL`` or the constructor
args — the same endpoint as the judge works). ``execute`` is synchronous and
thread-safe so it runs under :meth:`ToolEnvironment.astep`'s executor across
concurrent trajectories. If no summarizer is configured it returns the truncated
raw page content, so the tool is usable without a summarizer for smoke tests.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import requests

from unirl.rollout.loop.tools.tool import Tool

_JINA_READ = "https://r.jina.ai/"
_EXTRACT_PROMPT = (
    "Extract and summarize the information from the webpage that is relevant to the goal. "
    "Keep concrete facts, figures, dates, and quotes; drop boilerplate.\n\n"
    "## Goal\n{goal}\n\n## Webpage content\n{content}\n\n## Summary:"
)


class VisitTool(Tool):
    """Visit URL(s) via the Jina reader and summarize the content toward a goal.
    Requires ``$JINA_API_KEYS``; the summarizer endpoint comes from ``$SUMMARY_URL``
    / ``$SUMMARY_MODEL`` or the constructor args."""

    name = "visit"

    def __init__(
        self,
        *,
        endpoint: str = "",
        model: str = "",
        timeout: float = 60.0,
        max_content_chars: int = 60000,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self._timeout = float(timeout)
        self._max_content_chars = int(max_content_chars)

    def json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Visit webpage(s) and return a summary of the content relevant to a goal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": ["string", "array"],
                            "items": {"type": "string"},
                            "description": "The URL, or an array of URLs, to visit.",
                        },
                        "goal": {
                            "type": "string",
                            "description": "The specific information to extract from the page(s).",
                        },
                    },
                    "required": ["url", "goal"],
                },
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        url = arguments.get("url")
        goal = str(arguments.get("goal", ""))
        urls: List[str] = [url] if isinstance(url, str) else list(url or [])
        if not urls:
            raise ValueError("visit requires a 'url' string or array")
        return "\n=======\n".join(self._visit_one(str(u), goal) for u in urls)

    def _visit_one(self, url: str, goal: str) -> str:
        content = self._read(url)
        if content.startswith("[visit] "):
            return f"{url}: {content}"
        summary = self._summarize(content[: self._max_content_chars], goal)
        return f"Content of {url} for goal {goal!r}:\n{summary}"

    def _read(self, url: str) -> str:
        key = os.environ.get("JINA_API_KEYS", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        try:
            resp = requests.get(_JINA_READ + url, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # noqa: BLE001 — surfaced to the model as text
            return f"[visit] failed to read {url}: {exc}"

    def _summarize(self, content: str, goal: str) -> str:
        endpoint = os.environ.get("SUMMARY_URL", self._endpoint)
        model = os.environ.get("SUMMARY_MODEL", self._model)
        if not endpoint:
            return content  # no summarizer configured — return raw (truncated) content
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": _EXTRACT_PROMPT.format(goal=goal, content=content)}],
            "temperature": 0.2,
        }
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("SUMMARY_API_KEY", "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            return str(resp.json()["choices"][0]["message"]["content"]).strip()
        except Exception:  # noqa: BLE001 — fall back to raw content on summarizer failure
            return content
