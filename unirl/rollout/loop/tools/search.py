"""SearchTool — batched web search via Serper (LIN-519).

A concrete :class:`~unirl.rollout.loop.tools.tool.Tool` for the deep-research
agent: given an array of query strings it returns the top web results per query
as text. Uses the Serper Google Search API (needs ``$SERPER_KEY_ID``). ``execute``
is synchronous and thread-safe (it holds no state) so it runs cleanly under
:meth:`ToolEnvironment.astep`'s executor across concurrent trajectories.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import requests

from unirl.rollout.loop.tools.tool import Tool

_SERPER_URL = "https://google.serper.dev/search"


class SearchTool(Tool):
    """Google web search via Serper. Accepts one or more queries; returns the top
    results per query as text. Requires ``$SERPER_KEY_ID``."""

    name = "search"

    def __init__(self, *, top_k: int = 10, timeout: float = 30.0) -> None:
        self._top_k = int(top_k)
        self._timeout = float(timeout)

    def json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Search the web. Provide an array of query strings; returns the top "
                    "results for each query. Use multiple complementary queries in one call."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "One or more search query strings.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        queries = arguments.get("query")
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, list) or not queries:
            raise ValueError("search requires a non-empty 'query' string or array")
        return "\n=======\n".join(self._search_one(str(q)) for q in queries)

    def _search_one(self, query: str) -> str:
        headers = {"X-API-KEY": os.environ.get("SERPER_KEY_ID", ""), "Content-Type": "application/json"}
        try:
            resp = requests.post(_SERPER_URL, json={"q": query}, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 — surfaced to the model as text, not raised
            return f"[search] error for {query!r}: {exc}"
        organic = data.get("organic") or []
        if not organic:
            return f"No results for {query!r}. Try a less specific query."
        lines: List[str] = []
        for i, page in enumerate(organic[: self._top_k], start=1):
            title = page.get("title", "")
            link = page.get("link", "")
            snippet = page.get("snippet", "")
            lines.append(f"{i}. [{title}]({link})\n{snippet}")
        return f"Results for {query!r}:\n" + "\n\n".join(lines)
