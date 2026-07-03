r"""LLM-as-judge reward scorer (LIN-519).

Grades a free-form prediction against a reference answer by asking an
OpenAI-compatible chat endpoint whether they are equivalent, then parsing a
correct/incorrect verdict into ``1.0`` / ``0.0``. The text-reward path for
deep-research / open-ended QA where a symbolic verifier (math-verify) does not
apply.

The judge is hosted OUT-OF-BAND (a separate SGLang / vLLM OpenAI server, or any
OpenAI-compatible API); this scorer only POSTs to it, so no weight sync / GPU
placement is involved. ``base_device`` is ignored (HTTP, no local model). The
endpoint / model / key come from the recipe :class:`LLMJudgeSpec`, overridable at
load time by ``$JUDGE_URL`` / ``$JUDGE_MODEL`` / ``$JUDGE_API_KEY``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List

import requests

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend

logger = logging.getLogger(__name__)

_DEFAULT_JUDGE_PROMPT = (
    "You are an evaluation assistant. Decide whether the predicted answer is "
    "equivalent to the reference answer for the question. Minor differences in "
    "formatting or phrasing do not matter — only whether they convey the same "
    "answer.\n\n"
    "Question: {question}\n"
    "Reference answer: {answer}\n"
    "Predicted answer: {prediction}\n\n"
    "Reply with a single word: 'correct' if they are equivalent, otherwise "
    "'incorrect'."
)


class LLMJudgeRewardScorer(LocalRewardBackend):
    """Reward = 1.0 if an LLM judge deems the prediction equivalent to the
    reference answer, else 0.0. POSTs to an OpenAI-compatible chat endpoint."""

    canonical_model_name = "llm_judge"
    input_kind = "text"

    def __init__(self, *, config: "LLMJudgeSpec", base_device: str) -> None:
        del base_device  # HTTP backend — no local model / device
        self._spec = config
        super().__init__()  # sets self.model_name/batch_size/timeout; then calls _load_model

    def _load_model(self) -> None:
        # Nothing to load; resolve the endpoint (env override wins for secrets).
        self.model = "llm_judge"
        self._endpoint = os.environ.get("JUDGE_URL", self._spec.endpoint)
        self._judge_model = os.environ.get("JUDGE_MODEL", self._spec.model)
        self._api_key = os.environ.get("JUDGE_API_KEY", self._spec.api_key)
        if not self._endpoint:
            raise ValueError(
                "LLMJudgeRewardScorer requires an endpoint (LLMJudgeSpec.endpoint or $JUDGE_URL), "
                "e.g. http://<judge-host>:<port>/v1/chat/completions"
            )

    def _judge_one(self, question: str, prediction: str, answer: str) -> float:
        prompt = self._spec.prompt_template.format(
            question=question,
            answer=answer,
            prediction=(prediction or "")[: self._spec.max_prediction_chars],
        )
        payload = {
            "model": self._judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": self._spec.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        for attempt in range(self._spec.max_retries):
            try:
                resp = requests.post(self._endpoint, json=payload, headers=headers, timeout=self._spec.timeout)
                resp.raise_for_status()
                content = str(resp.json()["choices"][0]["message"]["content"]).strip().lower()
                # "incorrect" contains "correct" as a substring — test it first.
                return 1.0 if ("incorrect" not in content and "correct" in content) else 0.0
            except Exception as exc:  # noqa: BLE001 — judge failures score 0, never crash training
                logger.warning("LLM judge attempt %d/%d failed: %s", attempt + 1, self._spec.max_retries, exc)
        logger.warning("LLM judge failed after %d attempts; scoring 0.", self._spec.max_retries)
        return 0.0

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        predictions = request.texts
        if predictions is None:
            raise ValueError("LLMJudgeRewardScorer requires request.texts (predicted answers).")
        prompts = request.prompts or [""] * len(predictions)
        metadata_list = request.metadata or [None] * len(predictions)
        rewards: List[float] = []
        for question, prediction, meta in zip(prompts, predictions, metadata_list):
            if meta is None or meta.get("answer") is None:
                rewards.append(0.0)
                continue
            rewards.append(self._judge_one(question, prediction, str(meta["answer"])))
        return rewards


@dataclass
class LLMJudgeSpec(BaseRewardComponentSpec):
    """Config for the LLM-as-judge scorer. ``endpoint`` / ``model`` / ``api_key``
    are overridable at load time by ``$JUDGE_URL`` / ``$JUDGE_MODEL`` /
    ``$JUDGE_API_KEY`` (keep secrets out of the recipe)."""

    endpoint: str = ""  # OpenAI-compatible /v1/chat/completions URL
    model: str = ""
    api_key: str = ""
    timeout: float = 60.0
    max_tokens: int = 512
    max_retries: int = 3
    max_prediction_chars: int = 4000
    prompt_template: str = _DEFAULT_JUDGE_PROMPT
