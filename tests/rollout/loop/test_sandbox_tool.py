"""CPU tests for SandboxTool — persistent Python-REPL subprocess (LIN-533).

Drives the StatefulTool lifecycle directly (no engine): ``session_start`` -> ``execute_session``* ->
``session_end``. Verifies cross-turn variable persistence (the point of a stateful tool), per-session
isolation, lazy subprocess spawn, errors/timeouts surfaced as text, and that ``session_end`` kills
the process idempotently. Each test spawns real subprocesses and tears them down in a ``finally``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")  # keep the loop suite uniform (SandboxTool itself needs no torch)

from unirl.rollout.loop.tools.sandbox import SandboxTool  # noqa: E402


def _run(tool: SandboxTool, sid: str, code: str) -> str:
    return tool.execute_session(sid, {"code": code})


def test_cross_turn_state_persists():
    """The core capability: a variable bound on one turn is visible on the next."""
    tool = SandboxTool()
    tool.session_start("s0", {})
    try:
        _run(tool, "s0", "x = 40")
        assert "42" in _run(tool, "s0", "x + 2")  # x survived across calls; trailing expr echoed
    finally:
        tool.session_end("s0")


def test_sessions_are_isolated():
    tool = SandboxTool()
    tool.session_start("a", {})
    tool.session_start("b", {})
    try:
        _run(tool, "a", "x = 40")
        assert "NameError" in _run(tool, "b", "x")  # b's interpreter never saw x
    finally:
        tool.session_end("a")
        tool.session_end("b")


def test_subprocess_opens_lazily():
    tool = SandboxTool()
    tool.session_start("s0", {})
    try:
        assert tool._sessions["s0"].proc is None  # no subprocess until the first execute
        _run(tool, "s0", "1 + 1")
        assert tool._sessions["s0"].proc is not None  # spawned lazily, in-executor
    finally:
        tool.session_end("s0")


def test_error_is_surfaced_and_session_survives():
    tool = SandboxTool()
    tool.session_start("s0", {})
    try:
        assert "ZeroDivisionError" in _run(tool, "s0", "1 / 0")  # returned as text, not raised
        assert "2" in _run(tool, "s0", "1 + 1")  # the REPL survives the exception
    finally:
        tool.session_end("s0")


def test_timeout_kills_and_surfaces():
    tool = SandboxTool(timeout_s=0.5)
    tool.session_start("s0", {})
    try:
        out = _run(tool, "s0", "while True:\n    pass")
        assert "timed out" in out.lower()  # does not hang; error surfaced
    finally:
        tool.session_end("s0")


def test_session_end_kills_process_and_is_idempotent():
    tool = SandboxTool()
    tool.session_start("s0", {})
    _run(tool, "s0", "1 + 1")  # spawn
    proc = tool._sessions["s0"].proc
    assert proc is not None and proc.poll() is None  # alive

    tool.session_end("s0")
    assert proc.poll() is not None  # killed
    assert "s0" not in tool._sessions

    tool.session_end("s0")  # idempotent — already gone
    tool.session_end("never-started")  # no-op on an unknown id


def test_empty_code_raises_for_toolenvironment_to_catch():
    tool = SandboxTool()
    tool.session_start("s0", {})
    try:
        with pytest.raises(ValueError):
            tool.execute_session("s0", {"code": "   "})
    finally:
        tool.session_end("s0")
