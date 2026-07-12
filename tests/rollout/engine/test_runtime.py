"""Contract tests for engine-owned local coroutine sessions."""

import asyncio
import threading

import pytest

from unirl.rollout.engine.runtime import LocalAsyncRuntime
from unirl.rollout.engine.sglang.backends.base import SessionRunner


def test_runtime_runs_factory_and_closes_idempotently():
    runtime = LocalAsyncRuntime()

    async def value() -> int:
        await asyncio.sleep(0)
        return 7

    assert runtime.run(value) == 7
    runtime.close()
    runtime.close()


def test_closed_runtime_rejects_before_constructing_coroutine():
    runtime = LocalAsyncRuntime()
    runtime.close()
    constructed = False

    async def value() -> int:
        return 7

    def factory():
        nonlocal constructed
        constructed = True
        return value()

    with pytest.raises(RuntimeError, match="closed"):
        runtime.run(factory)
    assert constructed is False


def test_runtime_rejects_nested_drive_before_constructing_coroutine():
    runtime = LocalAsyncRuntime()
    constructed = False

    async def inner() -> None:
        nonlocal constructed

        async def value() -> int:
            return 7

        def factory():
            nonlocal constructed
            constructed = True
            return value()

        with pytest.raises(RuntimeError, match="running event loop"):
            runtime.run(factory)
        with pytest.raises(RuntimeError, match="running event loop"):
            runtime.close()

    runtime.run(inner)
    assert constructed is False
    runtime.close()


def test_sglang_session_does_not_construct_idle_control():
    loop = asyncio.new_event_loop()
    session = SessionRunner(loop, label="test")
    constructed = False

    async def control() -> None:
        return None

    def factory():
        nonlocal constructed
        constructed = True
        return control()

    assert session.run_control(factory) is None
    assert constructed is False
    loop.close()


def test_sglang_session_interleaves_control_with_active_operation():
    loop = asyncio.new_event_loop()
    session = SessionRunner(loop, label="test")
    entered = threading.Event()
    release: asyncio.Event | None = None
    result: dict[str, int] = {}

    async def operation() -> int:
        nonlocal release
        release = asyncio.Event()
        entered.set()
        await release.wait()
        return 7

    def drive() -> None:
        result["value"] = session.run(operation)

    worker = threading.Thread(target=drive)
    worker.start()
    assert entered.wait(timeout=5.0)

    async def control() -> None:
        assert release is not None
        release.set()

    session.run_control(control)
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert result == {"value": 7}
    loop.close()


def test_sglang_session_rejects_same_runner_reentry_before_locking():
    loop = asyncio.new_event_loop()
    session = SessionRunner(loop, label="test")
    constructed = False

    async def nested() -> None:
        nonlocal constructed

        async def value() -> int:
            return 7

        def factory():
            nonlocal constructed
            constructed = True
            return value()

        with pytest.raises(RuntimeError, match="running event loop"):
            session.run(factory)
        with pytest.raises(RuntimeError, match="running event loop"):
            session.close(finalizer=loop.close)

    session.run(nested)
    assert constructed is False
    session.close(finalizer=loop.close)


def test_sglang_session_requires_owned_active_loop_for_async_work():
    loop = asyncio.new_event_loop()
    session = SessionRunner(loop, label="test")

    async def wrong_loop() -> None:
        with pytest.raises(RuntimeError, match="active run_session"):
            session.assert_active_loop()

    asyncio.run(wrong_loop())

    async def owned_loop() -> None:
        session.assert_active_loop()

    session.run(owned_loop)
    session.close(finalizer=loop.close)


def test_sglang_session_close_waits_for_active_operation_and_rejects_later_factory():
    loop = asyncio.new_event_loop()
    session = SessionRunner(loop, label="test")
    entered = threading.Event()
    finalized = threading.Event()
    release: asyncio.Event | None = None

    async def operation() -> None:
        nonlocal release
        release = asyncio.Event()
        entered.set()
        await release.wait()

    driver = threading.Thread(target=lambda: session.run(operation))
    driver.start()
    assert entered.wait(timeout=5.0)

    def finalize() -> None:
        loop.close()
        finalized.set()

    closer = threading.Thread(target=lambda: session.close(finalizer=finalize))
    closer.start()
    assert not finalized.wait(timeout=0.05)

    async def finish() -> None:
        assert release is not None
        release.set()

    session.run_control(finish)
    driver.join(timeout=5.0)
    closer.join(timeout=5.0)
    assert not driver.is_alive() and not closer.is_alive()
    assert finalized.is_set()

    constructed = False

    async def value() -> int:
        return 7

    def factory():
        nonlocal constructed
        constructed = True
        return value()

    with pytest.raises(RuntimeError, match="not available"):
        session.run(factory)
    assert constructed is False


def test_sglang_session_serializes_sync_operation_with_active_session():
    loop = asyncio.new_event_loop()
    session = SessionRunner(loop, label="test")
    entered = threading.Event()
    sync_ran = threading.Event()
    release: asyncio.Event | None = None

    async def operation() -> None:
        nonlocal release
        release = asyncio.Event()
        entered.set()
        await release.wait()

    driver = threading.Thread(target=lambda: session.run(operation))
    driver.start()
    assert entered.wait(timeout=5.0)

    sync_worker = threading.Thread(target=lambda: session.run_sync(sync_ran.set))
    sync_worker.start()
    assert not sync_ran.wait(timeout=0.05)

    async def finish() -> None:
        assert release is not None
        release.set()

    session.run_control(finish)
    driver.join(timeout=5.0)
    sync_worker.join(timeout=5.0)
    assert not driver.is_alive() and not sync_worker.is_alive()
    assert sync_ran.is_set()
    session.close(finalizer=loop.close)


def test_sglang_session_bounds_cooperative_best_effort_control():
    loop = asyncio.new_event_loop()
    session = SessionRunner(loop, label="test", control_timeout_s=0.05)
    entered = threading.Event()
    release: asyncio.Event | None = None

    async def operation() -> None:
        nonlocal release
        release = asyncio.Event()
        entered.set()
        await release.wait()

    driver = threading.Thread(target=lambda: session.run(operation))
    driver.start()
    assert entered.wait(timeout=5.0)

    async def stuck_control() -> None:
        await asyncio.Event().wait()

    assert session.run_control(stuck_control) is None

    async def finish() -> None:
        assert release is not None
        release.set()

    session.run_control(finish)
    driver.join(timeout=5.0)
    assert not driver.is_alive()
    session.close(finalizer=loop.close)
