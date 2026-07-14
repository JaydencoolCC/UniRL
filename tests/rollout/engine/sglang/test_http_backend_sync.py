"""HTTPBackend sync-generation mechanics against a stdlib HTTP stub (CPU).

The pure-sync HTTP backend has no event loop: concurrency = caller threads
bounded by one ``threading.Semaphore``, and the server sees the in-flight
POSTs together. The stub (``ThreadingHTTPServer``) counts concurrent
``/generate`` handlers so the tests assert real overlap, not just results.
``HTTPBackend`` is constructed directly (``boot`` is the only sglang path);
``shutdown()`` is never called — it would kill a real process tree.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("torch")

from unirl.rollout.engine.sglang.backends.http import HTTPBackend  # noqa: E402


class _FakeProc:
    pid = 0

    @staticmethod
    def is_alive() -> bool:
        return True


class _Stub:
    """One stub server per test: concurrency counters + a per-test barrier."""

    def __init__(self, *, expected_concurrent: int = 0) -> None:
        self.lock = threading.Lock()
        self.inflight = 0
        self.peak = 0
        # When set, every /generate handler waits until this many are inside
        # (deterministic overlap); 0 disables the rendezvous.
        self.barrier = threading.Barrier(expected_concurrent, timeout=10) if expected_concurrent else None

        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # keep test output quiet
                del args

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/generate":
                    with stub.lock:
                        stub.inflight += 1
                        stub.peak = max(stub.peak, stub.inflight)
                    try:
                        if stub.barrier is not None:
                            stub.barrier.wait()
                        else:
                            time.sleep(0.05)
                        body = {
                            "text": str(payload.get("text", "")),
                            "meta_info": {
                                "output_token_logprobs": [[-0.5, 3]],
                                "finish_reason": {"type": "stop"},
                            },
                        }
                        self._reply(200, body)
                    finally:
                        with stub.lock:
                            stub.inflight -= 1
                elif self.path in ("/abort_request", "/pause_generation"):
                    self._reply(200, {})
                else:
                    self._reply(404, {"error": "no such endpoint"})

            def _reply(self, status: int, body: dict) -> None:
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _backend(stub: _Stub, concurrency: int) -> HTTPBackend:
    return HTTPBackend(_FakeProc(), stub.base_url, concurrency=concurrency, runtime={})


def test_concurrent_caller_threads_overlap_on_the_server():
    stub = _Stub(expected_concurrent=4)
    try:
        backend = _backend(stub, concurrency=4)
        results = [None] * 4
        threads = [
            threading.Thread(target=lambda i=i: results.__setitem__(i, backend.generate([{"text": f"p{i}"}])))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert stub.peak == 4  # the barrier held all four handlers inside at once
        assert sorted(r[0].text for r in results) == ["p0", "p1", "p2", "p3"]
    finally:
        stub.stop()


def test_semaphore_bounds_in_flight_posts():
    stub = _Stub()  # 50ms handler hold
    try:
        backend = _backend(stub, concurrency=2)
        threads = [threading.Thread(target=lambda i=i: backend.generate([{"text": f"p{i}"}])) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert 1 <= stub.peak <= 2  # never more than the semaphore admits
    finally:
        stub.stop()


def test_batch_generate_preserves_prompt_order():
    stub = _Stub()
    try:
        backend = _backend(stub, concurrency=3)
        out = backend.generate([{"text": f"p{i}"} for i in range(5)])
        assert [r.text for r in out] == [f"p{i}" for i in range(5)]
        assert [r.token_ids for r in out] == [[3]] * 5
        assert [r.finish_reason for r in out] == ["stop"] * 5
    finally:
        stub.stop()


def test_controls_are_best_effort_and_never_raise():
    stub = _Stub()
    try:
        backend = _backend(stub, concurrency=2)
        backend.abort(abort_all=True)  # 200 route
        backend.pause()  # 200 route
        backend.resume()  # 404 route (/continue_generation) — warn, no raise
    finally:
        stub.stop()
