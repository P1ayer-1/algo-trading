"""The overnight-crash regression: no loop may take the recorder down with it.

On 2026-09-08 at 03:02 UTC a collection run died seven hours in. The cause was
one dropped HTTP keep-alive connection on the chart's candle endpoint —
`RemoteDisconnected`, which any long-lived connection pool meets eventually.
Two separate defects turned that into a total loss of collection:

  1. `refresh_candles_loop` was the only loop in server.py without a handler.
  2. `run_servers` gathered the raw loops, and `asyncio.gather` propagates the
     first exception, so ANY loop dying ended the process — including the
     microstructure feed, the recorder and the raw archive.

The second is the one that matters. The chart is a convenience; raw events are
irreplaceable, because that market moment is over. These tests pin both.

There is no pytest-asyncio here, so each test drives its own event loop.
"""

import asyncio

import pytest

pytest.importorskip("blofin", reason="BloFin SDK not installed")
pytest.importorskip("websockets", reason="websockets not installed")

import server  # noqa: E402


def run(coro, timeout=5.0):
    return asyncio.run(asyncio.wait_for(coro, timeout))


class Boom(RuntimeError):
    """Stands in for RemoteDisconnected and everything like it."""


# ---------------------------------------------------------------------------
# supervise
# ---------------------------------------------------------------------------


def test_a_crashing_loop_is_restarted_rather_than_propagating(monkeypatch):
    monkeypatch.setattr(server, "MAX_RESTART_BACKOFF_SECONDS", 0.0)
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) < 4:
            raise Boom("remote end closed connection")
        await asyncio.sleep(10)  # settled; will be cancelled by the timeout

    async def drive():
        task = asyncio.create_task(server.supervise("flaky", flaky))
        while len(attempts) < 4:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(drive())
    assert len(attempts) == 4


def test_one_loop_crashing_does_not_stop_another(monkeypatch):
    """The actual regression. The candle loop died; the recorder must not."""
    monkeypatch.setattr(server, "MAX_RESTART_BACKOFF_SECONDS", 0.0)
    collected = []

    async def chart_loop():
        raise Boom("RemoteDisconnected")

    async def collection_loop():
        while True:
            collected.append(1)
            await asyncio.sleep(0.01)

    async def drive():
        # `gather` already returns a future; this mirrors what run_servers
        # does with the real loops.
        gathered = asyncio.gather(
            server.supervise("chart", chart_loop),
            server.supervise("collection", collection_loop),
        )
        await asyncio.sleep(0.2)
        still_running = not gathered.done()
        gathered.cancel()
        try:
            await gathered
        except asyncio.CancelledError:
            pass
        return still_running

    assert run(drive()) is True
    # The collection loop kept ticking throughout, despite the other one
    # failing on every single restart.
    assert len(collected) > 5


def test_cancellation_is_not_swallowed():
    """Ctrl+C must still stop the process. A supervisor that catches
    CancelledError would make it unkillable."""
    started = asyncio.Event()

    async def forever():
        started.set()
        await asyncio.sleep(30)

    async def drive():
        task = asyncio.create_task(server.supervise("forever", forever))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(drive())


def test_a_loop_that_returns_is_restarted_not_silently_dropped(monkeypatch):
    # These loops are all `while True`. One returning means something is
    # wrong, and quietly losing it is how you find out days later that the
    # recorder stopped.
    monkeypatch.setattr(server, "MAX_RESTART_BACKOFF_SECONDS", 0.0)
    calls = []

    async def returns_immediately():
        calls.append(1)

    async def drive():
        task = asyncio.create_task(
            server.supervise("quitter", returns_immediately))
        while len(calls) < 3:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(drive())
    assert len(calls) >= 3


def test_backoff_resets_after_a_healthy_run(monkeypatch):
    # A loop that ran for hours and then hit one bad response should retry
    # promptly, not inherit the backoff from last night's failure.
    monkeypatch.setattr(server, "HEALTHY_AFTER_SECONDS", 0.0)
    monkeypatch.setattr(server, "MAX_RESTART_BACKOFF_SECONDS", 30.0)
    delays = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        # `server.asyncio` IS the global module, so this patch is global --
        # call the captured original or fake_sleep recurses into itself.
        delays.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    async def always_fails():
        raise Boom("nope")

    async def drive():
        task = asyncio.create_task(server.supervise("flaky", always_fails))
        while len(delays) < 4:
            await real_sleep(0)   # not the patched one; it would be recorded
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(drive())
    # Every attempt counts as "healthy" under this monkeypatch, so the backoff
    # is reset each time instead of doubling away.
    assert delays[:4] == [1.0, 1.0, 1.0, 1.0]


def test_backoff_grows_when_failures_are_immediate(monkeypatch):
    monkeypatch.setattr(server, "HEALTHY_AFTER_SECONDS", 1e9)
    monkeypatch.setattr(server, "MAX_RESTART_BACKOFF_SECONDS", 8.0)
    delays = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        # `server.asyncio` IS the global module, so this patch is global --
        # call the captured original or fake_sleep recurses into itself.
        delays.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    async def always_fails():
        raise Boom("nope")

    async def drive():
        task = asyncio.create_task(server.supervise("flaky", always_fails))
        while len(delays) < 6:
            await real_sleep(0)   # not the patched one
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(drive())
    assert delays[:5] == [1.0, 2.0, 4.0, 8.0, 8.0]  # doubles, then caps


# ---------------------------------------------------------------------------
# The loop that actually broke
# ---------------------------------------------------------------------------


def test_candle_refresh_survives_a_dropped_connection(monkeypatch):
    """The direct cause: this loop had no handler at all."""
    monkeypatch.setattr(server, "RECALCULATE_SECONDS", 0)
    calls = []

    def fetch(market_api):
        calls.append(1)
        if len(calls) <= 2:
            raise ConnectionError(
                "('Connection aborted.', RemoteDisconnected(...))")
        return [{"time": 1, "open": 1, "high": 1, "low": 1, "close": 1}]

    monkeypatch.setattr(server, "fetch_rest_candles", fetch)

    applied = []

    class FakeState:
        async def set_candles(self, candles):
            applied.append(candles)

    async def no_broadcast(state):
        return None

    monkeypatch.setattr(server, "broadcast", no_broadcast)

    async def drive():
        task = asyncio.create_task(
            server.refresh_candles_loop(FakeState(), object()))
        while len(applied) < 2:
            await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    run(drive())
    # It kept going through both failures and then applied real candles.
    assert len(calls) >= 4
    assert len(applied) >= 2


def test_candle_refresh_reraises_cancellation(monkeypatch):
    monkeypatch.setattr(server, "RECALCULATE_SECONDS", 0)

    def fetch(market_api):
        raise ConnectionError("down")

    monkeypatch.setattr(server, "fetch_rest_candles", fetch)

    async def drive():
        task = asyncio.create_task(
            server.refresh_candles_loop(object(), object()))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(drive())


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


def test_log_lines_carry_a_utc_timestamp_and_flush(capsys):
    # An overnight failure is only diagnosable if its line can be lined up
    # against the exchange's logs, and only survives the crash if it is
    # flushed rather than sitting in a block buffer.
    server.log("something broke")
    out = capsys.readouterr().out.strip()
    assert out.endswith("something broke")
    assert out.count(":") >= 2
    assert "Z  " in out
