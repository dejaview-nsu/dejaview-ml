"""Who may run a model now: concurrent search queries share the models without rejecting each
other on contact (#19099).

Each model runs one query at a time, different models run in parallel, but at most `slots`
heavy steps (model runs and media decoding) at once, since they share the CPU. A query that
finds its model and a slot free runs at once; otherwise it waits for its turn in a short queue:
no more than `queue_limit` requests wait for one model, and one request waits only until
`wait_seconds` after it arrived, over all of its steps. Past either limit the request gets
503 SERVER_BUSY. A request whose client has disconnected leaves the
queue and does not start a model run.
"""

import asyncio
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from dejaview_ml.errors import MlError

T = TypeVar("T")

DECODING = "decoding"  # no model: ffmpeg / Pillow only take a CPU slot
DISCONNECT_POLL_SECONDS = 0.05


def busy() -> MlError:
    return MlError(503, "SERVER_BUSY", "inference capacity is occupied; retry later")


def gone() -> MlError:
    return MlError(503, "NOT_READY", "client disconnected while waiting")


class Budget:
    """Until when one request may wait in queues, and how to tell that its client is gone."""

    def __init__(self, deadline: float, disconnected: Callable[[], Awaitable[bool]] | None = None):
        self.deadline = deadline
        self._disconnected = disconnected

    def remaining(self) -> float:
        return max(self.deadline - time.monotonic(), 0.0)

    async def is_gone(self) -> bool:
        return self._disconnected is not None and await self._disconnected()

    async def wait_gone(self) -> None:
        """Returns once the client has disconnected; never, if that cannot be known."""
        if self._disconnected is None:
            await asyncio.Event().wait()
        while not await self._disconnected():
            await asyncio.sleep(DISCONNECT_POLL_SECONDS)


class Capacity:
    def __init__(self, slots: int, queue_limit: int, wait_seconds: float):
        self.queue_limit = queue_limit
        self.wait_seconds = wait_seconds
        self._slots = asyncio.Semaphore(slots)
        self._models: dict[str, asyncio.Lock] = {}
        self._waiting: Counter[str] = Counter()

    def budget(
        self,
        started: float | None = None,
        disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> Budget:
        """Budget of a request that arrived at `started` (time.monotonic(); now by default)."""
        started = time.monotonic() if started is None else started
        return Budget(started + self.wait_seconds, disconnected)

    async def run(self, model: str, budget: Budget, function: Callable[..., T], *args: Any) -> T:
        """function(*args) in a worker thread, once the model and a CPU slot are free."""
        if await budget.is_gone():
            raise gone()
        lock = None if model == DECODING else self._models.setdefault(model, asyncio.Lock())
        if self._must_wait(lock):
            if self._waiting[model] >= self.queue_limit:
                raise busy()
            self._waiting[model] += 1
            try:
                await self._wait_turn(lock, budget)
            finally:
                self._waiting[model] -= 1
        else:
            await self._take(lock)

        def release(work: asyncio.Future) -> None:
            if not work.cancelled():
                work.exception()  # retrieved here if the request is gone
            self._free(lock)

        # The model is released when the thread is done, not when the request is: a client
        # that disconnects cancels the request, but cannot stop the thread using the model.
        work = asyncio.ensure_future(asyncio.to_thread(function, *args))
        work.add_done_callback(release)
        return await asyncio.shield(work)

    def _must_wait(self, lock: asyncio.Lock | None) -> bool:
        return (lock is not None and lock.locked()) or self._slots.locked()

    async def _take(self, lock: asyncio.Lock | None) -> None:
        if lock is not None:
            await lock.acquire()
        try:
            await self._slots.acquire()
        except BaseException:
            if lock is not None:
                lock.release()
            raise

    def _free(self, lock: asyncio.Lock | None) -> None:
        self._slots.release()
        if lock is not None:
            lock.release()

    async def _wait_turn(self, lock: asyncio.Lock | None, budget: Budget) -> None:
        """Takes the model and a slot, unless the budget runs out or the client leaves first."""
        turn = asyncio.ensure_future(self._take(lock))
        left = asyncio.ensure_future(budget.wait_gone())
        try:
            await asyncio.wait(
                {turn, left}, timeout=budget.remaining(), return_when=asyncio.FIRST_COMPLETED
            )
        except BaseException:
            left.cancel()
            await self._give_up(turn, lock)
            raise
        left.cancel()
        if left.done() and not left.cancelled() and left.exception() is None:
            await self._give_up(turn, lock)
            raise gone()
        if not turn.done():
            await self._give_up(turn, lock)
            raise busy()

    async def _give_up(self, turn: asyncio.Future, lock: asyncio.Lock | None) -> None:
        """Leaves the queue; a turn that came at the same moment is given back."""
        if not turn.done():
            turn.cancel()
            await asyncio.gather(turn, return_exceptions=True)
        if not turn.cancelled() and turn.exception() is None:
            self._free(lock)
