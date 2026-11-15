"""Who may run a model now: concurrent search queries share the models without rejecting each
other on contact (#19099).

Each model runs one query at a time, different models run in parallel, but at most `slots`
heavy steps (model runs and media decoding) at once, since they share the CPU. A query waits
for its turn in a short queue: in all no longer than `wait_seconds` for one request, and no more
than `queue_limit` requests wait for one model. Past either limit the request gets 503 NOT_READY,
which the backend may retry.
"""

import asyncio
import time
from collections import Counter
from collections.abc import Callable
from typing import Any, TypeVar

from dejaview_ml.errors import MlError

T = TypeVar("T")

DECODING = "decoding"  # no model: ffmpeg / Pillow only take a CPU slot


def busy() -> MlError:
    return MlError(503, "NOT_READY", "inference capacity is occupied; retry later")


class Budget:
    """How long one request may still wait in queues, over all of its steps."""

    def __init__(self, seconds: float):
        self.seconds = seconds


class Capacity:
    def __init__(self, slots: int, queue_limit: int, wait_seconds: float):
        self.queue_limit = queue_limit
        self.wait_seconds = wait_seconds
        self._slots = asyncio.Semaphore(slots)
        self._models: dict[str, asyncio.Lock] = {}
        self._waiting: Counter[str] = Counter()

    def budget(self) -> Budget:
        return Budget(self.wait_seconds)

    async def run(self, model: str, budget: Budget, function: Callable[..., T], *args: Any) -> T:
        """function(*args) in a worker thread, once the model and a CPU slot are free."""
        lock = None if model == DECODING else self._models.setdefault(model, asyncio.Lock())
        if self._waiting[model] >= self.queue_limit:
            raise busy()
        self._waiting[model] += 1
        started = time.monotonic()
        try:
            async with asyncio.timeout(max(budget.seconds, 0.0)):
                if lock is not None:
                    await lock.acquire()
                try:
                    await self._slots.acquire()
                except BaseException:
                    if lock is not None:
                        lock.release()
                    raise
        except TimeoutError:
            raise busy() from None
        finally:
            self._waiting[model] -= 1
            budget.seconds -= time.monotonic() - started

        def release(work: asyncio.Future) -> None:
            if not work.cancelled():
                work.exception()  # retrieved here if the request is gone
            self._slots.release()
            if lock is not None:
                lock.release()

        # The model is released when the thread is done, not when the request is: a client
        # that disconnects cancels the request, but cannot stop the thread using the model.
        work = asyncio.ensure_future(asyncio.to_thread(function, *args))
        work.add_done_callback(release)
        return await asyncio.shield(work)
