import time
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        requests_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second должно быть больше 0")
        self._min_interval = 1.0 / requests_per_second
        self._clock = clock
        self._sleep = sleep
        self._next_allowed_at = clock()

    def wait(self) -> None:
        now = self._clock()
        delay = self._next_allowed_at - now
        if delay > 0:
            self._sleep(delay)
            now += delay
        self._next_allowed_at = now + self._min_interval

    def pause(self, seconds: float) -> None:
        """Откладывает следующий запрос, например по заголовку Retry-After."""
        self._next_allowed_at = max(self._next_allowed_at, self._clock() + seconds)
