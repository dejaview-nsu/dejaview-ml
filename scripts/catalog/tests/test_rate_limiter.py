import pytest

from scripts.catalog.tmdb.rate_limiter import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_first_request_is_not_delayed_and_next_ones_are_spaced():
    clock = FakeClock()
    limiter = RateLimiter(4, clock=clock.time, sleep=clock.sleep)

    limiter.wait()
    limiter.wait()
    limiter.wait()

    assert clock.sleeps == pytest.approx([0.25, 0.25])


def test_no_delay_when_requests_are_rare():
    clock = FakeClock()
    limiter = RateLimiter(4, clock=clock.time, sleep=clock.sleep)

    limiter.wait()
    clock.now += 10
    limiter.wait()

    assert clock.sleeps == []


def test_pause_postpones_next_request():
    clock = FakeClock()
    limiter = RateLimiter(10, clock=clock.time, sleep=clock.sleep)

    limiter.wait()
    limiter.pause(3)
    limiter.wait()

    assert clock.sleeps == pytest.approx([3])


@pytest.mark.parametrize("rate", [0, -1])
def test_rejects_non_positive_rate(rate):
    with pytest.raises(ValueError):
        RateLimiter(rate)
