from datetime import date

import pytest
import requests

from scripts.catalog.tmdb.client import MAX_ATTEMPTS, TmdbClient, backoff_delay
from scripts.catalog.tmdb.errors import (
    TmdbAuthError,
    TmdbNotFoundError,
    TmdbRequestError,
    TmdbUnavailableError,
)


class FakeResponse:
    def __init__(self, status_code: int, payload=None, headers=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    """Отдаёт заранее заданные ответы или бросает заданные исключения."""

    def __init__(self, *outcomes) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []

    def get(self, url, params, headers, timeout):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeRateLimiter:
    def __init__(self) -> None:
        self.waits = 0
        self.pauses: list[float] = []

    def wait(self) -> None:
        self.waits += 1

    def pause(self, seconds: float) -> None:
        self.pauses.append(seconds)


def make_client(*outcomes, api_key=None, access_token="token"):
    session = FakeSession(*outcomes)
    limiter = FakeRateLimiter()
    client = TmdbClient(limiter, access_token=access_token, api_key=api_key, session=session)
    return client, session, limiter


def test_get_movie_requests_everything_in_one_call_with_bearer_token():
    client, session, limiter = make_client(FakeResponse(200, {"id": 550}))

    assert client.get_movie(550) == {"id": 550}

    call = session.calls[0]
    assert call["url"].endswith("/movie/550")
    assert call["params"]["language"] == "ru-RU"
    assert call["params"]["append_to_response"] == "credits,release_dates,translations"
    assert call["headers"]["Authorization"] == "Bearer token"
    assert "api_key" not in call["params"]
    assert limiter.waits == 1


def test_api_key_goes_to_query_params():
    client, session, _ = make_client(FakeResponse(200, {"id": 1}), access_token=None, api_key="secret")

    client.get_movie(1)

    assert session.calls[0]["params"]["api_key"] == "secret"
    assert "Authorization" not in session.calls[0]["headers"]


def test_client_requires_some_credentials():
    with pytest.raises(ValueError):
        TmdbClient(FakeRateLimiter())


@pytest.mark.parametrize("movie_id", [0, -1, "550", True, None])
def test_get_movie_validates_id(movie_id):
    client, _, _ = make_client()
    with pytest.raises(ValueError):
        client.get_movie(movie_id)


def test_retries_server_errors_and_network_failures_then_succeeds():
    client, session, limiter = make_client(
        FakeResponse(503),
        requests.ConnectionError("обрыв"),
        requests.Timeout("таймаут"),
        FakeResponse(200, {"id": 7}),
    )

    assert client.get_movie(7) == {"id": 7}
    assert len(session.calls) == 4
    assert limiter.pauses == [backoff_delay(1), backoff_delay(2), backoff_delay(3)]


def test_rate_limit_response_respects_retry_after():
    client, _, limiter = make_client(FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, {"id": 7}))

    client.get_movie(7)

    assert limiter.pauses == [7.0]


def test_invalid_retry_after_falls_back_to_backoff():
    client, _, limiter = make_client(
        FakeResponse(429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}),
        FakeResponse(200, {"id": 7}),
    )

    client.get_movie(7)

    assert limiter.pauses == [backoff_delay(1)]


def test_gives_up_after_max_attempts():
    client, session, limiter = make_client(*[FakeResponse(500) for _ in range(MAX_ATTEMPTS)])

    with pytest.raises(TmdbUnavailableError):
        client.get_movie(7)

    assert len(session.calls) == MAX_ATTEMPTS
    assert len(limiter.pauses) == MAX_ATTEMPTS - 1


def test_auth_error_is_not_retried():
    client, session, _ = make_client(FakeResponse(401))

    with pytest.raises(TmdbAuthError):
        client.get_movie(7)

    assert len(session.calls) == 1


def test_not_found():
    client, _, _ = make_client(FakeResponse(404))
    with pytest.raises(TmdbNotFoundError):
        client.get_movie(7)


def test_other_client_errors_are_reported_without_retry():
    client, session, _ = make_client(FakeResponse(422, text="invalid page"))

    with pytest.raises(TmdbRequestError, match="422"):
        client.get_movie(7)

    assert len(session.calls) == 1


@pytest.mark.parametrize("payload", [ValueError("not json"), ["список"], None])
def test_malformed_body_is_request_error(payload):
    client, _, _ = make_client(FakeResponse(200, payload))
    with pytest.raises(TmdbRequestError):
        client.get_movie(7)


def test_api_key_is_redacted_from_network_errors():
    error = requests.ConnectionError("failed: https://api.themoviedb.org/3/movie/7?api_key=secret")
    client, _, _ = make_client(*[error] * MAX_ATTEMPTS, access_token=None, api_key="secret")

    with pytest.raises(TmdbUnavailableError) as raised:
        client.get_movie(7)

    assert "secret" not in str(raised.value)


def test_discover_movies_builds_filters():
    client, session, _ = make_client(FakeResponse(200, {"results": [], "total_pages": 0}))

    client.discover_movies(35, date(1990, 1, 1), date(1999, 12, 31), 200, 2)

    params = session.calls[0]["params"]
    assert params["with_genres"] == "35"
    assert params["primary_release_date.gte"] == "1990-01-01"
    assert params["primary_release_date.lte"] == "1999-12-31"
    assert params["vote_count.gte"] == "200"
    assert params["page"] == "2"
    assert params["include_adult"] == "false"


@pytest.mark.parametrize("page", [0, 501])
def test_discover_validates_page(page):
    client, _, _ = make_client()
    with pytest.raises(ValueError):
        client.discover_movies(35, date(1990, 1, 1), date(1999, 12, 31), 200, page)


def test_get_genres_requires_list():
    client, _, _ = make_client(FakeResponse(200, {"genres": None}))
    with pytest.raises(TmdbRequestError):
        client.get_genres()
