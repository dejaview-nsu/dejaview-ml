"""Клиент TMDB API v3. Переиспользуется утилитой индексации (#17650)."""

import logging
from datetime import date
from http import HTTPStatus
from typing import Any

import requests

from scripts.catalog.json_values import as_positive_int
from scripts.catalog.tmdb.errors import (
    TmdbAuthError,
    TmdbNotFoundError,
    TmdbRequestError,
    TmdbUnavailableError,
)
from scripts.catalog.tmdb.rate_limiter import RateLimiter

API_BASE_URL = "https://api.themoviedb.org/3"
LANGUAGE = "ru-RU"
MOVIE_APPENDS = "credits,release_dates,translations"
REQUEST_TIMEOUT_SECONDS = 15
MAX_ATTEMPTS = 5
BASE_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0
MAX_DISCOVER_PAGE = 500
ERROR_BODY_PREVIEW_LENGTH = 200

logger = logging.getLogger(__name__)


class _RetryableError(Exception):
    def __init__(self, problem: str, delay: float) -> None:
        super().__init__(problem)
        self.problem = problem
        self.delay = delay


class TmdbClient:
    def __init__(
        self,
        rate_limiter: RateLimiter,
        access_token: str | None = None,
        api_key: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if not access_token and not api_key:
            raise ValueError("Нужен access_token или api_key")
        self._rate_limiter = rate_limiter
        self._session = session or requests.Session()
        self._secrets = [secret for secret in (access_token, api_key) if secret]
        self._headers = {"Accept": "application/json"}
        self._auth_params: dict[str, str] = {}
        if access_token:
            self._headers["Authorization"] = f"Bearer {access_token}"
        else:
            self._auth_params["api_key"] = api_key

    def get_movie(self, movie_id: int) -> dict[str, Any]:
        """Детали фильма вместе с credits, release_dates и translations одним запросом."""
        if as_positive_int(movie_id) is None:
            raise ValueError(f"movie_id должен быть положительным целым, получено: {movie_id!r}")
        return self._get(f"/movie/{movie_id}", {"language": LANGUAGE, "append_to_response": MOVIE_APPENDS})

    def get_genres(self) -> list[dict[str, Any]]:
        genres = self._get("/genre/movie/list", {"language": LANGUAGE}).get("genres")
        if not isinstance(genres, list):
            raise TmdbRequestError("В ответе /genre/movie/list нет списка genres")
        return genres

    def discover_movies(
        self,
        genre_id: int,
        released_from: date,
        released_to: date,
        min_vote_count: int,
        page: int,
    ) -> dict[str, Any]:
        """Страница фильмов жанра за период, самые известные первыми."""
        if not 1 <= page <= MAX_DISCOVER_PAGE:
            raise ValueError(f"page должен быть в диапазоне 1..{MAX_DISCOVER_PAGE}")
        params = {
            "language": LANGUAGE,
            "with_genres": str(genre_id),
            "primary_release_date.gte": released_from.isoformat(),
            "primary_release_date.lte": released_to.isoformat(),
            "vote_count.gte": str(min_vote_count),
            "sort_by": "vote_count.desc",
            "include_adult": "false",
            "include_video": "false",
            "page": str(page),
        }
        return self._get("/discover/movie", params)

    def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        last_problem = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._rate_limiter.wait()
            try:
                return self._request_once(path, params, attempt)
            except _RetryableError as error:
                last_problem = error.problem
                if attempt < MAX_ATTEMPTS:
                    logger.warning(
                        "%s: %s, повтор %d/%d через %.1f с", path, error.problem, attempt + 1, MAX_ATTEMPTS, error.delay
                    )
                    self._rate_limiter.pause(error.delay)
        raise TmdbUnavailableError(f"{path}: TMDB недоступен после {MAX_ATTEMPTS} попыток, {last_problem}")

    def _request_once(self, path: str, params: dict[str, str], attempt: int) -> dict[str, Any]:
        try:
            response = self._session.get(
                API_BASE_URL + path,
                params={**params, **self._auth_params},
                headers=self._headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except (requests.ConnectionError, requests.Timeout) as error:
            raise _RetryableError(f"сетевая ошибка: {self._redact(str(error))}", backoff_delay(attempt)) from None

        match response.status_code:
            case HTTPStatus.OK:
                return parse_json_object(response, path)
            case HTTPStatus.UNAUTHORIZED:
                raise TmdbAuthError("TMDB отклонил ключ (401): проверьте TMDB_ACCESS_TOKEN / TMDB_API_KEY")
            case HTTPStatus.NOT_FOUND:
                raise TmdbNotFoundError(f"{path}: не найдено в TMDB")
            case HTTPStatus.TOO_MANY_REQUESTS:
                delay = retry_after_delay(response) or backoff_delay(attempt)
                raise _RetryableError("превышен лимит запросов (429)", delay)
            case status if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
                raise _RetryableError(f"ошибка сервера TMDB ({status})", backoff_delay(attempt))
            case status:
                preview = response.text[:ERROR_BODY_PREVIEW_LENGTH].replace("\n", " ")
                raise TmdbRequestError(f"{path}: TMDB вернул {status}: {preview}")

    def _redact(self, text: str) -> str:
        """api_key приходит в тексте сетевых ошибок вместе с URL, в лог он попасть не должен."""
        for secret in self._secrets:
            text = text.replace(secret, "***")
        return text


def backoff_delay(attempt: int) -> float:
    return min(BASE_BACKOFF_SECONDS * 2 ** (attempt - 1), MAX_BACKOFF_SECONDS)


def retry_after_delay(response: requests.Response) -> float | None:
    """Retry-After в секундах. Формат с датой не поддерживаем: тогда сработает обычная пауза."""
    try:
        seconds = float(response.headers.get("Retry-After", ""))
    except ValueError:
        return None
    return min(seconds, MAX_BACKOFF_SECONDS) if seconds >= 0 else None


def parse_json_object(response: requests.Response, path: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        raise TmdbRequestError(f"{path}: ответ TMDB не является JSON") from None
    if not isinstance(payload, dict):
        raise TmdbRequestError(f"{path}: ожидался JSON-объект, получено {type(payload).__name__}")
    return payload
