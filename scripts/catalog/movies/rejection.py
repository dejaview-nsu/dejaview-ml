from enum import StrEnum
from typing import Any

from scripts.catalog.json_values import clean_text


class RejectReason(StrEnum):
    NOT_FOUND = "not_found"
    ADULT = "adult"
    NO_POSTER = "no_poster"
    NO_RUSSIAN_TITLE = "no_russian_title"
    NO_RUSSIAN_OVERVIEW = "no_russian_overview"


class MovieRejectedError(Exception):
    def __init__(self, movie_id: int, reason: RejectReason) -> None:
        super().__init__(f"Фильм {movie_id} не подходит для каталога: {reason}")
        self.movie_id = movie_id
        self.reason = reason


def extract_poster_path(item: dict[str, Any]) -> str | None:
    path = clean_text(item.get("poster_path"))
    return path if path is not None and path.startswith("/") else None


def listing_rejection_reason(item: dict[str, Any]) -> RejectReason | None:
    """Проверка по элементу выдачи /discover: отсеивает заведомо неподходящие фильмы без запроса деталей.

    Русское название так не проверить: в выдаче нет переводов, это делает parse_movie.
    """
    if item.get("adult") is True:
        return RejectReason.ADULT
    if extract_poster_path(item) is None:
        return RejectReason.NO_POSTER
    if clean_text(item.get("overview")) is None:
        return RejectReason.NO_RUSSIAN_OVERVIEW
    return None
