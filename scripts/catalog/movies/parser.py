"""Ответ TMDB /movie/{id} -> запись каталога."""

import math
from collections.abc import Iterable
from datetime import date
from typing import Any, TypeVar

from scripts.catalog.json_values import as_dict, as_int, as_list, clean_text
from scripts.catalog.movies.model import Actor, Movie, Person
from scripts.catalog.movies.rejection import MovieRejectedError, RejectReason, extract_poster_path

POSTER_BASE_URL = "https://image.tmdb.org/t/p/"
RUSSIAN_AGE_RATINGS = frozenset({"0+", "6+", "12+", "16+", "18+"})
THEATRICAL_RELEASE_TYPE = 3
COUNTRY_CODE_LENGTH = 2
MAX_ACTORS = 30

DIRECTOR_JOBS = frozenset({"Director"})
WRITER_JOBS = frozenset({"Screenplay", "Writer"})
COMPOSER_JOBS = frozenset({"Original Music Composer", "Music"})
PRODUCER_JOBS = frozenset({"Producer"})

T = TypeVar("T")


class MovieDataError(Exception):
    pass


def parse_movie(details: dict[str, Any], expected_movie_id: int, poster_size: str) -> Movie:
    """Собирает запись каталога или бросает MovieRejectedError, если фильм не проходит отбор."""
    movie_id = details.get("id")
    if movie_id != expected_movie_id:
        raise MovieDataError(f"TMDB вернул фильм {movie_id!r} вместо {expected_movie_id}")
    if details.get("adult") is True:
        raise MovieRejectedError(movie_id, RejectReason.ADULT)
    poster_path = _require(extract_poster_path(details), movie_id, RejectReason.NO_POSTER)
    title_ru = _require(extract_russian_title(details), movie_id, RejectReason.NO_RUSSIAN_TITLE)
    overview_ru = _require(clean_text(details.get("overview")), movie_id, RejectReason.NO_RUSSIAN_OVERVIEW)

    credits = as_dict(details.get("credits"))
    crew = as_list(credits.get("crew"))
    return Movie(
        movie_id=movie_id,
        title_ru=title_ru,
        original_title=clean_text(details.get("original_title")) or title_ru,
        release_date=parse_release_date(details.get("release_date")),
        age_rating=extract_russian_age_rating(details),
        poster_url=f"{POSTER_BASE_URL}{poster_size}{poster_path}",
        overview_ru=overview_ru,
        runtime_minutes=parse_runtime(details.get("runtime")),
        country_codes=extract_country_codes(details),
        genres=extract_genre_names(details),
        directors=extract_crew(crew, DIRECTOR_JOBS),
        writers=extract_crew(crew, WRITER_JOBS),
        composers=extract_crew(crew, COMPOSER_JOBS),
        producers=extract_crew(crew, PRODUCER_JOBS),
        actors=extract_actors(as_list(credits.get("cast"))),
    )


def extract_russian_title(details: dict[str, Any]) -> str | None:
    """Смотрим translations: без перевода TMDB кладёт в title оригинальное название."""
    if details.get("original_language") == "ru":
        return clean_text(details.get("original_title"))
    return clean_text(_find_russian_translation(details).get("title"))


def extract_russian_age_rating(details: dict[str, Any]) -> str | None:
    countries = map(as_dict, as_list(as_dict(details.get("release_dates")).get("results")))
    russia = next((country for country in countries if country.get("iso_3166_1") == "RU"), {})
    releases = sorted(
        map(as_dict, as_list(russia.get("release_dates"))),
        key=lambda release: release.get("type") != THEATRICAL_RELEASE_TYPE,
    )
    ratings = (normalize_age_rating(release.get("certification")) for release in releases)
    return next((rating for rating in ratings if rating is not None), None)


def normalize_age_rating(raw_rating: Any) -> str | None:
    rating = clean_text(raw_rating)
    if rating is None:
        return None
    rating = rating if rating.endswith("+") else f"{rating}+"
    return rating if rating in RUSSIAN_AGE_RATINGS else None


def parse_release_date(raw_date: Any) -> date | None:
    text = clean_text(raw_date)
    if text is None:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def parse_runtime(raw_runtime: Any) -> int | None:
    runtime = as_int(raw_runtime)
    # 0 в TMDB означает «неизвестно».
    return runtime if runtime is not None and runtime > 0 else None


def extract_country_codes(details: dict[str, Any]) -> list[str]:
    codes = (clean_text(as_dict(country).get("iso_3166_1")) for country in as_list(details.get("production_countries")))
    return _unique(code.upper() for code in codes if code is not None and len(code) == COUNTRY_CODE_LENGTH)


def extract_genre_names(details: dict[str, Any]) -> list[str]:
    names = (clean_text(as_dict(genre).get("name")) for genre in as_list(details.get("genres")))
    return _unique(name for name in names if name is not None)


def extract_crew(crew: list[Any], jobs: frozenset[str]) -> list[Person]:
    """Человек с двумя подходящими должностями попадает в список один раз."""
    people: dict[int, Person] = {}
    for member in map(as_dict, crew):
        if member.get("job") in jobs and (person := _parse_person(member)):
            people.setdefault(person.person_id, person)
    return list(people.values())


def extract_actors(cast: list[Any]) -> list[Actor]:
    actors: dict[int, Actor] = {}
    for member in sorted(map(as_dict, cast), key=_billing_order):
        if person := _parse_person(member):
            actor = Actor(person.person_id, person.name, clean_text(member.get("character")))
            actors.setdefault(actor.person_id, actor)
    return list(actors.values())[:MAX_ACTORS]


def _require(value: T | None, movie_id: int, reason: RejectReason) -> T:
    if value is None:
        raise MovieRejectedError(movie_id, reason)
    return value


def _find_russian_translation(details: dict[str, Any]) -> dict[str, Any]:
    translations = map(as_dict, as_list(as_dict(details.get("translations")).get("translations")))
    russian = sorted(
        (item for item in translations if item.get("iso_639_1") == "ru"),
        key=lambda item: item.get("iso_3166_1") != "RU",
    )
    return as_dict(russian[0].get("data")) if russian else {}


def _parse_person(member: dict[str, Any]) -> Person | None:
    person_id = as_int(member.get("id"))
    name = clean_text(member.get("name"))
    if person_id is None or name is None:
        return None
    return Person(person_id, name)


def _billing_order(member: dict[str, Any]) -> float:
    order = as_int(member.get("order"))
    return math.inf if order is None else order


def _unique(values: Iterable[T]) -> list[T]:
    return list(dict.fromkeys(values))
