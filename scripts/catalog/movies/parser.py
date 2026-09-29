"""Ответ TMDB /movie/{id} -> запись каталога в формате таблицы movies."""

import math
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, TypeVar

from scripts.catalog.json_values import as_dict, as_int, as_list, as_tmdb_path, clean_text
from scripts.catalog.movies.model import Actor, Country, CrewMember, CrewRole, Genre, Movie
from scripts.catalog.movies.rejection import MovieRejectedError, RejectReason, extract_poster_path

RUSSIAN_AGE_RATINGS = frozenset({"0+", "6+", "12+", "16+", "18+"})
THEATRICAL_RELEASE_TYPE = 3
COUNTRY_CODE_LENGTH = 2
MAX_ACTORS = 30
# runtime_min в таблице movies - smallint.
MAX_RUNTIME_MIN = 32767

# Порядок ролей - порядок показа в карточке.
CREW_JOBS = {
    CrewRole.DIRECTOR: frozenset({"Director"}),
    CrewRole.WRITER: frozenset({"Screenplay", "Writer"}),
    CrewRole.COMPOSER: frozenset({"Original Music Composer", "Music"}),
    CrewRole.PRODUCER: frozenset({"Producer"}),
}

T = TypeVar("T")


class MovieDataError(Exception):
    pass


def parse_movie(
    details: dict[str, Any], expected_movie_id: int, country_names: Mapping[str, str], cached_at: datetime
) -> Movie:
    """Собирает запись каталога или бросает MovieRejectedError, если фильм не проходит отбор.

    country_names - русские названия стран по коду ISO 3166-1, см. parse_country_names.
    """
    movie_id = details.get("id")
    if movie_id != expected_movie_id:
        raise MovieDataError(f"TMDB вернул фильм {movie_id!r} вместо {expected_movie_id}")
    if details.get("adult") is True:
        raise MovieRejectedError(movie_id, RejectReason.ADULT)
    poster_path = _require(extract_poster_path(details), movie_id, RejectReason.NO_POSTER)
    title = _require(extract_russian_title(details), movie_id, RejectReason.NO_RUSSIAN_TITLE)
    overview = _require(clean_text(details.get("overview")), movie_id, RejectReason.NO_RUSSIAN_OVERVIEW)

    credits = as_dict(details.get("credits"))
    return Movie(
        movie_id=movie_id,
        title=title,
        original_title=clean_text(details.get("original_title")) or title,
        release_date=parse_release_date(details.get("release_date")),
        age_rating=extract_russian_age_rating(details),
        poster_path=poster_path,
        runtime_min=parse_runtime(details.get("runtime")),
        overview=overview,
        genres=parse_genres(as_list(details.get("genres"))),
        countries=extract_countries(details, country_names),
        crew=extract_crew(as_list(credits.get("crew"))),
        actors=extract_actors(as_list(credits.get("cast"))),
        cached_at=cached_at,
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
    return runtime if runtime is not None and 0 < runtime <= MAX_RUNTIME_MIN else None


def parse_genres(items: list[Any]) -> list[Genre]:
    """Жанры из деталей фильма или из /genre/movie/list, названия на языке запроса."""
    genres: dict[int, Genre] = {}
    for item in map(as_dict, items):
        genre_id = as_int(item.get("id"))
        name = clean_text(item.get("name"))
        if genre_id is not None and name is not None:
            genres.setdefault(genre_id, Genre(genre_id, name))
    return list(genres.values())


def parse_country_names(items: list[Any]) -> dict[str, str]:
    """Ответ /configuration/countries -> {код: название}. native_name TMDB отдаёт на языке запроса."""
    names: dict[str, str] = {}
    for item in map(as_dict, items):
        code = _country_code(item)
        name = clean_text(item.get("native_name"))
        if code is not None and name is not None:
            names[code] = name
    return names


def extract_countries(details: dict[str, Any], country_names: Mapping[str, str]) -> list[Country]:
    """В production_countries названия только английские, английское остаётся, если русского нет."""
    countries: dict[str, Country] = {}
    for item in map(as_dict, as_list(details.get("production_countries"))):
        code = _country_code(item)
        if code is None:
            continue
        name = country_names.get(code) or clean_text(item.get("name"))
        if name is not None:
            countries.setdefault(code, Country(code, name))
    return list(countries.values())


def extract_crew(crew: list[Any]) -> list[CrewMember]:
    """Роли по порядку CREW_JOBS. Человек с двумя должностями одной роли попадает в неё один раз."""
    jobs_and_people = [
        (member.get("job"), person) for member in map(as_dict, crew) if (person := _parse_person(member))
    ]
    members: dict[tuple[int, CrewRole], CrewMember] = {}
    for role, jobs in CREW_JOBS.items():
        for job, (person_id, name) in jobs_and_people:
            if job in jobs:
                members.setdefault((person_id, role), CrewMember(person_id, name, role))
    return list(members.values())


def extract_actors(cast: list[Any]) -> list[Actor]:
    actors: dict[int, Actor] = {}
    for member in sorted(map(as_dict, cast), key=_billing_order):
        if person := _parse_person(member):
            person_id, name = person
            character = clean_text(member.get("character"))
            actors.setdefault(person_id, Actor(person_id, name, character, as_tmdb_path(member.get("profile_path"))))
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


def _country_code(item: dict[str, Any]) -> str | None:
    code = clean_text(item.get("iso_3166_1"))
    return code.upper() if code is not None and len(code) == COUNTRY_CODE_LENGTH else None


def _parse_person(member: dict[str, Any]) -> tuple[int, str] | None:
    person_id = as_int(member.get("id"))
    name = clean_text(member.get("name"))
    if person_id is None or name is None:
        return None
    return person_id, name


def _billing_order(member: dict[str, Any]) -> float:
    order = as_int(member.get("order"))
    return math.inf if order is None else order
