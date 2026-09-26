from datetime import date

import pytest

from scripts.catalog.movies.model import Actor, Person
from scripts.catalog.movies.parser import (
    MAX_ACTORS,
    MovieDataError,
    extract_actors,
    extract_crew,
    extract_russian_age_rating,
    extract_russian_title,
    normalize_age_rating,
    parse_movie,
    parse_release_date,
    parse_runtime,
)
from scripts.catalog.movies.rejection import MovieRejectedError, RejectReason
from tests.catalog.samples import movie_details


def test_parse_movie_collects_all_card_fields():
    movie = parse_movie(movie_details(), 550, "w500")

    assert movie.movie_id == 550
    assert movie.title_ru == "Бойцовский клуб"
    assert movie.original_title == "Fight Club"
    assert movie.release_date == date(1999, 10, 15)
    assert movie.age_rating == "18+"
    assert movie.poster_url == "https://image.tmdb.org/t/p/w500/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg"
    assert movie.overview_ru.startswith("Сотрудник")
    assert movie.runtime_minutes == 139
    assert movie.country_codes == ["US", "DE"]
    assert movie.genres == ["драма", "триллер"]
    assert movie.directors == [Person(7467, "David Fincher")]
    assert movie.writers == [Person(7468, "Jim Uhls")]
    assert movie.composers == [Person(1060, "Dust Brothers")]
    assert movie.producers == [Person(7474, "Art Linson")]
    assert movie.actors == [Actor(819, "Edward Norton", "The Narrator"), Actor(287, "Brad Pitt", "Tyler Durden")]


def test_parse_movie_rejects_foreign_id():
    with pytest.raises(MovieDataError):
        parse_movie(movie_details(movie_id=551), 550, "w500")


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"adult": True}, RejectReason.ADULT),
        ({"poster_path": None}, RejectReason.NO_POSTER),
        ({"poster_path": ""}, RejectReason.NO_POSTER),
        ({"poster_path": "без-слеша.jpg"}, RejectReason.NO_POSTER),
        ({"translations": {"translations": []}}, RejectReason.NO_RUSSIAN_TITLE),
        ({"overview": "   "}, RejectReason.NO_RUSSIAN_OVERVIEW),
        ({"overview": None}, RejectReason.NO_RUSSIAN_OVERVIEW),
    ],
)
def test_parse_movie_rejects_unsuitable_movie(overrides, reason):
    with pytest.raises(MovieRejectedError) as raised:
        parse_movie(movie_details(**overrides), 550, "w500")

    assert raised.value.reason is reason
    assert raised.value.movie_id == 550


def test_parse_movie_survives_missing_optional_blocks():
    details = movie_details(
        credits=None,
        release_dates="мусор",
        genres=[{"name": "  "}, "не словарь"],
        production_countries=None,
        runtime=0,
        release_date="",
    )

    movie = parse_movie(details, 550, "original")

    assert movie.age_rating is None
    assert movie.release_date is None
    assert movie.runtime_minutes is None
    assert movie.genres == []
    assert movie.country_codes == []
    assert movie.directors == movie.actors == []
    assert movie.poster_url.startswith("https://image.tmdb.org/t/p/original/")


def test_original_title_falls_back_to_russian_title():
    movie = parse_movie(movie_details(original_title="  "), 550, "w500")
    assert movie.original_title == "Бойцовский клуб"


def test_russian_title_uses_original_for_russian_movies():
    details = movie_details(original_language="ru", original_title="Брат", translations={})
    assert extract_russian_title(details) == "Брат"


def test_russian_title_ignores_tmdb_fallback_to_original_title():
    details = movie_details(title="Fight Club", translations={"translations": [{"iso_639_1": "ru", "data": {}}]})
    assert extract_russian_title(details) is None


def test_russian_title_prefers_russia_region():
    translations = [
        {"iso_639_1": "ru", "iso_3166_1": "UA", "data": {"title": "Вариант UA"}},
        {"iso_639_1": "ru", "iso_3166_1": "RU", "data": {"title": "Вариант RU"}},
    ]
    assert extract_russian_title(movie_details(translations={"translations": translations})) == "Вариант RU"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("18+", "18+"), (" 16 ", "16+"), ("0+", "0+"), ("PG-13", None), ("21+", None), ("", None), (None, None)],
)
def test_normalize_age_rating(raw, expected):
    assert normalize_age_rating(raw) == expected


def test_age_rating_absent_for_russia():
    details = movie_details(release_dates={"results": [{"iso_3166_1": "US", "release_dates": []}]})
    assert extract_russian_age_rating(details) is None


def test_age_rating_prefers_theatrical_release():
    releases = [{"certification": "12+", "type": 4}, {"certification": "16+", "type": 3}]
    details = movie_details(release_dates={"results": [{"iso_3166_1": "RU", "release_dates": releases}]})
    assert extract_russian_age_rating(details) == "16+"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("2001-02-03", date(2001, 2, 3)), ("2001", None), ("", None), (None, None), (20010203, None)],
)
def test_parse_release_date(raw, expected):
    assert parse_release_date(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"), [(90, 90), (0, None), (-5, None), (None, None), (True, None), ("90", None)]
)
def test_parse_runtime(raw, expected):
    assert parse_runtime(raw) == expected


def test_crew_member_with_two_matching_jobs_is_listed_once():
    crew = [
        {"id": 1, "name": "Автор", "job": "Writer"},
        {"id": 1, "name": "Автор", "job": "Screenplay"},
        {"id": 2, "name": "Второй", "job": "Screenplay"},
        {"id": 3, "name": "Не сценарист", "job": "Novel"},
    ]
    assert extract_crew(crew, frozenset({"Writer", "Screenplay"})) == [Person(1, "Автор"), Person(2, "Второй")]


def test_actors_are_deduplicated_ordered_and_capped():
    cast = [{"id": index, "name": f"Актёр {index}", "order": 100 - index} for index in range(MAX_ACTORS + 10)]
    cast.append({"id": 5, "name": "Повтор", "order": -1})
    cast.append({"id": None, "name": "Без id", "order": -2})
    cast.append({"id": 999, "name": "Без порядка"})

    actors = extract_actors(cast)

    assert len(actors) == MAX_ACTORS
    assert actors[0] == Actor(5, "Повтор", None)
    assert len({actor.person_id for actor in actors}) == MAX_ACTORS
    assert 999 not in {actor.person_id for actor in actors}
