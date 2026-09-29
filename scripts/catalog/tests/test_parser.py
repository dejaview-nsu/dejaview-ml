from datetime import date

import pytest

from scripts.catalog.movies.model import Actor, Country, CrewMember, CrewRole, Genre
from scripts.catalog.movies.parser import (
    MAX_ACTORS,
    MAX_RUNTIME_MIN,
    MovieDataError,
    extract_actors,
    extract_countries,
    extract_crew,
    extract_russian_age_rating,
    extract_russian_title,
    normalize_age_rating,
    parse_country_names,
    parse_genres,
    parse_movie,
    parse_release_date,
    parse_runtime,
)
from scripts.catalog.movies.rejection import MovieRejectedError, RejectReason
from scripts.catalog.tests.samples import CACHED_AT, COUNTRIES, COUNTRY_NAMES, movie_details, sample_movie


def test_parse_movie_collects_all_card_fields():
    movie = sample_movie()

    assert movie.movie_id == 550
    assert movie.title == "Бойцовский клуб"
    assert movie.original_title == "Fight Club"
    assert movie.release_date == date(1999, 10, 15)
    assert movie.age_rating == "18+"
    assert movie.poster_path == "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg"
    assert movie.runtime_min == 139
    assert movie.overview.startswith("Сотрудник")
    assert movie.genres == [Genre(18, "драма"), Genre(53, "триллер")]
    assert movie.countries == [Country("US", "США"), Country("DE", "Germany")]
    assert movie.crew == [
        CrewMember(7467, "David Fincher", CrewRole.DIRECTOR),
        CrewMember(7468, "Jim Uhls", CrewRole.WRITER),
        CrewMember(1060, "Dust Brothers", CrewRole.COMPOSER),
        CrewMember(7474, "Art Linson", CrewRole.PRODUCER),
    ]
    assert movie.actors == [
        Actor(819, "Edward Norton", "The Narrator", "/8nytsqL59SFJTVYVrN72k6qkGgJ.jpg"),
        Actor(287, "Brad Pitt", "Tyler Durden", None),
    ]
    assert movie.cached_at == CACHED_AT


def test_parse_movie_rejects_foreign_id():
    with pytest.raises(MovieDataError):
        parse_movie(movie_details(movie_id=551), 550, COUNTRY_NAMES, CACHED_AT)


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
        sample_movie(**overrides)

    assert raised.value.reason is reason
    assert raised.value.movie_id == 550


def test_parse_movie_survives_missing_optional_blocks():
    movie = sample_movie(
        credits=None,
        release_dates="мусор",
        genres=[{"name": "  "}, "не словарь"],
        production_countries=None,
        runtime=0,
        release_date="",
    )

    assert movie.age_rating is None
    assert movie.release_date is None
    assert movie.runtime_min is None
    assert movie.genres == movie.countries == movie.crew == movie.actors == []


def test_original_title_falls_back_to_russian_title():
    movie = sample_movie(original_title="  ")
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
    ("raw", "expected"),
    [
        (90, 90),
        (MAX_RUNTIME_MIN, MAX_RUNTIME_MIN),
        (MAX_RUNTIME_MIN + 1, None),
        (0, None),
        (-5, None),
        (None, None),
        (True, None),
        ("90", None),
    ],
)
def test_parse_runtime(raw, expected):
    assert parse_runtime(raw) == expected


def test_genres_keep_id_and_skip_duplicates_and_garbage():
    items = [{"id": 18, "name": "драма"}, {"id": 18, "name": "драма"}, {"id": 35}, {"name": "без id"}, "мусор"]
    assert parse_genres(items) == [Genre(18, "драма")]


def test_country_names_skip_untranslated_and_malformed():
    items = [*COUNTRIES, {"iso_3166_1": "SUN", "native_name": "Не код"}, "мусор"]
    assert parse_country_names(items) == {"US": "США"}


def test_countries_use_russian_name_and_fall_back_to_tmdb_name():
    details = movie_details(
        production_countries=[
            {"iso_3166_1": "us", "name": "United States of America"},
            {"iso_3166_1": "US", "name": "дубль"},
            {"iso_3166_1": "SU", "name": "Soviet Union"},
            {"iso_3166_1": "XX"},
            {"name": "без кода"},
        ]
    )
    assert extract_countries(details, COUNTRY_NAMES) == [Country("US", "США"), Country("SU", "Soviet Union")]


def test_crew_is_grouped_by_role_in_display_order():
    crew = [
        {"id": 3, "name": "Продюсер", "job": "Producer"},
        {"id": 1, "name": "Автор", "job": "Writer"},
        {"id": 1, "name": "Автор", "job": "Screenplay"},
        {"id": 1, "name": "Автор", "job": "Director"},
        {"id": 2, "name": "Не сценарист", "job": "Novel"},
        {"id": None, "name": "Без id", "job": "Director"},
    ]
    assert extract_crew(crew) == [
        CrewMember(1, "Автор", CrewRole.DIRECTOR),
        CrewMember(1, "Автор", CrewRole.WRITER),
        CrewMember(3, "Продюсер", CrewRole.PRODUCER),
    ]


@pytest.mark.parametrize(
    ("raw", "expected"), [("/a.jpg", "/a.jpg"), (" /a.jpg ", "/a.jpg"), ("a.jpg", None), ("", None)]
)
def test_actor_profile_path(raw, expected):
    actors = extract_actors([{"id": 1, "name": "Актёр", "profile_path": raw}])
    assert actors[0].profile_path == expected


def test_actors_are_deduplicated_ordered_and_capped():
    cast = [{"id": index, "name": f"Актёр {index}", "order": 100 - index} for index in range(MAX_ACTORS + 10)]
    cast.append({"id": 5, "name": "Повтор", "order": -1})
    cast.append({"id": None, "name": "Без id", "order": -2})
    cast.append({"id": 999, "name": "Без порядка"})

    actors = extract_actors(cast)

    assert len(actors) == MAX_ACTORS
    assert actors[0] == Actor(5, "Повтор", None, None)
    assert len({actor.person_id for actor in actors}) == MAX_ACTORS
    assert 999 not in {actor.person_id for actor in actors}
