import os

import pytest
from psycopg.types.json import Jsonb

from scripts.catalog.tests.samples import sample_movie
from scripts.index.errors import IndexingError
from scripts.index.metadata import MovieTable, movie_to_row

MOVIES_DDL = """
CREATE TABLE movies (
    movie_id        bigint PRIMARY KEY CHECK (movie_id > 0),
    title           text COLLATE "ru-RU-x-icu" NOT NULL CHECK (title <> ''),
    original_title  text NOT NULL CHECK (original_title <> ''),
    release_date    date,
    age_rating      text CHECK (age_rating IN ('0+', '6+', '12+', '16+', '18+')),
    poster_path     text CHECK (poster_path LIKE '/%'),
    runtime_min     smallint CHECK (runtime_min > 0),
    overview        text CHECK (overview <> ''),
    genres          jsonb NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(genres) = 'array'),
    countries       jsonb NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(countries) = 'array'),
    crew            jsonb NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(crew) = 'array'),
    actors          jsonb NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(actors) = 'array'),
    cached_at       timestamptz NOT NULL DEFAULT now(),
    indexed_at      timestamptz
)
"""
DATABASE_URL = os.getenv("INDEX_TEST_DATABASE_URL")


def test_movie_to_row_wraps_lists_as_jsonb():
    row = movie_to_row(sample_movie(550, credits={"cast": [], "crew": []}))
    assert isinstance(row["genres"], Jsonb)
    assert row["actors"].obj == []
    assert row["genres"].obj == [{"id": 18, "name": "драма"}, {"id": 53, "name": "триллер"}]
    assert row["release_date"].isoformat() == "1999-10-15"
    assert "indexed_at" not in row


def test_unreachable_database_is_readable_error():
    table = MovieTable("postgresql://user:secret@127.0.0.1:1/dejaview?connect_timeout=1")
    with pytest.raises(IndexingError) as error:
        table.check()
    assert "DATABASE_URL" in str(error.value)
    assert "secret" not in str(error.value)


@pytest.fixture
def database_url():
    if not DATABASE_URL:
        pytest.skip("INDEX_TEST_DATABASE_URL не задан")
    import psycopg  # noqa: PLC0415

    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        connection.execute("DROP SCHEMA IF EXISTS index_test CASCADE")
        connection.execute("CREATE SCHEMA index_test")
    yield f"{DATABASE_URL}{'&' if '?' in DATABASE_URL else '?'}options=-csearch_path%3Dindex_test"
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        connection.execute("DROP SCHEMA index_test CASCADE")


def test_missing_table(database_url):
    with pytest.raises(IndexingError, match="нет таблицы movies"):
        MovieTable(database_url).check()


def test_upsert_and_indexed_at_against_real_schema(database_url):
    import psycopg  # noqa: PLC0415

    with psycopg.connect(database_url) as connection:
        has_icu = connection.execute("SELECT 1 FROM pg_collation WHERE collname = 'ru-RU-x-icu'").fetchone()
        connection.execute(MOVIES_DDL if has_icu else MOVIES_DDL.replace('COLLATE "ru-RU-x-icu" ', ""))
    table = MovieTable(database_url)
    table.check()

    def stored():
        with psycopg.connect(database_url) as connection:
            return connection.execute(
                "SELECT title, runtime_min, actors, indexed_at FROM movies WHERE movie_id = 550"
            ).fetchone()

    table.upsert(sample_movie(550))
    title, runtime_min, actors, indexed_at = stored()
    assert (title, runtime_min) == ("Бойцовский клуб", 139)
    assert actors[0]["name"] == "Edward Norton"
    assert indexed_at is None

    table.mark_indexed(550)
    assert stored()[3] is not None

    table.upsert(sample_movie(550, runtime=151))
    _, runtime_min, _, indexed_at = stored()
    assert runtime_min == 151
    assert indexed_at is None

    with pytest.raises(IndexingError, match="пропала из movies"):
        table.mark_indexed(603)
