import os
from dataclasses import asdict

import pytest
from psycopg.types.json import Jsonb

from scripts.catalog.tests.samples import sample_movie
from scripts.index.errors import IndexingError
from scripts.index.metadata import MovieTable, movie_from_row, movie_to_row

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

    indexed_at = stored()[3]
    table.upsert(sample_movie(550, runtime=151))
    _, runtime_min, _, still_indexed_at = stored()
    assert runtime_min == 151
    assert still_indexed_at == indexed_at

    table.reset_indexed(550)
    assert stored()[3] is None

    with pytest.raises(IndexingError, match="пропала из movies"):
        table.mark_indexed(603)

    assert table.load(550) == sample_movie(550, runtime=151)
    assert table.load(603) is None
    assert table.indexed_ids() == []
    table.mark_indexed(550)
    assert table.indexed_ids() == [550]


def test_movie_row_converts_back_to_movie():
    movie = sample_movie(550)
    # Так строку отдаёт psycopg: JSONB уже разобран в списки словарей.
    assert movie_from_row(asdict(movie)) == movie


def test_delete_cascades_and_is_repeatable(database_url):
    import psycopg  # noqa: PLC0415

    # Минимальные связанные таблицы с теми же внешними ключами, что в db-schema.sql.
    with psycopg.connect(database_url) as connection:
        connection.execute(MOVIES_DDL.replace('COLLATE "ru-RU-x-icu" ', ""))
        connection.execute("""
            CREATE TABLE user_movie_lists (
                movie_id bigint REFERENCES movies ON DELETE CASCADE
            );
            CREATE TABLE ratings (
                movie_id bigint REFERENCES movies ON DELETE CASCADE
            );
            CREATE TABLE reviews (
                review_id bigint PRIMARY KEY,
                movie_id bigint REFERENCES movies ON DELETE CASCADE
            );
            CREATE TABLE review_reactions (
                review_id bigint REFERENCES reviews ON DELETE CASCADE
            );
        """)
    table = MovieTable(database_url)
    table.upsert(sample_movie(550))
    table.upsert(sample_movie(603))
    with psycopg.connect(database_url) as connection:
        connection.execute("INSERT INTO user_movie_lists VALUES (550), (603)")
        connection.execute("INSERT INTO ratings VALUES (550), (603)")
        connection.execute("INSERT INTO reviews VALUES (1, 550), (2, 603)")
        connection.execute("INSERT INTO review_reactions VALUES (1), (2)")

    assert table.delete(550) is True
    assert table.delete(550) is False
    assert table.delete(999) is False
    assert table.load(550) is None
    assert table.load(603) is not None
    with psycopg.connect(database_url) as connection:
        assert connection.execute("SELECT movie_id FROM user_movie_lists").fetchall() == [(603,)]
        assert connection.execute("SELECT movie_id FROM ratings").fetchall() == [(603,)]
        assert connection.execute("SELECT movie_id FROM reviews").fetchall() == [(603,)]
        assert connection.execute("SELECT review_id FROM review_reactions").fetchall() == [(2,)]
