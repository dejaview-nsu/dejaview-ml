from dataclasses import asdict
from typing import Any

import psycopg
from psycopg import errors as pg_errors
from psycopg.types.json import Jsonb

from scripts.catalog.importer import fetch_movie
from scripts.catalog.movies.model import Movie
from scripts.catalog.movies.parser import parse_country_names
from scripts.catalog.tmdb.client import TmdbClient
from scripts.catalog.tmdb.rate_limiter import RateLimiter
from scripts.index.errors import IndexingError

CONNECT_TIMEOUT_SECONDS = 15
JSONB_COLUMNS = ("genres", "countries", "crew", "actors")

UPSERT_MOVIE = """
INSERT INTO movies (movie_id, title, original_title, release_date, age_rating, poster_path,
                    runtime_min, overview, genres, countries, crew, actors, cached_at, indexed_at)
VALUES (%(movie_id)s, %(title)s, %(original_title)s, %(release_date)s, %(age_rating)s, %(poster_path)s,
        %(runtime_min)s, %(overview)s, %(genres)s, %(countries)s, %(crew)s, %(actors)s, %(cached_at)s, NULL)
ON CONFLICT (movie_id) DO UPDATE SET
    title = EXCLUDED.title, original_title = EXCLUDED.original_title,
    release_date = EXCLUDED.release_date, age_rating = EXCLUDED.age_rating,
    poster_path = EXCLUDED.poster_path, runtime_min = EXCLUDED.runtime_min,
    overview = EXCLUDED.overview, genres = EXCLUDED.genres, countries = EXCLUDED.countries,
    crew = EXCLUDED.crew, actors = EXCLUDED.actors, cached_at = EXCLUDED.cached_at
"""


def fetch_metadata(movie_id: int, access_token: str | None, api_key: str | None, requests_per_second: float) -> Movie:
    client = TmdbClient(RateLimiter(requests_per_second), access_token, api_key)
    return fetch_movie(client, movie_id, parse_country_names(client.get_countries()))


def movie_to_row(movie: Movie) -> dict[str, Any]:
    row = asdict(movie)
    for column in JSONB_COLUMNS:
        row[column] = Jsonb(row[column])
    return row


class MovieTable:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def check(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute("SELECT movie_id, indexed_at FROM movies LIMIT 0")
        except pg_errors.UndefinedTable:
            raise IndexingError(
                "В базе нет таблицы movies: примените миграции dejaview-backend (схема contracts/db-schema.sql)"
            ) from None

    def upsert(self, movie: Movie) -> None:
        with self._connect() as connection:
            connection.execute(UPSERT_MOVIE, movie_to_row(movie))

    def reset_indexed(self, movie_id: int) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE movies SET indexed_at = NULL WHERE movie_id = %s", (movie_id,))

    def mark_indexed(self, movie_id: int) -> None:
        with self._connect() as connection:
            cursor = connection.execute("UPDATE movies SET indexed_at = now() WHERE movie_id = %s", (movie_id,))
            if cursor.rowcount != 1:
                raise IndexingError(
                    f"Строка фильма {movie_id} пропала из movies во время индексации, indexed_at не выставлен. "
                    "Векторы записаны: запустите индексацию повторно или удалите их"
                )

    def _connect(self) -> psycopg.Connection:
        try:
            return psycopg.connect(self._database_url, connect_timeout=CONNECT_TIMEOUT_SECONDS)
        except psycopg.OperationalError as error:
            raise IndexingError(
                f"Не удалось подключиться к PostgreSQL: {str(error).strip()}. Проверьте DATABASE_URL и SSH-туннель"
            ) from None
