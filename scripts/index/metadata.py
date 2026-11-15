from dataclasses import asdict
from typing import Any

import psycopg
from psycopg import errors as pg_errors
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from scripts.catalog.importer import fetch_movie
from scripts.catalog.movies.model import Actor, Country, CrewMember, CrewRole, Genre, Movie
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


class TmdbMetadata:
    """Метаданные фильма из TMDB кодом каталога #17640. Клиент и справочник стран создаются при первом
    запросе и переиспользуются: при индексации каталога это один запрос стран на весь прогон."""

    def __init__(self, access_token: str | None, api_key: str | None, requests_per_second: float) -> None:
        self._access_token = access_token
        self._api_key = api_key
        self._requests_per_second = requests_per_second
        self._client: TmdbClient | None = None
        self._country_names: dict[str, str] = {}

    def __call__(self, movie_id: int) -> Movie:
        if self._client is None:
            client = TmdbClient(RateLimiter(self._requests_per_second), self._access_token, self._api_key)
            self._country_names = parse_country_names(client.get_countries())
            self._client = client
        return fetch_movie(self._client, movie_id, self._country_names)


SELECT_MOVIE = """
SELECT movie_id, title, original_title, release_date, age_rating, poster_path, runtime_min, overview,
       genres, countries, crew, actors, cached_at
FROM movies WHERE movie_id = %s
"""


def movie_from_row(row: dict[str, Any]) -> Movie:
    """Строка movies -> Movie, как его отдаёт скрипт каталога: поля и ключи JSONB по db-schema.md."""
    return Movie(
        movie_id=row["movie_id"],
        title=row["title"],
        original_title=row["original_title"],
        release_date=row["release_date"],
        age_rating=row["age_rating"],
        poster_path=row["poster_path"],
        runtime_min=row["runtime_min"],
        overview=row["overview"],
        genres=[Genre(id=item["id"], name=item["name"]) for item in row["genres"]],
        countries=[Country(code=item["code"], name=item["name"]) for item in row["countries"]],
        crew=[
            CrewMember(person_id=item["person_id"], name=item["name"], role=CrewRole(item["role"]))
            for item in row["crew"]
        ],
        actors=[
            Actor(
                person_id=item["person_id"],
                name=item["name"],
                character=item.get("character"),
                profile_path=item.get("profile_path"),
            )
            for item in row["actors"]
        ],
        cached_at=row["cached_at"],
    )


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

    def load(self, movie_id: int) -> Movie | None:
        with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
            row = cursor.execute(SELECT_MOVIE, (movie_id,)).fetchone()
        return movie_from_row(row) if row else None

    def indexed_ids(self) -> list[int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT movie_id FROM movies WHERE indexed_at IS NOT NULL ORDER BY movie_id"
            ).fetchall()
        return [movie_id for (movie_id,) in rows]

    def upsert(self, movie: Movie) -> None:
        with self._connect() as connection:
            connection.execute(UPSERT_MOVIE, movie_to_row(movie))

    def reset_indexed(self, movie_id: int) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE movies SET indexed_at = NULL WHERE movie_id = %s", (movie_id,))

    def delete(self, movie_id: int) -> bool:
        """Удалить строку movies; связанные данные удаляются внешними ключами ON DELETE CASCADE."""
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM movies WHERE movie_id = %s", (movie_id,))
            return cursor.rowcount > 0

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
