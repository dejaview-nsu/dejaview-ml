"""Удаление фильма из Qdrant и PostgreSQL (#18846).

Запуск из корня репозитория: python -m scripts.index.delete --movie-id 603
"""

import argparse
import logging
import signal
import sys
from contextlib import suppress
from pathlib import Path

from scripts.index.cli import (
    ExitCode,
    configure_logging,
    connect_vectors,
    describe_error,
    interrupt_on_sigterm,
    parse_movie_id,
    use_utf8_output,
)
from scripts.index.config import load_settings
from scripts.index.metadata import MovieTable
from scripts.index.vectors import VectorWriter

STOPPED_HINT = "Удаление могло выполниться частично. Устраните причину и повторите команду с тем же movie_id"

logger = logging.getLogger("scripts.index")


def main(argv: list[str] | None = None) -> int:
    use_utf8_output()
    signal.signal(signal.SIGTERM, interrupt_on_sigterm)
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)
    try:
        run(args.movie_id, args.env_file)
    except KeyboardInterrupt:
        logger.warning("Прервано. %s", STOPPED_HINT)
        return ExitCode.INTERRUPTED
    except Exception as error:
        code, message = describe_error(error, STOPPED_HINT)
        logger.error("%s", message)
        logger.debug("Трассировка", exc_info=True)
        return code
    logger.info("Готово: фильм %d отсутствует в Qdrant и movies", args.movie_id)
    return ExitCode.OK


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.index.delete",
        description="Удаляет фильм из четырёх коллекций Qdrant, затем из movies. "
        "Списки, оценки и отзывы удаляются каскадом. Повторный запуск безопасен.",
    )
    parser.add_argument("--movie-id", type=parse_movie_id, required=True, help="id фильма в TMDB, например 603")
    parser.add_argument("--env-file", type=Path, help="файл окружения, по умолчанию scripts/index/.env")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог и трассировка ошибок")
    return parser.parse_args(argv)


def run(movie_id: int, env_file: Path | None) -> None:
    settings = load_settings(env_file, require_tmdb=False)
    movies = MovieTable(settings.database_url)
    # Проверка доступности БД до первого изменения в Qdrant.
    movies.check()
    writer = connect_vectors(settings)
    try:
        delete_movie(movie_id, movies, writer)
    finally:
        with suppress(Exception):
            writer.close()


def delete_movie(movie_id: int, movies: MovieTable, writer: VectorWriter) -> None:
    logger.info("Этап 1/2: удаление точек фильма %d из Qdrant", movie_id)
    writer.delete_movie(movie_id)
    # При любой ошибке Qdrant до DELETE не доходим. Общей транзакции у двух хранилищ нет.
    logger.info("Этап 2/2: удаление фильма %d из PostgreSQL", movie_id)
    if movies.delete(movie_id):
        logger.info("Строка фильма %d удалена из movies", movie_id)
    else:
        logger.info("Фильм %d уже отсутствует в movies", movie_id)


if __name__ == "__main__":
    sys.exit(main())
