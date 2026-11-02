"""Точки коллекции text без видеофайла (#18244, #18913): описание и метаданные фильма.

Запуск из корня репозитория:
    python -m scripts.index.text 603 550 27205
    python -m scripts.index.text --ids-file ids.txt
    python -m scripts.index.text --all          # все проиндексированные фильмы из movies
"""

import argparse
import logging
import signal
import sys
from contextlib import suppress
from pathlib import Path

from dejaview_ml.embeddings import Encoders
from scripts.index.cli import (
    REJECT_REASONS,
    ExitCode,
    configure_logging,
    connect_vectors,
    describe_error,
    interrupt_on_sigterm,
    parse_movie_id,
    tmdb_metadata,
    use_utf8_output,
)
from scripts.index.config import load_settings
from scripts.index.embedding import load_model
from scripts.index.errors import ConfigError
from scripts.index.metadata import MovieTable
from scripts.index.pipeline import Services, TextResult, index_texts

STOPPED_HINT = "Прогон остановлен: фильмы, записанные до сбоя, остаются в поиске, повторный запуск безопасен"

logger = logging.getLogger("scripts.index")


def main(argv: list[str] | None = None) -> int:
    use_utf8_output()
    signal.signal(signal.SIGTERM, interrupt_on_sigterm)
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)
    try:
        movie_ids = read_movie_ids(args.movie_ids, args.ids_file, required=not args.all)
        result = run(movie_ids, args.env_file, all_indexed=args.all)
    except KeyboardInterrupt:
        logger.warning("Прервано. Фильмы, записанные до этого, остаются в поиске, повторный запуск безопасен")
        return ExitCode.INTERRUPTED
    except Exception as error:
        code, message = describe_error(error, STOPPED_HINT)
        logger.error(message)
        logger.debug("Трассировка", exc_info=error)
        return code
    return summarize(result)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.index.text",
        description="Записывает точки коллекции text без видеофайла: описание и метаданные фильма из movies "
        "(фильма нет в movies - из TMDB). Кадры, видео и звук фильма не трогаются.",
    )
    parser.add_argument("movie_ids", nargs="*", type=parse_movie_id, metavar="MOVIE_ID", help="id фильмов в TMDB")
    parser.add_argument("--ids-file", type=Path, help="файл с id фильмов в TMDB: по одному на строку, # - комментарий")
    parser.add_argument(
        "--all",
        action="store_true",
        help="все проиндексированные фильмы из movies, например после смены строки метаданных",
    )
    parser.add_argument("--env-file", type=Path, help="файл окружения, по умолчанию scripts/index/.env")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог и трассировка ошибок")
    return parser.parse_args(argv)


def read_movie_ids(movie_ids: list[int], ids_file: Path | None, *, required: bool = True) -> list[int]:
    ids = list(movie_ids)
    if ids_file is not None:
        try:
            lines = ids_file.read_text(encoding="utf-8-sig").splitlines()
        except OSError as error:
            raise ConfigError(f"Не удалось прочитать файл с id фильмов {ids_file}: {error}") from None
        for number, line in enumerate(lines, start=1):
            value = line.split("#", 1)[0].strip()
            if not value:
                continue
            try:
                ids.append(parse_movie_id(value))
            except argparse.ArgumentTypeError as error:
                raise ConfigError(f"{ids_file}, строка {number}: {error}, получено {value!r}") from None
    if not ids and required:
        raise ConfigError("Не задан ни один id фильма: перечислите их в команде, передайте --ids-file или --all")
    return list(dict.fromkeys(ids))


def run(movie_ids: list[int], env_file: Path | None, *, all_indexed: bool = False) -> TextResult:
    settings = load_settings(env_file)
    writer = connect_vectors(settings)
    services = Services(
        fetch_movie=tmdb_metadata(settings),
        movies=MovieTable(settings.database_url),
        vectors=writer,
        encoders=Encoders(settings.ml),
    )
    try:
        logger.info("Проверка Qdrant и PostgreSQL")
        services.vectors.check_collections()
        services.movies.check()
        if all_indexed:
            movie_ids = list(dict.fromkeys([*movie_ids, *services.movies.indexed_ids()]))
        load_model(services.encoders, "text")
        logger.info("Фильмов к индексации: %d", len(movie_ids))
        return index_texts(movie_ids, services)
    finally:
        with suppress(Exception):
            writer.close()


def summarize(result: TextResult) -> ExitCode:
    minutes, seconds = divmod(round(result.elapsed_sec), 60)
    logger.info(
        "Готово за %d мин %d с: текст записан для %d из %d фильмов", minutes, seconds, len(result.indexed), result.total
    )
    for movie_id, reason in result.rejected.items():
        logger.warning("Отклонён %d: %s", movie_id, REJECT_REASONS.get(reason, reason))
    for movie_id, error in result.failed.items():
        logger.error("Не получен из TMDB %d: %s", movie_id, error)
    if result.failed:
        logger.error("Повторите запуск для фильмов, не полученных из TMDB: записанные фильмы дублей не дадут")
        return ExitCode.FAILURE
    return ExitCode.REJECTED if result.rejected else ExitCode.OK


if __name__ == "__main__":
    sys.exit(main())
