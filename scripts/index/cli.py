import argparse
import io
import logging
import signal
import sys
from enum import IntEnum
from functools import partial
from pathlib import Path

import psycopg

from dejaview_ml.embeddings import Encoders
from scripts.catalog.movies.parser import MovieDataError
from scripts.catalog.movies.rejection import MovieRejectedError, RejectReason
from scripts.catalog.tmdb.errors import TmdbAuthError, TmdbError
from scripts.index import vectors
from scripts.index.config import load_settings
from scripts.index.errors import ConfigError, IndexingError, UnsuitableFileError
from scripts.index.metadata import MovieTable, fetch_metadata
from scripts.index.pipeline import Result, Services, index_movie

NOT_INDEXED_HINT = "Фильм не проиндексирован (indexed_at не выставлен), исходный файл не тронут"
REJECT_REASONS = {
    RejectReason.NOT_FOUND: "такого movie_id нет в TMDB",
    RejectReason.ADULT: "фильм для взрослых",
    RejectReason.NO_POSTER: "у фильма нет постера",
    RejectReason.NO_RUSSIAN_TITLE: "у фильма нет русского названия",
    RejectReason.NO_RUSSIAN_OVERVIEW: "у фильма нет русского описания, без него не построить вектор text",
}

logger = logging.getLogger(__name__)


class ExitCode(IntEnum):
    OK = 0
    FAILURE = 1
    CONFIG_ERROR = 2
    REJECTED = 3
    INTERRUPTED = 130


def main(argv: list[str] | None = None) -> int:
    _use_utf8_output()
    signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)
    try:
        result = run(args)
    except KeyboardInterrupt:
        logger.warning("Прервано. %s, временные файлы удалены", NOT_INDEXED_HINT)
        return ExitCode.INTERRUPTED
    except Exception as error:
        return _fail(*describe_error(error))
    summarize(result)
    return ExitCode.OK


def describe_error(error: Exception) -> tuple[ExitCode, str]:
    if isinstance(error, ConfigError | TmdbAuthError):
        code, message = ExitCode.CONFIG_ERROR, str(error)
    elif isinstance(error, UnsuitableFileError):
        code, message = ExitCode.REJECTED, str(error)
    elif isinstance(error, MovieRejectedError):
        reason = REJECT_REASONS.get(error.reason, error.reason)
        code, message = ExitCode.REJECTED, f"Фильм {error.movie_id} не подходит для каталога: {reason}"
    elif isinstance(error, IndexingError | TmdbError | MovieDataError):
        code, message = ExitCode.FAILURE, f"{error}. {NOT_INDEXED_HINT}"
    elif isinstance(error, psycopg.Error):
        code, message = ExitCode.FAILURE, f"Ошибка PostgreSQL: {str(error).strip()}. {NOT_INDEXED_HINT}"
    else:
        code, message = ExitCode.FAILURE, f"Непредвиденная ошибка {type(error).__name__}: {error}. {NOT_INDEXED_HINT}"
    return code, message


def _fail(code: ExitCode, message: str) -> ExitCode:
    logger.error("%s", message)
    logger.debug("Трассировка:", exc_info=True)
    if not logger.isEnabledFor(logging.DEBUG) and code is ExitCode.FAILURE:
        logger.info("Подробности: запустите ещё раз с -v")
    return code


def _use_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def _interrupt_on_sigterm(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.index",
        description="Индексирует один фильм: видеофайл -> эмбеддинги в Qdrant, метаданные TMDB -> таблица movies",
    )
    parser.add_argument("file", type=Path, help="видеофайл MP4 (H.264 + AAC, 480p)")
    parser.add_argument("--movie-id", type=_movie_id, required=True, help="id фильма в TMDB, например 603")
    parser.add_argument("--env-file", type=Path, help="путь к .env (по умолчанию scripts/index/.env)")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог и трассировка ошибок")
    return parser.parse_args(argv)


def configure_logging(*, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("urllib3", "httpx", "httpcore", "filelock", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)


def run(args: argparse.Namespace) -> Result:
    settings = load_settings(args.env_file)
    ml = settings.ml
    writer = vectors.connect(
        ml.qdrant_url,
        ml.qdrant_grpc_port,
        api_key=ml.qdrant_api_key.get_secret_value() if ml.qdrant_api_key else None,
        prefer_grpc=ml.qdrant_prefer_grpc,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )
    services = Services(
        fetch_movie=partial(
            fetch_metadata,
            access_token=settings.tmdb_access_token,
            api_key=settings.tmdb_api_key,
            requests_per_second=settings.tmdb_requests_per_second,
        ),
        movies=MovieTable(settings.database_url),
        vectors=writer,
        encoders=Encoders(ml),
    )
    try:
        return index_movie(args.file, args.movie_id, settings.work_dir, services)
    finally:
        writer.close()


def summarize(result: Result) -> None:
    minutes, seconds = divmod(round(result.elapsed_sec), 60)
    counts = ", ".join(f"{name} {count}" for name, count in result.point_counts.items())
    logger.info(
        "Готово за %d мин %d с: фильм %d «%s» проиндексирован. Точек: %s",
        minutes,
        seconds,
        result.movie.movie_id,
        result.movie.title,
        counts,
    )


def _movie_id(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError:
        raise argparse.ArgumentTypeError("нужно целое число, id фильма в TMDB") from None
    if not 0 < value < 2**63:
        raise argparse.ArgumentTypeError("id фильма должен быть положительным")
    return value
