import argparse
import io
import logging
import signal
import sys
from dataclasses import replace
from datetime import date
from enum import IntEnum
from pathlib import Path

from scripts.catalog.candidates.discovery import CandidateDiscovery
from scripts.catalog.candidates.priority_list import PriorityFileError, load_priority_ids
from scripts.catalog.config import MAX_TARGET_SIZE, ConfigError, Settings, load_settings
from scripts.catalog.importer import CatalogAbortError, CatalogImporter, RunSummary
from scripts.catalog.report import build_report, format_report
from scripts.catalog.storage.catalog_store import CatalogStore
from scripts.catalog.storage.json_files import CatalogFileError
from scripts.catalog.tmdb.client import TmdbClient
from scripts.catalog.tmdb.errors import TmdbAuthError, TmdbUnavailableError
from scripts.catalog.tmdb.rate_limiter import RateLimiter

RESUME_HINT = "Уже сохранённые фильмы не потеряются, повторный запуск продолжит с места остановки"

logger = logging.getLogger(__name__)


class ExitCode(IntEnum):
    OK = 0
    FAILURE = 1
    CONFIG_ERROR = 2
    TARGET_NOT_REACHED = 3
    INTERRUPTED = 130


def main(argv: list[str] | None = None) -> int:
    _use_utf8_output()
    signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)
    try:
        return run(args)
    except (ConfigError, PriorityFileError, CatalogFileError, TmdbAuthError) as error:
        logger.error("%s", error)
        return ExitCode.CONFIG_ERROR
    except (TmdbUnavailableError, CatalogAbortError) as error:
        logger.error("%s. %s", error, RESUME_HINT)
        return ExitCode.FAILURE
    except OSError as error:
        logger.error("Не удалось записать каталог: %s", error)
        return ExitCode.FAILURE
    except KeyboardInterrupt:
        logger.warning("Прервано. %s", RESUME_HINT)
        return ExitCode.INTERRUPTED


def _use_utf8_output() -> None:
    """При выводе в файл или конвейер Python берёт кодировку системы, а в cp1252 кириллица не кодируется."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def _interrupt_on_sigterm(_signum: int, _frame: object) -> None:
    """kill и docker stop завершают процесс как Ctrl+C: прогресс успевает сохраниться."""
    raise KeyboardInterrupt


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.catalog",
        description="Собирает каталог фильмов из TMDB в JSON",
    )
    parser.add_argument("--env-file", type=Path, help="путь к .env (по умолчанию scripts/catalog/.env)")
    parser.add_argument("--target", type=_target_size, help="сколько фильмов должно быть в каталоге")
    parser.add_argument("--report-only", action="store_true", help="только напечатать отчёт о полноте данных")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог, включая отклонённые фильмы")
    return parser.parse_args(argv)


def configure_logging(*, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # urllib3 в DEBUG печатает URL запросов вместе с api_key.
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def run(args: argparse.Namespace) -> int:
    today = date.today()
    settings = load_settings(args.env_file, today.year, require_tmdb_key=not args.report_only)
    if args.target is not None:
        settings = replace(settings, target_size=args.target)
    store = CatalogStore.open(settings.output_dir)

    if args.report_only:
        print(format_report(build_report(store)))
        return ExitCode.OK

    summary = collect_catalog(store, settings, today)
    print(format_report(build_report(store)))
    logger.info("Каталог: %s", store.movies_path.resolve())
    return summarize(summary)


def collect_catalog(store: CatalogStore, settings: Settings, today: date) -> RunSummary:
    priority_ids = load_priority_ids(settings.priority_ids_file) if settings.priority_ids_file else []
    if priority_ids:
        logger.info("Приоритетных фильмов в списке: %d", len(priority_ids))
    client = TmdbClient(RateLimiter(settings.requests_per_second), settings.tmdb_access_token, settings.tmdb_api_key)
    discovery = CandidateDiscovery(client, settings.year_from, settings.min_vote_count, today)
    importer = CatalogImporter(client, store, discovery, settings.target_size, settings.poster_size)
    return importer.run(priority_ids)


def summarize(summary: RunSummary) -> int:
    logger.info(
        "Запуск завершён: сохранено %d, отклонено %d, не загрузилось %d. В каталоге %d из %d",
        summary.imported,
        summary.rejected,
        summary.failed,
        summary.total_movies,
        summary.target_size,
    )
    if summary.target_reached:
        return ExitCode.OK
    if summary.failed:
        logger.warning("Цель не достигнута, %d фильмов не загрузились. %s", summary.failed, RESUME_HINT)
    else:
        logger.warning(
            "Цель не достигнута: подходящие кандидаты закончились. Уменьшите CATALOG_MIN_VOTE_COUNT "
            "или CATALOG_YEAR_FROM либо дополните список CATALOG_PRIORITY_IDS_FILE"
        )
    return ExitCode.TARGET_NOT_REACHED


def _target_size(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError:
        raise argparse.ArgumentTypeError("нужно целое число") from None
    if not 1 <= value <= MAX_TARGET_SIZE:
        raise argparse.ArgumentTypeError(f"допустимо от 1 до {MAX_TARGET_SIZE}")
    return value
