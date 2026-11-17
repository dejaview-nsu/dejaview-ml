"""Точки text с сюжетом фильма из Википедии (source=plot): текстовый поиск по описанию сцены.

Статья находится по id TMDB через Wikidata, раздел сюжета режется на куски примерно по 150 слов,
каждый кусок - точка text. Описание, метаданные, кадры, видео, звук и реплики не трогаются.
Ответы Википедии кешируются в INDEX_WORK_DIR/wiki: повторный запуск в неё не ходит без --refresh.

Запуск из корня репозитория:
    python -m scripts.index.plot 120 603
    python -m scripts.index.plot --all              # все проиндексированные фильмы из movies
    python -m scripts.index.plot --all --dry-run    # только найти сюжеты, ничего не записывать
"""

import argparse
import logging
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path

from dejaview_ml.embeddings import Encoders
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
from scripts.index.embedding import embed_plot, load_model
from scripts.index.metadata import MovieTable
from scripts.index.text import read_movie_ids
from scripts.index.wiki import Wikipedia, plot_chunks

STOPPED_HINT = "Прогон остановлен: сюжеты, записанные до сбоя, остаются в поиске, повторный запуск безопасен"

logger = logging.getLogger("scripts.index")


def main(argv: list[str] | None = None) -> int:
    use_utf8_output()
    signal.signal(signal.SIGTERM, interrupt_on_sigterm)
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)
    try:
        movie_ids = read_movie_ids(args.movie_ids, args.ids_file, required=not args.all)
        return run(movie_ids, args)
    except KeyboardInterrupt:
        logger.warning("Прервано. Сюжеты, записанные до этого, остаются в поиске, повторный запуск безопасен")
        return ExitCode.INTERRUPTED
    except Exception as error:
        code, message = describe_error(error, STOPPED_HINT)
        logger.error(message)
        logger.debug("Трассировка", exc_info=error)
        return code


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.index.plot",
        description="Записывает сюжет фильма из Википедии точками text (source=plot). Остальные точки фильма "
        "не трогаются.",
    )
    parser.add_argument("movie_ids", nargs="*", type=parse_movie_id, metavar="MOVIE_ID", help="id фильмов в TMDB")
    parser.add_argument("--ids-file", type=Path, help="файл с id фильмов в TMDB: по одному на строку, # - комментарий")
    parser.add_argument("--all", action="store_true", help="все проиндексированные фильмы из movies")
    parser.add_argument("--refresh", action="store_true", help="заново запросить Википедию, не глядя в кеш")
    parser.add_argument("--dry-run", action="store_true", help="найти сюжеты и показать куски, ничего не записывать")
    parser.add_argument("--env-file", type=Path, help="файл окружения, по умолчанию scripts/index/.env")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог и трассировка ошибок")
    return parser.parse_args(argv)


def run(movie_ids: list[int], args: argparse.Namespace) -> ExitCode:
    started = time.monotonic()
    settings = load_settings(args.env_file, require_tmdb=False)
    movies = MovieTable(settings.database_url)
    movies.check()
    if args.all:
        movie_ids = list(dict.fromkeys([*movie_ids, *movies.indexed_ids()]))
    known = [movie_id for movie_id in movie_ids if movies.load(movie_id) is not None]
    for movie_id in movie_ids:
        if movie_id not in known:
            logger.warning("Фильм %d пропущен: его нет в movies, сначала проиндексируйте его", movie_id)
    cache = settings.work_dir / "wiki"
    logger.info("Поиск сюжетов в Википедии для %d фильмов (кеш %s)", len(known), cache)
    plots = Wikipedia(cache, refresh=args.refresh).plots(known)
    found = {movie_id: plot_chunks(plot.text) for movie_id, plot in plots.items() if plot}
    for movie_id, plot in plots.items():
        if plot is None:
            logger.warning("Фильм %d: нет статьи с разделом сюжета, прежний сюжет удаляется", movie_id)
        else:
            logger.info(
                "Фильм %d: «%s» (%s), %d слов, кусков %d",
                movie_id,
                plot.title,
                plot.language,
                len(plot.text.split()),
                len(found[movie_id]),
            )
    if args.dry_run:
        logger.info("Сухой прогон: сюжет найден у %d из %d фильмов, ничего не записано", len(found), len(known))
        return ExitCode.OK

    writer = connect_vectors(settings)
    try:
        writer.check_collections()
        encoders = Encoders(settings.ml)
        load_model(encoders, "text")
        for number, movie_id in enumerate(known, start=1):
            count = writer.replace_plot(movie_id, embed_plot(encoders, found.get(movie_id, [])))
            logger.info("%d/%d: фильм %d, точек сюжета %d", number, len(known), movie_id, count)
    finally:
        with suppress(Exception):
            writer.close()
    minutes, seconds = divmod(round(time.monotonic() - started), 60)
    logger.info("Готово за %d мин %d с: сюжет записан для %d из %d фильмов", minutes, seconds, len(found), len(known))
    return ExitCode.OK


if __name__ == "__main__":
    sys.exit(main())
