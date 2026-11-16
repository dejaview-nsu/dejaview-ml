"""Переиндексация только видео: python -m scripts.index.video movie.mp4 --movie-id 603."""

import argparse
import logging
import signal
import sys
import tempfile
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
from scripts.index.embedding import embed_video
from scripts.index.errors import ConfigError
from scripts.index.metadata import MovieTable
from scripts.index.validation import probe_video, require_ffmpeg

STOPPED_HINT = (
    "Видеоиндекс мог обновиться частично: повторите команду. Остальные коллекции, метаданные и исходный файл сохранены"
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    use_utf8_output()
    signal.signal(signal.SIGTERM, interrupt_on_sigterm)
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)
    try:
        count = run(args)
    except KeyboardInterrupt:
        logger.warning("Прервано. %s", STOPPED_HINT)
        return ExitCode.INTERRUPTED
    except Exception as error:
        code, message = describe_error(error, STOPPED_HINT)
        logger.error("%s", message)
        logger.debug("Трассировка", exc_info=True)
        return code
    logger.info("Готово: фильм %d, точек video %d, исходный файл сохранён", args.movie_id, count)
    return ExitCode.OK


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.index.video",
        description="Обновляет только video существующего фильма: один вектор на секунду. "
        "Другие коллекции, метаданные и исходный файл сохраняются. Ключ TMDB не нужен.",
    )
    parser.add_argument("file", type=Path, help="полный фильм MP4 (H.264, 480p); звук необязателен")
    parser.add_argument("--movie-id", type=parse_movie_id, required=True, help="id существующего фильма в movies")
    parser.add_argument("--env-file", type=Path, help="путь к .env (по умолчанию scripts/index/.env)")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог и трассировка ошибок")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    settings = load_settings(args.env_file, require_tmdb=False)
    require_ffmpeg()
    info = probe_video(args.file, require_audio=False)
    movies = MovieTable(settings.database_url)
    movies.check()
    if movies.load(args.movie_id) is None:
        raise ConfigError(f"Фильма {args.movie_id} нет в movies: сначала добавьте его в каталог")

    writer = connect_vectors(settings)
    try:
        writer.check_collections(("video",))
        settings.work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"video-{args.movie_id}-", dir=settings.work_dir) as temp_dir:
            fragments = embed_video(Encoders(settings.ml), args.file, info, Path(temp_dir))
        # indexed_at не меняется: остальные виды поиска остаются доступны.
        return writer.update_video(args.movie_id, fragments)
    finally:
        with suppress(Exception):
            writer.close()


if __name__ == "__main__":
    sys.exit(main())
