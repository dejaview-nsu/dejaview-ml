import logging
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dejaview_ml.embeddings import Encoders
from scripts.catalog.movies.model import Movie
from scripts.index.embedding import embed_movie
from scripts.index.metadata import MovieTable
from scripts.index.validation import probe_video, require_ffmpeg
from scripts.index.vectors import VectorWriter

STAGES = 6

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Services:
    fetch_movie: Callable[[int], Movie]
    movies: MovieTable
    vectors: VectorWriter
    encoders: Encoders


@dataclass(frozen=True)
class Result:
    movie: Movie
    point_counts: dict[str, int]
    source_removed: bool
    elapsed_sec: float


def index_movie(source: Path, movie_id: int, work_dir: Path, services: Services) -> Result:
    started = time.monotonic()
    _stage(1, "проверка файла")
    require_ffmpeg()
    info = probe_video(source)
    logger.info(
        "Файл подходит: %d×%d, %s", info.width, info.height, time.strftime("%H:%M:%S", time.gmtime(info.duration_sec))
    )

    _stage(2, "проверка Qdrant и PostgreSQL")
    services.vectors.check_collections()
    services.movies.check()

    _stage(3, f"метаданные фильма {movie_id} из TMDB в таблицу movies")
    movie = services.fetch_movie(movie_id)
    services.movies.upsert(movie)
    logger.info("Фильм %d «%s» записан в movies, indexed_at = NULL до записи векторов", movie_id, movie.title)

    work_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"movie-{movie_id}-", dir=work_dir))
    try:
        _stage(4, "нарезка и эмбеддинги")
        fragments = embed_movie(services.encoders, movie.overview, source, info, temp_dir)

        _stage(5, "запись векторов в Qdrant")
        point_counts = services.vectors.replace_movie(movie_id, fragments)
        services.movies.mark_indexed(movie_id)
        logger.info("indexed_at выставлен: фильм находится поиском")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    _stage(6, "очистка")
    source_removed = remove_source(source, work_dir)
    return Result(movie, point_counts, source_removed, time.monotonic() - started)


def remove_source(source: Path, work_dir: Path) -> bool:
    if not source.resolve().is_relative_to(work_dir.resolve()):
        logger.info("Исходный файл вне рабочего каталога %s, оставлен на месте: %s", work_dir, source)
        return False
    source.unlink()
    logger.info("Исходный файл удалён из рабочего каталога: %s", source)
    return True


def _stage(number: int, title: str) -> None:
    logger.info("Этап %d/%d: %s", number, STAGES, title)
