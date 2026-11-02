import logging
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dejaview_ml.embeddings import Encoders
from scripts.catalog.movies.model import Movie
from scripts.catalog.movies.parser import MovieDataError
from scripts.catalog.movies.rejection import MovieRejectedError, RejectReason
from scripts.catalog.tmdb.errors import TmdbAuthError, TmdbError
from scripts.index.embedding import embed_movie, embed_text
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


@dataclass
class TextResult:
    total: int = 0
    indexed: list[Movie] = field(default_factory=list)
    rejected: dict[int, RejectReason] = field(default_factory=dict)
    failed: dict[int, str] = field(default_factory=dict)
    elapsed_sec: float = 0.0


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
    logger.info("Фильм %d «%s» записан в movies", movie_id, movie.title)

    work_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"movie-{movie_id}-", dir=work_dir))
    try:
        _stage(4, "нарезка и эмбеддинги")
        fragments = embed_movie(services.encoders, movie, source, info, temp_dir)

        _stage(5, "запись векторов в Qdrant")
        # Уже проиндексированный фильм остаётся в поиске, пока считаются эмбеддинги, и снимается
        # только на время замены точек.
        services.movies.reset_indexed(movie_id)
        point_counts = services.vectors.replace_movie(movie_id, fragments)
        services.movies.mark_indexed(movie_id)
        logger.info("indexed_at выставлен: фильм находится поиском")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    _stage(6, "очистка")
    try:
        source_removed = remove_source(source, work_dir)
    except (OSError, KeyboardInterrupt) as error:
        logger.warning("Фильм проиндексирован, но исходный файл не удалён (%s): %s", type(error).__name__, source)
        source_removed = False
    return Result(movie, point_counts, source_removed, time.monotonic() - started)


def index_texts(movie_ids: list[int], services: Services) -> TextResult:
    """Точки text без видеофайла (#18244, #18913): описание и метаданные фильма из movies. В TMDB - только
    за фильмом, которого в movies ещё нет. Кадры, видео и звук фильма не трогаются. Сбой TMDB по одному
    фильму не останавливает прогон, сбой PostgreSQL, Qdrant или отказ ключа TMDB - останавливает: дальше
    упадут все фильмы."""
    started = time.monotonic()
    result = TextResult(total=len(movie_ids))
    for number, movie_id in enumerate(movie_ids, start=1):
        prefix = f"{number}/{len(movie_ids)}: фильм {movie_id}"
        movie = services.movies.load(movie_id)
        if movie is None:
            try:
                movie = services.fetch_movie(movie_id)
            except MovieRejectedError as error:
                result.rejected[movie_id] = error.reason
                logger.warning("%s отклонён: %s", prefix, error.reason)
                continue
            except TmdbAuthError:
                raise
            except (TmdbError, MovieDataError) as error:
                result.failed[movie_id] = str(error)
                logger.error("%s не получен из TMDB: %s", prefix, error)
                continue
            services.movies.upsert(movie)
        fragments = embed_text(services.encoders, movie)
        # indexed_at не сбрасывается: точки перезаписываются по ID, и при сбое у фильма остаются прежние
        # или уже новые точки, а --all подхватит его повторным запуском.
        services.vectors.update_text(movie_id, fragments)
        services.movies.mark_indexed(movie_id)
        result.indexed.append(movie)
        logger.info("%s «%s»: точек text %d", prefix, movie.title, len(fragments))
    result.elapsed_sec = time.monotonic() - started
    return result


def remove_source(source: Path, work_dir: Path) -> bool:
    if not source.resolve().is_relative_to(work_dir.resolve()):
        logger.info("Исходный файл вне рабочего каталога %s, оставлен на месте: %s", work_dir, source)
        return False
    source.unlink()
    logger.info("Исходный файл удалён из рабочего каталога: %s", source)
    return True


def _stage(number: int, title: str) -> None:
    logger.info("Этап %d/%d: %s", number, STAGES, title)
