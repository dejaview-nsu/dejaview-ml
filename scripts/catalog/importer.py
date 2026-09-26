"""Сборка каталога: очередь кандидатов -> TMDB -> JSON, с продолжением после обрыва."""

import logging
from dataclasses import dataclass

from scripts.catalog.candidates.discovery import CandidateDiscovery
from scripts.catalog.candidates.model import CandidateStatus
from scripts.catalog.candidates.priority_list import build_manual_candidates
from scripts.catalog.movies.model import Movie
from scripts.catalog.movies.parser import MovieDataError, parse_movie
from scripts.catalog.movies.rejection import MovieRejectedError, RejectReason
from scripts.catalog.storage.catalog_store import CatalogStore
from scripts.catalog.tmdb.client import TmdbClient
from scripts.catalog.tmdb.errors import TmdbNotFoundError, TmdbRequestError, TmdbUnavailableError

MAX_ATTEMPTS_PER_MOVIE = 3
MAX_CONSECUTIVE_FAILURES = 5
MAX_TOP_UP_ROUNDS = 10

logger = logging.getLogger(__name__)


class CatalogAbortError(Exception):
    """Фильмы не загружаются один за другим: продолжать значит впустую тратить попытки всей очереди."""


@dataclass
class RunSummary:
    target_size: int
    total_movies: int = 0
    imported: int = 0
    rejected: int = 0
    failed: int = 0

    @property
    def target_reached(self) -> bool:
        return self.total_movies >= self.target_size


def fetch_movie(client: TmdbClient, movie_id: int, poster_size: str) -> Movie:
    """Переиспользуется утилитой индексации (#17650)."""
    try:
        details = client.get_movie(movie_id)
    except TmdbNotFoundError:
        raise MovieRejectedError(movie_id, RejectReason.NOT_FOUND) from None
    return parse_movie(details, movie_id, poster_size)


class CatalogImporter:
    def __init__(
        self,
        client: TmdbClient,
        store: CatalogStore,
        discovery: CandidateDiscovery,
        target_size: int,
        poster_size: str,
    ) -> None:
        self._client = client
        self._store = store
        self._discovery = discovery
        self._poster_size = poster_size
        self._summary = RunSummary(target_size)
        self._attempted_ids: set[int] = set()
        self._failure_streak = 0

    def run(self, priority_ids: list[int]) -> RunSummary:
        """Прогресс сохраняется на диск при любом завершении, в том числе по Ctrl+C."""
        try:
            self._collect(priority_ids)
        finally:
            self._store.save()
        return self._summary

    def _collect(self, priority_ids: list[int]) -> None:
        self._store.add_candidates(build_manual_candidates(priority_ids))
        self._store.mark_already_imported()
        self._summary.total_movies = self._store.movie_count()
        logger.info("В каталоге %d фильмов, цель %d", self._summary.total_movies, self._summary.target_size)

        top_up_rounds = 0
        while not self._summary.target_reached:
            if queue := self._next_queue():
                self._process_queue(queue)
            elif top_up_rounds < MAX_TOP_UP_ROUNDS and self._top_up_candidates():
                top_up_rounds += 1
            else:
                break

    def _next_queue(self) -> list[int]:
        """Каждый фильм пробуем не больше раза за запуск, упавшие повторятся при следующем."""
        pending = self._store.pending_movie_ids(MAX_ATTEMPTS_PER_MOVIE)
        return [movie_id for movie_id in pending if movie_id not in self._attempted_ids]

    def _top_up_candidates(self) -> bool:
        needed_count = self._summary.target_size - self._summary.total_movies
        new_candidates = self._discovery.find(self._store.known_movie_ids(), needed_count)
        self._store.add_candidates(new_candidates)
        pending_count = sum(c.status is CandidateStatus.PENDING for c in new_candidates)
        logger.info("Новых кандидатов: %d, отсеяно по выдаче: %d", pending_count, len(new_candidates) - pending_count)
        return pending_count > 0

    def _process_queue(self, queue: list[int]) -> None:
        for movie_id in queue:
            if self._summary.target_reached:
                return
            self._attempted_ids.add(movie_id)
            status = self._process_candidate(movie_id)
            self._count(status)
            self._track_failure_streak(status)

    def _process_candidate(self, movie_id: int) -> CandidateStatus:
        try:
            movie = fetch_movie(self._client, movie_id, self._poster_size)
        except MovieRejectedError as rejection:
            self._store.mark_rejected(movie_id, rejection.reason)
            logger.debug("Фильм %d отклонён: %s", movie_id, rejection.reason)
            return CandidateStatus.REJECTED
        except (TmdbUnavailableError, TmdbRequestError, MovieDataError) as error:
            self._store.mark_failed(movie_id, str(error))
            logger.warning("Фильм %d не загружен: %s", movie_id, error)
            return CandidateStatus.FAILED

        self._store.save_movie(movie)
        year = movie.release_date.year if movie.release_date else "год неизвестен"
        logger.info(
            "[%d/%d] %d «%s» (%s)",
            self._summary.total_movies + 1,
            self._summary.target_size,
            movie.movie_id,
            movie.title_ru,
            year,
        )
        return CandidateStatus.IMPORTED

    def _count(self, status: CandidateStatus) -> None:
        match status:
            case CandidateStatus.IMPORTED:
                self._summary.imported += 1
                self._summary.total_movies += 1
            case CandidateStatus.REJECTED:
                self._summary.rejected += 1
            case CandidateStatus.FAILED:
                self._summary.failed += 1

    def _track_failure_streak(self, status: CandidateStatus) -> None:
        self._failure_streak = self._failure_streak + 1 if status is CandidateStatus.FAILED else 0
        if self._failure_streak >= MAX_CONSECUTIVE_FAILURES:
            raise CatalogAbortError(
                f"{MAX_CONSECUTIVE_FAILURES} фильмов подряд не загрузились, работа остановлена. "
                "Проверьте доступность TMDB и запустите скрипт повторно"
            )
