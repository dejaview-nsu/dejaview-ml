"""Каталог в двух JSON-файлах: movies.json со списком фильмов и candidates.json с очередью отбора.

movies.json пишется первым. Если процесс упадёт между двумя записями, фильм уже сохранён,
а очередь догонит его при следующем запуске через mark_already_imported.
"""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar

from scripts.catalog.candidates.model import Candidate, CandidateStatus
from scripts.catalog.movies.model import Movie
from scripts.catalog.movies.rejection import RejectReason
from scripts.catalog.storage.json_files import CatalogFileError, read_json_list, write_json_atomically
from scripts.catalog.storage.records import (
    InvalidRecordError,
    candidate_from_json,
    candidate_to_json,
    movie_id_of,
    movie_to_json,
)

MOVIES_FILE_NAME = "movies.json"
CANDIDATES_FILE_NAME = "candidates.json"
SAVE_EVERY_CHANGES = 20
MAX_ERROR_TEXT_LENGTH = 1000

T = TypeVar("T")


class CatalogStore:
    """Каталог и очередь в памяти, на диск сбрасываются каждые SAVE_EVERY_CHANGES изменений и по save()."""

    def __init__(self, output_dir: Path, movies: dict[int, dict[str, Any]], candidates: dict[int, Candidate]) -> None:
        self.movies_path = output_dir / MOVIES_FILE_NAME
        self.candidates_path = output_dir / CANDIDATES_FILE_NAME
        self._movies = movies
        self._candidates = candidates
        self._unsaved_changes = 0

    @classmethod
    def open(cls, output_dir: Path) -> "CatalogStore":
        movies = _load_keyed(output_dir / MOVIES_FILE_NAME, _movie_entry)
        candidates = _load_keyed(output_dir / CANDIDATES_FILE_NAME, _candidate_entry)
        return cls(output_dir, movies, candidates)

    def movie_count(self) -> int:
        return len(self._movies)

    def movies(self) -> list[dict[str, Any]]:
        return [self._movies[movie_id] for movie_id in sorted(self._movies)]

    def candidates(self) -> list[Candidate]:
        return sorted(self._candidates.values(), key=lambda candidate: candidate.queue_order)

    def known_movie_ids(self) -> set[int]:
        return self._movies.keys() | self._candidates.keys()

    def pending_movie_ids(self, max_attempts: int) -> list[int]:
        """Новые кандидаты и упавшие, у которых ещё остались попытки."""
        return [
            candidate.movie_id
            for candidate in self.candidates()
            if candidate.status is CandidateStatus.PENDING
            or (candidate.status is CandidateStatus.FAILED and candidate.attempts < max_attempts)
        ]

    def add_candidates(self, new_candidates: list[Candidate]) -> None:
        """Новые встают в конец очереди. Уже известные не теряют статус, но могут подняться в приоритете."""
        next_position = max((c.queue_position for c in self._candidates.values()), default=0) + 1
        for candidate in new_candidates:
            existing = self._candidates.get(candidate.movie_id)
            if existing is None:
                self._put(replace(candidate, queue_position=next_position))
                next_position += 1
            elif candidate.priority < existing.priority:
                self._put(replace(existing, priority=candidate.priority))

    def mark_already_imported(self) -> None:
        """Фильм мог попасть в каталог без отметки в очереди: сбой между записями или другая утилита."""
        for candidate in list(self._candidates.values()):
            if candidate.movie_id in self._movies and candidate.status is not CandidateStatus.IMPORTED:
                self._put(replace(candidate, status=CandidateStatus.IMPORTED))

    def save_movie(self, movie: Movie) -> None:
        """Повторное сохранение того же movie_id обновляет запись, а не создаёт дубликат."""
        self._movies[movie.movie_id] = movie_to_json(movie)
        if movie.movie_id in self._candidates:
            self._finish_attempt(movie.movie_id, CandidateStatus.IMPORTED)
        else:
            self._record_change()

    def mark_rejected(self, movie_id: int, reason: RejectReason) -> None:
        self._finish_attempt(movie_id, CandidateStatus.REJECTED, reject_reason=reason)

    def mark_failed(self, movie_id: int, error_text: str) -> None:
        self._finish_attempt(movie_id, CandidateStatus.FAILED, last_error=error_text[:MAX_ERROR_TEXT_LENGTH])

    def save(self) -> None:
        write_json_atomically(self.movies_path, self.movies())
        write_json_atomically(self.candidates_path, [candidate_to_json(c) for c in self.candidates()])
        self._unsaved_changes = 0

    def _finish_attempt(self, movie_id: int, status: CandidateStatus, **changes: Any) -> None:
        candidate = self._candidates[movie_id]
        updates = {"status": status, "attempts": candidate.attempts + 1, "last_error": None, **changes}
        self._put(replace(candidate, **updates))

    def _put(self, candidate: Candidate) -> None:
        self._candidates[candidate.movie_id] = candidate
        self._record_change()

    def _record_change(self) -> None:
        self._unsaved_changes += 1
        if self._unsaved_changes >= SAVE_EVERY_CHANGES:
            self.save()


def _movie_entry(record: Any) -> tuple[int, dict[str, Any]]:
    return movie_id_of(record), record


def _candidate_entry(record: Any) -> tuple[int, Candidate]:
    candidate = candidate_from_json(record)
    return candidate.movie_id, candidate


def _load_keyed(path: Path, parse: Callable[[Any], tuple[int, T]]) -> dict[int, T]:
    items: dict[int, T] = {}
    for index, record in enumerate(read_json_list(path)):
        try:
            key, item = parse(record)
        except InvalidRecordError as error:
            raise CatalogFileError(f"{path}: элемент {index}: {error}") from None
        if key in items:
            raise CatalogFileError(f"{path}: movie_id {key} встречается дважды")
        items[key] = item
    return items
