"""Преобразование записей каталога в JSON и обратно с проверкой того, что прочитано с диска."""

from dataclasses import asdict
from typing import Any

from scripts.catalog.candidates.model import Candidate, CandidateStatus, Priority
from scripts.catalog.json_values import as_int, as_positive_int
from scripts.catalog.movies.model import Movie
from scripts.catalog.movies.rejection import RejectReason


class InvalidRecordError(ValueError):
    pass


def movie_to_json(movie: Movie) -> dict[str, Any]:
    record = asdict(movie)
    record["release_date"] = movie.release_date.isoformat() if movie.release_date else None
    return record


def movie_id_of(record: Any) -> int:
    """Фильмы хранятся как есть, от записи требуется только корректный movie_id."""
    if not isinstance(record, dict):
        raise InvalidRecordError("запись не является объектом")
    movie_id = as_positive_int(record.get("movie_id"))
    if movie_id is None:
        raise InvalidRecordError("нет корректного movie_id")
    return movie_id


def candidate_to_json(candidate: Candidate) -> dict[str, Any]:
    return asdict(candidate)


def candidate_from_json(record: Any) -> Candidate:
    if not isinstance(record, dict):
        raise InvalidRecordError("запись не является объектом")
    unknown_fields = record.keys() - Candidate.__dataclass_fields__.keys()
    if unknown_fields:
        raise InvalidRecordError(f"неизвестные поля {sorted(unknown_fields)}")
    try:
        return Candidate(
            movie_id=_required(as_positive_int(record.get("movie_id")), "movie_id"),
            priority=Priority(_required(as_int(record.get("priority")), "priority")),
            source_bucket=_required_str(record.get("source_bucket"), "source_bucket"),
            status=CandidateStatus(record.get("status", CandidateStatus.PENDING)),
            reject_reason=_optional_reason(record.get("reject_reason")),
            last_error=_optional_str(record.get("last_error"), "last_error"),
            attempts=_non_negative(record.get("attempts", 0), "attempts"),
            queue_position=_non_negative(record.get("queue_position", 0), "queue_position"),
        )
    except ValueError as error:
        raise InvalidRecordError(str(error)) from None


def _required(value: int | None, field: str) -> int:
    if value is None:
        raise InvalidRecordError(f"поле {field} отсутствует или некорректно")
    return value


def _non_negative(value: Any, field: str) -> int:
    number = as_int(value)
    if number is None or number < 0:
        raise InvalidRecordError(f"поле {field} должно быть неотрицательным целым")
    return number


def _required_str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise InvalidRecordError(f"поле {field} должно быть строкой")
    return value


def _optional_str(value: Any, field: str) -> str | None:
    return None if value is None else _required_str(value, field)


def _optional_reason(value: Any) -> RejectReason | None:
    return None if value is None else RejectReason(value)
