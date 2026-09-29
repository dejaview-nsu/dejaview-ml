from dataclasses import dataclass
from enum import IntEnum, StrEnum

from scripts.catalog.movies.rejection import RejectReason


class CandidateStatus(StrEnum):
    PENDING = "pending"
    IMPORTED = "imported"
    REJECTED = "rejected"
    FAILED = "failed"


class Priority(IntEnum):
    """Меньше значение - раньше в очереди."""

    MANUAL = 0
    DISCOVERED = 1


@dataclass(frozen=True)
class Candidate:
    movie_id: int
    priority: Priority
    source_bucket: str
    status: CandidateStatus = CandidateStatus.PENDING
    reject_reason: RejectReason | None = None
    last_error: str | None = None
    attempts: int = 0
    queue_position: int = 0

    @property
    def queue_order(self) -> tuple[int, int]:
        return self.priority, self.queue_position
