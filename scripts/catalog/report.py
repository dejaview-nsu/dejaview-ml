"""Отчёт о полноте каталога: входные данные для решения по риску #17300."""

from collections import Counter
from dataclasses import dataclass
from typing import Any

from scripts.catalog.candidates.model import CandidateStatus
from scripts.catalog.movies.rejection import RejectReason
from scripts.catalog.storage.catalog_store import CatalogStore

RISK_INCOMPLETE_THRESHOLD_PERCENT = 10
RISK_NO_TRANSLATION_THRESHOLD_PERCENT = 5

# Название, постер и описание не проверяются: без них фильм в каталог не попадает.
OPTIONAL_FIELDS = {
    "age_rating": "Возрастное ограничение (RU)",
    "release_date": "Год выпуска",
    "runtime_minutes": "Длительность",
    "country_codes": "Страны производства",
    "genres": "Жанры",
    "directors": "Режиссёры",
    "writers": "Сценаристы",
    "composers": "Композиторы",
    "producers": "Продюсеры",
    "actors": "Актёры",
}
RISK_FIELDS = ("poster_url", "overview_ru", "actors")

REJECT_REASON_LABELS = {
    RejectReason.NO_RUSSIAN_TITLE: "нет русского названия",
    RejectReason.NO_RUSSIAN_OVERVIEW: "нет русского описания",
    RejectReason.NO_POSTER: "нет постера",
    RejectReason.ADULT: "контент для взрослых",
    RejectReason.NOT_FOUND: "не найден в TMDB",
}
NO_TRANSLATION_REASONS = (RejectReason.NO_RUSSIAN_TITLE, RejectReason.NO_RUSSIAN_OVERVIEW)


@dataclass(frozen=True)
class CompletenessReport:
    total_movies: int
    incomplete_movies: int
    risk_incomplete_movies: int
    missing_by_field: dict[str, int]
    status_counts: Counter[CandidateStatus]
    reject_counts: Counter[RejectReason]

    @property
    def checked_candidates(self) -> int:
        return self.status_counts[CandidateStatus.IMPORTED] + self.status_counts[CandidateStatus.REJECTED]

    @property
    def no_translation_rejects(self) -> int:
        return sum(self.reject_counts[reason] for reason in NO_TRANSLATION_REASONS)


def build_report(store: CatalogStore) -> CompletenessReport:
    movies = store.movies()
    candidates = store.candidates()
    return CompletenessReport(
        total_movies=len(movies),
        incomplete_movies=sum(_has_empty_field(movie, OPTIONAL_FIELDS) for movie in movies),
        risk_incomplete_movies=sum(_has_empty_field(movie, RISK_FIELDS) for movie in movies),
        missing_by_field={
            label: sum(_is_empty(movie.get(field)) for movie in movies) for field, label in OPTIONAL_FIELDS.items()
        },
        status_counts=Counter(candidate.status for candidate in candidates),
        reject_counts=Counter(
            candidate.reject_reason for candidate in candidates if candidate.status is CandidateStatus.REJECTED
        ),
    )


def format_report(report: CompletenessReport) -> str:
    total = report.total_movies
    lines = [
        f"Фильмов в каталоге: {total}",
        f"С неполными данными (пустое хотя бы одно поле ниже): {_share(report.incomplete_movies, total)}",
        f"Индикатор риска #17300 «нет постера, описания или актёров»: "
        f"{_share(report.risk_incomplete_movies, total)}, порог {RISK_INCOMPLETE_THRESHOLD_PERCENT}%",
        "",
        "Пустые поля:",
        *(f"  {label}: {_share(count, total)}" for label, count in report.missing_by_field.items()),
        "",
        *_selection_lines(report),
    ]
    return "\n".join(lines)


def _selection_lines(report: CompletenessReport) -> list[str]:
    checked = report.checked_candidates
    rejected = report.status_counts[CandidateStatus.REJECTED]
    lines = [f"Отбор: проверено {checked} фильмов, отклонено {_share(rejected, checked)}"]
    lines += [
        f"  {REJECT_REASON_LABELS.get(reason, str(reason))}: {_share(count, checked)}"
        for reason, count in report.reject_counts.most_common()
    ]
    lines.append(
        f"Без русской локализации среди проверенных: {_share(report.no_translation_rejects, checked)} "
        f"(порог риска #17300 {RISK_NO_TRANSLATION_THRESHOLD_PERCENT}%; в каталог такие фильмы не попадают)"
    )
    failed = report.status_counts[CandidateStatus.FAILED]
    pending = report.status_counts[CandidateStatus.PENDING]
    if failed or pending:
        lines.append(f"В очереди: не загрузились {failed}, ещё не проверены {pending}")
    return lines


def _is_empty(value: Any) -> bool:
    return value is None or value in ("", [])


def _has_empty_field(movie: dict[str, Any], fields: dict[str, str] | tuple[str, ...]) -> bool:
    return any(_is_empty(movie.get(field)) for field in fields)


def _share(count: int, total: int) -> str:
    if total == 0:
        return f"{count} (—)"
    return f"{count} ({count / total:.1%})"
