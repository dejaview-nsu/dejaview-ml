"""Поиск кандидатов в TMDB по корзинам «жанр × десятилетие», чтобы каталог был разнообразным."""

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from itertools import zip_longest
from typing import Any

from scripts.catalog.candidates.model import Candidate, CandidateStatus, Priority
from scripts.catalog.json_values import as_dict, as_int, as_list, as_positive_int, clean_text
from scripts.catalog.movies.rejection import listing_rejection_reason
from scripts.catalog.tmdb.client import MAX_DISCOVER_PAGE, TmdbClient
from scripts.catalog.tmdb.errors import TmdbRequestError

# Запас на отсев: часть найденных фильмов не пройдёт проверку русского названия.
CANDIDATE_POOL_FACTOR = 2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Genre:
    genre_id: int
    name: str


@dataclass(frozen=True)
class Decade:
    released_from: date
    released_to: date

    @property
    def label(self) -> str:
        return f"{self.released_from.year // 10 * 10}-е"


@dataclass(frozen=True)
class Bucket:
    genre: Genre
    decade: Decade

    @property
    def label(self) -> str:
        return f"{self.decade.label} / {self.genre.name}"


class CandidateDiscovery:
    def __init__(self, client: TmdbClient, year_from: int, min_vote_count: int, today: date) -> None:
        self._client = client
        self._year_from = year_from
        self._min_vote_count = min_vote_count
        self._today = today

    def find(self, known_ids: set[int], needed_count: int) -> list[Candidate]:
        """Новые кандидаты, по одному из каждой корзины по кругу. Фильмы из known_ids пропускаются."""
        buckets = self._build_buckets()
        if not buckets:
            return []
        quota = bucket_quota(needed_count, len(buckets))
        logger.info("Поиск кандидатов: %d корзин жанр × десятилетие, до %d фильмов в каждой", len(buckets), quota)
        seen_ids = set(known_ids)
        return interleave([self._scan_bucket(bucket, seen_ids, quota) for bucket in buckets])

    def _build_buckets(self) -> list[Bucket]:
        genres = self._fetch_genres()
        return [Bucket(genre, decade) for decade in build_decades(self._year_from, self._today) for genre in genres]

    def _fetch_genres(self) -> list[Genre]:
        genres = []
        for item in map(as_dict, self._client.get_genres()):
            genre_id = as_int(item.get("id"))
            name = clean_text(item.get("name"))
            if genre_id is not None and name is not None:
                genres.append(Genre(genre_id, name))
        return genres

    def _scan_bucket(self, bucket: Bucket, seen_ids: set[int], quota: int) -> list[Candidate]:
        """Идёт по выдаче корзины, пока не наберёт quota подходящих фильмов. seen_ids пополняется."""
        found: list[Candidate] = []
        accepted_count = 0
        for item in self._listing(bucket):
            movie_id = as_positive_int(item.get("id"))
            if movie_id is None or movie_id in seen_ids:
                continue
            seen_ids.add(movie_id)
            candidate = listing_item_to_candidate(movie_id, item, bucket)
            found.append(candidate)
            if candidate.status is CandidateStatus.PENDING:
                accepted_count += 1
                if accepted_count == quota:
                    break
        return found

    def _listing(self, bucket: Bucket) -> Iterator[dict[str, Any]]:
        """Элементы выдачи корзины страница за страницей; следующая страница запрашивается по требованию."""
        page, total_pages = 1, 1
        while page <= min(total_pages, MAX_DISCOVER_PAGE):
            try:
                payload = self._client.discover_movies(
                    bucket.genre.genre_id,
                    bucket.decade.released_from,
                    bucket.decade.released_to,
                    self._min_vote_count,
                    page,
                )
            except TmdbRequestError as error:
                logger.warning("Корзина «%s» пропущена: %s", bucket.label, error)
                return
            total_pages = as_int(payload.get("total_pages")) or 0
            yield from map(as_dict, as_list(payload.get("results")))
            page += 1


def build_decades(year_from: int, today: date) -> list[Decade]:
    return [
        Decade(date(max(start, year_from), 1, 1), min(date(start + 9, 12, 31), today))
        for start in range(year_from // 10 * 10, today.year + 1, 10)
    ]


def bucket_quota(needed_count: int, bucket_count: int) -> int:
    return max(1, math.ceil(needed_count * CANDIDATE_POOL_FACTOR / bucket_count))


def listing_item_to_candidate(movie_id: int, item: dict[str, Any], bucket: Bucket) -> Candidate:
    reason = listing_rejection_reason(item)
    status = CandidateStatus.PENDING if reason is None else CandidateStatus.REJECTED
    return Candidate(movie_id, Priority.DISCOVERED, bucket.label, status, reason)


def interleave(per_bucket: list[list[Candidate]]) -> list[Candidate]:
    """Первыми идут самые известные фильмы всех корзин, затем вторые по известности и так далее."""
    return [candidate for rank in zip_longest(*per_bucket) for candidate in rank if candidate is not None]
