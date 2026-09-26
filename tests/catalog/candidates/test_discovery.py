from datetime import date

import pytest

from scripts.catalog.candidates.discovery import CandidateDiscovery, bucket_quota, build_decades, interleave
from scripts.catalog.candidates.model import Candidate, CandidateStatus, Priority
from scripts.catalog.movies.rejection import RejectReason
from scripts.catalog.tmdb.errors import TmdbRequestError
from tests.catalog.samples import discover_item

COMEDY = 35


class FakeDiscoverClient:
    """pages[genre_id] - список страниц выдачи, каждая страница - список элементов."""

    def __init__(self, pages: dict[int, list[list[dict]]], genres=None, failing_genres=()) -> None:
        self._pages = pages
        self._genres = genres if genres is not None else [{"id": gid, "name": f"жанр {gid}"} for gid in pages]
        self._failing_genres = set(failing_genres)
        self.requested_pages: list[tuple[int, int]] = []

    def get_genres(self):
        return self._genres

    def discover_movies(self, genre_id, released_from, released_to, min_vote_count, page):
        self.requested_pages.append((genre_id, page))
        if genre_id in self._failing_genres:
            raise TmdbRequestError("422")
        pages = self._pages.get(genre_id, [])
        results = pages[page - 1] if page <= len(pages) else []
        return {"results": results, "total_pages": len(pages)}


def single_bucket_discovery(client: FakeDiscoverClient) -> CandidateDiscovery:
    """Одно десятилетие: число корзин равно числу жанров."""
    return CandidateDiscovery(client, year_from=1990, min_vote_count=100, today=date(1995, 1, 1))


def test_build_decades_aligns_to_ten_years_and_stops_today():
    decades = build_decades(1985, date(2006, 5, 1))

    assert [(d.released_from, d.released_to) for d in decades] == [
        (date(1985, 1, 1), date(1989, 12, 31)),
        (date(1990, 1, 1), date(1999, 12, 31)),
        (date(2000, 1, 1), date(2006, 5, 1)),
    ]
    assert decades[0].label == "1980-е"


def test_build_decades_for_current_year_only():
    assert len(build_decades(2026, date(2026, 1, 1))) == 1


@pytest.mark.parametrize(("needed", "buckets", "expected"), [(1000, 152, 14), (1, 152, 1), (0, 10, 1), (10, 1, 20)])
def test_bucket_quota(needed, buckets, expected):
    assert bucket_quota(needed, buckets) == expected


def test_interleave_round_robin():
    first = [Candidate(movie_id, Priority.DISCOVERED, "a") for movie_id in (1, 2, 3)]
    second = [Candidate(10, Priority.DISCOVERED, "b")]

    assert [c.movie_id for c in interleave([first, [], second])] == [1, 10, 2, 3]
    assert interleave([]) == []


def test_pages_until_quota_and_skips_known_and_broken_items():
    first_page = [discover_item(1), discover_item(2, poster_path=None), {"id": "x"}, discover_item(3)]
    client = FakeDiscoverClient({COMEDY: [first_page, [discover_item(4)], [discover_item(5)]]})
    known_ids = {3}

    found = single_bucket_discovery(client).find(known_ids, needed_count=1)

    assert [(c.movie_id, c.status, c.reject_reason) for c in found] == [
        (1, CandidateStatus.PENDING, None),
        (2, CandidateStatus.REJECTED, RejectReason.NO_POSTER),
        (4, CandidateStatus.PENDING, None),
    ]
    assert all(c.priority is Priority.DISCOVERED and c.source_bucket == "1990-е / жанр 35" for c in found)
    assert client.requested_pages == [(COMEDY, 1), (COMEDY, 2)]
    assert known_ids == {3}


def test_stops_when_results_run_out():
    client = FakeDiscoverClient({COMEDY: [[discover_item(1)]]})

    found = single_bucket_discovery(client).find(set(), needed_count=5)

    assert [c.movie_id for c in found] == [1]
    assert client.requested_pages == [(COMEDY, 1)]


def test_failing_bucket_is_skipped():
    client = FakeDiscoverClient({COMEDY: [[discover_item(1)]], 2: [[discover_item(7)]]}, failing_genres={COMEDY})
    assert [c.movie_id for c in single_bucket_discovery(client).find(set(), 5)] == [7]


def test_mixes_buckets_and_deduplicates_across_them():
    client = FakeDiscoverClient(
        {1: [[discover_item(100), discover_item(101)]], 2: [[discover_item(100), discover_item(200)]]}
    )

    found = single_bucket_discovery(client).find(set(), needed_count=2)

    assert [c.movie_id for c in found] == [100, 200, 101]


def test_no_valid_genres_means_no_candidates():
    client = FakeDiscoverClient({}, genres=[{"id": "мусор"}])
    assert single_bucket_discovery(client).find(set(), 10) == []
