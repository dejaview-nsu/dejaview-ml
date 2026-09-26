"""Сквозные тесты сборки каталога: TMDB подменён, файлы пишутся во временную папку."""

import json
from datetime import date

import pytest

from scripts.catalog.candidates.discovery import CandidateDiscovery
from scripts.catalog.candidates.model import CandidateStatus
from scripts.catalog.importer import MAX_CONSECUTIVE_FAILURES, CatalogAbortError, CatalogImporter, fetch_movie
from scripts.catalog.movies.rejection import RejectReason
from scripts.catalog.report import build_report, format_report
from scripts.catalog.storage.catalog_store import CatalogStore
from scripts.catalog.tmdb.errors import TmdbNotFoundError, TmdbUnavailableError
from tests.catalog.samples import discover_item, movie_details

TODAY = date(2026, 9, 26)


class FakeTmdb:
    """movies[movie_id] - ответ деталей либо исключение, которое нужно бросить."""

    def __init__(self, movies: dict, discover_pages: list[list[dict]]) -> None:
        self.movies = movies
        self.discover_pages = discover_pages
        self.requested_movies: list[int] = []

    def get_genres(self):
        return [{"id": 1, "name": "драма"}]

    def discover_movies(self, genre_id, released_from, released_to, min_vote_count, page):
        results = self.discover_pages[page - 1] if page <= len(self.discover_pages) else []
        return {"results": results, "total_pages": len(self.discover_pages)}

    def get_movie(self, movie_id):
        self.requested_movies.append(movie_id)
        outcome = self.movies.get(movie_id, TmdbNotFoundError("нет"))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "output"


def good_movies(*movie_ids: int) -> dict:
    return {movie_id: movie_details(movie_id) for movie_id in movie_ids}


def listing(*movie_ids: int) -> list[list[dict]]:
    return [[discover_item(movie_id) for movie_id in movie_ids]]


def run_importer(output_dir, tmdb, target_size=3, priority_ids=()):
    """Как настоящий запуск: каждый раз открывает файлы с диска заново."""
    store = CatalogStore.open(output_dir)
    discovery = CandidateDiscovery(tmdb, year_from=2020, min_vote_count=0, today=TODAY)
    summary = CatalogImporter(tmdb, store, discovery, target_size, "w500").run(list(priority_ids))
    return summary, CatalogStore.open(output_dir)


def candidate(store, movie_id):
    return next(c for c in store.candidates() if c.movie_id == movie_id)


def test_collects_target_and_skips_unsuitable(output_dir):
    movies = good_movies(1, 3, 4, 5)
    movies[2] = movie_details(2, translations={"translations": []})

    summary, store = run_importer(output_dir, FakeTmdb(movies, listing(1, 2, 3, 4, 5)))

    assert summary.target_reached
    assert (summary.imported, summary.rejected, summary.failed) == (3, 1, 0)
    assert store.movie_count() == 3
    assert candidate(store, 2).reject_reason is RejectReason.NO_RUSSIAN_TITLE


def test_movies_file_is_readable_list_of_cards(output_dir):
    run_importer(output_dir, FakeTmdb(good_movies(550), listing(550)), target_size=1)

    movies = json.loads((output_dir / "movies.json").read_text(encoding="utf-8"))

    assert movies == [
        {
            "movie_id": 550,
            "title_ru": "Бойцовский клуб",
            "original_title": "Fight Club",
            "release_date": "1999-10-15",
            "age_rating": "18+",
            "poster_url": "https://image.tmdb.org/t/p/w500/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
            "overview_ru": "Сотрудник страховой компании страдает хронической бессонницей.",
            "runtime_minutes": 139,
            "country_codes": ["US", "DE"],
            "genres": ["драма", "триллер"],
            "directors": [{"person_id": 7467, "name": "David Fincher"}],
            "writers": [{"person_id": 7468, "name": "Jim Uhls"}],
            "composers": [{"person_id": 1060, "name": "Dust Brothers"}],
            "producers": [{"person_id": 7474, "name": "Art Linson"}],
            "actors": [
                {"person_id": 819, "name": "Edward Norton", "character": "The Narrator"},
                {"person_id": 287, "name": "Brad Pitt", "character": "Tyler Durden"},
            ],
        }
    ]


def test_rerun_does_not_duplicate_or_refetch(output_dir):
    tmdb = FakeTmdb(good_movies(1, 2, 3), listing(1, 2, 3))
    run_importer(output_dir, tmdb)
    tmdb.requested_movies.clear()

    summary, store = run_importer(output_dir, tmdb)

    assert summary.imported == 0
    assert tmdb.requested_movies == []
    assert store.movie_count() == 3


def test_resumes_after_interruption(output_dir):
    class InterruptingTmdb(FakeTmdb):
        def get_movie(self, movie_id):
            if movie_id == 3:
                raise KeyboardInterrupt
            return super().get_movie(movie_id)

    with pytest.raises(KeyboardInterrupt):
        run_importer(output_dir, InterruptingTmdb(good_movies(1, 2, 3), listing(1, 2, 3)))
    assert CatalogStore.open(output_dir).movie_count() == 2

    tmdb = FakeTmdb(good_movies(1, 2, 3), listing(1, 2, 3))
    summary, store = run_importer(output_dir, tmdb)

    assert tmdb.requested_movies == [3]
    assert summary.imported == 1
    assert store.movie_count() == 3


def test_failed_movie_is_retried_on_next_run(output_dir):
    movies = good_movies(1, 3, 4)
    movies[2] = TmdbUnavailableError("503")
    _, store = run_importer(output_dir, FakeTmdb(movies, listing(1, 2, 3, 4)))

    failed = candidate(store, 2)
    assert (failed.status, failed.attempts, failed.last_error) == (CandidateStatus.FAILED, 1, "503")

    tmdb = FakeTmdb(good_movies(2), listing())
    _, store = run_importer(output_dir, tmdb, target_size=4)

    assert tmdb.requested_movies == [2]
    assert store.movie_count() == 4


def test_priority_movies_go_first(output_dir):
    tmdb = FakeTmdb(good_movies(1, 2, 3, 99), listing(1, 2, 3))

    _, store = run_importer(output_dir, tmdb, priority_ids=[99])

    assert tmdb.requested_movies[0] == 99
    assert 99 in {movie["movie_id"] for movie in store.movies()}


def test_priority_movie_missing_in_tmdb_is_rejected(output_dir):
    _, store = run_importer(output_dir, FakeTmdb(good_movies(1, 2, 3), listing(1, 2, 3)), priority_ids=[404])
    assert candidate(store, 404).reject_reason is RejectReason.NOT_FOUND


def test_stops_when_candidates_run_out(output_dir):
    summary, _ = run_importer(output_dir, FakeTmdb(good_movies(1), listing(1)))

    assert not summary.target_reached
    assert summary.total_movies == 1


def test_aborts_on_consecutive_failures_and_keeps_progress(output_dir):
    movie_ids = list(range(1, MAX_CONSECUTIVE_FAILURES + 3))
    movies = {movie_id: TmdbUnavailableError("503") for movie_id in movie_ids}
    movies[1] = movie_details(1)

    with pytest.raises(CatalogAbortError):
        run_importer(output_dir, FakeTmdb(movies, listing(*movie_ids)), target_size=len(movie_ids))

    store = CatalogStore.open(output_dir)
    assert store.movie_count() == 1
    assert sum(c.status is CandidateStatus.FAILED for c in store.candidates()) == MAX_CONSECUTIVE_FAILURES


def test_movies_added_elsewhere_count_toward_target(output_dir):
    store = CatalogStore.open(output_dir)
    store.save_movie(fetch_movie(FakeTmdb(good_movies(7), []), 7, "w500"))
    store.save()
    tmdb = FakeTmdb(good_movies(1, 2, 7), listing(7, 1, 2))

    _, store = run_importer(output_dir, tmdb)

    assert 7 not in tmdb.requested_movies
    assert store.movie_count() == 3


def test_report_counts_gaps_and_rejections(output_dir):
    movies = good_movies(1, 2)
    movies[2] = movie_details(2, credits={}, release_dates={})
    movies[3] = movie_details(3, overview="")

    _, store = run_importer(output_dir, FakeTmdb(movies, listing(1, 2, 3)))
    report = build_report(store)
    text = format_report(report)

    assert report.total_movies == 2
    assert report.incomplete_movies == 1
    assert report.risk_incomplete_movies == 1
    assert report.missing_by_field["Актёры"] == 1
    assert report.checked_candidates == 3
    assert report.no_translation_rejects == 1
    assert "Фильмов в каталоге: 2" in text
    assert "нет русского описания: 1 (33.3%)" in text


def test_report_on_empty_catalog(tmp_path):
    text = format_report(build_report(CatalogStore.open(tmp_path)))
    assert "Фильмов в каталоге: 0" in text
