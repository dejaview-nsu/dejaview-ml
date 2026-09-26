import json
from dataclasses import replace

import pytest

from scripts.catalog.candidates.model import Candidate, CandidateStatus, Priority
from scripts.catalog.movies.parser import parse_movie
from scripts.catalog.movies.rejection import RejectReason
from scripts.catalog.storage.catalog_store import MAX_ERROR_TEXT_LENGTH, SAVE_EVERY_CHANGES, CatalogStore
from scripts.catalog.storage.json_files import CatalogFileError, write_json_atomically
from tests.catalog.samples import movie_details

DUPLICATE_CANDIDATE = {"movie_id": 1, "priority": 1, "source_bucket": "a"}


def sample_movie(movie_id: int = 550):
    return parse_movie(movie_details(movie_id), movie_id, "w500")


def discovered(movie_id: int, bucket: str = "a") -> Candidate:
    return Candidate(movie_id, Priority.DISCOVERED, bucket)


def test_missing_files_mean_empty_catalog(tmp_path):
    store = CatalogStore.open(tmp_path / "нет такой папки")

    assert store.movie_count() == 0
    assert store.candidates() == []


def test_save_and_reopen_roundtrip(tmp_path):
    store = CatalogStore.open(tmp_path)
    store.add_candidates([discovered(550, "1990-е / драма"), discovered(2)])
    store.save_movie(sample_movie(550))
    store.mark_rejected(2, RejectReason.NO_POSTER)
    store.save()

    reopened = CatalogStore.open(tmp_path)

    assert reopened.movies() == store.movies()
    assert reopened.candidates() == store.candidates()
    assert [(c.movie_id, c.status, c.attempts) for c in reopened.candidates()] == [
        (550, CandidateStatus.IMPORTED, 1),
        (2, CandidateStatus.REJECTED, 1),
    ]


def test_files_keep_cyrillic_readable(tmp_path):
    store = CatalogStore.open(tmp_path)
    store.save_movie(sample_movie())
    store.save()

    assert "Бойцовский клуб" in store.movies_path.read_text(encoding="utf-8")


def test_save_movie_twice_updates_instead_of_duplicating(tmp_path):
    store = CatalogStore.open(tmp_path)
    store.save_movie(sample_movie())
    store.save_movie(replace(sample_movie(), title_ru="Новое название"))

    assert store.movie_count() == 1
    assert store.movies()[0]["title_ru"] == "Новое название"


def test_movies_are_sorted_by_id(tmp_path):
    store = CatalogStore.open(tmp_path)
    for movie_id in (30, 10, 20):
        store.save_movie(sample_movie(movie_id))

    assert [movie["movie_id"] for movie in store.movies()] == [10, 20, 30]


def test_add_candidates_keeps_existing_status_and_raises_priority(tmp_path):
    store = CatalogStore.open(tmp_path)
    store.add_candidates([discovered(1), discovered(2)])
    store.mark_rejected(1, RejectReason.NO_POSTER)

    store.add_candidates([Candidate(1, Priority.MANUAL, "manual"), Candidate(2, Priority.MANUAL, "manual")])

    assert [(c.movie_id, c.priority, c.status) for c in store.candidates()] == [
        (1, Priority.MANUAL, CandidateStatus.REJECTED),
        (2, Priority.MANUAL, CandidateStatus.PENDING),
    ]


def test_add_candidates_does_not_modify_caller_objects(tmp_path):
    store = CatalogStore.open(tmp_path)
    candidate = discovered(1)

    store.add_candidates([candidate])

    assert candidate.queue_position == 0
    assert store.candidates()[0].queue_position == 1


def test_new_candidates_continue_queue_positions(tmp_path):
    store = CatalogStore.open(tmp_path)
    store.add_candidates([discovered(1)])
    store.add_candidates([discovered(2)])

    assert [c.queue_position for c in store.candidates()] == [1, 2]


def test_pending_includes_failed_with_attempts_left(tmp_path):
    store = CatalogStore.open(tmp_path)
    store.add_candidates([discovered(movie_id) for movie_id in (1, 2, 3)])
    store.mark_failed(1, "503")
    store.mark_failed(2, "503")
    store.mark_failed(2, "503")

    assert store.pending_movie_ids(max_attempts=2) == [1, 3]


def test_long_error_text_is_truncated(tmp_path):
    store = CatalogStore.open(tmp_path)
    store.add_candidates([discovered(1)])
    store.mark_failed(1, "x" * 5000)

    failed = store.candidates()[0]
    assert failed.status is CandidateStatus.FAILED
    assert len(failed.last_error) == MAX_ERROR_TEXT_LENGTH


def test_crash_between_files_is_reconciled(tmp_path):
    """movies.json успел записаться, candidates.json - нет: фильм не должен скачиваться заново."""
    store = CatalogStore.open(tmp_path)
    store.add_candidates([discovered(550)])
    store.save()
    store.save_movie(sample_movie())
    write_json_atomically(store.movies_path, store.movies())

    reopened = CatalogStore.open(tmp_path)
    reopened.mark_already_imported()

    assert reopened.pending_movie_ids(max_attempts=3) == []


def test_autosaves_every_n_changes(tmp_path):
    store = CatalogStore.open(tmp_path)
    for movie_id in range(1, SAVE_EVERY_CHANGES + 1):
        store.save_movie(sample_movie(movie_id))

    assert CatalogStore.open(tmp_path).movie_count() == SAVE_EVERY_CHANGES


@pytest.mark.parametrize(
    ("file_name", "content", "message"),
    [
        ("movies.json", "[1]", "не является объектом"),
        ("movies.json", '[{"title_ru": "Без id"}]', "movie_id"),
        ("movies.json", '[{"movie_id": 1}, {"movie_id": 1}]', "дважды"),
        ("candidates.json", '[{"movie_id": 1}]', "priority"),
        ("candidates.json", [DUPLICATE_CANDIDATE, DUPLICATE_CANDIDATE], "дважды"),
    ],
)
def test_broken_records_are_reported_with_file_and_index(tmp_path, file_name, content, message):
    if not isinstance(content, str):
        content = json.dumps(content)
    (tmp_path / file_name).write_text(content, encoding="utf-8")

    with pytest.raises(CatalogFileError, match=message) as raised:
        CatalogStore.open(tmp_path)

    assert file_name in str(raised.value)
