import json
from datetime import date, datetime

import pytest

from scripts.catalog.candidates.model import Candidate, CandidateStatus, Priority
from scripts.catalog.movies.rejection import RejectReason
from scripts.catalog.storage.records import (
    MOVIE_FIELDS,
    InvalidRecordError,
    candidate_from_json,
    candidate_to_json,
    movie_id_of,
    movie_to_json,
)
from scripts.catalog.tests.samples import sample_movie


def test_movie_to_json_matches_movies_table():
    record = json.loads(json.dumps(movie_to_json(sample_movie()), ensure_ascii=False))

    assert record["release_date"] == "1999-10-15"
    assert record["cached_at"] == "2026-09-26T12:30:00+00:00"
    assert record["genres"][0] == {"id": 18, "name": "драма"}
    assert record["countries"][0] == {"code": "US", "name": "США"}
    assert record["crew"][0] == {"person_id": 7467, "name": "David Fincher", "role": "director"}
    assert record["actors"][0] == {
        "person_id": 819,
        "name": "Edward Norton",
        "character": "The Narrator",
        "profile_path": "/8nytsqL59SFJTVYVrN72k6qkGgJ.jpg",
    }


def test_movie_without_release_date():
    movie = sample_movie(release_date="")
    assert movie.release_date is None
    assert movie_to_json(movie)["release_date"] is None


def test_candidate_roundtrip():
    candidate = Candidate(
        movie_id=7,
        priority=Priority.MANUAL,
        source_bucket="manual",
        status=CandidateStatus.REJECTED,
        reject_reason=RejectReason.NO_POSTER,
        last_error=None,
        attempts=2,
        queue_position=5,
    )
    assert candidate_from_json(candidate_to_json(candidate)) == candidate


def test_candidate_defaults_for_optional_fields():
    candidate = candidate_from_json({"movie_id": 1, "priority": 1, "source_bucket": "a"})
    assert (candidate.status, candidate.attempts, candidate.reject_reason) == (CandidateStatus.PENDING, 0, None)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ([1], "не является объектом"),
        ({"movie_id": 1, "source_bucket": "a"}, "priority"),
        ({"movie_id": -1, "priority": 1, "source_bucket": "a"}, "movie_id"),
        ({"movie_id": True, "priority": 1, "source_bucket": "a"}, "movie_id"),
        ({"movie_id": 1, "priority": 7, "source_bucket": "a"}, "Priority"),
        ({"movie_id": 1, "priority": 1, "source_bucket": 5}, "source_bucket"),
        ({"movie_id": 1, "priority": 1, "source_bucket": "a", "status": "?"}, "CandidateStatus"),
        ({"movie_id": 1, "priority": 1, "source_bucket": "a", "reject_reason": "?"}, "RejectReason"),
        ({"movie_id": 1, "priority": 1, "source_bucket": "a", "attempts": -1}, "attempts"),
        ({"movie_id": 1, "priority": 1, "source_bucket": "a", "lol": 1}, "неизвестные поля"),
    ],
)
def test_invalid_candidate_records(record, message):
    with pytest.raises(InvalidRecordError, match=message):
        candidate_from_json(record)


def test_movie_id_of_valid_record():
    assert movie_id_of(movie_to_json(sample_movie())) == 550


@pytest.mark.parametrize(("movie_id", "message"), [(0, "movie_id"), ("1", "movie_id"), (True, "movie_id")])
def test_invalid_movie_id(movie_id, message):
    record = movie_to_json(sample_movie()) | {"movie_id": movie_id}
    with pytest.raises(InvalidRecordError, match=message):
        movie_id_of(record)


@pytest.mark.parametrize("record", [[], {}, {"movie_id": 550, "title_ru": "Старый формат"}])
def test_records_outside_movies_table_are_rejected(record):
    with pytest.raises(InvalidRecordError):
        movie_id_of(record)


def test_extra_field_is_rejected():
    record = movie_to_json(sample_movie()) | {"poster_url": "https://image.tmdb.org/t/p/w500/x.jpg"}
    with pytest.raises(InvalidRecordError, match="poster_url"):
        movie_id_of(record)


def test_record_has_exactly_movies_table_fields():
    assert movie_to_json(sample_movie()).keys() == MOVIE_FIELDS


def test_date_types_are_not_leaked_into_json():
    record = movie_to_json(sample_movie())
    assert not any(isinstance(value, date | datetime) for value in record.values())
