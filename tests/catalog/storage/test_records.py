from datetime import date

import pytest

from scripts.catalog.candidates.model import Candidate, CandidateStatus, Priority
from scripts.catalog.movies.parser import parse_movie
from scripts.catalog.movies.rejection import RejectReason
from scripts.catalog.storage.records import (
    InvalidRecordError,
    candidate_from_json,
    candidate_to_json,
    movie_id_of,
    movie_to_json,
)
from tests.catalog.samples import movie_details


def test_movie_to_json_uses_iso_date_and_plain_people():
    record = movie_to_json(parse_movie(movie_details(), 550, "w500"))

    assert record["release_date"] == "1999-10-15"
    assert record["directors"] == [{"person_id": 7467, "name": "David Fincher"}]
    assert record["actors"][0] == {"person_id": 819, "name": "Edward Norton", "character": "The Narrator"}


def test_movie_without_release_date():
    movie = parse_movie(movie_details(release_date=""), 550, "w500")
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


@pytest.mark.parametrize("record", [[], {"movie_id": 0}, {"movie_id": "1"}, {}])
def test_invalid_movie_records(record):
    with pytest.raises(InvalidRecordError):
        movie_id_of(record)


def test_date_type_is_not_leaked_into_json():
    record = movie_to_json(parse_movie(movie_details(), 550, "w500"))
    assert not any(isinstance(value, date) for value in record.values())
