import logging
from dataclasses import replace

import pytest

from scripts.catalog.movies.rejection import MovieRejectedError, RejectReason
from scripts.catalog.tests.samples import sample_movie
from scripts.catalog.tmdb.errors import TmdbAuthError, TmdbUnavailableError
from scripts.index import metadata, text
from scripts.index.cli import ExitCode
from scripts.index.embedding import embed_text
from scripts.index.errors import ConfigError, IndexingError
from scripts.index.pipeline import Services, TextResult, index_texts


class TextLog:
    def __init__(self):
        self.texts = []

    def encode_text(self, text, passage):
        assert passage
        self.texts.append(text)
        return [[float(len(self.texts))]]


def test_film_gets_overview_and_metadata_points():
    movie = sample_movie(550)
    encoders = TextLog()
    fragments = embed_text(encoders, movie)
    assert [fragment.source for fragment in fragments] == ["overview", "metadata"]
    assert all(fragment.offset_sec is None for fragment in fragments)
    assert encoders.texts == [
        movie.overview,
        "Жанр: драма, триллер. Год: 1999, 1990-е. Страна: США, Germany. "
        "В ролях: Edward Norton (The Narrator); Brad Pitt (Tyler Durden).",
    ]


def test_point_is_skipped_when_its_fields_are_empty():
    movie = sample_movie(550)
    no_metadata = replace(movie, genres=[], release_date=None, countries=[], actors=[])
    assert [fragment.source for fragment in embed_text(TextLog(), no_metadata)] == ["overview"]
    assert [fragment.source for fragment in embed_text(TextLog(), replace(movie, overview=""))] == ["metadata"]


class Recorder:
    def __init__(self):
        self.calls = []
        self.fail = {}

    def __call__(self, name, *args):
        self.calls.append((name, *args))
        if name in self.fail:
            raise self.fail[name]


class FakeMovies:
    def __init__(self, log, stored=()):
        self.log = log
        self.stored = {movie_id: sample_movie(movie_id) for movie_id in stored}

    def load(self, movie_id):
        return self.stored.get(movie_id)

    def upsert(self, movie):
        self.log("upsert", movie.movie_id)

    def reset_indexed(self, movie_id):
        self.log("reset_indexed", movie_id)

    def mark_indexed(self, movie_id):
        self.log("mark_indexed", movie_id)


class FakeVectors:
    def __init__(self, log):
        self.log = log

    def update_text(self, movie_id, fragments):
        self.log("update_text", movie_id, [fragment.source for fragment in fragments])
        return len(fragments)
        return {name: len(items) for name, items in fragments.items()}


class FakeEncoders:
    def encode_text(self, text, passage):
        assert passage
        return [[1.0]]


def services(log, outcomes=None, stored=()):
    outcomes = outcomes or {}

    def fetch(movie_id):
        log("tmdb", movie_id)
        if movie_id in outcomes:
            raise outcomes[movie_id]
        return sample_movie(movie_id)

    movies = FakeMovies(log, stored)
    return Services(fetch_movie=fetch, movies=movies, vectors=FakeVectors(log), encoders=FakeEncoders())


def test_movie_from_movies_table_is_not_fetched_from_tmdb():
    log = Recorder()
    result = index_texts([603, 550], services(log, stored=[603]))
    assert [movie.movie_id for movie in result.indexed] == [603, 550]
    assert result.total == 2
    sources = ["overview", "metadata"]
    assert log.calls == [
        ("update_text", 603, sources),
        ("mark_indexed", 603),
        ("tmdb", 550),
        ("upsert", 550),
        ("update_text", 550, sources),
        ("mark_indexed", 550),
    ]


def test_failed_text_write_keeps_indexed_at_for_the_next_all_run():
    # Сценарий из ревью: запись упала - indexed_at не сброшен, поэтому --all снова выберет фильм.
    log = Recorder()
    log.fail["update_text"] = IndexingError("Сбой Qdrant")
    with pytest.raises(IndexingError):
        index_texts([603], services(log, stored=[603]))
    assert [call[0] for call in log.calls] == ["update_text"]


def test_tmdb_problems_with_one_movie_do_not_stop_the_run():
    log = Recorder()
    outcomes = {
        1: MovieRejectedError(1, RejectReason.NO_RUSSIAN_OVERVIEW),
        2: TmdbUnavailableError("TMDB недоступен"),
    }
    result = index_texts([1, 603, 2, 550], services(log, outcomes))
    assert [movie.movie_id for movie in result.indexed] == [603, 550]
    assert result.rejected == {1: RejectReason.NO_RUSSIAN_OVERVIEW}
    assert list(result.failed) == [2]
    assert ("upsert", 1) not in log.calls
    assert ("upsert", 2) not in log.calls


def test_all_mode_needs_no_explicit_ids():
    assert text.read_movie_ids([], None, required=False) == []


@pytest.mark.parametrize("error", [TmdbAuthError("ключ отклонён"), IndexingError("Сбой Qdrant")])
def test_key_and_storage_failures_stop_the_run(error):
    log = Recorder()
    if isinstance(error, IndexingError):
        log.fail["update_text"] = error
        outcomes = {}
    else:
        outcomes = {603: error}
    with pytest.raises(type(error)):
        index_texts([603, 550], services(log, outcomes))
    assert ("upsert", 550) not in log.calls


def test_movie_ids_from_arguments_and_file(tmp_path):
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("﻿# каталог\n550\n\n27205  # Начало\n603\n", encoding="utf-8")
    assert text.read_movie_ids([603, 280], ids_file) == [603, 280, 550, 27205]


@pytest.mark.parametrize(("content", "message"), [("550\nabc\n", "строка 2"), ("# пусто\n", "Не задан ни один id")])
def test_invalid_movie_ids_file(tmp_path, content, message):
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        text.read_movie_ids([], ids_file)


@pytest.mark.parametrize(
    ("result", "code"),
    [
        (TextResult(indexed=[sample_movie(603)]), ExitCode.OK),
        (TextResult(rejected={1: RejectReason.NOT_FOUND}), ExitCode.REJECTED),
        (TextResult(rejected={1: RejectReason.NOT_FOUND}, failed={2: "TMDB недоступен"}), ExitCode.FAILURE),
    ],
)
def test_exit_code_reflects_the_worst_outcome(monkeypatch, result, code):
    monkeypatch.setattr(text, "run", lambda movie_ids, env_file, all_indexed=False: result)
    assert text.main(["603"]) == code


def test_storage_failure_is_one_line_without_traceback(monkeypatch, caplog):
    def run(movie_ids, env_file, all_indexed=False):
        raise IndexingError("Сбой Qdrant (http://127.0.0.1:6333)")

    monkeypatch.setattr(text, "run", run)
    with caplog.at_level(logging.INFO):
        assert text.main(["603"]) == ExitCode.FAILURE
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert errors[0].getMessage() == f"Сбой Qdrant (http://127.0.0.1:6333). {text.STOPPED_HINT}"
    assert errors[0].exc_info is None


def test_tmdb_countries_are_requested_once(monkeypatch):
    clients = []

    class FakeClient:
        def __init__(self, rate_limiter, access_token, api_key):
            clients.append(self)
            self.country_requests = 0

        def get_countries(self):
            self.country_requests += 1
            return [{"iso_3166_1": "US", "english_name": "United States", "native_name": "США"}]

    monkeypatch.setattr(metadata, "TmdbClient", FakeClient)
    monkeypatch.setattr(metadata, "fetch_movie", lambda client, movie_id, countries: sample_movie(movie_id))
    fetch = metadata.TmdbMetadata("token", None, 20)
    assert [fetch(movie_id).movie_id for movie_id in (603, 550, 280)] == [603, 550, 280]
    assert len(clients) == 1
    assert clients[0].country_requests == 1
