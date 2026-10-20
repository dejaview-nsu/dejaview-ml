import logging
import os

import psycopg
import pytest

from scripts.catalog.movies.parser import MovieDataError
from scripts.catalog.movies.rejection import MovieRejectedError, RejectReason
from scripts.catalog.tmdb.errors import TmdbAuthError, TmdbUnavailableError
from scripts.index import cli, vectors
from scripts.index.cli import ExitCode, describe_error, main, parse_args
from scripts.index.errors import ConfigError, IndexingError, UnsuitableFileError


@pytest.mark.parametrize(
    ("error", "code", "fragment"),
    [
        (ConfigError("Не задан DATABASE_URL"), ExitCode.CONFIG_ERROR, "DATABASE_URL"),
        (TmdbAuthError("TMDB отклонил ключ"), ExitCode.CONFIG_ERROR, "отклонил ключ"),
        (UnsuitableFileError("контейнер MOV"), ExitCode.REJECTED, "MOV"),
        (MovieRejectedError(603, RejectReason.NO_RUSSIAN_OVERVIEW), ExitCode.REJECTED, "нет русского описания"),
        (MovieRejectedError(1, RejectReason.NOT_FOUND), ExitCode.REJECTED, "нет в TMDB"),
        (IndexingError("Сбой Qdrant"), ExitCode.FAILURE, "indexed_at не выставлен"),
        (TmdbUnavailableError("TMDB недоступен"), ExitCode.FAILURE, "TMDB недоступен"),
        (MovieDataError("не тот фильм"), ExitCode.FAILURE, "не тот фильм"),
        (psycopg.errors.CheckViolation("check"), ExitCode.FAILURE, "Ошибка PostgreSQL"),
        (RuntimeError("boom"), ExitCode.FAILURE, "Непредвиденная ошибка RuntimeError: boom"),
    ],
)
def test_describe_error(error, code, fragment):
    actual_code, message = describe_error(error)
    assert actual_code == code
    assert fragment in message


def test_main_prints_reason_without_traceback(monkeypatch, caplog, tmp_path):
    def run(args):
        raise UnsuitableFileError("Файл не подходит: контейнер MOV")

    monkeypatch.setattr(cli, "run", run)
    with caplog.at_level(logging.DEBUG):
        code = main([str(tmp_path / "movie.mov"), "--movie-id", "603"])
    assert code == ExitCode.REJECTED
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert [record.getMessage() for record in errors] == ["Файл не подходит: контейнер MOV"]
    assert errors[0].exc_info is None


def test_main_interrupt(monkeypatch, tmp_path):
    def run(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run", run)
    assert main([str(tmp_path / "movie.mp4"), "--movie-id", "603"]) == ExitCode.INTERRUPTED


@pytest.mark.parametrize("movie_id", ["0", "-5", "abc"])
def test_invalid_movie_id(movie_id, capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse_args(["movie.mp4", "--movie-id", movie_id])
    assert exit_info.value.code == ExitCode.CONFIG_ERROR


def test_movie_id_is_required():
    with pytest.raises(SystemExit):
        parse_args(["movie.mp4"])


def test_run_passes_qdrant_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TMDB_ACCESS_TOKEN=token\nDATABASE_URL=postgresql://localhost/dejaview\nML_QDRANT_API_KEY=secret\n",
        encoding="utf-8",
    )
    received = {}

    def client(**kwargs):
        received.update(kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(vectors, "QdrantClient", client)
    with pytest.raises(KeyboardInterrupt):
        cli.run(parse_args([str(tmp_path / "movie.mp4"), "--movie-id", "603", "--env-file", str(env_file)]))
    assert received["api_key"] == "secret"
