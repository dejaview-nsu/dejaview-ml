import logging
from unittest.mock import Mock

import psycopg
import pytest

from scripts.index import delete
from scripts.index.cli import ExitCode
from scripts.index.errors import ConfigError, IndexingError
from scripts.index.metadata import MovieTable
from scripts.index.vectors import VectorWriter


@pytest.mark.parametrize("movie_id", ["0", "-1", "abc", str(2**63)])
def test_invalid_id_is_rejected_before_connecting(movie_id, monkeypatch):
    run = Mock()
    monkeypatch.setattr(delete, "run", run)
    with pytest.raises(SystemExit) as error:
        delete.main(["--movie-id", movie_id])
    assert error.value.code == ExitCode.CONFIG_ERROR
    run.assert_not_called()


def test_id_is_required():
    with pytest.raises(SystemExit) as error:
        delete.parse_args([])
    assert error.value.code == ExitCode.CONFIG_ERROR


@pytest.mark.parametrize("exists", [True, False])
def test_postgres_is_deleted_only_after_vectors(exists):
    calls = Mock()
    movies = Mock(spec=MovieTable)
    writer = Mock(spec=VectorWriter)
    calls.attach_mock(writer.delete_movie, "vectors")
    calls.attach_mock(movies.delete, "postgres")
    movies.delete.return_value = exists

    delete.delete_movie(603, movies, writer)

    assert [entry[0] for entry in calls.mock_calls] == ["vectors", "postgres"]
    writer.delete_movie.assert_called_once_with(603)
    movies.delete.assert_called_once_with(603)


@pytest.mark.parametrize("error", [IndexingError("Qdrant недоступен"), KeyboardInterrupt()])
def test_qdrant_failure_or_interrupt_does_not_delete_postgres(error):
    movies = Mock(spec=MovieTable)
    writer = Mock(spec=VectorWriter)
    writer.delete_movie.side_effect = error

    with pytest.raises(type(error)):
        delete.delete_movie(603, movies, writer)

    movies.delete.assert_not_called()


def test_retry_after_postgres_failure_repeats_vector_cleanup():
    movies = Mock(spec=MovieTable)
    writer = Mock(spec=VectorWriter)
    movies.delete.side_effect = [psycopg.OperationalError("connection lost"), True, False]

    with pytest.raises(psycopg.OperationalError):
        delete.delete_movie(603, movies, writer)
    delete.delete_movie(603, movies, writer)
    delete.delete_movie(603, movies, writer)

    assert writer.delete_movie.call_count == 3
    assert movies.delete.call_count == 3


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ConfigError("Не задан DATABASE_URL"), ExitCode.CONFIG_ERROR),
        (IndexingError("Qdrant недоступен"), ExitCode.FAILURE),
        (psycopg.OperationalError("connection lost"), ExitCode.FAILURE),
        (KeyboardInterrupt(), ExitCode.INTERRUPTED),
    ],
)
def test_main_reports_failure_without_success(error, code, monkeypatch, caplog):
    monkeypatch.setattr(delete, "run", Mock(side_effect=error))
    with caplog.at_level(logging.INFO):
        assert delete.main(["--movie-id", "603"]) == code
    assert "Готово" not in caplog.text
    assert "не проиндексирован" not in caplog.text
    if code != ExitCode.CONFIG_ERROR:
        assert "частично" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_main_success(monkeypatch, caplog):
    run = Mock()
    monkeypatch.setattr(delete, "run", run)
    with caplog.at_level(logging.INFO):
        assert delete.main(["--movie-id", "603"]) == ExitCode.OK
    run.assert_called_once_with(603, None)
    assert "Готово" in caplog.text


@pytest.mark.parametrize("failure", [None, IndexingError("delete failed"), KeyboardInterrupt()])
def test_run_closes_writer_on_success_failure_and_interrupt(failure, monkeypatch):
    settings = Mock(database_url="postgresql://test")
    load_settings = Mock(return_value=settings)
    movies = Mock(spec=MovieTable)
    writer = Mock(spec=VectorWriter)
    writer.delete_movie.side_effect = failure
    monkeypatch.setattr(delete, "load_settings", load_settings)
    monkeypatch.setattr(delete, "MovieTable", Mock(return_value=movies))
    monkeypatch.setattr(delete, "connect_vectors", Mock(return_value=writer))

    if failure:
        with pytest.raises(type(failure)):
            delete.run(603, None)
    else:
        delete.run(603, None)

    load_settings.assert_called_once_with(None, require_tmdb=False)
    movies.check.assert_called_once_with()
    writer.close.assert_called_once_with()


def test_unavailable_postgres_stops_before_qdrant_connection(monkeypatch):
    monkeypatch.setattr(delete, "load_settings", Mock(return_value=Mock(database_url="postgresql://test")))
    movies = Mock(spec=MovieTable)
    movies.check.side_effect = IndexingError("PostgreSQL недоступен")
    monkeypatch.setattr(delete, "MovieTable", Mock(return_value=movies))
    connect = Mock()
    monkeypatch.setattr(delete, "connect_vectors", connect)

    with pytest.raises(IndexingError):
        delete.run(603, None)
    connect.assert_not_called()
