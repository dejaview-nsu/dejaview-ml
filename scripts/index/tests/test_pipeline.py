from fractions import Fraction

import pytest

from scripts.catalog.tests.samples import sample_movie
from scripts.index import pipeline
from scripts.index.embedding import Fragment
from scripts.index.errors import IndexingError
from scripts.index.pipeline import Services, index_movie
from scripts.index.validation import VideoInfo

INFO = VideoInfo(
    duration_sec=12, video_stream=0, audio_stream=1, width=854, height=480, sample_aspect_ratio=Fraction(1)
)


class Recorder:
    def __init__(self):
        self.calls = []
        self.fail_on = None

    def record(self, name, *args):
        self.calls.append(name)
        if name == self.fail_on:
            raise IndexingError(f"{name} упал")


class FakeMovies:
    def __init__(self, log):
        self.log = log

    def check(self):
        self.log.record("movies.check")

    def upsert(self, movie):
        self.log.record("movies.upsert")

    def reset_indexed(self, movie_id):
        self.log.record("movies.reset_indexed")

    def mark_indexed(self, movie_id):
        self.log.record("movies.mark_indexed")


class FakeVectors:
    def __init__(self, log):
        self.log = log

    def check_collections(self):
        self.log.record("vectors.check")

    def replace_movie(self, movie_id, fragments):
        self.log.record("vectors.replace")
        return {name: len(items) for name, items in fragments.items()}


@pytest.fixture
def log(monkeypatch):
    log = Recorder()
    temp_dirs = []

    def embed(encoders, overview, source, info, temp_dir):
        temp_dirs.append(temp_dir)
        assert temp_dir.is_dir()
        log.record("embed")
        return {"text": [Fragment(None, [1.0])], "image": [Fragment(0, [1.0])]}

    monkeypatch.setattr(pipeline, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(pipeline, "probe_video", lambda source: log.record("probe") or INFO)
    monkeypatch.setattr(pipeline, "embed_movie", embed)
    log.temp_dirs = temp_dirs
    return log


def services(log):
    def fetch(movie_id):
        log.record("tmdb")
        return sample_movie(movie_id)

    return Services(fetch_movie=fetch, movies=FakeMovies(log), vectors=FakeVectors(log), encoders=None)


def test_stage_order(log, tmp_path):
    source = tmp_path / "movie.mp4"
    source.write_bytes(b"video")
    result = index_movie(source, 603, tmp_path / "work", services(log))
    assert log.calls == [
        "probe",
        "vectors.check",
        "movies.check",
        "tmdb",
        "movies.upsert",
        "embed",
        "movies.reset_indexed",
        "vectors.replace",
        "movies.mark_indexed",
    ]
    assert result.point_counts == {"text": 1, "image": 1}
    assert result.movie.movie_id == 603


def test_source_inside_work_dir_is_removed_after_success(log, tmp_path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    source = work_dir / "movie.mp4"
    source.write_bytes(b"video")
    result = index_movie(source, 603, work_dir, services(log))
    assert result.source_removed
    assert list(work_dir.iterdir()) == []


@pytest.mark.parametrize("error", [PermissionError("файл открыт в плеере"), KeyboardInterrupt()])
def test_failed_source_removal_keeps_indexed_result(log, tmp_path, monkeypatch, error):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    source = work_dir / "movie.mp4"
    source.write_bytes(b"video")

    def unlink(self, missing_ok=False):
        raise error

    monkeypatch.setattr(pipeline.Path, "unlink", unlink)
    result = index_movie(source, 603, work_dir, services(log))
    assert log.calls[-1] == "movies.mark_indexed"
    assert not result.source_removed
    assert source.exists()


def test_source_outside_work_dir_is_kept(log, tmp_path):
    source = tmp_path / "storage" / "movie.mp4"
    source.parent.mkdir()
    source.write_bytes(b"video")
    result = index_movie(source, 603, tmp_path / "work", services(log))
    assert not result.source_removed
    assert source.exists()
    assert list((tmp_path / "work").iterdir()) == []


@pytest.mark.parametrize("failing_step", ["embed", "vectors.replace", "movies.mark_indexed"])
def test_failure_keeps_source_and_removes_temp_files(log, tmp_path, failing_step):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    source = work_dir / "movie.mp4"
    source.write_bytes(b"video")
    log.fail_on = failing_step
    with pytest.raises(IndexingError):
        index_movie(source, 603, work_dir, services(log))
    assert source.exists()
    assert not log.temp_dirs[0].exists()
    assert list(work_dir.iterdir()) == [source]


@pytest.mark.parametrize("failing_step", ["vectors.check", "movies.check", "tmdb"])
def test_checks_fail_before_anything_is_written(log, tmp_path, failing_step):
    source = tmp_path / "movie.mp4"
    source.write_bytes(b"video")
    log.fail_on = failing_step
    with pytest.raises(IndexingError):
        index_movie(source, 603, tmp_path / "work", services(log))
    assert "movies.upsert" not in log.calls
    assert "embed" not in log.calls


def test_interrupt_removes_temp_files(log, tmp_path, monkeypatch):
    def interrupted(*args):
        log.temp_dirs.append(args[-1])
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "embed_movie", interrupted)
    source = tmp_path / "movie.mp4"
    source.write_bytes(b"video")
    with pytest.raises(KeyboardInterrupt):
        index_movie(source, 603, tmp_path / "work", services(log))
    assert not log.temp_dirs[0].exists()
