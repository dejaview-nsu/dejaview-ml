import os
from contextlib import closing
from fractions import Fraction
from unittest.mock import Mock

import pytest
from qdrant_client import QdrantClient, models

from dejaview_ml.config import COLLECTIONS
from scripts.index import video
from scripts.index.cli import ExitCode
from scripts.index.embedding import Fragment, embed_video
from scripts.index.errors import ConfigError, IndexingError
from scripts.index.metadata import MovieTable
from scripts.index.tests.test_vectors import fragments, line, movie_points
from scripts.index.validation import VideoInfo
from scripts.index.vectors import VectorWriter


def video_fragments(count):
    return [Fragment(i, [1.0] + [0.0] * 511) for i in range(count)]


def test_video_update_migrates_old_windows_and_keeps_other_data():
    with closing(QdrantClient(":memory:")) as client:
        for name, size in COLLECTIONS.items():
            client.create_collection(
                name, vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE)
            )
        writer = VectorWriter(client, "memory")
        writer.replace_movie(603, fragments(3))
        writer.replace_movie(550, fragments(2))
        writer.replace_speech(603, [line(0)])
        writer.replace_movie(603, {"video": [Fragment(0, [1.0] * 512), Fragment(20, [1.0] * 512)]})
        other_collections = {name: client.scroll(name)[0] for name in COLLECTIONS if name != "video"}
        other_movie = movie_points(client, "video", 550)
        for _ in range(2):
            assert writer.update_video(603, video_fragments(12)) == 12
            assert sorted(p.payload["offset_sec"] for p in movie_points(client, "video", 603)) == list(range(12))
            assert movie_points(client, "video", 550) == other_movie
            assert {name: client.scroll(name)[0] for name in other_collections} == other_collections


def test_failed_upsert_keeps_old_points_and_does_not_delete_tail(monkeypatch):
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection("video", vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE))
        writer = VectorWriter(client, "memory")
        writer.check_collections(("video",))
        writer.update_video(603, video_fragments(3))
        previous = movie_points(client, "video", 603)
        monkeypatch.setattr(client, "upsert", Mock(side_effect=TimeoutError("timeout")))
        delete = Mock(wraps=client.delete)
        monkeypatch.setattr(client, "delete", delete)
        with pytest.raises(IndexingError, match="запись в video"):
            writer.update_video(603, video_fragments(1))
        assert movie_points(client, "video", 603) == previous
        delete.assert_not_called()
        with pytest.raises(IndexingError, match="Нет видеофрагментов"):
            writer.update_video(603, [])


@pytest.fixture
def setup_run(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    env = tmp_path / ".env"
    env.write_text("DATABASE_URL=postgresql://unused/db\n", encoding="utf-8")
    monkeypatch.delenv("TMDB_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.setenv("TMDB_REQUESTS_PER_SECOND", "unused")
    monkeypatch.setenv("INDEX_WORK_DIR", str(tmp_path / "work"))
    source = tmp_path / "work" / "movie.mp4"
    source.parent.mkdir()
    source.write_bytes(b"source")
    args = video.parse_args([str(source), "--movie-id", "603", "--env-file", str(env)])
    movies, writer = Mock(spec=MovieTable), Mock(spec=VectorWriter)
    writer.update_video.return_value = 2
    monkeypatch.setattr(video, "MovieTable", Mock(return_value=movies))
    monkeypatch.setattr(video, "connect_vectors", Mock(return_value=writer))
    monkeypatch.setattr(video, "require_ffmpeg", Mock())
    monkeypatch.setattr(video, "probe_video", Mock())
    temp_dirs = []

    def embed(encoders, path, info, temp_dir):
        temp_dirs.append(temp_dir)
        assert temp_dir.is_dir()
        return video_fragments(2)

    monkeypatch.setattr(video, "embed_video", embed)
    return args, movies, writer, temp_dirs


def test_video_only_preserves_metadata_source_and_closes_resources(setup_run):
    args, movies, writer, temp_dirs = setup_run
    assert video.run(args) == 2
    writer.check_collections.assert_called_once_with(("video",))
    movies.load.assert_called_once_with(603)
    movies.upsert.assert_not_called()
    movies.reset_indexed.assert_not_called()
    movies.mark_indexed.assert_not_called()
    assert args.file.read_bytes() == b"source"
    assert temp_dirs and all(not path.exists() for path in temp_dirs)
    writer.close.assert_called_once_with()


def test_missing_movie_does_not_write_vectors(setup_run):
    args, movies, writer, _ = setup_run
    movies.load.return_value = None
    with pytest.raises(ConfigError, match="нет в movies"):
        video.run(args)
    writer.update_video.assert_not_called()


@pytest.mark.parametrize("failure", [RuntimeError("model failed"), KeyboardInterrupt()])
def test_encoding_failure_keeps_index_source_and_cleans_up(setup_run, monkeypatch, failure):
    args, movies, writer, temp_dirs = setup_run

    def fail(encoders, path, info, temp_dir):
        temp_dirs.append(temp_dir)
        raise failure

    monkeypatch.setattr(video, "embed_video", fail)
    with pytest.raises(type(failure)):
        video.run(args)
    writer.update_video.assert_not_called()
    movies.reset_indexed.assert_not_called()
    assert args.file.exists()
    assert temp_dirs and all(not path.exists() for path in temp_dirs)
    writer.close.assert_called_once_with()


def test_only_video_model_is_loaded_and_frames_are_batched(monkeypatch, tmp_path):
    from PIL import Image  # noqa: PLC0415

    frames = [Image.new("RGB", (224, 224), (i, 0, 0)) for i in range(35)]
    monkeypatch.setattr(
        "scripts.index.embedding.video_segments_stream",
        lambda *args: ((i, [frame]) for i, frame in enumerate(frames)),
    )
    encoders = Mock()
    encoders.encode_video_frames.side_effect = lambda batch: [[float(f.getpixel((0, 0))[0])] for f in batch]
    info = VideoInfo(35, 0, None, 224, 224, Fraction(1))
    result = embed_video(encoders, tmp_path / "movie.mp4", info, tmp_path)
    encoders.load.assert_called_once_with("video")
    assert [len(call.args[0]) for call in encoders.encode_video_frames.call_args_list] == [32, 3]
    assert [f.offset_sec for f in result] == list(range(35))
    assert [f.vector for f in result] == [[float(i)] for i in range(35)]


@pytest.mark.parametrize("movie_id", ["0", "-1", "abc"])
def test_invalid_id(movie_id):
    with pytest.raises(SystemExit) as error:
        video.parse_args(["movie.mp4", "--movie-id", movie_id])
    assert error.value.code == ExitCode.CONFIG_ERROR
