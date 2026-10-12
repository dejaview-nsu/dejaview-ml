from pathlib import Path

import numpy as np
import pytest

from dejaview_ml.media import decode_audio, decode_video
from scripts.index.slicing import (
    FfmpegError,
    audio_chunks,
    audio_windows,
    expected_counts,
    image_batches,
    read_ffmpeg,
    video_segments_stream,
)
from scripts.index.tests.videos import make_video, requires_ffmpeg
from scripts.index.validation import probe_video

RATE = 48_000


def windows_for(seconds: float, block_sec: float = 5) -> list[tuple[int, float]]:
    samples = np.arange(round(seconds * RATE), dtype=np.float32)
    block = round(block_sec * RATE)
    blocks = [samples[start : start + block] for start in range(0, len(samples), block)]
    return [(offset, len(window) / RATE) for offset, window in audio_windows(blocks)]


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (20, [(0, 10), (5, 10), (10, 10), (15, 5)]),
        (12, [(0, 10), (5, 7), (10, 2)]),
        (10.5, [(0, 10), (5, 5.5)]),
        (7, [(0, 7), (5, 2)]),
        (0.5, []),
    ],
)
def test_audio_windows_step_and_tail(seconds, expected):
    assert windows_for(seconds) == expected


def test_audio_windows_do_not_depend_on_block_size():
    assert windows_for(23.3, block_sec=5) == windows_for(23.3, block_sec=0.7)


def test_audio_window_contents():
    samples = np.arange(15 * RATE, dtype=np.float32)
    windows = dict(audio_windows([samples]))
    np.testing.assert_array_equal(windows[5], samples[5 * RATE : 15 * RATE])


def test_expected_counts():
    assert expected_counts(12) == {"image": 3, "video": 2, "audio": 3}
    assert expected_counts(6000) == {"image": 1200, "video": 600, "audio": 1200}


@requires_ffmpeg
def test_offsets_of_real_movie(tmp_path):
    source = make_video(tmp_path / "movie.mp4", duration=12)
    info = probe_video(source)

    batches = list(image_batches(source, info, tmp_path))
    assert [offset for batch in batches for offset in batch.offsets_sec] == [0, 5, 10]
    assert batches[0].images[0].size == (854, 480)

    segments = list(video_segments_stream(source, info, tmp_path))
    assert [(offset, len(frames)) for offset, frames in segments] == [(0, 10), (10, 2)]

    windows = list(audio_chunks(source, info, tmp_path))
    assert [offset for offset, _ in windows] == [0, 5, 10]
    assert len(windows[0][1]) == 10 * RATE


@requires_ffmpeg
def test_movie_frames_and_samples_match_query_decoding(tmp_path):
    source = make_video(tmp_path / "movie.mp4", duration=10)
    info = probe_video(source)

    query_segments = decode_video(source.read_bytes(), "video/mp4", 30)
    movie_segments = [frames for _, frames in video_segments_stream(source, info, tmp_path)]
    assert len(movie_segments) == len(query_segments)
    for movie_frames, query_frames in zip(movie_segments, query_segments, strict=True):
        assert [frame.tobytes() for frame in movie_frames] == [frame.tobytes() for frame in query_frames]

    query_waveform = decode_audio(source.read_bytes(), "audio/mp4", 30)
    _, movie_window = next(audio_chunks(source, info, tmp_path))
    np.testing.assert_array_equal(movie_window[: len(query_waveform)], query_waveform)


@requires_ffmpeg
def test_anamorphic_frames_get_square_pixels(tmp_path):
    source = make_video(tmp_path / "dvd.mp4", size="720x480", extra=("-vf", "setsar=32/27"))
    batch = next(image_batches(source, probe_video(source), tmp_path))
    assert batch.images[0].size == (854, 480)


@requires_ffmpeg
def test_ffmpeg_failure_is_reported(tmp_path):
    with pytest.raises(FfmpegError, match="ffmpeg завершился с кодом"):
        list(read_ffmpeg(["-i", str(tmp_path / "missing.mp4"), "-f", "null", "-"], 1024, tmp_path))


@requires_ffmpeg
def test_abandoned_reading_stops_ffmpeg(tmp_path):
    source = make_video(tmp_path / "movie.mp4", duration=12)
    chunks = read_ffmpeg(["-i", str(source), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], 1024, tmp_path)
    next(chunks)
    chunks.close()
    assert list(tmp_path.iterdir()) == [source]


@requires_ffmpeg
def test_relative_source_path(tmp_path, monkeypatch):
    make_video(tmp_path / "movie.mp4", duration=6)
    temp_dir = tmp_path / "work" / "tmp"
    temp_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    source = Path("movie.mp4")
    info = probe_video(source)
    assert [offset for batch in image_batches(source, info, temp_dir) for offset in batch.offsets_sec] == [0, 5]
    assert [offset for offset, _ in video_segments_stream(source, info, temp_dir)] == [0]
    assert [offset for offset, _ in audio_chunks(source, info, temp_dir)] == [0, 5]
