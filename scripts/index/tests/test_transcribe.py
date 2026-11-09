import numpy as np
import pytest

from scripts.index import transcribe
from scripts.index.errors import IndexingError
from scripts.index.transcribe import Cue, speech_windows

RATE = 16_000


class FakeEncoders:
    def __init__(self):
        self.batches = []

    def transcribe_segments(self, chunks):
        self.batches.append(len(chunks))
        return [[(1.0, 4.0, f"реплика {len(self.batches)}.{index}")] for index in range(len(chunks))]


def chunks(levels):
    for index, level in enumerate(levels):
        yield index * transcribe.CHUNK_SEC, np.full(transcribe.CHUNK_SEC * RATE, level, dtype=np.float32)


def test_cues_are_timed_from_film_start_and_silence_is_skipped(monkeypatch, tmp_path):
    # Ten chunks, the third silent: nine go to Whisper in batches of eight and one.
    levels = [0.1] * 10
    levels[2] = 0.0
    monkeypatch.setattr(transcribe, "_chunks", lambda source, temp_dir: chunks(levels))
    encoders = FakeEncoders()
    cues = transcribe.transcribe_film(encoders, tmp_path / "film.mp4", 300, tmp_path)
    assert encoders.batches == [8, 1]
    assert cues[0] == Cue(1.0, 4.0, "реплика 1.0")
    assert [cue.start_sec for cue in cues[:3]] == [1.0, 31.0, 91.0]  # 61 s: silence
    assert cues[-1] == Cue(271.0, 274.0, "реплика 2.0")


def test_film_without_speech_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(transcribe, "_chunks", lambda source, temp_dir: chunks([0.0, 0.0]))
    with pytest.raises(IndexingError, match="ни одной реплики"):
        transcribe.transcribe_film(FakeEncoders(), tmp_path / "film.mp4", 60, tmp_path)


def test_speech_windows_overlap_so_any_ten_seconds_fit_in_one():
    cues = [Cue(5, 8, "один два три"), Cue(18, 21, "четыре пять шесть"), Cue(45, 46, "да")]
    assert [(window.offset_sec, window.text) for window in speech_windows(cues)] == [
        (0, "один два три четыре пять шесть"),
        (10, "четыре пять шесть"),
        (20, "четыре пять шесть"),
        # 30 s and 40 s: only "да", too short to identify a film.
    ]
