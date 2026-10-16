import subprocess

import pytest

from scripts.index import slicing
from scripts.index.embedding import embed_movie
from scripts.index.tests.videos import make_video, requires_ffmpeg
from scripts.index.validation import probe_video


class FailingEncoders:
    def __init__(self, failing_modality):
        self.failing_modality = failing_modality

    def load(self, modality):
        pass

    def encode_text(self, text, passage):
        return [[1.0]]

    def _encode_frames(self, modality, frames):
        return self._encode(modality)

    def _encode_audio(self, window):
        return self._encode("audio")

    def _encode(self, modality):
        if modality == self.failing_modality:
            raise RuntimeError(f"{modality}: модель упала")
        return [1.0]


@requires_ffmpeg
@pytest.mark.parametrize("modality", ["image", "video", "audio"])
def test_encoder_failure_stops_ffmpeg_before_cleanup(tmp_path, monkeypatch, modality):
    source = make_video(tmp_path / "movie.mp4", duration=30)
    info = probe_video(source)
    temp_dir = tmp_path / "work" / "tmp"
    temp_dir.mkdir(parents=True)
    processes = []
    original_popen = subprocess.Popen

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(slicing.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="модель упала") as failure:
        embed_movie(FailingEncoders(modality), "описание", source, info, temp_dir)
    # Ошибка ещё не обработана и держит кадры генераторов: так её видит finally в pipeline
    assert failure.value
    assert processes
    assert all(process.poll() is not None for process in processes)
