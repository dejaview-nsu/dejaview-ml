import os
import subprocess

import numpy as np
import pytest

from dejaview_ml.config import Settings
from dejaview_ml.embeddings import Encoders
from dejaview_ml.storage import metadata_passage
from scripts.catalog.tests.samples import sample_movie
from scripts.index.embedding import embed_movie
from scripts.index.tests.videos import make_video, requires_ffmpeg
from scripts.index.validation import probe_video

pytestmark = [
    pytest.mark.integration,
    requires_ffmpeg,
    pytest.mark.skipif(os.getenv("RUN_MODEL_TESTS") != "1", reason="RUN_MODEL_TESTS=1 загружает настоящие веса"),
]


def cut(source, start, duration, name, *options):
    path = source.with_name(name)
    args = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        str(start),
        "-i",
        str(source),
        "-t",
        str(duration),
        *options,
        str(path),
    ]
    subprocess.run(args, check=True, capture_output=True)
    return path.read_bytes()


def test_indexed_fragments_match_queries(tmp_path):
    source = make_video(tmp_path / "movie.mp4", duration=22)
    encoders = Encoders(Settings(_env_file=None, device="cpu"))
    movie = sample_movie(550)
    fragments = embed_movie(encoders, movie, source, probe_video(source), tmp_path)

    assert [fragment.offset_sec for fragment in fragments["image"]] == [0, 5, 10, 15, 20]
    assert [fragment.offset_sec for fragment in fragments["video"]] == [0, 10, 20]
    assert [fragment.offset_sec for fragment in fragments["audio"]] == [0, 5, 10, 15, 20]

    def similarity(indexed, query):
        return float(np.dot(indexed, query))

    frame = cut(source, 5, 1, "frame.png", "-frames:v", "1")
    (image_query,) = encoders.encode("image", frame, "image/png")
    assert similarity(fragments["image"][1].vector, image_query) > 0.999

    clip = cut(source, 0, 10, "clip.mp4", "-an", "-c", "copy")
    video_query = encoders.encode("video", clip, "video/mp4")[0]
    assert similarity(fragments["video"][0].vector, video_query) > 0.999

    sound = cut(source, 0, 10, "sound.wav", "-vn", "-c:a", "pcm_f32le")
    (audio_query,) = encoders.encode("audio", sound, "audio/wav")
    assert similarity(fragments["audio"][0].vector, audio_query) > 0.999

    overview, metadata = fragments["text"]
    assert (overview.source, metadata.source) == ("overview", "metadata")
    (overview_query,) = encoders.encode_text(movie.overview, passage=True)
    assert similarity(overview.vector, overview_query) > 0.999
    (metadata_query,) = encoders.encode_text(metadata_passage(movie), passage=True)
    assert similarity(metadata.vector, metadata_query) > 0.999
