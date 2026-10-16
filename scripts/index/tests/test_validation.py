import subprocess
from fractions import Fraction

import pytest

from scripts.index.errors import ConfigError, UnsuitableFileError
from scripts.index.tests.videos import make_video, requires_ffmpeg
from scripts.index.validation import probe_video

pytestmark = requires_ffmpeg


def test_suitable_mp4(tmp_path):
    info = probe_video(make_video(tmp_path / "movie.mp4"))
    assert (info.width, info.height) == (854, 480)
    assert info.duration_sec == pytest.approx(12, abs=0.1)
    assert (info.video_stream, info.audio_stream) == (0, 1)
    assert info.sample_aspect_ratio == 1


def test_widescreen_480p_is_accepted(tmp_path):
    info = probe_video(make_video(tmp_path / "scope.mp4", size="854x358"))
    assert (info.width, info.height) == (854, 358)


def test_anamorphic_sample_aspect_ratio(tmp_path):
    path = make_video(tmp_path / "dvd.mp4", size="720x480", extra=("-vf", "setsar=32/27"))
    assert probe_video(path).sample_aspect_ratio == Fraction(32, 27)


@pytest.mark.parametrize(
    ("name", "options", "problem"),
    [
        ("movie.mov", {}, "контейнер MOV, нужен MP4"),
        ("movie.webm", {"video_codec": "libvpx-vp9", "audio_codec": "libopus"}, "контейнер matroska,webm"),
        ("movie.mp4", {"size": "1280x720"}, "кадр 1280×720"),
        ("movie.mp4", {"audio_codec": None}, "нет звуковой дорожки"),
        ("movie.mp4", {"audio_codec": "libmp3lame"}, "аудиокодек mp3, нужен AAC"),
        ("movie.mp4", {"video_codec": "mpeg4"}, "видеокодек mpeg4, нужен H.264"),
    ],
)
def test_unsuitable_file_is_rejected_with_reason_and_hint(tmp_path, name, options, problem):
    path = make_video(tmp_path / name, duration=2, **options)
    with pytest.raises(UnsuitableFileError) as error:
        probe_video(path)
    assert problem in str(error.value)
    assert "ffmpeg -i in.mkv" in str(error.value)


def test_rotated_video_is_rejected(tmp_path):
    source = make_video(tmp_path / "source.mp4", duration=2)
    path = tmp_path / "rotated.mp4"
    args = ["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", str(source), "-c", "copy", str(path)]
    subprocess.run(args, check=True)
    with pytest.raises(UnsuitableFileError, match="поворот кадра"):
        probe_video(path)


def test_all_problems_are_reported_at_once(tmp_path):
    path = make_video(tmp_path / "movie.mov", duration=2, size="1280x720", audio_codec=None)
    with pytest.raises(UnsuitableFileError) as error:
        probe_video(path)
    message = str(error.value)
    assert "контейнер MOV" in message
    assert "1280×720" in message
    assert "нет звуковой дорожки" in message


def test_not_a_video(tmp_path):
    path = tmp_path / "movie.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(UnsuitableFileError, match="не читается как видео"):
        probe_video(path)


def test_missing_file_is_an_argument_error(tmp_path):
    with pytest.raises(ConfigError, match="не найден"):
        probe_video(tmp_path / "missing.mp4")
