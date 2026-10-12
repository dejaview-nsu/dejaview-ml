import shutil
import subprocess

import pytest

from dejaview_ml.errors import MlError
from dejaview_ml.media import decode_video

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")


def video_file(tmp_path, duration, extension="mp4"):
    path = tmp_path / f"clip-{duration}.{extension}"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=64x48:rate=5:duration={duration}",
            "-an",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.mark.parametrize(
    ("duration", "segment_sizes"), [(0.2, [1]), (10, [10]), (12, [10, 2]), (30, [10, 10, 10])]
)
def test_real_video_sampling(tmp_path, duration, segment_sizes):
    data = video_file(tmp_path, duration).read_bytes()
    segments = decode_video(data, "video/mp4", 10)
    assert [len(segment) for segment in segments] == segment_sizes
    assert segments[0][0].size == (224, 224)


def test_video_duration_and_corruption(tmp_path):
    data = video_file(tmp_path, 31).read_bytes()
    with pytest.raises(MlError) as error:
        decode_video(data, "video/mp4", 10)
    assert error.value.code == "DURATION_TOO_LONG"
    with pytest.raises(MlError) as error:
        decode_video(b"not a video", "video/mp4", 10)
    assert error.value.code == "FILE_CORRUPTED"


@pytest.mark.parametrize(
    ("extension", "media_type"), [("mov", "video/quicktime"), ("webm", "video/webm")]
)
def test_other_video_containers(tmp_path, extension, media_type):
    data = video_file(tmp_path, 1, extension).read_bytes()
    assert len(decode_video(data, media_type, 10)) == 1
