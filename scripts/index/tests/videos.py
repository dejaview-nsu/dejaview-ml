import shutil
import subprocess
from pathlib import Path

import pytest

requires_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="нужны ffmpeg и ffprobe"
)


def make_video(  # noqa: PLR0913
    path: Path,
    *,
    duration: float = 12,
    size: str = "854x480",
    video_codec: str = "libx264",
    audio_codec: str | None = "aac",
    extra: tuple[str, ...] = (),
) -> Path:
    args = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={size}:rate=10:duration={duration}",
    ]
    if audio_codec:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={duration}"]
    args += ["-c:v", video_codec, "-pix_fmt", "yuv420p"]
    args += ["-c:a", audio_codec] if audio_codec else ["-an"]
    args += [*extra, str(path)]
    subprocess.run(args, check=True, capture_output=True)
    return path
