import logging
import math
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.index.errors import IndexingError
from scripts.index.validation import VideoInfo

IMAGE_STEP_SEC = 5
VIDEO_SEGMENT_SEC = 10
AUDIO_WINDOW_SEC = 10
AUDIO_STEP_SEC = 5
AUDIO_MIN_WINDOW_SEC = 1
IMAGE_BATCH_SIZE = 32
STDERR_PREVIEW_LENGTH = 500

VIDEO_FRAME_SIZE = 224
VIDEO_FRAME_BYTES = VIDEO_FRAME_SIZE * VIDEO_FRAME_SIZE * 3
VIDEO_SEGMENT_FRAMES = 10
VIDEO_OUTPUT_ARGS = [
    "-vf",
    "fps=1:round=up,scale=224:224:force_original_aspect_ratio=increase:flags=bicubic,crop=224:224",
    "-f",
    "rawvideo",
    "-pix_fmt",
    "rgb24",
]
AUDIO_SAMPLE_RATE = 48_000
AUDIO_OUTPUT_ARGS = ["-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE), "-f", "f32le"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageBatch:
    offsets_sec: list[int]
    images: list[Image.Image]


class FfmpegError(IndexingError):
    pass


def expected_counts(duration_sec: float) -> dict[str, int]:
    audio_windows = math.floor((duration_sec - AUDIO_MIN_WINDOW_SEC) / AUDIO_STEP_SEC) + 1
    return {
        "image": math.ceil(duration_sec / IMAGE_STEP_SEC),
        "video": math.ceil(duration_sec / VIDEO_SEGMENT_SEC),
        "audio": max(audio_windows, 0),
    }


def image_batches(source: Path, info: VideoInfo, temp_dir: Path) -> Iterator[ImageBatch]:
    width, height = info.width, info.height
    video_filter = f"fps=1/{IMAGE_STEP_SEC}:round=up"
    if info.sample_aspect_ratio != 1:
        width = max(2, round(info.width * info.sample_aspect_ratio / 2) * 2)
        video_filter += f",scale={width}:{height}:flags=bicubic,setsar=1"
    frame_bytes = width * height * 3
    args = [
        "-i",
        str(source.resolve()),
        "-map",
        f"0:{info.video_stream}",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        video_filter,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    index = 0
    for chunk in read_ffmpeg(args, IMAGE_BATCH_SIZE * frame_bytes, temp_dir):
        if len(chunk) % frame_bytes:
            raise FfmpegError(f"ffmpeg вернул неполный кадр {width}×{height}")
        frames = np.frombuffer(chunk, dtype=np.uint8).reshape(-1, height, width, 3)
        offsets = [(index + position) * IMAGE_STEP_SEC for position in range(len(frames))]
        index += len(frames)
        yield ImageBatch(offsets, [Image.fromarray(frame) for frame in frames])


def video_segments_stream(source: Path, info: VideoInfo, temp_dir: Path) -> Iterator[tuple[int, list[Image.Image]]]:
    args = ["-threads", "1", "-i", str(source.resolve()), "-map", f"0:{info.video_stream}", "-an", "-sn", "-dn"]
    args += [*VIDEO_OUTPUT_ARGS, "pipe:1"]
    for index, chunk in enumerate(read_ffmpeg(args, VIDEO_SEGMENT_FRAMES * VIDEO_FRAME_BYTES, temp_dir)):
        if len(chunk) % VIDEO_FRAME_BYTES:
            raise FfmpegError("ffmpeg вернул неполный кадр 224×224")
        frames = np.frombuffer(chunk, dtype=np.uint8).reshape(-1, VIDEO_FRAME_SIZE, VIDEO_FRAME_SIZE, 3)
        yield index * VIDEO_SEGMENT_SEC, [Image.fromarray(frame) for frame in frames]


def audio_chunks(source: Path, info: VideoInfo, temp_dir: Path) -> Iterator[tuple[int, np.ndarray]]:
    args = ["-threads", "1", "-i", str(source.resolve()), "-map", f"0:{info.audio_stream}", "-vn", "-sn", "-dn"]
    args += [*AUDIO_OUTPUT_ARGS, "pipe:1"]
    step_bytes = AUDIO_STEP_SEC * AUDIO_SAMPLE_RATE * 4

    def blocks() -> Iterator[np.ndarray]:
        for chunk in read_ffmpeg(args, step_bytes, temp_dir):
            if len(chunk) % 4:
                raise FfmpegError("ffmpeg вернул неполный аудиосэмпл")
            yield np.frombuffer(chunk, dtype="<f4")

    with closing(blocks()) as stream:
        yield from audio_windows(stream)


def audio_windows(blocks: Iterable[np.ndarray]) -> Iterator[tuple[int, np.ndarray]]:
    window = AUDIO_WINDOW_SEC * AUDIO_SAMPLE_RATE
    step = AUDIO_STEP_SEC * AUDIO_SAMPLE_RATE
    buffer = np.empty(0, dtype=np.float32)
    offset_sec = 0
    for block in blocks:
        buffer = np.concatenate([buffer, block])
        while len(buffer) >= window:
            yield offset_sec, buffer[:window]
            buffer = buffer[step:]
            offset_sec += AUDIO_STEP_SEC
    while len(buffer) >= AUDIO_MIN_WINDOW_SEC * AUDIO_SAMPLE_RATE:
        yield offset_sec, buffer[:window]
        buffer = buffer[step:]
        offset_sec += AUDIO_STEP_SEC


def read_ffmpeg(args: list[str], chunk_bytes: int, temp_dir: Path) -> Iterator[bytes]:
    command = ["ffmpeg", "-nostdin", "-v", "error", *args]
    with (
        tempfile.TemporaryFile(dir=temp_dir) as stderr,
        subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr, cwd=temp_dir
        ) as process,
    ):
        try:
            while chunk := process.stdout.read(chunk_bytes):
                yield chunk
        except BaseException:
            process.kill()
            raise
        return_code = process.wait()
        stderr.seek(0)
        errors = stderr.read().decode(errors="replace").strip()
    if return_code != 0:
        raise FfmpegError(
            f"ffmpeg завершился с кодом {return_code}: {errors[-STDERR_PREVIEW_LENGTH:] or 'без сообщения'}"
        )
    if errors:
        logger.warning("ffmpeg сообщил об ошибках декодирования: %s", errors[-STDERR_PREVIEW_LENGTH:])
