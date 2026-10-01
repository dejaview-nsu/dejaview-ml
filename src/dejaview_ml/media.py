import io
import json
import math
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from dejaview_ml.errors import MlError

MIB = 1024 * 1024
LIMITS = {"image": 10 * MIB, "video": 50 * MIB, "audio": 20 * MIB, "text": 64 * 1024}
MEDIA_TYPES = {
    "text": {"application/json": ".json"},
    "image": {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"},
    "video": {"video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm"},
    "audio": {"audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/ogg": ".ogg", "audio/mp4": ".m4a"},
}


def validate_text(query: str, *, passage: bool = False) -> str:
    if not isinstance(query, str) or not query.strip():
        raise MlError(422, "INVALID_QUERY", "query must be a nonempty string")
    if not passage and not 3 <= len(query) <= 250:
        raise MlError(422, "INVALID_QUERY", "query must contain 3-250 characters")
    try:
        query.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise MlError(422, "INVALID_QUERY", "query contains invalid Unicode") from exc
    return query


def parse_text_query(data: bytes) -> str:
    try:
        body = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise MlError(422, "INVALID_QUERY", "invalid UTF-8 JSON") from exc
    if not isinstance(body, dict) or set(body) != {"query"}:
        raise MlError(422, "INVALID_QUERY", "expected an object containing only query")
    return validate_text(body["query"])


def decode_image(data: bytes, content_type: str) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format != MEDIA_TYPES["image"][content_type]:
                    raise MlError(
                        415, "UNSUPPORTED_MEDIA_TYPE", "image does not match Content-Type"
                    )
                source.load()
                return ImageOps.exif_transpose(source).convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise MlError(422, "FILE_CORRUPTED", "cannot decode image") from exc


def _run(args: list[str], timeout: float) -> bytes:
    try:
        return subprocess.run(
            args,
            capture_output=True,
            check=True,
            timeout=timeout,
        ).stdout
    except subprocess.TimeoutExpired as exc:
        raise MlError(500, "INTERNAL_ERROR", "media decoder timed out") from exc
    except subprocess.CalledProcessError as exc:
        raise MlError(422, "FILE_CORRUPTED", "cannot decode media") from exc


def decode_audio(data: bytes, content_type: str, timeout: float) -> np.ndarray:
    """Decode one <=10-second clip as mono float32 PCM at CLAP's 48 kHz rate."""
    with tempfile.TemporaryDirectory(prefix="dejaview-audio-") as directory:
        path = Path(directory) / ("query" + MEDIA_TYPES["audio"][content_type])
        path.write_bytes(data)
        probe = _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-show_entries",
                "format=format_name:stream=codec_type,codec_name,profile,sample_rate,duration",
                "-of",
                "json",
                str(path),
            ],
            timeout,
        )
        try:
            metadata = json.loads(probe)
            stream = next(s for s in metadata["streams"] if s["codec_type"] == "audio")
            formats = set(metadata["format"]["format_name"].split(","))
        except (KeyError, ValueError, TypeError, StopIteration) as exc:
            raise MlError(422, "FILE_CORRUPTED", "invalid audio stream") from exc
        expected = {
            "audio/mpeg": {"mp3"},
            "audio/wav": {"wav"},
            "audio/ogg": {"ogg"},
            "audio/mp4": {"mov", "mp4"},
        }[content_type]
        if not formats & expected:
            raise MlError(415, "UNSUPPORTED_MEDIA_TYPE", "audio does not match Content-Type")
        # Decode a little beyond the limit to detect oversize audio without trusting
        # container duration (e.g. MP3 headers include encoder delay/padding).
        raw = _run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-threads",
                "1",
                "-protocol_whitelist",
                "file,pipe",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-t",
                "10.1",
                "-ac",
                "1",
                "-ar",
                "48000",
                "-f",
                "f32le",
                "pipe:1",
            ],
            timeout,
        )
    if not raw or len(raw) % 4:
        raise MlError(422, "FILE_CORRUPTED", "audio has no valid samples")
    waveform = np.frombuffer(raw, dtype="<f4").copy()
    if content_type == "audio/mp4" and stream.get("codec_name") == "aac":
        # FFmpeg can leave a partial final AAC frame beyond the MP4 track duration.
        # Trim only a bounded codec tail, never arbitrary audio beyond the time limit.
        try:
            track_duration = float(stream["duration"])
            source_rate = int(stream["sample_rate"])
            frame_size = 2048 if str(stream.get("profile", "")).startswith("HE") else 1024
            if math.isfinite(track_duration) and 0 < track_duration <= 10 and source_rate > 0:
                track_samples = round(track_duration * 48_000)
                padding_limit = math.ceil(frame_size * 48_000 / source_rate)
                if 0 < len(waveform) - track_samples <= padding_limit:
                    waveform = waveform[:track_samples]
        except (KeyError, ValueError, TypeError):
            pass  # Missing duration cannot authorize trimming; use decoded sample count.
    if len(waveform) > 480_000:
        raise MlError(422, "DURATION_TOO_LONG", "audio exceeds 10 s")
    if not np.isfinite(waveform).all():
        raise MlError(422, "FILE_CORRUPTED", "audio has non-finite samples")
    return waveform


def decode_video(data: bytes, content_type: str, timeout: float) -> list[list[Image.Image]]:
    """One frame/second; chunks of <=10 frames correspond to <=10-second segments.

    FFmpeg performs bicubic resize and center crop; the model processor only normalizes.
    This function is shared by query inference and indexing.
    """
    with tempfile.TemporaryDirectory(prefix="dejaview-") as directory:
        path = Path(directory) / ("query" + MEDIA_TYPES["video"][content_type])
        path.write_bytes(data)
        probe = _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-show_entries",
                "format=duration,format_name:stream=codec_type,width,height,duration",
                "-of",
                "json",
                str(path),
            ],
            timeout,
        )
        try:
            metadata = json.loads(probe)
            stream = next(s for s in metadata["streams"] if s["codec_type"] == "video")
            duration = float(metadata["format"].get("duration", stream.get("duration", "nan")))
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("missing duration")
            if stream["width"] * stream["height"] > 16_777_216:
                raise ValueError("video frame exceeds 16 megapixels")
            formats = set(metadata["format"]["format_name"].split(","))
        except (KeyError, ValueError, TypeError, StopIteration) as exc:
            raise MlError(422, "FILE_CORRUPTED", "invalid video stream or duration") from exc
        expected = {"webm"} if content_type == "video/webm" else {"mov", "mp4"}
        if not formats & expected:
            raise MlError(415, "UNSUPPORTED_MEDIA_TYPE", "video does not match Content-Type")
        if duration > 30:
            raise MlError(422, "DURATION_TOO_LONG", f"video is {duration:g} s, limit 30 s")
        raw = _run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-threads",
                "1",
                "-protocol_whitelist",
                "file,pipe",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-vf",
                "fps=1:round=up,scale=224:224:force_original_aspect_ratio=increase:flags=bicubic,"
                "crop=224:224",
                "-t",
                "30",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            timeout,
        )
    frame_bytes = 224 * 224 * 3
    if not raw or len(raw) % frame_bytes or len(raw) > 30 * frame_bytes:
        raise MlError(422, "FILE_CORRUPTED", "video has no valid sampled frames")
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 224, 224, 3)
    return [
        [Image.fromarray(frame) for frame in frames[start : start + 10]]
        for start in range(0, len(frames), 10)
    ]
