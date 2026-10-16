import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from scripts.index.errors import ConfigError, UnsuitableFileError

MAX_WIDTH = 854
MAX_HEIGHT = 480
PROBE_TIMEOUT_SECONDS = 60
STDERR_PREVIEW_LENGTH = 300
REENCODE_HINT = (
    'ffmpeg -i in.mkv -map 0:v:0 -map 0:a:0 -c:v libx264 -c:a aac -vf "scale=854:480:'
    'force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1" out.mp4'
)


@dataclass(frozen=True)
class VideoInfo:
    duration_sec: float
    video_stream: int
    audio_stream: int
    width: int
    height: int
    sample_aspect_ratio: Fraction


def require_ffmpeg() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise ConfigError(
            f"Не найдены {', '.join(missing)} в PATH: установите ffmpeg (https://ffmpeg.org/download.html)"
        )


def probe_video(path: Path) -> VideoInfo:
    if not path.is_file():
        raise ConfigError(f"Видеофайл не найден: {path}")
    metadata = _run_ffprobe(path)
    problems: list[str] = []

    format_section = metadata.get("format") or {}
    format_names = set(str(format_section.get("format_name", "")).split(","))
    major_brand = str((format_section.get("tags") or {}).get("major_brand", "")).strip()
    if "mp4" not in format_names or not major_brand or major_brand == "qt":
        container = "MOV" if major_brand == "qt" else format_section.get("format_name", "неизвестен")
        problems.append(f"контейнер {container}, нужен MP4")

    streams = metadata.get("streams") or []
    video = _first_stream(streams, "video")
    if video is None:
        problems.append("нет видеодорожки")
    else:
        problems.extend(_video_problems(video))
    audio = _first_stream(streams, "audio")
    if audio is None:
        problems.append("нет звуковой дорожки, она нужна для коллекции audio")
    elif audio.get("codec_name") != "aac":
        problems.append(f"аудиокодек {audio.get('codec_name', 'неизвестен')}, нужен AAC")

    duration = _as_float(format_section.get("duration"))
    if duration is None or duration <= 0:
        problems.append("не удалось определить длительность")

    if problems or video is None or audio is None or duration is None:
        raise UnsuitableFileError(
            f"Файл {path} не подходит для индексации (#17395 п. 2.3: MP4, H.264, AAC, не больше "
            f"{MAX_WIDTH}×{MAX_HEIGHT}):\n" + "".join(f"  - {problem}\n" for problem in problems) + "Пережмите его, "
            f"например:\n  {REENCODE_HINT}"
        )
    return VideoInfo(
        duration_sec=duration,
        video_stream=int(video["index"]),
        audio_stream=int(audio["index"]),
        width=int(video["width"]),
        height=int(video["height"]),
        sample_aspect_ratio=_sample_aspect_ratio(video.get("sample_aspect_ratio")),
    )


def _run_ffprobe(path: Path) -> dict[str, Any]:
    args = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=format_name,duration:format_tags=major_brand"
        ":stream=index,codec_type,codec_name,width,height,sample_aspect_ratio"
        ":stream_disposition=attached_pic:stream_side_data=rotation",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(args, capture_output=True, check=False, timeout=PROBE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        raise UnsuitableFileError(f"ffprobe не прочитал {path} за {PROBE_TIMEOUT_SECONDS} с") from None
    if result.returncode != 0:
        details = result.stderr.decode(errors="replace").strip()[:STDERR_PREVIEW_LENGTH]
        raise UnsuitableFileError(f"Файл {path} не читается как видео: {details or 'ffprobe завершился с ошибкой'}")
    try:
        metadata = json.loads(result.stdout)
    except ValueError:
        raise UnsuitableFileError(f"ffprobe вернул непонятный ответ для {path}") from None
    return metadata if isinstance(metadata, dict) else {}


def _first_stream(streams: list[dict[str, Any]], codec_type: str) -> dict[str, Any] | None:
    return next(
        (
            stream
            for stream in streams
            if stream.get("codec_type") == codec_type and not (stream.get("disposition") or {}).get("attached_pic")
        ),
        None,
    )


def _video_problems(video: dict[str, Any]) -> list[str]:
    problems = []
    if video.get("codec_name") != "h264":
        problems.append(f"видеокодек {video.get('codec_name', 'неизвестен')}, нужен H.264")
    width, height = video.get("width"), video.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        problems.append("не удалось определить размер кадра")
    elif width > MAX_WIDTH or height > MAX_HEIGHT:
        problems.append(f"кадр {width}×{height}, нужен 480p: не больше {MAX_WIDTH}×{MAX_HEIGHT}")
    rotation = next((item.get("rotation") for item in video.get("side_data_list") or [] if "rotation" in item), 0)
    if rotation:
        problems.append(f"в файле задан поворот кадра на {rotation}°")
    return problems


def _sample_aspect_ratio(value: Any) -> Fraction:
    try:
        numerator, denominator = (int(part) for part in str(value).split(":"))
    except ValueError:
        return Fraction(1)
    return Fraction(numerator, denominator) if numerator > 0 and denominator > 0 else Fraction(1)


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
