import logging
import math
import re
import time
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dejaview_ml.embeddings import SPEECH_RATE, Encoders
from scripts.index.slicing import FfmpegError, read_ffmpeg

# Whisper слушает по 30 с; несколько кусков за раз заметно быстрее на GPU.
CHUNK_SEC = 30
BATCH_CHUNKS = 8
# На тишине Whisper сочиняет текст; такие куски пропускаются целиком.
SILENCE_DB = -50
PROGRESS_STEP_PERCENT = 10
# Запрос пользователя не длиннее 10 с: при окне 20 с и шаге 10 с любой такой отрывок целиком
# попадает хотя бы в одно окно.
SPEECH_WINDOW_SEC = 20
SPEECH_STEP_SEC = 10
SPEECH_MIN_WORDS = 3
WORD = re.compile(r"\w+")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cue:
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class SpeechWindow:
    offset_sec: int
    text: str


def transcribe_film(encoders: Encoders, source: Path, duration_sec: float, temp_dir: Path) -> list[Cue]:
    """Реплики звуковой дорожки фильма с временем от начала файла."""
    started = time.monotonic()
    cues: list[Cue] = []
    expected = max(1, int(duration_sec // CHUNK_SEC) + 1)
    next_report = PROGRESS_STEP_PERCENT
    done = 0
    with closing(_chunks(source, temp_dir)) as chunks:
        batch: list[tuple[float, np.ndarray]] = []
        for offset, chunk in chunks:
            done += 1
            if _level_db(chunk) > SILENCE_DB:
                batch.append((offset, chunk))
            if len(batch) == BATCH_CHUNKS:
                cues += _transcribe(encoders, batch)
                batch = []
            percent = 100 * done // expected
            if percent >= next_report:
                logger.info(
                    "распознавание: %d%%, реплик %d, %.0f с", min(percent, 100), len(cues), time.monotonic() - started
                )
                next_report = (percent // PROGRESS_STEP_PERCENT + 1) * PROGRESS_STEP_PERCENT
        if batch:
            cues += _transcribe(encoders, batch)
    if not cues:
        raise FfmpegError(f"Whisper не распознал в {source.name} ни одной реплики: в файле нет речи или нет звука")
    logger.info("распознавание: %d реплик за %.0f с", len(cues), time.monotonic() - started)
    return cues


def _chunks(source: Path, temp_dir: Path) -> Iterator[tuple[float, np.ndarray]]:
    args = ["-threads", "1", "-i", str(source.resolve()), "-vn", "-sn", "-dn", "-ac", "1"]
    args += ["-ar", str(SPEECH_RATE), "-f", "f32le", "pipe:1"]
    for index, chunk in enumerate(read_ffmpeg(args, CHUNK_SEC * SPEECH_RATE * 4, temp_dir)):
        if len(chunk) % 4:
            raise FfmpegError("ffmpeg вернул неполный аудиосэмпл")
        yield index * CHUNK_SEC, np.frombuffer(chunk, dtype="<f4")


def _transcribe(encoders: Encoders, batch: list[tuple[float, np.ndarray]]) -> list[Cue]:
    segments = encoders.transcribe_segments([chunk for _, chunk in batch])
    return [
        Cue(offset + start, offset + end, text)
        for (offset, _), chunk_segments in zip(batch, segments, strict=True)
        for start, end, text in chunk_segments
    ]


def _level_db(chunk: np.ndarray) -> float:
    return 20 * float(np.log10(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))) + 1e-9))


def speech_windows(
    cues: list[Cue], window_sec: int = SPEECH_WINDOW_SEC, step_sec: int = SPEECH_STEP_SEC
) -> list[SpeechWindow]:
    """Окна по window_sec с шагом step_sec; в окно попадает каждая реплика, которая с ним пересекается."""
    if not cues:
        return []
    windows = []
    first = math.floor(cues[0].start_sec / step_sec) * step_sec
    for offset in range(first, math.ceil(cues[-1].end_sec), step_sec):
        text = " ".join(cue.text for cue in cues if cue.start_sec < offset + window_sec and cue.end_sec > offset)
        if len(WORD.findall(text)) >= SPEECH_MIN_WORDS:
            windows.append(SpeechWindow(offset, text))
    return windows
