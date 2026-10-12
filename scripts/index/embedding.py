import logging
import time
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from dejaview_ml.config import MODEL_IDS
from dejaview_ml.embeddings import Encoders
from scripts.index.errors import IndexingError
from scripts.index.slicing import audio_chunks, expected_counts, image_batches, video_segments_stream
from scripts.index.validation import VideoInfo

PROGRESS_STEP_PERCENT = 10

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fragment:
    offset_sec: int | None
    vector: list[float]


def embed_movie(
    encoders: Encoders, overview: str, source: Path, info: VideoInfo, temp_dir: Path
) -> dict[str, list[Fragment]]:
    expected = expected_counts(info.duration_sec)
    fragments: dict[str, list[Fragment]] = {}

    _load(encoders, "text")
    fragments["text"] = [Fragment(None, encoders.encode_text(overview, passage=True)[0])]
    logger.info("text: описание фильма закодировано")

    _load(encoders, "image")
    fragments["image"] = _collect("image", expected["image"], _encode_images(encoders, source, info, temp_dir))

    # Нарезки закрываются явно: при ошибке модели ffmpeg останавливается до удаления временной папки,
    # а не когда сборщик мусора доберётся до генератора.
    _load(encoders, "video")
    with closing(video_segments_stream(source, info, temp_dir)) as segments:
        fragments["video"] = _collect(
            "video",
            expected["video"],
            (Fragment(offset, encoders._encode_frames("video", frames)) for offset, frames in segments),
        )

    _load(encoders, "audio")
    with closing(audio_chunks(source, info, temp_dir)) as windows:
        fragments["audio"] = _collect(
            "audio",
            expected["audio"],
            (Fragment(offset, encoders._encode_audio(window)) for offset, window in windows),
        )
    return fragments


def _load(encoders: Encoders, modality: str) -> None:
    logger.info("%s: загрузка модели %s (первый запуск скачивает веса)", modality, MODEL_IDS[modality])
    try:
        encoders.load(modality)
    except Exception as error:
        raise IndexingError(
            f"Не удалось загрузить модель {MODEL_IDS[modality]}: {error}. Проверьте ML_CACHE_DIR, "
            "ML_LOCAL_FILES_ONLY и доступ к huggingface.co"
        ) from error


def _encode_images(encoders: Encoders, source: Path, info: VideoInfo, temp_dir: Path) -> Iterator[Fragment]:
    with closing(image_batches(source, info, temp_dir)) as batches:
        for batch in batches:
            for offset, image in zip(batch.offsets_sec, batch.images, strict=True):
                yield Fragment(offset, encoders._encode_frames("image", [image]))


def _collect(modality: str, expected: int, fragments: Iterator[Fragment]) -> list[Fragment]:
    started = time.monotonic()
    result: list[Fragment] = []
    next_report = PROGRESS_STEP_PERCENT
    for fragment in fragments:
        result.append(fragment)
        percent = 100 * len(result) // max(expected, 1)
        if percent >= next_report:
            logger.info("%s: %d из ~%d (%d%%)", modality, len(result), expected, min(percent, 100))
            next_report = (percent // PROGRESS_STEP_PERCENT + 1) * PROGRESS_STEP_PERCENT
    if not result:
        raise IndexingError(f"{modality}: ffmpeg не выдал ни одного фрагмента, файл слишком короткий или повреждён")
    logger.info("%s: %d фрагментов за %.0f с", modality, len(result), time.monotonic() - started)
    return result
