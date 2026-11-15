"""Реплики фильма -> коллекция speech в Qdrant (поиск записи речи по её тексту).

    python -m scripts.index.speech --movie-id 603 --transcribe matrix.mp4
    python -m scripts.index.speech --movie-id 603 --transcribe matrix.mp4 --dry-run   # показать окна, не писать

Реплики распознаёт Whisper по звуковой дорожке файла - того же, что проиндексирован (или
скачанного с того же источника): слова и время совпадают с озвучкой, которую запишет пользователь.
Несколько минут на фильм. Повторный запуск заменяет реплики фильма целиком.
"""

import argparse
import logging
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path

from dotenv import load_dotenv

from dejaview_ml.embeddings import Encoders
from scripts.index import vectors
from scripts.index.cli import ExitCode, configure_logging, describe_error, parse_movie_id, use_utf8_output
from scripts.index.config import (
    DEFAULT_ENV_FILE,
    DEFAULT_QDRANT_TIMEOUT_SECONDS,
    MAX_QDRANT_TIMEOUT_SECONDS,
    _read_ml_settings,
    _read_seconds,
)
from scripts.index.embedding import load_model
from scripts.index.errors import ConfigError, UnsuitableFileError
from scripts.index.transcribe import speech_windows, transcribe_film
from scripts.index.validation import probe_audio

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    use_utf8_output()
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)
    try:
        run(args)
    except KeyboardInterrupt:
        logger.warning("Прервано, реплики фильма %d могли записаться не полностью: запустите ещё раз", args.movie_id)
        return ExitCode.INTERRUPTED
    except Exception as error:
        code, message = describe_error(error)
        logger.error("%s", message.replace(" (indexed_at не выставлен), исходный файл не тронут", ""))
        logger.debug("Трассировка:", exc_info=True)
        return code
    return ExitCode.OK


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    if args.env_file is not None and not args.env_file.is_file():
        raise ConfigError(f"Файл окружения не найден: {args.env_file}")
    load_dotenv(args.env_file or DEFAULT_ENV_FILE, override=False)
    ml = _read_ml_settings()
    encoders = Encoders(ml)
    load_model(encoders, "speech")
    with tempfile.TemporaryDirectory(prefix=f"speech-{args.movie_id}-") as temp_dir:
        cues = transcribe_film(encoders, args.transcribe, probe_audio(args.transcribe), Path(temp_dir))
    windows = speech_windows(cues)
    if not windows:
        raise UnsuitableFileError(f"В {args.transcribe} не нашлось реплик длиннее двух слов: записывать нечего")
    logger.info(
        "Окон по 20 с: %d, с %s по %s", len(windows), _clock(windows[0].offset_sec), _clock(windows[-1].offset_sec)
    )
    if args.dry_run is not None:
        for window in windows[: args.dry_run]:
            print(f"{_clock(window.offset_sec)}  {window.text}")
        return

    writer = vectors.connect(
        ml.qdrant_url,
        ml.qdrant_grpc_port,
        api_key=ml.qdrant_api_key.get_secret_value() if ml.qdrant_api_key else None,
        prefer_grpc=ml.qdrant_prefer_grpc,
        timeout_seconds=_read_seconds(
            "INDEX_QDRANT_TIMEOUT_SECONDS", DEFAULT_QDRANT_TIMEOUT_SECONDS, MAX_QDRANT_TIMEOUT_SECONDS
        ),
    )
    try:
        load_model(encoders, "text")
        encoded = [
            (window.offset_sec, window.text, encoders._encode_text(window.text, passage=True)) for window in windows
        ]
        count = writer.replace_speech(args.movie_id, encoded)
    finally:
        with suppress(Exception):
            writer.close()
    logger.info(
        "Готово за %.0f с: у фильма %d в поиске %d окон реплик", time.monotonic() - started, args.movie_id, count
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.index.speech",
        description="Распознаёт реплики фильма Whisper по озвучке и индексирует их в коллекцию speech",
    )
    parser.add_argument("--movie-id", type=parse_movie_id, required=True, help="id фильма в TMDB, например 603")
    parser.add_argument(
        "--transcribe",
        type=_video,
        required=True,
        metavar="FILE",
        help="видеофайл фильма: реплики распознаёт Whisper по озвучке",
    )
    parser.add_argument(
        "--dry-run",
        type=_positive,
        nargs="?",
        const=20,
        default=None,
        metavar="N",
        help="ничего не записывать, показать первые N окон (по умолчанию 20)",
    )
    parser.add_argument("--env-file", type=Path, help="путь к .env (по умолчанию scripts/index/.env)")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог и трассировка ошибок")
    return parser.parse_args(argv)


def _video(raw_value: str) -> Path:
    if not Path(raw_value).is_file():
        raise argparse.ArgumentTypeError(f"файл не найден: {raw_value}")
    return Path(raw_value)


def _positive(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"нужно целое число больше нуля, получено: {raw_value}")
    return value


def _clock(seconds: int) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


if __name__ == "__main__":
    sys.exit(main())
