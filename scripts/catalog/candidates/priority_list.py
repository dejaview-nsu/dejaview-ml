"""Ручной список movie_id, которые берутся в каталог первыми: например, фильмы с доступным видео."""

from pathlib import Path

from scripts.catalog.candidates.model import Candidate, Priority

MANUAL_BUCKET = "manual"


class PriorityFileError(Exception):
    pass


def load_priority_ids(path: Path) -> list[int]:
    """Один movie_id на строку, пустые строки и всё после # пропускаются."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise PriorityFileError(f"Не удалось прочитать {path}: {error.strerror or error}") from None
    except UnicodeDecodeError:
        raise PriorityFileError(f"{path}: файл должен быть в кодировке UTF-8") from None

    movie_ids = (_parse_line(line, f"{path}:{line_number}") for line_number, line in enumerate(lines, start=1))
    return list(dict.fromkeys(movie_id for movie_id in movie_ids if movie_id is not None))


def build_manual_candidates(movie_ids: list[int]) -> list[Candidate]:
    return [Candidate(movie_id, Priority.MANUAL, MANUAL_BUCKET) for movie_id in movie_ids]


def _parse_line(line: str, location: str) -> int | None:
    text = line.split("#", 1)[0].strip()
    if not text:
        return None
    if not text.isdigit() or int(text) == 0:
        raise PriorityFileError(f"{location}: ожидался положительный movie_id, получено {text!r}")
    return int(text)
