"""Настройки из переменных окружения и файла .env. Переменные окружения процесса важнее файла."""

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv

DEFAULT_ENV_FILE = Path(__file__).with_name(".env")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("output")

DEFAULT_TARGET_SIZE = 1000
MAX_TARGET_SIZE = 10_000
DEFAULT_MIN_VOTE_COUNT = 200
MAX_MIN_VOTE_COUNT = 1_000_000
DEFAULT_YEAR_FROM = 1950
EARLIEST_YEAR = 1900
DEFAULT_REQUESTS_PER_SECOND = 20.0
MAX_REQUESTS_PER_SECOND = 50.0
DEFAULT_POSTER_SIZE = "w500"
POSTER_SIZE_PATTERN = re.compile(r"w\d+|original")

T = TypeVar("T")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Settings:
    tmdb_access_token: str | None
    tmdb_api_key: str | None
    output_dir: Path
    target_size: int
    min_vote_count: int
    year_from: int
    requests_per_second: float
    poster_size: str
    priority_ids_file: Path | None


def load_settings(env_file: Path | None, current_year: int, *, require_tmdb_key: bool = True) -> Settings:
    _load_env_file(env_file)
    access_token = _read_text("TMDB_ACCESS_TOKEN")
    api_key = _read_text("TMDB_API_KEY")
    if require_tmdb_key and access_token is None and api_key is None:
        raise ConfigError("Не задан ключ TMDB: укажите TMDB_ACCESS_TOKEN или TMDB_API_KEY")
    return Settings(
        tmdb_access_token=access_token,
        tmdb_api_key=api_key,
        output_dir=_read_output_dir(),
        target_size=_read_int("CATALOG_TARGET_SIZE", DEFAULT_TARGET_SIZE, 1, MAX_TARGET_SIZE),
        min_vote_count=_read_int("CATALOG_MIN_VOTE_COUNT", DEFAULT_MIN_VOTE_COUNT, 0, MAX_MIN_VOTE_COUNT),
        year_from=_read_int("CATALOG_YEAR_FROM", DEFAULT_YEAR_FROM, EARLIEST_YEAR, current_year),
        requests_per_second=_read_number(
            "TMDB_REQUESTS_PER_SECOND",
            DEFAULT_REQUESTS_PER_SECOND,
            float,
            lambda value: 0 < value <= MAX_REQUESTS_PER_SECOND,
            f"числом больше 0 и не больше {MAX_REQUESTS_PER_SECOND:g}",
        ),
        poster_size=_read_poster_size(),
        priority_ids_file=_read_path("CATALOG_PRIORITY_IDS_FILE"),
    )


def _load_env_file(env_file: Path | None) -> None:
    if env_file is not None and not env_file.is_file():
        raise ConfigError(f"Файл окружения не найден: {env_file}")
    load_dotenv(env_file or DEFAULT_ENV_FILE, override=False)


def _read_text(name: str) -> str | None:
    return os.environ.get(name, "").strip() or None


def _read_path(name: str) -> Path | None:
    value = _read_text(name)
    return Path(value) if value else None


def _read_int(name: str, default: int, minimum: int, maximum: int) -> int:
    return _read_number(
        name, default, int, lambda value: minimum <= value <= maximum, f"целым числом от {minimum} до {maximum}"
    )


def _read_number(name: str, default: T, parse: Callable[[str], T], is_valid: Callable[[T], bool], rule: str) -> T:
    raw_value = _read_text(name)
    if raw_value is None:
        return default
    try:
        value = parse(raw_value)
    except ValueError:
        value = None
    if value is None or not is_valid(value):
        raise ConfigError(f"{name} должно быть {rule}, получено: {raw_value!r}")
    return value


def _read_poster_size() -> str:
    value = _read_text("TMDB_POSTER_SIZE") or DEFAULT_POSTER_SIZE
    if not POSTER_SIZE_PATTERN.fullmatch(value):
        raise ConfigError(f"TMDB_POSTER_SIZE должно быть вида w500 или original, получено: {value!r}")
    return value


def _read_output_dir() -> Path:
    output_dir = _read_path("CATALOG_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR
    if output_dir.exists() and not output_dir.is_dir():
        raise ConfigError(f"CATALOG_OUTPUT_DIR указывает на файл, а нужна папка: {output_dir}")
    return output_dir
