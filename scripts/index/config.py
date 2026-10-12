import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from dejaview_ml.config import Settings as MlSettings
from scripts.catalog.config import DEFAULT_REQUESTS_PER_SECOND, MAX_REQUESTS_PER_SECOND
from scripts.index.errors import ConfigError

DEFAULT_ENV_FILE = Path(__file__).with_name(".env")
DEFAULT_WORK_DIR = Path(__file__).with_name("work")
DEFAULT_QDRANT_TIMEOUT_SECONDS = 60
MAX_QDRANT_TIMEOUT_SECONDS = 3600


@dataclass(frozen=True)
class Settings:
    tmdb_access_token: str | None
    tmdb_api_key: str | None
    tmdb_requests_per_second: float
    database_url: str
    work_dir: Path
    qdrant_timeout_seconds: int
    ml: MlSettings


def load_settings(env_file: Path | None) -> Settings:
    if env_file is not None and not env_file.is_file():
        raise ConfigError(f"Файл окружения не найден: {env_file}")
    load_dotenv(env_file or DEFAULT_ENV_FILE, override=False)

    access_token = _read_text("TMDB_ACCESS_TOKEN")
    api_key = _read_text("TMDB_API_KEY")
    if access_token is None and api_key is None:
        raise ConfigError("Не задан ключ TMDB: укажите TMDB_ACCESS_TOKEN или TMDB_API_KEY")
    database_url = _read_text("DATABASE_URL")
    if database_url is None:
        raise ConfigError(
            "Не задан DATABASE_URL, строка подключения к PostgreSQL вида postgresql://user:password@host:5432/dejaview"
        )
    return Settings(
        tmdb_access_token=access_token,
        tmdb_api_key=api_key,
        tmdb_requests_per_second=_read_number(
            "TMDB_REQUESTS_PER_SECOND", DEFAULT_REQUESTS_PER_SECOND, 0, MAX_REQUESTS_PER_SECOND
        ),
        database_url=database_url,
        work_dir=_read_work_dir(),
        qdrant_timeout_seconds=int(
            _read_number("INDEX_QDRANT_TIMEOUT_SECONDS", DEFAULT_QDRANT_TIMEOUT_SECONDS, 0, MAX_QDRANT_TIMEOUT_SECONDS)
        ),
        ml=_read_ml_settings(),
    )


def _read_text(name: str) -> str | None:
    return os.environ.get(name, "").strip() or None


def _read_number(name: str, default: float, above: float, maximum: float) -> float:
    raw_value = _read_text(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        value = None
    if value is None or not above < value <= maximum:
        raise ConfigError(
            f"{name} должно быть числом больше {above:g} и не больше {maximum:g}, получено: {raw_value!r}"
        )
    return value


def _read_work_dir() -> Path:
    work_dir = Path(_read_text("INDEX_WORK_DIR") or DEFAULT_WORK_DIR)
    if work_dir.exists() and not work_dir.is_dir():
        raise ConfigError(f"INDEX_WORK_DIR указывает на файл, а нужна папка: {work_dir}")
    return work_dir


def _read_ml_settings() -> MlSettings:
    try:
        return MlSettings(_env_file=None)
    except ValidationError as error:
        problems = "; ".join(
            f"ML_{'_'.join(str(part) for part in item['loc']).upper()}: {item['msg']}" for item in error.errors()
        )
        raise ConfigError(f"Неверные настройки ML: {problems}") from None
