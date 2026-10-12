import os
from pathlib import Path

import pytest

from scripts.index.config import DEFAULT_QDRANT_TIMEOUT_SECONDS, DEFAULT_WORK_DIR, load_settings
from scripts.index.errors import ConfigError

VARIABLES = (
    "TMDB_ACCESS_TOKEN",
    "TMDB_API_KEY",
    "TMDB_REQUESTS_PER_SECOND",
    "DATABASE_URL",
    "INDEX_WORK_DIR",
    "INDEX_QDRANT_TIMEOUT_SECONDS",
    "ML_QDRANT_URL",
    "ML_QDRANT_API_KEY",
    "ML_DEVICE",
)


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("TMDB_ACCESS_TOKEN", "token")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/dejaview")
    return env_file


def test_defaults(env):
    settings = load_settings(env)
    assert settings.tmdb_access_token == "token"
    assert settings.tmdb_requests_per_second == 20
    assert settings.work_dir == DEFAULT_WORK_DIR
    assert settings.qdrant_timeout_seconds == DEFAULT_QDRANT_TIMEOUT_SECONDS
    assert settings.ml.qdrant_url == "http://localhost:6333"
    assert settings.ml.qdrant_api_key is None


def test_values_from_env_file(env, monkeypatch):
    monkeypatch.delenv("DATABASE_URL")
    env.write_text(
        "DATABASE_URL=postgresql://tunnel/dejaview\nINDEX_WORK_DIR=/data/index\n"
        "ML_QDRANT_URL=http://localhost:16333\nML_QDRANT_API_KEY=secret\nINDEX_QDRANT_TIMEOUT_SECONDS=120\n",
        encoding="utf-8",
    )
    settings = load_settings(env)
    assert settings.database_url == "postgresql://tunnel/dejaview"
    assert settings.work_dir == Path("/data/index")
    assert settings.ml.qdrant_url == "http://localhost:16333"
    assert settings.qdrant_timeout_seconds == 120
    assert settings.ml.qdrant_api_key.get_secret_value() == "secret"


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("TMDB_ACCESS_TOKEN", "", "Не задан ключ TMDB"),
        ("DATABASE_URL", " ", "Не задан DATABASE_URL"),
        ("TMDB_REQUESTS_PER_SECOND", "0", "TMDB_REQUESTS_PER_SECOND должно быть"),
        ("INDEX_QDRANT_TIMEOUT_SECONDS", "soon", "INDEX_QDRANT_TIMEOUT_SECONDS должно быть"),
        ("ML_DEVICE", "tpu", "ML_DEVICE"),
    ],
)
def test_invalid_settings(env, monkeypatch, variable, value, message):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ConfigError, match=message):
        load_settings(env)


def test_work_dir_must_not_be_a_file(env, monkeypatch, tmp_path):
    monkeypatch.setenv("INDEX_WORK_DIR", str(env))
    with pytest.raises(ConfigError, match="INDEX_WORK_DIR указывает на файл"):
        load_settings(env)


def test_missing_env_file(env, tmp_path):
    with pytest.raises(ConfigError, match="Файл окружения не найден"):
        load_settings(tmp_path / "missing.env")
