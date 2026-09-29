from pathlib import Path

import pytest

from scripts.catalog.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POSTER_SIZE,
    DEFAULT_TARGET_SIZE,
    ConfigError,
    load_settings,
)

ALL_VARIABLES = (
    "TMDB_ACCESS_TOKEN",
    "TMDB_API_KEY",
    "CATALOG_OUTPUT_DIR",
    "CATALOG_TARGET_SIZE",
    "CATALOG_MIN_VOTE_COUNT",
    "CATALOG_YEAR_FROM",
    "TMDB_REQUESTS_PER_SECOND",
    "TMDB_POSTER_SIZE",
    "CATALOG_PRIORITY_IDS_FILE",
)


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    for name in ALL_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text("TMDB_ACCESS_TOKEN=token\n", encoding="utf-8")
    return path


def test_defaults(env_file):
    settings = load_settings(env_file, 2026)

    assert settings.tmdb_access_token == "token"
    assert settings.tmdb_api_key is None
    assert settings.output_dir == DEFAULT_OUTPUT_DIR
    assert settings.target_size == DEFAULT_TARGET_SIZE
    assert settings.poster_size == DEFAULT_POSTER_SIZE
    assert settings.priority_ids_file is None


def test_environment_overrides_file(env_file, monkeypatch):
    monkeypatch.setenv("CATALOG_TARGET_SIZE", "600")
    monkeypatch.setenv("CATALOG_PRIORITY_IDS_FILE", "ids.txt")
    monkeypatch.setenv("CATALOG_OUTPUT_DIR", "catalog_out")

    settings = load_settings(env_file, 2026)

    assert settings.target_size == 600
    assert settings.priority_ids_file == Path("ids.txt")
    assert settings.output_dir == Path("catalog_out")


def test_missing_tmdb_key(env_file):
    env_file.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="TMDB"):
        load_settings(env_file, 2026)


def test_tmdb_key_not_required_for_report(env_file):
    env_file.write_text("", encoding="utf-8")
    assert load_settings(env_file, 2026, require_tmdb_key=False).tmdb_access_token is None


def test_output_dir_must_not_be_a_file(env_file, monkeypatch):
    monkeypatch.setenv("CATALOG_OUTPUT_DIR", str(env_file))
    with pytest.raises(ConfigError, match="CATALOG_OUTPUT_DIR"):
        load_settings(env_file, 2026)


def test_missing_env_file(tmp_path):
    with pytest.raises(ConfigError, match="не найден"):
        load_settings(tmp_path / "нет.env", 2026)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CATALOG_TARGET_SIZE", "много"),
        ("CATALOG_TARGET_SIZE", "0"),
        ("CATALOG_TARGET_SIZE", "10001"),
        ("CATALOG_YEAR_FROM", "2030"),
        ("CATALOG_YEAR_FROM", "1800"),
        ("CATALOG_MIN_VOTE_COUNT", "-1"),
        ("TMDB_REQUESTS_PER_SECOND", "0"),
        ("TMDB_REQUESTS_PER_SECOND", "100"),
        ("TMDB_REQUESTS_PER_SECOND", "быстро"),
        ("TMDB_POSTER_SIZE", "huge"),
    ],
)
def test_invalid_values(env_file, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError, match=name):
        load_settings(env_file, 2026)
