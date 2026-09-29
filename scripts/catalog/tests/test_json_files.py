import json
from pathlib import Path

import pytest

from scripts.catalog.storage import json_files
from scripts.catalog.storage.json_files import CatalogFileError, read_json_list, write_json_atomically


def test_missing_file_is_empty_list(tmp_path):
    assert read_json_list(tmp_path / "нет.json") == []


def test_write_then_read(tmp_path):
    path = tmp_path / "папка" / "data.json"

    write_json_atomically(path, [{"title": "Фильм"}])

    assert read_json_list(path) == [{"title": "Фильм"}]
    assert not list(path.parent.glob("*.tmp"))


def test_failed_write_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "data.json"
    write_json_atomically(path, [1])

    def broken_dump(*args, **kwargs):
        raise OSError("диск заполнен")

    monkeypatch.setattr(json_files.json, "dump", broken_dump)
    with pytest.raises(OSError):
        write_json_atomically(path, [2])
    monkeypatch.undo()

    assert read_json_list(path) == [1]


def test_replace_retries_while_file_is_busy(tmp_path, monkeypatch):
    calls = []

    def flaky_replace(source: Path, target: Path) -> None:
        calls.append(source)
        if len(calls) < 3:
            raise PermissionError("занят")
        source.rename(target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(json_files.time, "sleep", lambda seconds: None)

    write_json_atomically(tmp_path / "data.json", [])

    assert len(calls) == 3
    assert json.loads((tmp_path / "data.json").read_text(encoding="utf-8")) == []


def test_replace_gives_up_after_all_attempts(tmp_path, monkeypatch):
    def always_busy(source: Path, target: Path) -> None:
        raise PermissionError("занят")

    monkeypatch.setattr(Path, "replace", always_busy)
    monkeypatch.setattr(json_files.time, "sleep", lambda seconds: None)

    with pytest.raises(PermissionError):
        write_json_atomically(tmp_path / "data.json", [])


@pytest.mark.parametrize(
    ("content", "message"),
    [("{не json", "повреждён JSON"), ('{"movie_id": 1}', "ожидался JSON-список")],
)
def test_broken_json(tmp_path, content, message):
    path = tmp_path / "data.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(CatalogFileError, match=message):
        read_json_list(path)


def test_non_utf8_file(tmp_path):
    path = tmp_path / "data.json"
    path.write_bytes('[{"title_ru": "Фильм"}]'.encode("cp1251"))

    with pytest.raises(CatalogFileError, match="UTF-8"):
        read_json_list(path)
