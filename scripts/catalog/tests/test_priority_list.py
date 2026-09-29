import pytest

from scripts.catalog.candidates.model import Priority
from scripts.catalog.candidates.priority_list import PriorityFileError, build_manual_candidates, load_priority_ids


def test_load_priority_ids_skips_comments_blanks_and_duplicates(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("# фильмы с доступным видео\n550\n\n  603  # Матрица\n550\n", encoding="utf-8")

    assert load_priority_ids(path) == [550, 603]


@pytest.mark.parametrize("content", ["550\nabc\n", "0\n", "-5\n", "5.5\n"])
def test_garbage_is_reported_with_line_number(tmp_path, content):
    path = tmp_path / "ids.txt"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(PriorityFileError, match=r"ids\.txt:\d"):
        load_priority_ids(path)


def test_missing_file(tmp_path):
    with pytest.raises(PriorityFileError):
        load_priority_ids(tmp_path / "нет.txt")


def test_wrong_encoding(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_bytes("550 # фильм".encode("cp1251"))

    with pytest.raises(PriorityFileError, match="UTF-8"):
        load_priority_ids(path)


def test_manual_candidates_have_top_priority():
    assert all(candidate.priority is Priority.MANUAL for candidate in build_manual_candidates([1, 2]))
