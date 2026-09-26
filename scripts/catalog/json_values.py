"""Безопасное чтение значений из JSON, которому нельзя доверять: ответов TMDB и файлов каталога."""

from typing import Any


def as_int(value: Any) -> int | None:
    # bool - подкласс int, но True не должен превращаться в movie_id 1.
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def as_positive_int(value: Any) -> int | None:
    number = as_int(value)
    return number if number is not None and number > 0 else None


def clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
