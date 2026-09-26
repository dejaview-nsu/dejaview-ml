import json
import logging
import os
import time
from pathlib import Path
from typing import Any

REPLACE_ATTEMPTS = 5
REPLACE_RETRY_SECONDS = 0.2

logger = logging.getLogger(__name__)


class CatalogFileError(Exception):
    pass


def read_json_list(path: Path) -> list[Any]:
    """Отсутствующий файл - это пустой список: каталог ещё не собирали."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as error:
        raise CatalogFileError(f"Не удалось прочитать {path}: {error.strerror or error}") from None
    except UnicodeDecodeError:
        raise CatalogFileError(f"{path}: файл должен быть в кодировке UTF-8") from None

    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise CatalogFileError(
            f"{path}: повреждён JSON (строка {error.lineno}, позиция {error.colno}). Исправьте или удалите файл"
        ) from None
    if not isinstance(data, list):
        raise CatalogFileError(f"{path}: ожидался JSON-список")
    return data


def write_json_atomically(path: Path, data: Any) -> None:
    """Пишет во временный файл и подменяет им целевой: обрыв посреди записи оставляет прежнюю версию целой."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    _replace_with_retries(temp_path, path)


def _replace_with_retries(source: Path, target: Path) -> None:
    """В Windows подмена падает, пока целевой файл открыт другой программой."""
    for attempt in range(1, REPLACE_ATTEMPTS + 1):
        try:
            source.replace(target)
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS:
                raise
            logger.warning("%s занят другой программой, повтор записи", target)
            time.sleep(REPLACE_RETRY_SECONDS)
        else:
            return
