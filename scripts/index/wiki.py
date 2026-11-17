"""Сюжет фильма из Википедии для точек text с source=plot.

Статья находится по id TMDB через Wikidata (свойство P4947 «TMDB movie ID»): сначала русская, при её
отсутствии или без раздела сюжета - английская (e5 многоязычная). Из статьи берётся только раздел
сюжета, без подразделов-заголовков. Ответы кешируются в рабочем каталоге: повторный прогон не ходит в
Википедию, пока не передан --refresh.
"""

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any

import requests

SPARQL_URL = "https://query.wikidata.org/sparql"
API_URL = "https://{language}.wikipedia.org/w/api.php"
LANGUAGES = ("ru", "en")
PLOT_HEADINGS = {
    "ru": ("сюжет", "содержание", "сюжет фильма", "краткое содержание"),
    "en": ("plot", "synopsis", "plot summary", "story"),
}
USER_AGENT = "DejaViewIndexer/1.0 (https://github.com/dejaview-nsu/dejaview-ml) python-requests"
SPARQL_BATCH = 100
REQUEST_TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 4
PAUSE_SECONDS = 0.2

CHUNK_WORDS = 150
MAX_CHUNKS = 20
MIN_PLOT_WORDS = 40
SECTION_LEVEL = 2
RETRY_STATUSES = (HTTPStatus.TOO_MANY_REQUESTS, 500, 502, 503, 504)

logger = logging.getLogger(__name__)


class WikiError(RuntimeError):
    pass


@dataclass(frozen=True)
class Plot:
    language: str
    title: str
    text: str


def plot_section(extract: str, language: str) -> str | None:
    """Раздел сюжета из plain-text выдержки статьи (заголовки вида `== Сюжет ==`)."""
    headings = PLOT_HEADINGS[language]
    parts = re.split(r"^(={2,})\s*(.*?)\s*\1\s*$", extract, flags=re.MULTILINE)
    # parts: текст до первого заголовка, затем тройки (уровень, заголовок, текст).
    for index in range(1, len(parts) - 2, 3):
        level, heading = len(parts[index]), parts[index + 1].strip().lower()
        if level != SECTION_LEVEL or heading not in headings:
            continue
        body = [parts[index + 2]]
        for sub in range(index + 3, len(parts) - 2, 3):
            if len(parts[sub]) <= SECTION_LEVEL:
                break
            body.append(parts[sub + 2])  # подраздел сюжета: его заголовок не нужен
        text = "\n".join(line.strip() for line in "\n".join(body).splitlines() if line.strip())
        return text if len(text.split()) >= MIN_PLOT_WORDS else None
    return None


def plot_chunks(text: str, words: int = CHUNK_WORDS, limit: int = MAX_CHUNKS) -> list[str]:
    """Куски сюжета примерно по `words` слов по границам абзацев (длинный абзац - по предложениям),
    не больше `limit`: e5 читает до 512 токенов, а один кусок - одна точка."""
    pieces: list[str] = []
    for paragraph in text.splitlines():
        if len(paragraph.split()) <= words:
            pieces.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?…])\s+", paragraph)
        current: list[str] = []
        for sentence in sentences:
            if current and len(" ".join([*current, sentence]).split()) > words:
                pieces.append(" ".join(current))
                current = []
            current.append(sentence)
        if current:
            pieces.append(" ".join(current))
    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(chunks[-1].split()) + len(piece.split()) <= words:
            chunks[-1] += " " + piece
        else:
            chunks.append(piece)
    return chunks[:limit]


class Wikipedia:
    def __init__(self, cache_dir: Path, session: requests.Session | None = None, *, refresh: bool = False):
        self._cache_dir = cache_dir
        self._session = session or requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT
        self._refresh = refresh
        self._articles: dict[int, dict[str, str]] = {}

    def plots(self, movie_ids: list[int]) -> dict[int, Plot | None]:
        """Сюжет каждого фильма или None, если статьи или раздела сюжета нет."""
        result: dict[int, Plot | None] = {}
        missing = []
        for movie_id in movie_ids:
            cached = None if self._refresh else self._read_cache(movie_id)
            if cached is None:
                missing.append(movie_id)
            else:
                result[movie_id] = cached or None
        if missing:
            self._find_articles(missing)
        for movie_id in missing:
            plot = self._fetch_plot(self._articles.get(movie_id, {}))
            self._write_cache(movie_id, plot)
            result[movie_id] = plot
        return result

    def _find_articles(self, movie_ids: list[int]) -> None:
        for start in range(0, len(movie_ids), SPARQL_BATCH):
            batch = movie_ids[start : start + SPARQL_BATCH]
            values = " ".join(f'"{movie_id}"' for movie_id in batch)
            query = f"""
                SELECT ?tmdb ?lang ?title WHERE {{
                  VALUES ?tmdb {{ {values} }}
                  ?item wdt:P4947 ?tmdb .
                  ?article schema:about ?item ; schema:inLanguage ?lang ; schema:name ?title ;
                           schema:isPartOf ?wiki .
                  VALUES (?lang ?wiki) {{
                    ("ru" <https://ru.wikipedia.org/>) ("en" <https://en.wikipedia.org/>)
                  }}
                }}"""
            data = self._get(SPARQL_URL, {"query": query, "format": "json"})
            for row in data["results"]["bindings"]:
                movie_id = int(row["tmdb"]["value"])
                self._articles.setdefault(movie_id, {})[row["lang"]["value"]] = row["title"]["value"]

    def _fetch_plot(self, titles: dict[str, str]) -> Plot | None:
        for language in LANGUAGES:
            title = titles.get(language)
            if title is None:
                continue
            data = self._get(
                API_URL.format(language=language),
                {
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "exsectionformat": "wiki",
                    "redirects": 1,
                    "titles": title,
                    "format": "json",
                    "formatversion": 2,
                },
            )
            pages = data.get("query", {}).get("pages", [])
            extract = pages[0].get("extract", "") if pages else ""
            text = plot_section(extract, language)
            if text:
                return Plot(language, title, text)
        return None

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            time.sleep(PAUSE_SECONDS)
            try:
                response = self._session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            except requests.RequestException as error:
                problem = str(error)
            else:
                if response.status_code == HTTPStatus.OK:
                    data = response.json()
                    if "error" in data or "errors" in data:
                        raise WikiError(f"Ошибка API Википедии ({url}): {data.get('error', data.get('errors'))}")
                    return data
                problem = f"HTTP {response.status_code}"
                if response.status_code not in RETRY_STATUSES:
                    break
            if attempt < MAX_ATTEMPTS:
                time.sleep(2**attempt)
        raise WikiError(f"Википедия не ответила ({url}): {problem}")

    def _cache_path(self, movie_id: int) -> Path:
        return self._cache_dir / f"{movie_id}.json"

    def _read_cache(self, movie_id: int) -> Plot | bool | None:
        """Plot, False (сюжета нет - проверено) или None (не кешировано)."""
        try:
            data = json.loads(self._cache_path(movie_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return Plot(**data) if data else False

    def _write_cache(self, movie_id: int, plot: Plot | None) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        payload = asdict(plot) if plot else {}
        self._cache_path(movie_id).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
