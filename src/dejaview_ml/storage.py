import re
from collections import defaultdict
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import AsyncQdrantClient, models

from dejaview_ml.config import COLLECTIONS, DIMENSIONS, SPEECH_COLLECTION, Settings
from dejaview_ml.embeddings import AudioQuery
from dejaview_ml.errors import MlError

TEXT_SOURCES = ("overview", "metadata")
METADATA_ACTORS = 10


class SchemaMismatch(RuntimeError):
    pass


def point_payload(
    modality: str, movie_id: int, offset_sec: int | None = None, source: str | None = None
) -> dict:
    if modality not in COLLECTIONS or type(movie_id) is not int:
        raise ValueError("invalid modality or movie_id")
    if not -(2**63) <= movie_id < 2**63:
        raise ValueError("movie_id must fit int64")
    result = {"movie_id": movie_id, "type": modality}
    if modality == "text":
        if offset_sec is not None:
            raise ValueError("text must not have offset_sec")
        source = source or "overview"
        if source not in TEXT_SOURCES:
            raise ValueError(f"text source must be one of {TEXT_SOURCES}")
        result["source"] = source
    else:
        if type(offset_sec) is not int or offset_sec < 0:
            raise ValueError("offset_sec must be a nonnegative integer")
        if source is not None:
            raise ValueError("only text has source")
        result["offset_sec"] = offset_sec
    return result


def point_id(
    modality: str, movie_id: int, offset_sec: int | None = None, source: str | None = None
) -> str:
    payload = point_payload(modality, movie_id, offset_sec, source)
    if modality != "text":
        suffix = f":{offset_sec}"
    else:
        suffix = ":metadata" if payload["source"] == "metadata" else ""
    return str(uuid5(NAMESPACE_URL, f"dejaview:{modality}:{movie_id}{suffix}"))


def metadata_passage(movie: Any) -> str | None:
    """Строка точки metadata без префикса: `passage: ` добавляет Encoders.encode_text."""
    parts = []
    if movie.genres:
        parts.append("Жанр: " + ", ".join(g.name for g in movie.genres) + ".")
    if movie.release_date:
        year = movie.release_date.year
        parts.append(f"Год: {year}, {year // 10 * 10}-е.")
    if movie.countries:
        parts.append("Страна: " + ", ".join(c.name for c in movie.countries) + ".")
    directors = [member.name for member in movie.crew if member.role == "director"]
    if directors:
        parts.append("Режиссёр: " + ", ".join(directors) + ".")
    actors = [
        f"{a.name} ({a.character})" if a.character else a.name
        for a in movie.actors[:METADATA_ACTORS]
    ]
    if actors:
        parts.append("В ролях: " + "; ".join(actors) + ".")
    return " ".join(parts) or None


def aggregate(
    points: list[models.ScoredPoint], limit: int = 1, combine: Literal["sum", "max"] = "sum"
) -> list[int]:
    unique: dict[str, models.ScoredPoint] = {}
    for point in points:
        key = str(point.id)
        if key not in unique or point.score > unique[key].score:
            unique[key] = point
    totals: dict[int, float] = defaultdict(float)
    for point in unique.values():
        movie_id = (point.payload or {}).get("movie_id")
        if type(movie_id) is not int:
            raise RuntimeError("Qdrant point has invalid movie_id")
        if combine == "max":
            totals[movie_id] = max(totals.get(movie_id, point.score), point.score)
        else:
            totals[movie_id] += point.score
    return sorted(totals, key=lambda movie_id: (-totals[movie_id], movie_id))[:limit]


# Matches of consecutive query windows must land at consecutive moments of the same film.
# Tolerance covers the index grid: a query window rarely starts exactly on an indexed offset.
TEMPORAL_TOLERANCE_SEC = {"video": 10.0, "audio": 10.0}
TEMPORAL_CANDIDATES = 100

# Audio search scores every film twice: by CLAP (how well the query windows line up in time with
# the film's sound) and by its dialogue (the transcript among the film's transcribed lines) -
# and adds them.
CLAP_WEIGHT = 2.0
# A text query may be a quote: its dialogue line counts (shared meaningful words, as for audio)
# only if it is at least as close to the query as the film's best description. Tested on 291
# queries over 117 films (26 with dialogue): quotes from subtitles in another translation than
# the dub are found first in 51/72 instead of 4/72, while descriptions, directors and taglines
# keep 102 of 179 first answers, as without dialogue. Without the closeness condition, generic
# words ("фильм", "работа", "человек") pulled 44 of them to random lines.
TEXT_LIMIT = 10
SPEECH_CANDIDATES = 50

SPEECH_WORD_WEIGHT = 0.5
SPEECH_MIN_SHARED_WORDS = 0.34
SPEECH_MIN_WORDS = 2
STEM_LENGTH = 5
STOP_WORDS = frozenset(
    """что это как так вот ещё еще уже там тут где кто чем чего тебя тебе меня мне его ему её она
    они оно мы вы ты он нас вас них ним нам вам был была было были быть есть нет да не ни но или
    если когда тоже только даже ведь ваш ваша мой моя моё мое твой твоя наш наша свой этот эта эти
    тот того той тем всё все всех весь вся всем очень может можно надо нужно будет буду будем
    хорошо ладно ну ага пока хватит знаю знаешь думаю хочу хочешь давай почему зачем сейчас здесь
    теперь потом тогда просто всегда никогда ничего сам сама себя себе него неё нее этого этом
    этой""".replace("ё", "е").split()
)


def temporal_scores(
    windows: list[tuple[float, list[models.ScoredPoint]]], tolerance: float
) -> dict[int, float]:
    """windows: (query offset, hits) per query window.

    A film scores the best sum of one hit per window whose film offset minus query offset
    agrees within tolerance. Scattered look-alike fragments add up only if they line up in time.
    """
    hits: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for window, (query_offset, points) in enumerate(windows):
        for point in points:
            payload = point.payload or {}
            movie_id, offset = payload.get("movie_id"), payload.get("offset_sec")
            if type(movie_id) is not int or type(offset) is not int:
                raise RuntimeError("Qdrant point has invalid movie_id or offset_sec")
            hits[movie_id].append((window, offset - query_offset, point.score))
    totals = {}
    for movie_id, film_hits in hits.items():
        best = 0.0
        for _, anchor, _ in film_hits:
            per_window: dict[int, float] = {}
            for window, shift, score in film_hits:
                if abs(shift - anchor) <= tolerance and score > per_window.get(window, -1.0):
                    per_window[window] = score
            best = max(best, sum(per_window.values()))
        totals[movie_id] = best
    return totals


def temporal_aggregate(
    windows: list[tuple[float, list[models.ScoredPoint]]],
    tolerance: float,
    limit: int = 1,
) -> list[int]:
    totals = temporal_scores(windows, tolerance)
    return sorted(totals, key=lambda movie_id: (-totals[movie_id], movie_id))[:limit]


def word_stems(text: str) -> set[str]:
    """Meaningful words, crudely stemmed: Russian endings change with case and tense, the first
    letters rarely do. Common words and words shorter than 3 letters are dropped. "ё" is read as
    "е": Whisper and people write it either way."""
    return {
        word[:STEM_LENGTH]
        for word in re.findall(r"\w+", text.lower().replace("ё", "е"))
        if len(word) >= 3 and word not in STOP_WORDS
    }


def shared_words(transcript: str, line: str) -> float:
    query = word_stems(transcript)
    return len(query & word_stems(line)) / len(query) if query else 0.0


def speech_scores(hits: list[models.ScoredPoint], transcript: str) -> dict[int, float]:
    """Best dialogue match per film: text similarity plus shared words. Lines sharing too few
    meaningful words with the transcript, and transcripts with too few of them, give nothing."""
    if len(word_stems(transcript)) < SPEECH_MIN_WORDS:
        return {}
    totals: dict[int, float] = {}
    for hit in hits:
        payload = hit.payload or {}
        movie_id = payload.get("movie_id")
        if type(movie_id) is not int:
            raise RuntimeError("Qdrant point has invalid movie_id")
        words = shared_words(transcript, str(payload.get("text", "")))
        if words < SPEECH_MIN_SHARED_WORDS:
            continue
        score = hit.score + SPEECH_WORD_WEIGHT * words
        totals[movie_id] = max(totals.get(movie_id, score), score)
    return totals


def text_ranking(text: dict[int, float], speech: dict[int, float]) -> list[int]:
    """text: per-film best similarity to its description or metadata; speech: dialogue score of
    films whose line matched the query as a quote. A quoted film the description candidates
    missed gets the weakest description score among them."""
    floor = min(text.values(), default=0.0)
    totals = {
        movie_id: text.get(movie_id, floor) + speech.get(movie_id, 0.0)
        for movie_id in text.keys() | speech.keys()
    }
    return sorted(totals, key=lambda movie_id: (-totals[movie_id], movie_id))


def audio_ranking(clap: dict[int, float], speech: dict[int, float]) -> list[int]:
    """clap: per-film CLAP score averaged over query windows; speech: dialogue score. A film the
    CLAP candidates missed gets the weakest CLAP score among them, not zero: a phone recording
    of speech often leaves the right film out of CLAP's nearest fragments altogether."""
    floor = min(clap.values(), default=0.0)
    totals = {
        movie_id: CLAP_WEIGHT * clap.get(movie_id, floor) + speech.get(movie_id, 0.0)
        for movie_id in clap.keys() | speech.keys()
    }
    return sorted(totals, key=lambda movie_id: (-totals[movie_id], movie_id))


class VectorStore:
    def __init__(self, settings: Settings, client: AsyncQdrantClient | None = None):
        self.settings = settings
        api_key = settings.qdrant_api_key and settings.qdrant_api_key.get_secret_value()
        self.client = client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=api_key or None,
            grpc_port=settings.qdrant_grpc_port,
            prefer_grpc=settings.qdrant_prefer_grpc,
            timeout=settings.qdrant_timeout_seconds,
        )

    async def ensure_collections(self) -> None:
        for name, size in COLLECTIONS.items():
            if not await self.client.collection_exists(name):
                await self.client.create_collection(
                    name,
                    vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
                )
            info = await self.client.get_collection(name)
            vectors = info.config.params.vectors
            if not isinstance(vectors, models.VectorParams) or (
                vectors.size != size or vectors.distance != models.Distance.COSINE
            ):
                raise SchemaMismatch(
                    f"{name}: expected {size} dimensions and Cosine; got {vectors}"
                )
            index = info.payload_schema.get("movie_id")
            if index is not None and index.data_type != models.PayloadSchemaType.INTEGER:
                raise SchemaMismatch(f"{name}: movie_id payload index must be integer")
            if index is None:
                await self.client.create_payload_index(
                    name,
                    "movie_id",
                    field_schema=models.PayloadSchemaType.INTEGER,
                    wait=True,
                )

    async def healthy(self) -> bool:
        try:
            names = {c.name for c in (await self.client.get_collections()).collections}
            return set(COLLECTIONS) <= names
        except Exception:
            return False

    async def search(
        self, modality: str, vectors: list[list[float]], offsets: list[float] | None = None
    ) -> list[int]:
        """offsets: start of each query window in seconds. With several windows of a timed
        modality, films are ranked by time-consistent matches instead of a plain sum."""
        if offsets is not None and len(vectors) > 1 and modality in TEMPORAL_TOLERANCE_SEC:
            if len(offsets) != len(vectors):
                raise ValueError("offsets must match vectors")
            limit = max(self.settings.top_k, TEMPORAL_CANDIDATES)
            windows = [
                (offset, await self._query(modality, vector, limit))
                for offset, vector in zip(offsets, vectors, strict=True)
            ]
            return temporal_aggregate(windows, TEMPORAL_TOLERANCE_SEC[modality])

        # Global top-K across query segments; duplicate point hits count only once.
        points: dict[str, models.ScoredPoint] = {}
        for vector in vectors:
            for point in await self._query(modality, vector, self.settings.top_k):
                key = str(point.id)
                if key not in points or point.score > points[key].score:
                    points[key] = point
        nearest = sorted(points.values(), key=lambda point: (-point.score, str(point.id)))
        if modality == "text":
            return aggregate(nearest[: self.settings.top_k], limit=10, combine="max")
        return aggregate(nearest[: self.settings.top_k], limit=1)

    async def search_text(self, vector: list[float], query: str) -> list[int]:
        """Films by description and metadata, with quoted dialogue raised to the top."""
        text: dict[int, float] = {}
        for point in await self._query("text", vector, self.settings.top_k):
            movie_id = (point.payload or {}).get("movie_id")
            if type(movie_id) is not int:
                raise RuntimeError("Qdrant point has invalid movie_id")
            text[movie_id] = max(text.get(movie_id, point.score), point.score)
        best = max(text.values(), default=-1.0)
        lines = await self._query(SPEECH_COLLECTION, vector, SPEECH_CANDIDATES)
        speech = speech_scores([line for line in lines if line.score >= best], query)
        return text_ranking(text, speech)[:TEXT_LIMIT]

    async def search_audio(self, query: AudioQuery) -> list[int]:
        """CLAP and dialogue scores added per film, see audio_ranking."""
        windows = [
            (offset, await self._query("audio", vector, TEMPORAL_CANDIDATES))
            for offset, vector in zip(query.offsets, query.vectors, strict=True)
        ]
        clap = {
            movie_id: score / len(windows)
            for movie_id, score in temporal_scores(windows, TEMPORAL_TOLERANCE_SEC["audio"]).items()
        }
        speech: dict[int, float] = {}
        if query.speech_vector is not None:
            hits = await self._query(SPEECH_COLLECTION, query.speech_vector, SPEECH_CANDIDATES)
            speech = speech_scores(hits, query.transcript)
        return audio_ranking(clap, speech)[:1]

    async def _query(
        self, modality: str, vector: list[float], limit: int
    ) -> list[models.ScoredPoint]:
        # Dialogue lines have no similarity threshold: speech_scores filters them by shared words.
        threshold = self.settings.threshold(modality) if modality in DIMENSIONS else None
        try:
            result = await self.client.query_points(
                collection_name=modality,
                query=vector,
                limit=limit,
                score_threshold=threshold,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise MlError(503, "QDRANT_UNAVAILABLE", "Qdrant search failed") from exc
        return result.points

    async def close(self) -> None:
        await self.client.close()
