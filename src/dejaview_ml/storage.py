from collections import defaultdict
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import AsyncQdrantClient, models

from dejaview_ml.config import DIMENSIONS, Settings
from dejaview_ml.errors import MlError


class SchemaMismatch(RuntimeError):
    pass


def point_payload(modality: str, movie_id: int, offset_sec: int | None = None) -> dict:
    if modality not in DIMENSIONS or type(movie_id) is not int:
        raise ValueError("invalid modality or movie_id")
    if not -(2**63) <= movie_id < 2**63:
        raise ValueError("movie_id must fit int64")
    result = {"movie_id": movie_id, "type": modality}
    if modality == "text":
        if offset_sec is not None:
            raise ValueError("text must not have offset_sec")
    else:
        if type(offset_sec) is not int or offset_sec < 0:
            raise ValueError("offset_sec must be a nonnegative integer")
        result["offset_sec"] = offset_sec
    return result


def point_id(modality: str, movie_id: int, offset_sec: int | None = None) -> str:
    point_payload(modality, movie_id, offset_sec)
    suffix = f":{offset_sec}" if modality != "text" else ""
    return str(uuid5(NAMESPACE_URL, f"dejaview:{modality}:{movie_id}{suffix}"))


def aggregate(points: list[models.ScoredPoint], limit: int = 1) -> list[int]:
    # A video query can hit the same indexed fragment in several segments.
    # Keep its best score once, then sum distinct fragment scores per film.
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
        totals[movie_id] += point.score
    return sorted(totals, key=lambda movie_id: (-totals[movie_id], movie_id))[:limit]


class VectorStore:
    def __init__(self, settings: Settings, client: AsyncQdrantClient | None = None):
        self.settings = settings
        self.client = client or AsyncQdrantClient(
            url=settings.qdrant_url,
            grpc_port=settings.qdrant_grpc_port,
            prefer_grpc=settings.qdrant_prefer_grpc,
            timeout=settings.qdrant_timeout_seconds,
        )

    async def ensure_collections(self) -> None:
        for name, size in DIMENSIONS.items():
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
            return set(DIMENSIONS) <= names
        except Exception:
            return False

    async def search(self, modality: str, vectors: list[list[float]]) -> list[int]:
        # Global top-K across query segments; duplicate point hits count only once.
        points: dict[str, models.ScoredPoint] = {}
        for vector in vectors:
            try:
                result = await self.client.query_points(
                    collection_name=modality,
                    query=vector,
                    limit=self.settings.top_k,
                    score_threshold=self.settings.score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                raise MlError(503, "QDRANT_UNAVAILABLE", "Qdrant search failed") from exc
            for point in result.points:
                key = str(point.id)
                if key not in points or point.score > points[key].score:
                    points[key] = point
        nearest = sorted(points.values(), key=lambda point: (-point.score, str(point.id)))
        return aggregate(nearest[: self.settings.top_k], limit=10 if modality == "text" else 1)

    async def close(self) -> None:
        await self.client.close()
