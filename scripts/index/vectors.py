import logging
from collections.abc import Callable
from typing import TypeVar

from qdrant_client import QdrantClient, models

from dejaview_ml.config import DIMENSIONS
from dejaview_ml.storage import point_id, point_payload
from scripts.index.embedding import Fragment
from scripts.index.errors import IndexingError

UPSERT_BATCH_SIZE = 256

T = TypeVar("T")

logger = logging.getLogger(__name__)


class VectorWriter:
    def __init__(self, client: QdrantClient, url: str) -> None:
        self._client = client
        self._url = url

    def check_collections(self) -> None:
        for name, size in DIMENSIONS.items():
            if not self._call(f"проверка коллекции {name}", lambda name=name: self._client.collection_exists(name)):
                raise IndexingError(
                    f"В Qdrant ({self._url}) нет коллекции «{name}». Коллекции создаёт ML-сервис при старте: "
                    "запустите его или проверьте ML_QDRANT_URL"
                )
            info = self._call(f"проверка коллекции {name}", lambda name=name: self._client.get_collection(name))
            vectors = info.config.params.vectors
            if not isinstance(vectors, models.VectorParams) or (
                vectors.size != size or vectors.distance != models.Distance.COSINE
            ):
                raise IndexingError(
                    f"Коллекция «{name}» не совпадает с моделью: нужно {size} измерений и Cosine, в Qdrant {vectors}"
                )
            index = info.payload_schema.get("movie_id")
            if index is None:
                logger.warning(
                    "В коллекции «%s» нет индекса payload по movie_id, удаление по фильму будет медленным", name
                )
            elif index.data_type != models.PayloadSchemaType.INTEGER:
                raise IndexingError(
                    f"Индекс movie_id в коллекции «{name}» должен быть integer, в Qdrant {index.data_type}"
                )

    def replace_movie(self, movie_id: int, fragments: dict[str, list[Fragment]]) -> dict[str, int]:
        for name in DIMENSIONS:
            self._call(f"удаление старых точек из {name}", lambda name=name: self._delete_movie(name, movie_id))
        counts = {}
        for name in DIMENSIONS:
            points = [
                models.PointStruct(
                    id=point_id(name, movie_id, fragment.offset_sec),
                    vector=fragment.vector,
                    payload=point_payload(name, movie_id, fragment.offset_sec),
                )
                for fragment in fragments[name]
            ]
            for start in range(0, len(points), UPSERT_BATCH_SIZE):
                batch = points[start : start + UPSERT_BATCH_SIZE]
                self._call(
                    f"запись в {name}", lambda name=name, batch=batch: self._client.upsert(name, batch, wait=True)
                )
            counts[name] = len(points)
            logger.info("%s: записано %d точек", name, len(points))
        return counts

    def _delete_movie(self, collection: str, movie_id: int) -> None:
        self._client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="movie_id", match=models.MatchValue(value=movie_id))]
                )
            ),
            wait=True,
        )

    def _call(self, action: str, request: Callable[[], T]) -> T:
        try:
            return request()
        except Exception as error:
            raise IndexingError(
                f"Сбой Qdrant ({self._url}), {action}: {describe_qdrant_error(error)}. Проверьте ML_QDRANT_URL, "
                "ML_QDRANT_API_KEY и SSH-туннель (портов два: 6333 и 6334 для gRPC, иначе ML_QDRANT_PREFER_GRPC=false)"
            ) from error

    def close(self) -> None:
        self._client.close()


def describe_qdrant_error(error: Exception) -> str:
    details = getattr(error, "details", None)
    text = details() if callable(details) else str(error)
    return " ".join(str(text or type(error).__name__).split())


def connect(url: str, grpc_port: int, *, api_key: str | None, prefer_grpc: bool, timeout_seconds: int) -> VectorWriter:
    client = QdrantClient(
        url=url, api_key=api_key, grpc_port=grpc_port, prefer_grpc=prefer_grpc, timeout=timeout_seconds
    )
    return VectorWriter(client, url)
