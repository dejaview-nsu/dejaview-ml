from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import QdrantClient, models

from dejaview_ml.config import DIMENSIONS
from scripts.index.embedding import Fragment
from scripts.index.errors import IndexingError
from scripts.index.vectors import VectorWriter, describe_qdrant_error

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


@pytest.fixture
def client():
    client = QdrantClient(":memory:")
    for name, size in DIMENSIONS.items():
        client.create_collection(name, vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE))
    yield client
    client.close()


def fragments(image_count: int, value: float = 1.0) -> dict[str, list[Fragment]]:
    def vector(name: str) -> list[float]:
        return [value] + [0.0] * (DIMENSIONS[name] - 1)

    return {
        "text": [Fragment(None, vector("text"))],
        "image": [Fragment(5 * index, vector("image")) for index in range(image_count)],
        "video": [Fragment(0, vector("video"))],
        "audio": [Fragment(0, vector("audio")), Fragment(5, vector("audio"))],
    }


def movie_points(client: QdrantClient, collection: str, movie_id: int) -> list[models.Record]:
    selector = models.Filter(must=[models.FieldCondition(key="movie_id", match=models.MatchValue(value=movie_id))])
    points, _ = client.scroll(collection, scroll_filter=selector, limit=1000)
    return points


def test_points_follow_the_contract(client):
    counts = VectorWriter(client, "memory").replace_movie(603, fragments(3))
    assert counts == {"text": 1, "image": 3, "video": 1, "audio": 2}

    (text,) = movie_points(client, "text", 603)
    assert text.id == str(uuid5(NAMESPACE_URL, "dejaview:text:603"))
    assert text.payload == {"movie_id": 603, "type": "text"}

    image = {point.id: point.payload for point in movie_points(client, "image", 603)}
    assert image[str(uuid5(NAMESPACE_URL, "dejaview:image:603:10"))] == {
        "movie_id": 603,
        "type": "image",
        "offset_sec": 10,
    }


def test_repeated_run_does_not_duplicate_and_drops_stale_points(client):
    writer = VectorWriter(client, "memory")
    writer.replace_movie(603, fragments(5))
    writer.replace_movie(550, fragments(2))
    writer.replace_movie(603, fragments(3, value=0.5))

    assert sorted(point.payload["offset_sec"] for point in movie_points(client, "image", 603)) == [0, 5, 10]
    assert len(movie_points(client, "image", 550)) == 2
    assert {name: client.count(name).count for name in DIMENSIONS} == {"text": 2, "image": 5, "video": 2, "audio": 4}


def test_collections_are_checked_not_created():
    client = QdrantClient(":memory:")
    with pytest.raises(IndexingError, match=r"нет коллекции «text».*создаёт ML-сервис"):
        VectorWriter(client, "memory").check_collections()
    assert client.get_collections().collections == []


def test_collection_with_wrong_dimension_is_rejected(client):
    client.delete_collection("audio")
    client.create_collection("audio", vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE))
    with pytest.raises(IndexingError, match="«audio» не совпадает с моделью"):
        VectorWriter(client, "memory").check_collections()


def test_correct_collections_pass(client):
    VectorWriter(client, "memory").check_collections()


def test_qdrant_failure_becomes_readable_error():
    class BrokenClient:
        def collection_exists(self, name):
            raise ConnectionRefusedError("connection refused")

    with pytest.raises(IndexingError, match=r"Сбой Qdrant \(http://localhost:6333\).*SSH-туннель"):
        VectorWriter(BrokenClient(), "http://localhost:6333").check_collections()


def test_grpc_error_is_reported_in_one_line():
    class RpcError(Exception):
        def __str__(self):
            return "<_InactiveRpcError of RPC that terminated with:\n\tstatus = StatusCode.UNAVAILABLE\n>"

        def details(self):
            return "failed to connect to all addresses;\n last error: UNAVAILABLE"

    assert describe_qdrant_error(RpcError()) == "failed to connect to all addresses; last error: UNAVAILABLE"
    assert describe_qdrant_error(TimeoutError()) == "TimeoutError"
