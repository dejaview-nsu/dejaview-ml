from unittest.mock import AsyncMock
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import QueryResponse

from dejaview_ml.config import DIMENSIONS, Settings
from dejaview_ml.errors import MlError
from dejaview_ml.storage import SchemaMismatch, VectorStore, aggregate, point_id, point_payload


def test_metadata_and_stable_point_ids():
    assert point_payload("image", 603, 125) == {"movie_id": 603, "type": "image", "offset_sec": 125}
    assert point_payload("text", 603) == {"movie_id": 603, "type": "text"}
    assert point_id("image", 603, 125) == str(uuid5(NAMESPACE_URL, "dejaview:image:603:125"))
    for args in [("text", 603, 0), ("image", 603, None), ("image", 603, -1), ("video", True, 0)]:
        with pytest.raises(ValueError):
            point_payload(*args)


def test_aggregation_sums_distinct_fragments_and_breaks_ties():
    def hit(id, movie, score):
        return models.ScoredPoint(id=id, version=1, score=score, payload={"movie_id": movie})

    points = [hit(1, 1, 0.8), hit(2, 2, 0.6), hit(3, 2, 0.5), hit(1, 1, 0.7)]
    assert aggregate(points) == [2]
    assert aggregate([hit(4, 3, 1), hit(5, 1, 1)], limit=2) == [1, 3]


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
async def test_collections_search_filter_and_restart(tmp_path):
    path = str(tmp_path / "qdrant")
    client = AsyncQdrantClient(path=path)
    store = VectorStore(Settings(_env_file=None), client)
    await store.ensure_collections()
    await store.ensure_collections()
    assert await store.healthy()
    for name, size in DIMENSIONS.items():
        info = await client.get_collection(name)
        assert info.config.params.vectors.size == size
        assert info.config.params.vectors.distance == models.Distance.COSINE
    vector = [1.0] + [0.0] * 511
    point = models.PointStruct(
        id=point_id("image", 603, 0), vector=vector, payload=point_payload("image", 603, 0)
    )
    await client.upsert("image", [point])
    await client.upsert("image", [point])
    assert (await client.count("image")).count == 1
    assert await store.search("image", [vector]) == [603]
    result = await client.query_points(
        "image",
        query=vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(key="movie_id", match=models.MatchValue(value=999)),
            ]
        ),
    )
    assert not result.points
    await store.close()
    reopened = VectorStore(Settings(_env_file=None), AsyncQdrantClient(path=path))
    assert await reopened.search("image", [vector]) == [603]
    await reopened.close()


@pytest.mark.parametrize(
    ("size", "distance"), [(100, models.Distance.COSINE), (512, models.Distance.DOT)]
)
@pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
async def test_wrong_schema_preserved(size, distance):
    client = AsyncQdrantClient(":memory:")
    await client.create_collection(
        "image", vectors_config=models.VectorParams(size=size, distance=distance)
    )
    store = VectorStore(Settings(_env_file=None), client)
    with pytest.raises(SchemaMismatch):
        await store.ensure_collections()
    assert (await client.get_collection("image")).config.params.vectors.size == size
    await store.close()


async def test_query_failure_maps_to_unavailable():
    client = AsyncMock()
    client.query_points.side_effect = TimeoutError
    store = VectorStore(Settings(_env_file=None), client)
    with pytest.raises(MlError, match="Qdrant search failed") as error:
        await store.search("image", [[0.0] * 512])
    assert error.value.code == "QDRANT_UNAVAILABLE"


async def test_video_uses_global_top_k_not_k_per_segment():
    client = AsyncMock()

    def hits(points):
        return QueryResponse(
            points=[
                models.ScoredPoint(
                    id=id,
                    version=1,
                    score=score,
                    payload={"movie_id": movie},
                )
                for id, movie, score in points
            ]
        )

    client.query_points.side_effect = [
        hits([(1, 10, 0.9), (2, 20, 0.8)]),
        hits([(1, 10, 0.95), (3, 20, 0.7)]),
    ]
    store = VectorStore(Settings(_env_file=None, top_k=2), client)
    assert await store.search("video", [[1.0] * 512, [1.0] * 512]) == [10]
