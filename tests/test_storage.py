from unittest.mock import AsyncMock, Mock
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import QueryResponse

import dejaview_ml.storage as storage
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


@pytest.mark.parametrize(("key", "expected"), [(None, None), ("", None), ("secret", "secret")])
def test_qdrant_api_key_is_optional(monkeypatch, key, expected):
    client = Mock()
    monkeypatch.setattr(storage, "AsyncQdrantClient", client)
    VectorStore(Settings(_env_file=None, qdrant_api_key=key))
    assert client.call_args.kwargs["api_key"] == expected


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


MODALITIES = ("image", "video", "audio", "text")


def thresholds(settings):
    return [settings.threshold(modality) for modality in MODALITIES]


def test_score_threshold_defaults():
    assert thresholds(Settings(_env_file=None)) == [0.8614, 0.9743, 0.8297, None]


def test_common_threshold_replaces_defaults(monkeypatch):
    monkeypatch.setenv("ML_SCORE_THRESHOLD", "0.7")
    assert thresholds(Settings(_env_file=None)) == [0.7, 0.7, 0.7, 0.7]


def test_explicit_none_disables_threshold_despite_common(monkeypatch):
    monkeypatch.setenv("ML_SCORE_THRESHOLD", "0.7")
    monkeypatch.setenv("ML_SCORE_THRESHOLD_IMAGE", "0.5")
    monkeypatch.setenv("ML_SCORE_THRESHOLD_AUDIO", "none")
    assert thresholds(Settings(_env_file=None)) == [0.5, 0.7, None, 0.7]


def test_empty_threshold_variable_is_not_set(monkeypatch):
    # Так приходит переменная из compose вида ${ML_SCORE_THRESHOLD_IMAGE:-}, не заданная на сервере.
    monkeypatch.setenv("ML_SCORE_THRESHOLD", "")
    monkeypatch.setenv("ML_SCORE_THRESHOLD_IMAGE", "")
    assert thresholds(Settings(_env_file=None)) == [0.8614, 0.9743, 0.8297, None]


@pytest.mark.parametrize(
    ("modality", "expected"), [("image", 0.8614), ("audio", 0.8297), ("text", None)]
)
async def test_search_uses_threshold_of_its_modality(modality, expected):
    client = AsyncMock()
    client.query_points.return_value = QueryResponse(points=[])
    store = VectorStore(Settings(_env_file=None), client)
    assert await store.search(modality, [[1.0] * DIMENSIONS[modality]]) == []
    assert client.query_points.call_args.kwargs["score_threshold"] == expected


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
async def test_points_below_threshold_give_empty_answer(tmp_path):
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    stored = [1.0] + [0.0] * 511
    await VectorStore(Settings(_env_file=None), client).ensure_collections()
    point = models.PointStruct(
        id=point_id("image", 603, 0), vector=stored, payload=point_payload("image", 603, 0)
    )
    await client.upsert("image", [point])
    query = [0.8, 0.6] + [0.0] * 510  # похожесть 0.8, ниже порога image
    assert await VectorStore(Settings(_env_file=None), client).search("image", [query]) == []
    disabled = Settings(_env_file=None, score_threshold_image=None)
    assert await VectorStore(disabled, client).search("image", [query]) == [603]
    await client.close()
