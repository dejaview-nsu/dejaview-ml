import os
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient, models

import dejaview_ml.storage as storage
from dejaview_ml.config import Settings

pytestmark = pytest.mark.skipif(not os.getenv("QDRANT_TEST_URL"), reason="real Qdrant URL required")


@pytest.mark.parametrize("prefer_grpc", [False, True])
async def test_real_collection_indexes_and_idempotent_bootstrap(monkeypatch, prefer_grpc):
    prefix = "test_" + uuid4().hex + "_"
    dimensions = {prefix + name: size for name, size in storage.DIMENSIONS.items()}
    monkeypatch.setattr(storage, "DIMENSIONS", dimensions)
    client = AsyncQdrantClient(url=os.environ["QDRANT_TEST_URL"], prefer_grpc=prefer_grpc)
    store = storage.VectorStore(Settings(_env_file=None), client)
    try:
        await store.ensure_collections()
        await store.ensure_collections()
        assert await store.healthy()
        for name in dimensions:
            info = await client.get_collection(name)
            assert info.payload_schema["movie_id"].data_type == models.PayloadSchemaType.INTEGER
        name = prefix + "image"
        vector = [1.0] + [0.0] * 511
        await client.upsert(
            name,
            [
                models.PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload={"movie_id": 603, "type": "image", "offset_sec": 125},
                )
            ],
            wait=True,
        )
        assert await store.search(name, [vector]) == [603]
        points = await client.query_points(
            name,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="movie_id", match=models.MatchValue(value=999)),
                ]
            ),
        )
        assert not points.points
    finally:
        for name in dimensions:
            if await client.collection_exists(name):
                await client.delete_collection(name)
        await client.close()
