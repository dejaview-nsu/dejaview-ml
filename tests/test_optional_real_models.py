import os

import httpx
import numpy as np
import pytest
from qdrant_client import AsyncQdrantClient, models
from test_text_audio import wav_bytes

from dejaview_ml.app import create_app
from dejaview_ml.config import Settings
from dejaview_ml.embeddings import Encoders
from dejaview_ml.storage import VectorStore, point_id, point_payload

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_OPTIONAL_MODEL_TESTS") != "1",
        reason="set RUN_OPTIONAL_MODEL_TESTS=1",
    ),
]


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
async def test_real_e5_clap_search_and_offline_reload():
    settings = Settings(
        _env_file=None,
        device="cpu",
        text_enabled=True,
        audio_enabled=True,
        image_enabled=False,
        video_enabled=False,
    )
    encoders = Encoders(settings)
    for modality in ("text", "audio"):
        encoders.load(modality)
    query = "девушка в красном платье"
    sound = wav_bytes()
    vectors = {
        "text": encoders.encode_text(query)[0],
        "audio": encoders.encode("audio", sound, "audio/wav")[0],
    }
    for modality, dimension in (("text", 768), ("audio", 512)):
        assert len(vectors[modality]) == dimension
        assert np.isfinite(vectors[modality]).all()
        assert np.linalg.norm(vectors[modality]) == pytest.approx(1, abs=1e-5)
    store = VectorStore(settings, AsyncQdrantClient(":memory:"))
    app = create_app(settings, encoders, store)
    async with app.router.lifespan_context(app):
        for modality in ("text", "audio"):
            offset = None if modality == "text" else 0
            vector = (
                encoders.encode_text(query, passage=True)[0]
                if modality == "text"
                else vectors[modality]
            )
            await store.client.upsert(
                modality,
                [
                    models.PointStruct(
                        id=point_id(modality, 603, offset),
                        vector=vector,
                        payload=point_payload(modality, 603, offset),
                    )
                ],
            )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://ml"
        ) as c:
            text_response = await c.post("/v1/search/text", json={"query": query})
            audio_response = await c.post(
                "/v1/search/audio",
                content=sound,
                headers={
                    "Content-Type": "audio/wav",
                },
            )
            assert text_response.status_code == audio_response.status_code == 200
            assert text_response.json() == audio_response.json() == {"movie_ids": [603]}
    cached = Encoders(settings.model_copy(update={"local_files_only": True}))
    for modality in ("text", "audio"):
        cached.load(modality)
    np.testing.assert_allclose(cached.encode_text(query)[0], vectors["text"], atol=1e-6)
    np.testing.assert_allclose(
        cached.encode("audio", sound, "audio/wav")[0], vectors["audio"], atol=1e-6
    )
