import io
import os

import httpx
import numpy as np
import pytest
from PIL import Image
from qdrant_client import AsyncQdrantClient, models
from test_media import video_file

from dejaview_ml.app import create_app
from dejaview_ml.config import Settings
from dejaview_ml.embeddings import Encoders
from dejaview_ml.storage import VectorStore, point_id, point_payload

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_MODEL_TESTS") != "1",
        reason="set RUN_MODEL_TESTS=1 to load real weights",
    ),
]


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
async def test_real_image_video_and_offline_reload(tmp_path):
    settings = Settings(_env_file=None, device="cpu")
    encoders = Encoders(settings)
    image = io.BytesIO()
    Image.new("RGB", (128, 96), "orange").save(image, format="PNG")
    video = video_file(tmp_path, 2).read_bytes()
    inputs = {"image": (image.getvalue(), "image/png"), "video": (video, "video/mp4")}
    results = {}
    for modality, (data, media_type) in inputs.items():
        encoders.load(modality)
        vectors = encoders.encode(modality, data, media_type)
        vector = np.asarray(vectors[0])
        assert vector.shape == (512,)
        assert np.isfinite(vector).all()
        assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-5)
        results[modality] = vector
    store = VectorStore(settings, AsyncQdrantClient(":memory:"))
    app = create_app(settings, encoders, store)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://ml"
        ) as c:
            for modality, (data, media_type) in inputs.items():
                await store.client.upsert(
                    modality,
                    [
                        models.PointStruct(
                            id=point_id(modality, 603, 0),
                            vector=results[modality].tolist(),
                            payload=point_payload(modality, 603, 0),
                        )
                    ],
                )
                response = await c.post(
                    f"/v1/search/{modality}", content=data, headers={"Content-Type": media_type}
                )
                assert response.status_code == 200
                assert response.json() == {"movie_ids": [603]}
    del encoders
    cached = Encoders(settings.model_copy(update={"local_files_only": True}))
    for modality, (data, media_type) in inputs.items():
        cached.load(modality)
        np.testing.assert_allclose(
            cached.encode(modality, data, media_type)[0], results[modality], atol=1e-6
        )
