import asyncio
import io
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from dejaview_ml.app import create_app
from dejaview_ml.config import Settings
from dejaview_ml.embeddings import Encoders
from dejaview_ml.errors import MlError
from dejaview_ml.media import LIMITS
from dejaview_ml.storage import SchemaMismatch


class TestEncoders(Encoders):
    __test__ = False

    def __init__(self):
        super().__init__(Settings(_env_file=None))
        self.states.update(image="ready", video="ready")

    def _encode_frames(self, modality, frames):
        return [1.0] + [0.0] * 511


@pytest.fixture
def api():
    encoders = TestEncoders()
    store = AsyncMock()
    store.healthy.return_value = True
    store.search.return_value = [603]
    app = create_app(Settings(_env_file=None), encoders, store)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, encoders, store


def png():
    out = io.BytesIO()
    Image.new("RGB", (64, 32), "red").save(out, format="PNG")
    return out.getvalue()


def test_image_search_and_request_id(api, caplog):
    client, _, store = api
    with caplog.at_level("INFO"):
        response = client.post(
            "/v1/search/image",
            content=png(),
            headers={
                "Content-Type": "image/png",
                "X-Request-Id": "image-test-1",
            },
        )
    assert response.status_code == 200
    assert response.json() == {"movie_ids": [603]}
    assert "image-test-1" in caplog.text
    assert len(store.search.call_args.args[1][0]) == 512


@pytest.mark.parametrize(
    ("content", "media_type", "status", "code"),
    [
        (b"", "image/png", 422, "FILE_CORRUPTED"),
        (b"garbage", "image/jpeg", 422, "FILE_CORRUPTED"),
        (b"GIF89a", "image/gif", 415, "UNSUPPORTED_MEDIA_TYPE"),
        (png(), "image/jpeg", 415, "UNSUPPORTED_MEDIA_TYPE"),
    ],
)
def test_invalid_images(api, content, media_type, status, code):
    client, _, store = api
    response = client.post(
        "/v1/search/image", content=content, headers={"Content-Type": media_type}
    )
    assert response.status_code == status
    assert response.json()["code"] == code
    store.search.assert_not_called()


def test_size_limit_checks_header_and_actual_stream(api, monkeypatch):
    client, _, store = api
    monkeypatch.setitem(LIMITS, "image", 10)
    for headers in ({"Content-Length": "11"}, {"Content-Length": "1"}):
        response = client.post(
            "/v1/search/image", content=b"a" * 11, headers={"Content-Type": "image/png", **headers}
        )
        assert response.status_code == 413
        assert response.json()["code"] == "FILE_TOO_LARGE"
    store.search.assert_not_called()


def test_chunked_body_cannot_bypass_limit(api, monkeypatch):
    client, _, store = api
    monkeypatch.setitem(LIMITS, "image", 10)
    response = client.post(
        "/v1/search/image",
        content=iter([b"a" * 6, b"b" * 6]),
        headers={"Content-Type": "image/png"},
    )
    assert response.status_code == 413
    store.search.assert_not_called()


@pytest.mark.parametrize("modality", ["text", "audio"])
def test_disabled(api, modality):
    client, _, _ = api
    response = client.post(f"/v1/search/{modality}", content=b"unused")
    assert response.status_code == 503
    assert response.json()["code"] == "NOT_READY"


def test_health_states_and_empty_search(api):
    client, encoders, store = api
    assert client.get("/health").json() == {
        "status": "ok",
        "models": {"text": "disabled", "image": "ready", "video": "ready", "audio": "disabled"},
        "qdrant": "ok",
    }
    for state in ("loading", "failed"):
        encoders.states["video"] = state
        assert client.get("/health").status_code == 503
        response = client.post(
            "/v1/search/video", content=b"video", headers={"Content-Type": "video/mp4"}
        )
        assert response.json()["code"] == "NOT_READY"
    encoders.states["video"] = "ready"
    store.healthy.return_value = False
    assert client.get("/health").json()["qdrant"] == "unavailable"
    store.search.return_value = []
    assert client.post(
        "/v1/search/image", content=png(), headers={"Content-Type": "image/png"}
    ).json() == {"movie_ids": []}


def test_errors_are_contract_json(api):
    client, _, store = api
    for error, status, code in [
        (MlError(503, "QDRANT_UNAVAILABLE", "timeout"), 503, "QDRANT_UNAVAILABLE"),
        (RuntimeError("private diagnostic"), 500, "INTERNAL_ERROR"),
    ]:
        store.search.side_effect = error
        response = client.post(
            "/v1/search/image", content=png(), headers={"Content-Type": "image/png"}
        )
        assert response.status_code == status
        assert response.json()["code"] == code
        assert "private diagnostic" not in response.text


def test_openapi_matches_supplied_contract(api):
    client, _, _ = api
    assert client.get("/openapi.json").json() == yaml.safe_load(
        Path("ml-service.openapi.yaml").read_text(),
    )


async def test_overload_does_not_queue_and_health_remains_available():
    store = AsyncMock()
    store.healthy.return_value = True
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_search(*args):
        entered.set()
        await release.wait()
        return [603]

    store.search.side_effect = slow_search
    app = create_app(Settings(_env_file=None), TestEncoders(), store)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://ml"
        ) as c:
            first = asyncio.create_task(
                c.post("/v1/search/image", content=png(), headers={"Content-Type": "image/png"})
            )
            await asyncio.wait_for(entered.wait(), timeout=5)
            try:
                assert (await c.get("/health")).status_code == 200
                second = await c.post(
                    "/v1/search/image", content=png(), headers={"Content-Type": "image/png"}
                )
                assert second.status_code == 503
            finally:
                release.set()
            assert (await first).status_code == 200


async def test_schema_mismatch_aborts_startup_without_loading_models():
    store = AsyncMock()
    store.ensure_collections.side_effect = SchemaMismatch("image: expected 512 dimensions")
    app = create_app(Settings(_env_file=None), TestEncoders(), store)
    with pytest.raises(SchemaMismatch):
        async with app.router.lifespan_context(app):
            pytest.fail("startup must fail")
    store.close.assert_awaited_once()


async def test_model_load_failure_is_visible_in_health():
    class BrokenEncoders(TestEncoders):
        def load(self, modality):
            raise RuntimeError("weights missing")

    encoders = BrokenEncoders()
    encoders.states["video"] = "loading"
    store = AsyncMock()
    store.healthy.return_value = True
    app = create_app(Settings(_env_file=None), encoders, store)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://ml"
        ) as c:
            for _ in range(100):
                if encoders.states["video"] == "failed":
                    break
                await asyncio.sleep(0.01)
            response = await c.get("/health")
            assert response.status_code == 503
            assert response.json()["models"]["video"] == "failed"
