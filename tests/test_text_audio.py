import io
import subprocess
import wave
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient, models

from dejaview_ml.app import create_app
from dejaview_ml.config import Settings
from dejaview_ml.embeddings import Encoders
from dejaview_ml.errors import MlError
from dejaview_ml.media import LIMITS, _run, decode_audio
from dejaview_ml.storage import VectorStore


def wav_bytes(duration=1):
    out = io.BytesIO()
    samples = (np.sin(np.arange(int(16000 * duration)) * 0.08) * 8000).astype("<i2")
    with wave.open(out, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(samples.tobytes())
    return out.getvalue()


@pytest.fixture
def api():
    class EncodersForTest(Encoders):
        def _encode_text(self, text, *, passage):
            assert passage is False
            return [1.0] + [0.0] * 767

        def _encode_audio(self, waveform):
            assert waveform.dtype == np.float32
            return [1.0] + [0.0] * 511

        transcript = ""

        def _transcribe(self, waveform):
            assert waveform.dtype == np.float32
            return self.transcript

    settings = Settings(_env_file=None, text_enabled=True, audio_enabled=True)
    encoders = EncodersForTest(settings)
    encoders.states = dict.fromkeys(encoders.states, "ready")
    store = AsyncMock()
    store.healthy.return_value = True
    store.search.return_value = [603]
    store.search_audio.return_value = [603]
    store.search_text.return_value = [603]
    with TestClient(create_app(settings, encoders, store)) as client:
        yield client, encoders, store


@pytest.mark.parametrize("query", ["кот", "cat", "я" * 250])
def test_text_search(api, query):
    client, _, store = api
    response = client.post("/v1/search/text", json={"query": query})
    assert response.status_code == 200
    assert response.json() == {"movie_ids": [603]}
    vector, text = store.search_text.call_args.args
    assert (len(vector), text) == (768, query)


@pytest.mark.parametrize(
    "body",
    [
        {},
        [],
        {"query": 123},
        {"query": None},
        {"query": "ab"},
        {"query": "x" * 251},
        {"query": "   "},
        {"query": "valid", "extra": 1},
        {"query": "\ud800ab"},
    ],
)
def test_invalid_text_json(api, body):
    import json

    client, _, store = api
    response = client.post(
        "/v1/search/text", content=json.dumps(body), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_QUERY"
    store.search.assert_not_called()


@pytest.mark.parametrize("body", [b"", b"{", b'{"query":"\xff"}', b"[" * 2000])
def test_malformed_text(api, body):
    client, _, _ = api
    response = client.post(
        "/v1/search/text", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_QUERY"


def test_text_wrong_content_type_and_oversize(api):
    client, _, _ = api
    assert (
        client.post(
            "/v1/search/text",
            content="valid",
            headers={
                "Content-Type": "text/plain",
            },
        ).status_code
        == 415
    )
    response = client.post(
        "/v1/search/text",
        content=b" " * (LIMITS["text"] + 1),
        headers={
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_QUERY"


def test_audio_search_and_health(api):
    client, _, store = api
    response = client.post(
        "/v1/search/audio", content=wav_bytes(), headers={"Content-Type": "audio/wav"}
    )
    assert response.status_code == 200
    assert response.json() == {"movie_ids": [603]}
    query = store.search_audio.call_args.args[0]
    assert len(query.vectors[0]) == 512
    assert query.offsets == [0.0]
    # Silence or music: nothing to search among the lines of dialogue.
    assert (query.transcript, query.speech_vector) == ("", None)
    assert client.get("/health").json()["models"]["audio"] == "ready"


def test_audio_query_carries_encoded_transcript(api):
    client, encoders, store = api
    encoders.transcript = "Матрица повсюду, она с тобой на работе"
    response = client.post(
        "/v1/search/audio", content=wav_bytes(), headers={"Content-Type": "audio/wav"}
    )
    assert response.status_code == 200
    query = store.search_audio.call_args.args[0]
    assert query.transcript == encoders.transcript
    assert len(query.speech_vector) == 768


@pytest.mark.parametrize("state", ["loading", "failed", "disabled"])
def test_audio_search_works_without_speech_model(api, state):
    client, encoders, store = api
    encoders.transcript = "Матрица повсюду, она с тобой на работе"
    encoders.states["speech"] = state
    response = client.post(
        "/v1/search/audio", content=wav_bytes(), headers={"Content-Type": "audio/wav"}
    )
    assert response.json() == {"movie_ids": [603]}
    assert store.search_audio.call_args.args[0].speech_vector is None


def test_long_audio_query_adds_overlapping_windows(api):
    client, _, store = api
    response = client.post(
        "/v1/search/audio", content=wav_bytes(9), headers={"Content-Type": "audio/wav"}
    )
    assert response.status_code == 200
    # Whole clip plus 5 s windows every 2.5 s.
    query = store.search_audio.call_args.args[0]
    assert query.offsets == [0.0, 0.0, 2.5]
    assert len(query.vectors) == 3


@pytest.mark.parametrize("modality", ["text", "audio"])
@pytest.mark.parametrize("state", ["loading", "failed", "disabled"])
def test_optional_model_not_ready(api, modality, state):
    client, encoders, _ = api
    encoders.states[modality] = state
    response = client.post(f"/v1/search/{modality}")
    assert response.status_code == 503
    assert response.json()["code"] == "NOT_READY"
    assert client.get("/health").status_code == (200 if state == "disabled" else 503)


def test_audio_errors(api):
    client, _, store = api
    for body, media_type, code in [
        (b"invalid", "audio/wav", "FILE_CORRUPTED"),
        (b"", "audio/wav", "FILE_CORRUPTED"),
        (wav_bytes(), "audio/flac", "UNSUPPORTED_MEDIA_TYPE"),
        (wav_bytes(), "audio/mpeg", "UNSUPPORTED_MEDIA_TYPE"),
        (wav_bytes(11), "audio/wav", "DURATION_TOO_LONG"),
    ]:
        response = client.post(
            "/v1/search/audio", content=body, headers={"Content-Type": media_type}
        )
        assert response.json()["code"] == code
        assert response.status_code == (415 if code == "UNSUPPORTED_MEDIA_TYPE" else 422)
    response = client.post(
        "/v1/search/audio",
        content=b"x",
        headers={
            "Content-Type": "audio/wav",
            "Content-Length": str(20 * 1024 * 1024 + 1),
        },
    )
    assert response.status_code == 413
    store.search.assert_not_called()


@pytest.mark.parametrize(
    ("extension", "media_type"),
    [
        ("mp3", "audio/mpeg"),
        ("wav", "audio/wav"),
        ("ogg", "audio/ogg"),
        ("m4a", "audio/mp4"),
    ],
)
def test_audio_formats_resampling_and_mono(tmp_path, extension, media_type):
    path = tmp_path / f"clip.{extension}"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1:sample_rate=44100",
            "-ac",
            "2",
            str(path),
        ],
        check=True,
    )
    samples = decode_audio(path.read_bytes(), media_type, 10)
    assert samples.ndim == 1
    assert samples.dtype == np.float32
    assert 47000 <= len(samples) <= 50000
    assert np.isfinite(samples).all()


def test_audio_exact_duration_limit():
    assert len(decode_audio(wav_bytes(10), "audio/wav", 10)) == 480000


@pytest.mark.parametrize(
    ("extension", "media_type"),
    [
        ("mp3", "audio/mpeg"),
        ("wav", "audio/wav"),
        ("ogg", "audio/ogg"),
        ("m4a", "audio/mp4"),
    ],
)
@pytest.mark.parametrize("duration", [10, 10.01])
def test_audio_duration_boundary_with_codec_padding(tmp_path, extension, media_type, duration):
    path = tmp_path / f"clip.{extension}"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:duration={duration}",
            str(path),
        ],
        check=True,
    )
    if duration == 10:
        assert len(decode_audio(path.read_bytes(), media_type, 10)) == 480000
    else:
        with pytest.raises(MlError) as error:
            decode_audio(path.read_bytes(), media_type, 10)
        assert error.value.code == "DURATION_TOO_LONG"


def test_decoder_timeout_is_a_server_error():
    with patch(
        "dejaview_ml.media.subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 10)
    ):
        with pytest.raises(MlError) as error:
            _run(["ffmpeg"], 10)
    assert error.value.status == 500
    assert error.value.code == "INTERNAL_ERROR"


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
async def test_text_returns_up_to_ten_unique_movies_and_audio_one():
    store = VectorStore(Settings(_env_file=None), AsyncQdrantClient(":memory:"))
    try:
        await store.ensure_collections()
        for modality, dimension in [("text", 768), ("audio", 512)]:
            vector = [1.0] + [0.0] * (dimension - 1)
            await store.client.upsert(
                modality,
                [
                    models.PointStruct(
                        id=i,
                        vector=vector,
                        payload={"movie_id": i // 2, "type": modality},
                    )
                    for i in range(24)
                ],
            )
            result = await store.search(modality, [vector])
            assert len(result) == (10 if modality == "text" else 1)
            assert len(result) == len(set(result))
    finally:
        await store.close()


def test_downsample_keeps_speech_band_and_removes_aliases():
    from dejaview_ml.embeddings import downsample_to_speech_rate

    seconds = np.arange(48_000) / 48_000

    def level(frequency):
        out = downsample_to_speech_rate(np.sin(2 * np.pi * frequency * seconds).astype(np.float32))
        assert out.dtype == np.float32 and len(out) == 16_000
        return float(np.sqrt(np.mean(out[1000:-1000] ** 2)))

    assert level(1000) == pytest.approx(np.sqrt(0.5), rel=0.02)
    # 12 kHz would fold back to 4 kHz at 16 kHz sampling unless filtered out.
    assert level(12_000) < 0.01


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Матрица повсюду, она с тобой", True),
        ("Да, да.", False),
        ("Продолжение следует...", False),
        ("Субтитры сделал DimaTorzok", False),
        ("Спасибо за просмотр!", False),
    ],
)
def test_meaningful_speech(text, expected):
    from dejaview_ml.embeddings import meaningful_speech

    assert meaningful_speech(text) is expected


@pytest.mark.parametrize("disabled", ["text_enabled", "audio_enabled", "speech_enabled"])
def test_speech_needs_text_and_audio(disabled):
    encoders = Encoders(Settings(_env_file=None, **{disabled: False}))
    assert encoders.states["speech"] == "disabled"
