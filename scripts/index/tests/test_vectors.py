from unittest.mock import Mock
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import QdrantClient, models

from dejaview_ml.config import COLLECTIONS, DIMENSIONS
from scripts.index.delete import delete_movie
from scripts.index.embedding import Fragment
from scripts.index.errors import IndexingError
from scripts.index.metadata import MovieTable
from scripts.index.vectors import UPSERT_BATCH_SIZE, VectorWriter, describe_qdrant_error

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
        "text": [Fragment(None, vector("text"), "overview"), Fragment(None, vector("text"), "metadata")],
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
    assert counts == {"text": 2, "image": 3, "video": 1, "audio": 2}

    text = {point.id: point.payload for point in movie_points(client, "text", 603)}
    assert text == {
        str(uuid5(NAMESPACE_URL, "dejaview:text:603")): {"movie_id": 603, "type": "text", "source": "overview"},
        str(uuid5(NAMESPACE_URL, "dejaview:text:603:metadata")): {
            "movie_id": 603,
            "type": "text",
            "source": "metadata",
        },
    }

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
    assert {name: client.count(name).count for name in DIMENSIONS} == {"text": 4, "image": 5, "video": 2, "audio": 4}


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


def test_text_only_replacement_keeps_other_collections(client):
    writer = VectorWriter(client, "memory")
    writer.replace_movie(603, fragments(3))
    counts = writer.replace_movie(603, {"text": [Fragment(None, [0.5] + [0.0] * (DIMENSIONS["text"] - 1))]})
    assert counts == {"text": 1}
    assert len(movie_points(client, "text", 603)) == 1
    assert len(movie_points(client, "image", 603)) == 3
    assert len(movie_points(client, "video", 603)) == 1
    assert len(movie_points(client, "audio", 603)) == 2


def text_fragment(source: str, value: float) -> Fragment:
    return Fragment(None, [value] + [0.0] * (DIMENSIONS["text"] - 1), source)


def test_text_update_overwrites_points_in_place_and_keeps_other_collections(client):
    writer = VectorWriter(client, "memory")
    writer.replace_movie(603, fragments(3))
    writer.update_text(603, [text_fragment("overview", 0.5), text_fragment("metadata", 0.5)])

    text = movie_points(client, "text", 603)
    assert sorted(point.payload["source"] for point in text) == ["metadata", "overview"]
    assert len(movie_points(client, "image", 603)) == 3
    assert len(movie_points(client, "video", 603)) == 1
    assert len(movie_points(client, "audio", 603)) == 2


def test_text_update_removes_point_that_is_no_longer_produced(client):
    writer = VectorWriter(client, "memory")
    writer.update_text(603, [text_fragment("overview", 1.0), text_fragment("metadata", 1.0)])
    writer.update_text(550, [text_fragment("overview", 1.0), text_fragment("metadata", 1.0)])
    writer.update_text(603, [text_fragment("overview", 1.0)])

    (overview,) = movie_points(client, "text", 603)
    assert overview.id == str(uuid5(NAMESPACE_URL, "dejaview:text:603"))
    assert len(movie_points(client, "text", 550)) == 2


def test_replace_speech_keeps_line_text_and_replaces_old_lines(client):
    client.create_collection("speech", vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE))
    writer = VectorWriter(client, "memory")
    vector = [1.0] + [0.0] * 767
    assert writer.replace_speech(603, [(1680, "Она с тобой на работе", vector), (1690, "Какую правду?", vector)]) == 2
    assert writer.replace_speech(603, [(1680, "Матрица повсюду", vector)]) == 1
    (point,) = movie_points(client, "speech", 603)
    assert point.payload == {"movie_id": 603, "type": "speech", "offset_sec": 1680, "text": "Матрица повсюду"}
    assert point.id == str(uuid5(NAMESPACE_URL, "dejaview:speech:603:1680"))


def line(offset: int, text: str = "Матрица повсюду") -> tuple[int, str, list[float]]:
    return offset, text, [1.0] + [0.0] * 767


def test_replace_speech_requires_collection(client):
    with pytest.raises(IndexingError, match="нет коллекции «speech»"):
        VectorWriter(client, "memory").replace_speech(603, [line(0)])


@pytest.fixture
def speech_client(client):
    client.create_collection("speech", vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE))
    return client


def test_replace_speech_keeps_other_films(speech_client):
    writer = VectorWriter(speech_client, "memory")
    writer.replace_speech(550, [line(0), line(10)])
    writer.replace_speech(603, [line(0)])
    writer.replace_speech(603, [line(10)])
    assert [point.payload["offset_sec"] for point in movie_points(speech_client, "speech", 603)] == [10]
    assert len(movie_points(speech_client, "speech", 550)) == 2


def test_failed_write_keeps_previous_lines(speech_client):
    writer = VectorWriter(speech_client, "memory")
    writer.replace_speech(603, [line(offset, "прежняя реплика") for offset in range(0, 30, 10)])
    upsert = speech_client.upsert
    calls = []

    def failing_upsert(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise TimeoutError
        return upsert(*args, **kwargs)

    speech_client.upsert = failing_upsert
    new_lines = [line(offset, "новая реплика") for offset in range(0, 10 * (UPSERT_BATCH_SIZE + 1), 10)]
    with pytest.raises(IndexingError, match="запись в speech"):
        writer.replace_speech(603, new_lines)
    points = movie_points(speech_client, "speech", 603)
    # The first batch is written, the second failed: no old point was deleted.
    assert {point.payload["offset_sec"] for point in points} >= {0, 10, 20}
    assert len(points) >= 3


def test_empty_lines_keep_previous_index(speech_client):
    writer = VectorWriter(speech_client, "memory")
    writer.replace_speech(603, [line(0)])
    with pytest.raises(IndexingError, match="Нет реплик"):
        writer.replace_speech(603, [])
    assert len(movie_points(speech_client, "speech", 603)) == 1


def test_delete_removes_all_modalities_and_preserves_other_movies(speech_client):
    client = speech_client
    writer = VectorWriter(client, "memory")
    writer.replace_movie(603, fragments(3))
    writer.replace_movie(550, fragments(2))
    writer.replace_speech(603, [line(0), line(10)])
    writer.replace_speech(550, [line(0)])

    # Второй запуск и удаление неизвестного id тоже успешны.
    for movie_id in (603, 603, 999):
        writer.delete_movie(movie_id)
        for name in COLLECTIONS:
            assert movie_points(client, name, 603) == []
            assert movie_points(client, name, 550)
            result = client.query_points(name, query=[1.0] + [0.0] * (COLLECTIONS[name] - 1), limit=10)
            assert {point.payload["movie_id"] for point in result.points} == {550}


@pytest.mark.parametrize("missing", ["video", "speech"])
def test_delete_tolerates_missing_collection(speech_client, missing):
    client = speech_client
    writer = VectorWriter(client, "memory")
    writer.replace_movie(603, fragments(2))
    writer.replace_speech(603, [line(0)])
    client.delete_collection(missing)
    writer.delete_movie(603)
    writer.delete_movie(603)
    assert not client.collection_exists(missing)
    for name in COLLECTIONS.keys() - {missing}:
        assert movie_points(client, name, 603) == []


@pytest.mark.parametrize("failed_collection", ["image", "speech"])
def test_delete_retry_after_partial_qdrant_failure(speech_client, monkeypatch, failed_collection):
    client = speech_client
    writer = VectorWriter(client, "memory")
    movies = Mock(spec=MovieTable)
    writer.replace_movie(603, fragments(2))
    writer.replace_speech(603, [line(0), line(10)])
    original_delete = client.delete

    def fail_on_collection(*args, **kwargs):
        if kwargs.get("collection_name") == failed_collection:
            raise TimeoutError("timeout")
        return original_delete(*args, **kwargs)

    monkeypatch.setattr(client, "delete", fail_on_collection)
    with pytest.raises(IndexingError, match=f"удаление фильма 603 из {failed_collection}"):
        delete_movie(603, movies, writer)
    movies.delete.assert_not_called()
    assert movie_points(client, "text", 603) == []
    assert movie_points(client, failed_collection, 603)

    monkeypatch.setattr(client, "delete", original_delete)
    delete_movie(603, movies, writer)
    movies.delete.assert_called_once_with(603)
    assert all(not movie_points(client, name, 603) for name in COLLECTIONS)


def test_delete_requires_completion_and_waits(client, monkeypatch):
    def acknowledge(**kwargs):
        assert kwargs["wait"] is True
        return models.UpdateResult(operation_id=1, status=models.UpdateStatus.ACKNOWLEDGED)

    monkeypatch.setattr(client, "delete", acknowledge)
    with pytest.raises(IndexingError, match="не подтвердил завершение"):
        VectorWriter(client, "memory").delete_movie(603)


def plot_fragments(count: int, value: float = 1.0) -> list[Fragment]:
    return [Fragment(None, [value] + [0.0] * (DIMENSIONS["text"] - 1), "plot", chunk) for chunk in range(count)]


def sources(client: QdrantClient, movie_id: int) -> list[tuple[str, int | None]]:
    return sorted((p.payload["source"], p.payload.get("chunk")) for p in movie_points(client, "text", movie_id))


def test_plot_points_are_numbered_chunks_next_to_the_card(client):
    writer = VectorWriter(client, "memory")
    writer.replace_movie(120, fragments(1))
    assert writer.replace_plot(120, plot_fragments(3)) == 3
    assert sources(client, 120) == [("metadata", None), ("overview", None), ("plot", 0), ("plot", 1), ("plot", 2)]
    ids = {point.id for point in movie_points(client, "text", 120) if point.payload["source"] == "plot"}
    assert str(uuid5(NAMESPACE_URL, "dejaview:text:120:plot:2")) in ids


def test_shorter_plot_removes_stale_chunks_and_empty_plot_removes_all(client):
    writer = VectorWriter(client, "memory")
    writer.replace_movie(120, fragments(1))
    writer.replace_plot(120, plot_fragments(5))
    writer.replace_plot(120, plot_fragments(2, value=0.5))
    assert sources(client, 120) == [("metadata", None), ("overview", None), ("plot", 0), ("plot", 1)]
    writer.replace_plot(120, [])
    assert sources(client, 120) == [("metadata", None), ("overview", None)]


def test_plot_survives_reindexing_the_movie_and_its_card(client):
    writer = VectorWriter(client, "memory")
    writer.replace_movie(120, fragments(1))
    writer.replace_plot(120, plot_fragments(2))
    writer.replace_movie(120, fragments(2, value=0.5))
    writer.update_text(120, [Fragment(None, [0.3] + [0.0] * (DIMENSIONS["text"] - 1), "overview")])
    assert sources(client, 120) == [("overview", None), ("plot", 0), ("plot", 1)]


def test_replace_plot_takes_plot_fragments_only(client):
    with pytest.raises(ValueError):
        VectorWriter(client, "memory").replace_plot(120, fragments(1)["text"])
