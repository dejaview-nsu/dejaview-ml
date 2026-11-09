from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import QueryResponse

import dejaview_ml.storage as storage
from dejaview_ml.config import DIMENSIONS, Settings
from dejaview_ml.errors import MlError
from dejaview_ml.storage import (
    SchemaMismatch,
    VectorStore,
    aggregate,
    metadata_passage,
    point_id,
    point_payload,
    temporal_aggregate,
)


def test_metadata_and_stable_point_ids():
    assert point_payload("image", 603, 125) == {"movie_id": 603, "type": "image", "offset_sec": 125}
    assert point_payload("text", 603) == {"movie_id": 603, "type": "text", "source": "overview"}
    assert point_payload("text", 603, source="metadata") == {
        "movie_id": 603,
        "type": "text",
        "source": "metadata",
    }
    assert point_id("image", 603, 125) == str(uuid5(NAMESPACE_URL, "dejaview:image:603:125"))
    assert point_id("text", 603) == str(uuid5(NAMESPACE_URL, "dejaview:text:603"))
    assert point_id("text", 603, source="overview") == str(
        uuid5(NAMESPACE_URL, "dejaview:text:603")
    )
    assert point_id("text", 603, source="metadata") == str(
        uuid5(NAMESPACE_URL, "dejaview:text:603:metadata")
    )
    invalid = [
        ("text", 603, 0),
        ("image", 603, None),
        ("image", 603, -1),
        ("video", True, 0),
        ("text", 603, None, "title"),
        ("image", 603, 0, "overview"),
    ]
    for args in invalid:
        with pytest.raises(ValueError):
            point_payload(*args)


def named(*names):
    return [SimpleNamespace(name=name) for name in names]


def actor(name, character=None):
    return SimpleNamespace(name=name, character=character)


def test_metadata_passage_matches_contract_example():
    movie = SimpleNamespace(
        genres=named("боевик", "фантастика"),
        release_date=date(1999, 3, 31),
        countries=named("США"),
        crew=[
            SimpleNamespace(name="Лана Вачовски", role="director"),
            SimpleNamespace(name="Джоэл Сильвер", role="producer"),
            SimpleNamespace(name="Лилли Вачовски", role="director"),
        ],
        actors=[
            actor("Киану Ривз", "Neo"),
            actor("Лоренс Фишберн", "Morpheus"),
            actor("Кэрри-Энн Мосс", "Trinity"),
            actor("Хьюго Уивинг", "Agent Smith"),
        ],
    )
    assert metadata_passage(movie) == (
        "Жанр: боевик, фантастика. Год: 1999, 1990-е. Страна: США. "
        "Режиссёр: Лана Вачовски, Лилли Вачовски. В ролях: Киану Ривз (Neo); "
        "Лоренс Фишберн (Morpheus); Кэрри-Энн Мосс (Trinity); Хьюго Уивинг (Agent Smith)."
    )


def test_metadata_passage_skips_empty_fields_and_extra_actors():
    actors = [actor(f"Актёр {number}") for number in range(15)]
    movie = SimpleNamespace(genres=[], release_date=None, countries=[], crew=[], actors=actors)
    passage = metadata_passage(movie)
    assert passage.startswith("В ролях: Актёр 0; Актёр 1;")
    assert passage.endswith("Актёр 9.")
    empty = SimpleNamespace(genres=[], release_date=None, countries=[], crew=[], actors=[])
    assert metadata_passage(empty) is None


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


def test_text_film_score_is_its_best_point():
    def hit(id, movie, score):
        return models.ScoredPoint(id=id, version=1, score=score, payload={"movie_id": movie})

    # Два средних совпадения фильма 1 в сумме обгоняют одно сильное у фильма 2, максимум - нет.
    points = [hit(1, 1, 0.80), hit(2, 1, 0.79), hit(3, 2, 0.85)]
    assert aggregate(points, limit=2) == [1, 2]
    assert aggregate(points, limit=2, combine="max") == [2, 1]


def timed_hit(id, movie, offset, score):
    return models.ScoredPoint(
        id=id, version=1, score=score, payload={"movie_id": movie, "offset_sec": offset}
    )


def test_temporal_aggregation_prefers_matches_that_line_up_in_time():
    # Film 1: one window at 28:10, then 28:20, 28:30 - the query read in order.
    # Film 2: stronger single hits, but scattered over the film.
    windows = [
        (0, [timed_hit(1, 2, 300, 0.9), timed_hit(2, 1, 1690, 0.8)]),
        (10, [timed_hit(3, 2, 4000, 0.9), timed_hit(4, 1, 1700, 0.8)]),
        (20, [timed_hit(5, 2, 50, 0.9), timed_hit(6, 1, 1710, 0.8)]),
    ]
    assert aggregate([point for _, points in windows for point in points]) == [2]
    assert temporal_aggregate(windows, tolerance=10) == [1]


def test_temporal_aggregation_counts_one_hit_per_window_and_breaks_ties():
    windows = [
        (0, [timed_hit(1, 1, 100, 0.5), timed_hit(2, 1, 105, 0.6), timed_hit(3, 2, 100, 0.6)]),
        (5, [timed_hit(4, 2, 105, 0.5), timed_hit(5, 1, 400, 0.9)]),
    ]
    # Film 1: best aligned pair is 0.6 (the 0.9 hit is 300 s away); film 2: 0.6 + 0.5.
    assert temporal_aggregate(windows, tolerance=10, limit=2) == [2, 1]
    tie = [(0, [timed_hit(1, 7, 0, 0.5), timed_hit(2, 3, 0, 0.5)])]
    assert temporal_aggregate(tie, tolerance=10, limit=2) == [3, 7]


async def test_timed_query_with_offsets_uses_temporal_aggregation():
    client = AsyncMock()
    client.query_points.side_effect = [
        QueryResponse(points=[timed_hit(1, 2, 300, 0.9), timed_hit(2, 1, 1690, 0.8)]),
        QueryResponse(points=[timed_hit(3, 2, 4000, 0.9), timed_hit(4, 1, 1700, 0.8)]),
    ]
    store = VectorStore(Settings(_env_file=None, top_k=2), client)
    assert await store.search("audio", [[1.0] * 512, [1.0] * 512], [0, 10]) == [1]
    assert client.query_points.call_args.kwargs["limit"] == storage.TEMPORAL_CANDIDATES


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


async def test_text_search_ranks_films_by_best_point():
    client = AsyncMock()

    def hit(id, movie, score):
        return models.ScoredPoint(id=id, version=1, score=score, payload={"movie_id": movie})

    client.query_points.return_value = QueryResponse(
        points=[hit(1, 10, 0.80), hit(2, 10, 0.79), hit(3, 20, 0.85)]
    )
    store = VectorStore(Settings(_env_file=None), client)
    assert await store.search("text", [[1.0] * 768]) == [20, 10]


def speech_hit(movie, offset, score, text):
    return models.ScoredPoint(
        id=point_id("speech", movie, offset),
        version=1,
        score=score,
        payload={"movie_id": movie, "type": "speech", "offset_sec": offset, "text": text},
    )


def test_speech_point_has_offset():
    assert point_payload("speech", 603, 1680) == {
        "movie_id": 603,
        "type": "speech",
        "offset_sec": 1680,
    }


def test_shared_words_ignore_common_words():
    # "живете" and "жили" differ in their first five letters; "мире", "мир", "таков" match;
    # "Вы", "в" are too short and "сейчас" is a common word.
    transcript, line = "Вы живете в мире, таков мир сейчас", "Вы жили в мире. Таков мир сейчас"
    assert storage.word_stems(transcript) == {"живет", "мире", "мир", "таков"}
    assert storage.shared_words(transcript, line) == 0.75
    assert storage.shared_words("Ну всё, пока!", "Пока-пока") == 0.0


def test_yo_and_ye_are_the_same_letter():
    assert storage.word_stems("Я вижу мёртвых людей") == storage.word_stems("я вижу мертвых людей")
    assert storage.shared_words("Мы найдём способ", "Мы найдем способ, как всегда") == 1.0


def test_speech_scores_need_meaningful_shared_words():
    hits = [
        speech_hit(120, 10000, 0.88, "Выследите их, не останавливайтесь"),  # nothing shared
        speech_hit(597, 2690, 0.83, "Держись за белым кораблём"),  # 1 of 3: below a third
        speech_hit(603, 460, 0.82, "Следуй за белым кроликом. Кто там?"),  # 2 of 3 words
    ]
    scores = storage.speech_scores(hits, "Иди за белым кроликом")
    assert set(scores) == {603}
    assert scores[603] == pytest.approx(0.82 + storage.SPEECH_WORD_WEIGHT * 2 / 3)
    # A transcript of common words only says nothing about the film.
    assert storage.speech_scores(hits, "Ну всё, пока!") == {}


def test_audio_ranking_adds_clap_and_dialogue():
    clap = {6977: 0.80, 603: 0.75, 550: 0.70}
    assert storage.audio_ranking(clap, {}) == [6977, 603, 550]
    # The dialogue outweighs a small CLAP lead...
    assert storage.audio_ranking(clap, {603: 0.9})[0] == 603
    # ...and a film absent from the CLAP candidates gets the weakest CLAP score, not zero.
    assert storage.audio_ranking(clap, {120: 0.9})[0] == 120
    assert storage.audio_ranking({}, {}) == []


def audio_query(speech_vector=(1.0,), transcript="Иди за белым кроликом"):
    from dejaview_ml.embeddings import AudioQuery

    return AudioQuery(
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        offsets=[0.0, 2.5],
        transcript=transcript,
        speech_vector=list(speech_vector) if speech_vector else None,
    )


def speech_store(audio_hits, speech_hits):
    client = AsyncMock()

    async def query_points(collection_name, limit, **_):
        points = speech_hits if collection_name == "speech" else audio_hits
        return QueryResponse(points=list(points)[:limit])

    client.query_points.side_effect = query_points
    return VectorStore(Settings(_env_file=None), client), client


async def test_audio_search_combines_clap_and_dialogue():
    audio = [timed_hit(1, 6977, 100, 0.80), timed_hit(2, 603, 455, 0.76)]
    store, client = speech_store(audio, [speech_hit(603, 460, 0.82, "Следуй за белым кроликом")])
    assert await store.search_audio(audio_query()) == [603]
    calls = client.query_points.call_args_list
    assert [call.kwargs["collection_name"] for call in calls] == ["audio", "audio", "speech"]
    assert calls[0].kwargs["limit"] == storage.TEMPORAL_CANDIDATES


async def test_audio_threshold_applies_to_clap_but_not_to_dialogue():
    # A phone recording rarely clears the CLAP threshold; the dialogue still finds the film.
    store, client = speech_store([], [speech_hit(603, 460, 0.82, "Следуй за белым кроликом")])
    assert await store.search_audio(audio_query()) == [603]
    used = {
        call.kwargs["collection_name"]: call.kwargs["score_threshold"]
        for call in client.query_points.call_args_list
    }
    assert used == {"audio": Settings(_env_file=None).score_threshold_audio, "speech": None}


@pytest.mark.parametrize(
    ("query", "lines"),
    [
        # Nothing recognised: CLAP alone.
        (audio_query(speech_vector=None, transcript=""), []),
        # Recognised, but no line shares its meaningful words.
        (audio_query(), [speech_hit(120, 10000, 0.88, "Выследите их")]),
        (audio_query(transcript="Да, да, конечно"), [speech_hit(120, 10, 0.9, "Да, конечно")]),
    ],
)
async def test_audio_search_without_matching_dialogue_is_clap(query, lines):
    audio = [timed_hit(1, 6977, 100, 0.80), timed_hit(2, 603, 455, 0.76)]
    store, _ = speech_store(audio, lines)
    assert await store.search_audio(query) == [6977]


def text_hit(id, movie, score):
    return models.ScoredPoint(id=id, version=1, score=score, payload={"movie_id": movie})


def text_store(text_hits, speech_hits):
    client = AsyncMock()

    async def query_points(collection_name, limit, **_):
        points = speech_hits if collection_name == "speech" else text_hits
        return QueryResponse(points=list(points)[:limit])

    client.query_points.side_effect = query_points
    return VectorStore(Settings(_env_file=None), client), client


async def test_text_search_raises_quoted_film_to_the_top():
    descriptions = [text_hit(1, 550, 0.80), text_hit(2, 680, 0.79), text_hit(3, 550, 0.78)]
    lines = [speech_hit(603, 460, 0.86, "Следуй за белым кроликом. Кто там?")]
    store, client = text_store(descriptions, lines)
    assert await store.search_text([1.0], "Иди за белым кроликом") == [603, 550, 680]
    collections = [call.kwargs["collection_name"] for call in client.query_points.call_args_list]
    assert collections == ["text", "speech"]


@pytest.mark.parametrize(
    "lines",
    [
        # A line sharing words, but farther from the query than the best description.
        [speech_hit(603, 460, 0.79, "Хакер узнаёт, что мир - компьютерная симуляция")],
        # Close, but sharing only one meaningful word of five.
        [speech_hit(603, 460, 0.90, "Это симуляция")],
        [],
    ],
)
async def test_text_search_without_quote_ranks_by_description(lines):
    descriptions = [text_hit(1, 550, 0.80), text_hit(2, 680, 0.79), text_hit(3, 603, 0.70)]
    store, _ = text_store(descriptions, lines)
    query = "хакер узнаёт, что мир - компьютерная симуляция"
    assert await store.search_text([1.0], query) == [550, 680, 603]


async def test_text_search_returns_ten_films():
    descriptions = [text_hit(id, id, 0.9 - id / 100) for id in range(1, 15)]
    store, _ = text_store(descriptions, [])
    assert await store.search_text([1.0], "фильм") == list(range(1, 11))
