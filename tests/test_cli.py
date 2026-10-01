import json
import sys

import pytest
from qdrant_client import AsyncQdrantClient, QdrantClient, models
from test_text_audio import wav_bytes

from dejaview_ml import cli
from dejaview_ml.config import DIMENSIONS, Settings
from dejaview_ml.embeddings import Encoders
from dejaview_ml.storage import VectorStore, point_id


class CliEncoders(Encoders):
    seen_passage = None

    def load(self, modality):
        self.states[modality] = "ready"

    def _encode_text(self, text, *, passage):
        type(self).seen_passage = passage
        return [1.0] + [0.0] * 767

    def _encode_audio(self, waveform):
        return [1.0] + [0.0] * 511


@pytest.mark.parametrize("modality", ["text", "audio"])
def test_cli_upsert_uses_passage_prefix_and_correct_payload(
    tmp_path, monkeypatch, capsys, modality
):
    path = tmp_path / "qdrant"
    client = QdrantClient(path=str(path))
    client.create_collection(
        modality,
        vectors_config=models.VectorParams(
            size=DIMENSIONS[modality],
            distance=models.Distance.COSINE,
        ),
    )
    client.close()
    file = tmp_path / ("description.txt" if modality == "text" else "clip.wav")
    file.write_bytes(("a description " * 50).encode() if modality == "text" else wav_bytes())
    monkeypatch.setattr(cli, "Encoders", CliEncoders)
    monkeypatch.setattr(
        cli,
        "VectorStore",
        lambda settings: VectorStore(
            Settings(_env_file=None),
            AsyncQdrantClient(path=str(path)),
        ),
    )
    monkeypatch.setattr(
        sys, "argv", ["dejaview-embed", modality, str(file), "--upsert", "--movie-id", "603"]
    )
    cli.main()
    offset = None if modality == "text" else 0
    assert json.loads(capsys.readouterr().out)["point_ids"] == [point_id(modality, 603, offset)]
    client = QdrantClient(path=str(path))
    try:
        point = client.scroll(modality)[0][0]
        assert point.payload["movie_id"] == 603
        assert point.payload["type"] == modality
        if modality == "text":
            assert CliEncoders.seen_passage is True
            assert "offset_sec" not in point.payload
        else:
            assert point.payload["offset_sec"] == 0
    finally:
        client.close()


def test_cli_text_query_and_offset_rejection(tmp_path, monkeypatch, capsys):
    file = tmp_path / "query.txt"
    file.write_text("девушка в красном платье")
    monkeypatch.setattr(cli, "Encoders", CliEncoders)
    monkeypatch.setattr(sys, "argv", ["dejaview-embed", "text", str(file)])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["dimension"] == 768
    assert CliEncoders.seen_passage is False
    monkeypatch.setattr(sys, "argv", ["dejaview-embed", "text", str(file), "--offset-sec", "0"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
