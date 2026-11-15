import pytest

from scripts.index import speech
from scripts.index.transcribe import Cue


def test_cli_imports_and_parses(tmp_path):
    film = tmp_path / "film.mp4"
    film.write_bytes(b"")
    args = speech.parse_args(["--movie-id", "603", "--transcribe", str(film)])
    assert (args.movie_id, args.transcribe, args.dry_run) == (603, film, None)
    assert speech.parse_args(["--movie-id", "603", "--transcribe", str(film), "--dry-run"]).dry_run == 20
    assert speech.parse_args(["--movie-id", "603", "--transcribe", str(film), "--dry-run", "3"]).dry_run == 3


@pytest.mark.parametrize("value", ["0", "-1", "много"])
def test_dry_run_rejects_non_positive_count(tmp_path, value):
    film = tmp_path / "film.mp4"
    film.write_bytes(b"")
    with pytest.raises(SystemExit):
        speech.parse_args(["--movie-id", "603", "--transcribe", str(film), "--dry-run", value])


def test_dry_run_never_writes(monkeypatch, tmp_path, capsys):
    film = tmp_path / "film.mp4"
    film.write_bytes(b"")
    monkeypatch.setattr(speech, "Encoders", lambda settings: object())
    monkeypatch.setattr(speech, "load_model", lambda encoders, modality: None)
    monkeypatch.setattr(speech, "probe_audio", lambda path: None)
    monkeypatch.setattr(
        speech, "transcribe_film", lambda *args: [Cue(1, 4, "один два три"), Cue(31, 34, "четыре пять шесть")]
    )

    def connect(*args, **kwargs):
        raise AssertionError("--dry-run must not connect to Qdrant")

    monkeypatch.setattr(speech.vectors, "connect", connect)
    env = tmp_path / ".env"
    env.write_text("")
    args = speech.parse_args(["--movie-id", "603", "--transcribe", str(film), "--dry-run", "1", "--env-file", str(env)])
    speech.run(args)
    assert capsys.readouterr().out == "00:00:00  один два три\n"
