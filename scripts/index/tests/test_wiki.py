import json

import pytest

from scripts.index import wiki
from scripts.index.wiki import Plot, Wikipedia, plot_chunks, plot_section

WORDS = " ".join(["слово"] * 50)


def test_plot_section_takes_the_plot_with_its_subsections_only():
    extract = (
        "Вступление статьи.\n\n"
        f"== Сюжет ==\nГэндальф ведёт Братство через Морию. {WORDS}\n\n"
        "=== На мосту Хазад-дум ===\nБалрог обрушивает мост, Гэндальф падает.\n\n"
        "== В ролях ==\nИэн Маккеллен - Гэндальф\n"
    )
    text = plot_section(extract, "ru")
    assert text.startswith("Гэндальф ведёт Братство через Морию.")
    assert "Балрог обрушивает мост" in text
    assert "На мосту" not in text and "В ролях" not in text and "Маккеллен" not in text


@pytest.mark.parametrize(
    ("extract", "language"),
    [
        ("== В ролях ==\nкто-то\n", "ru"),
        ("== Сюжет ==\nКоротко.\n", "ru"),  # слишком коротко для сюжета
        (f"== Сюжет ==\n{WORDS}\n", "en"),  # заголовок другого языка
    ],
)
def test_no_plot_section(extract, language):
    assert plot_section(extract, language) is None


def test_english_plot_heading():
    assert plot_section(f"== Plot ==\nFrodo leaves the Shire. {WORDS}\n", "en").startswith("Frodo")


def test_chunks_follow_paragraphs_and_split_long_ones_by_sentence():
    short = " ".join(["а"] * 60)
    long = " ".join([f"{' '.join(['б'] * 40)}." for _ in range(5)])  # 200 слов, 5 предложений
    chunks = plot_chunks(f"{short}\n{short}\n{long}", words=150)
    assert [len(chunk.split()) for chunk in chunks] == [120, 120, 80]
    assert all(len(chunk.split()) <= 150 for chunk in chunks)
    assert len(plot_chunks("\n".join([short] * 50), words=60, limit=20)) == 20


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload, self.status_code = payload, status_code

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, extracts):
        self.headers = {}
        self.extracts = extracts
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(url)
        if "wikidata" in url:
            rows = [
                {"tmdb": {"value": "120"}, "lang": {"value": "ru"}, "title": {"value": "Братство Кольца"}},
                {"tmdb": {"value": "550"}, "lang": {"value": "en"}, "title": {"value": "Fight Club"}},
            ]
            return FakeResponse({"results": {"bindings": rows}})
        page = {"title": params["titles"], "extract": self.extracts.get(params["titles"], "")}
        return FakeResponse({"query": {"pages": [page]}})


def test_plots_by_tmdb_id_with_english_fallback_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(wiki, "PAUSE_SECONDS", 0)
    session = FakeSession(
        {
            "Братство Кольца": f"== Сюжет ==\nМория. {WORDS}\n",
            "Fight Club": f"== Plot ==\nThe narrator. {WORDS}\n",
        }
    )
    plots = Wikipedia(tmp_path, session).plots([120, 550, 999])
    assert plots[120] == Plot("ru", "Братство Кольца", f"Мория. {WORDS}")
    assert plots[550].language == "en"
    assert plots[999] is None
    assert "DejaView" in session.headers["User-Agent"]
    assert json.loads((tmp_path / "999.json").read_text()) == {}

    again = FakeSession({})
    assert Wikipedia(tmp_path, again).plots([120, 550, 999]) == plots
    assert again.calls == []  # всё из кеша


def test_wikipedia_errors_stop_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(wiki, "PAUSE_SECONDS", 0)
    monkeypatch.setattr(wiki.time, "sleep", lambda _: None)

    class Down(FakeSession):
        def get(self, url, params, timeout):
            return FakeResponse({}, status_code=503)

    with pytest.raises(wiki.WikiError, match="HTTP 503"):
        Wikipedia(tmp_path, Down({})).plots([120])
    assert not (tmp_path / "120.json").exists()


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "payload", [{"error": {"code": "maxlag", "info": "Lagged"}}, {"errors": [{"code": "internal_api_error"}]}]
)
def test_http_200_api_error_does_not_cache_missing_plot(tmp_path, monkeypatch, cached, payload):
    monkeypatch.setattr(wiki.time, "sleep", lambda _: None)
    path = tmp_path / "120.json"
    previous = json.dumps({"language": "ru", "title": "Братство Кольца", "text": WORDS})
    if cached:
        path.write_text(previous)

    class ApiError(FakeSession):
        def get(self, url, params, timeout):
            if "wikidata" in url:
                return super().get(url, params, timeout)
            return FakeResponse(payload)

    with pytest.raises(wiki.WikiError, match="Ошибка API Википедии"):
        Wikipedia(tmp_path, ApiError({}), refresh=True).plots([120])
    if cached:
        assert path.read_text() == previous
    else:
        assert not path.exists()
