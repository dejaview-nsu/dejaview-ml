import pytest

from dejaview_ml.query import strip_framing


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "Помню сцену, где влюблённые стоят на носу корабля. Что это за фильм?",
            "влюблённые стоят на носу корабля",
        ),
        (
            "Ищу фильм, где герою предлагают красную и синюю таблетку",
            "герою предлагают красную и синюю таблетку",
        ),
        ("Подскажите, пожалуйста, фильм про школу магии. Кто знает?", "школу магии"),
        (
            "Не могу вспомнить название фильма, там маньяк убивает по семи грехам.",
            "маньяк убивает по семи грехам",
        ),
        (
            "Есть такой фильм, где человек врезается в нарисованное небо",
            "человек врезается в нарисованное небо",
        ),
        (
            "Как называется фильм, в котором Сухов спасает гарем. Не помню, как называется, "
            "помогите!",
            "сухов спасает гарем",
        ),
        (
            "Фильм, где продюсер находит голову лошади. Как он называется?",
            "продюсер находит голову лошади",
        ),
        (
            "I remember a movie where a man ages backwards. What movie is this?",
            "a man ages backwards",
        ),
        (
            "Looking for a film about a thief who steals dreams. Any ideas?",
            "a thief who steals dreams",
        ),
        (
            "What's the name of the movie where hobbits destroy a ring? Thanks in advance",
            "hobbits destroy a ring",
        ),
        # The description keeps sentences that say something about the film.
        (
            "Фильм нулевых про школу магии. Над столами висят свечи. Что за фильм?",
            "фильм нулевых про школу магии. Над столами висят свечи",
        ),
    ],
)
def test_framing_is_removed(query, expected):
    assert strip_framing(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "Иди за белым кроликом",  # a quote
        "Что это? Что это такое?",  # a quote, not framing: no word about a film
        "Фильм режиссёра Кристофера Нолана",  # no joining word after "фильм"
        "фильм с Томом Хэнксом",
        "movie directed by Quentin Tarantino",
    ],
)
def test_content_is_kept(query):
    assert strip_framing(query) == query


@pytest.mark.parametrize("query", ["Что это за фильм?", "Кто знает?", "Help me!", "Спасибо"])
def test_query_of_framing_only_is_left_as_it_is(query):
    # Nothing would be left to search: the query goes as typed, a quote may be just that.
    assert strip_framing(query) == query
