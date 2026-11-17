"""Framing of a text query: words that say "I am looking for a film" rather than which film.

"Помню сцену, где ... Что это за фильм?" / "I remember a movie where ... What's it called?" -
the model encodes the framing together with the meaning, and the framing alone is close to any
film's description (0.825 for "Что это за фильм?"), so it pulls the answer to random films.
strip_framing removes whole framing sentences and a framing clause at the start of a sentence.
A query that is nothing but framing, or has none, is left as it is.
"""

import re

_RU_FILM = r"(?:фильм[а-я]*|кино|кинофильм[а-я]*|картин[а-я]*|мульт(?:ик|фильм)[а-я]*|сериал[а-я]*)"
_EN_FILM = r"(?:movie|film|cartoon|series|show)s?"
_RU_SOME = (
    r"(?:(?:один|одн[ау]|одного|такой|такую|такого|этот|эту|этого|как(?:ой|ую|ого)-то"
    r"|старый|старого)\s+)?"
)
_EN_SOME = r"(?:(?:a|an|the|this|that|one|some|an old|old)\s+)?"
_PLEASE = r"(?:,?\s*пожалуйста,?)?"

# A framing clause at the start of a sentence, with the word that joins it to the description:
# "Помню сцену, где", "Ищу фильм про", "I remember a movie where", "Looking for a film about".
_RU_LEAD = (
    rf"(?:я\s+)?(?:помню|вспоминаю|запомнил[аи]?)\s+{_RU_SOME}(?:сцену|момент|эпизод|кадр|{_RU_FILM})",
    rf"(?:я\s+)?(?:ищу|хочу\s+найти|пытаюсь\s+(?:найти|вспомнить))\s+{_RU_SOME}{_RU_FILM}",
    rf"(?:подскажите|посоветуйте|напомните|скажите){_PLEASE}\s*(?:название\s+)?{_RU_SOME}{_RU_FILM}",
    rf"помогите{_PLEASE}\s*(?:найти|вспомнить|определить|узнать)\s+(?:название\s+)?{_RU_SOME}{_RU_FILM}",
    rf"не\s+(?:могу\s+)?(?:вспомнить|помню)\s+(?:название|как\s+называется)\s+{_RU_SOME}{_RU_FILM}",
    rf"(?:как|подскажите,?\s+как)\s+называется\s+{_RU_SOME}{_RU_FILM}",
    rf"(?:есть|был[аи]?)\s+{_RU_SOME}{_RU_FILM}",
    rf"{_RU_SOME}{_RU_FILM}",
)
_RU_JOIN = r"(?:где|в\s+котор(?:ом|ой|ых)|когда|про\s+то,?\s+как|о\s+том,?\s+как|про|там|как)"
_EN_LEAD = (
    rf"i\s+(?:remember|recall)\s+{_EN_SOME}(?:scene|moment|part|{_EN_FILM})",
    rf"(?:i'?m\s+|i\s+am\s+)?(?:looking\s+for|searching\s+for|trying\s+to\s+(?:find|remember))\s+{_EN_SOME}{_EN_FILM}",
    rf"(?:(?:can|could)\s+you\s+)?help\s+me\s+(?:find|identify|remember)\s+{_EN_SOME}{_EN_FILM}",
    rf"what(?:'s|\s+is)\s+the\s+(?:name|title)\s+of\s+{_EN_SOME}{_EN_FILM}",
    rf"(?:do\s+you|does\s+anyone)\s+know\s+{_EN_SOME}{_EN_FILM}",
    rf"there(?:'s|\s+is|\s+was)\s+{_EN_SOME}{_EN_FILM}",
    rf"{_EN_SOME}{_EN_FILM}",
)
_EN_JOIN = r"(?:where|in\s+which|that|about|with|when|whose)"

# A sentence that is only framing: "Что это за фильм?", "Как он называется?", "What's it called?".
_META = (
    rf"(?:что|какой)\s+это\s+(?:за\s+)?{_RU_FILM}",
    rf"что\s+за\s+{_RU_FILM}",
    rf"как\s+(?:он|она|оно|этот\s+{_RU_FILM}|{_RU_FILM})?\s*называется",
    rf"(?:подскажите|скажите|напомните){_PLEASE}\s*(?:название|как\s+называется)?",
    rf"не\s+(?:могу\s+)?(?:вспомнить|помню),?\s*(?:название|как\s+(?:он\s+)?называется)?(?:,?\s*помогите{_PLEASE})?",
    r"кто\s+(?:знает|подскажет)",
    rf"помогите{_PLEASE}(?:\s+найти)?",
    r"(?:заранее\s+)?спасибо(?:\s+заранее)?",
    rf"what\s+{_EN_FILM}\s+is\s+(?:this|it|that)",
    rf"what(?:'s|\s+is)\s+(?:this|that|the|it)\s*(?:{_EN_FILM})?\s*(?:called)?",
    r"(?:does\s+anyone|do\s+you)\s+know\s*(?:the\s+(?:name|title))?(?:\s+of\s+(?:it|this|the\s+movie))?",
    r"any\s+ideas",
    r"(?:(?:can|could)\s+you\s+|please\s+)?help(?:\s+me)?",
    r"thanks(?:\s+in\s+advance)?|thank\s+you",
    r"i\s+can'?t\s+remember\s+the\s+(?:name|title)",
)

_LEAD = re.compile(
    rf"^\s*(?:(?:{'|'.join(_RU_LEAD)})\s*,?\s*{_RU_JOIN}|(?:{'|'.join(_EN_LEAD)})\s*,?\s*{_EN_JOIN})\s+",
    re.IGNORECASE,
)
_META_SENTENCE = re.compile(rf"^\s*(?:{'|'.join(_META)})\s*[.!?…]*\s*$", re.IGNORECASE)
_SENTENCES = re.compile(r"[^.!?…]+[.!?…]*")


def strip_framing(query: str) -> str:
    """The query without framing; the query itself if nothing meaningful would be left."""
    kept, changed = [], False
    for sentence in _SENTENCES.findall(query):
        if _META_SENTENCE.match(sentence):
            changed = True
            continue
        stripped = _LEAD.sub("", sentence, count=1).strip()
        changed = changed or stripped != sentence.strip()
        if stripped.strip(" ,.!?…"):
            kept.append(stripped)
    # What is left is a bare description: e5 finds it better without the final full stop and with
    # a lowercase first letter. On 99 framed queries the right film is first in 77 instead of 59
    # without cleanup (64 with the full stop and capital kept). A query without framing is left as
    # typed: in quotes the punctuation and case are part of the line.
    result = re.sub(r"[\s.!?…]+$", "", " ".join(kept))
    if not changed or len(re.findall(r"\w", result)) < 3:
        return query
    return result[0].lower() + result[1:]
