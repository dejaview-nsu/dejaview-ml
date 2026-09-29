"""Образцы ответов TMDB для тестов, по форме как настоящий /movie/{id}?append_to_response=..."""

import copy
from typing import Any

FIGHT_CLUB: dict[str, Any] = {
    "id": 550,
    "adult": False,
    "title": "Бойцовский клуб",
    "original_title": "Fight Club",
    "original_language": "en",
    "overview": "Сотрудник страховой компании страдает хронической бессонницей.",
    "poster_path": "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
    "release_date": "1999-10-15",
    "runtime": 139,
    "genres": [{"id": 18, "name": "драма"}, {"id": 53, "name": "триллер"}],
    "production_countries": [
        {"iso_3166_1": "US", "name": "United States of America"},
        {"iso_3166_1": "DE", "name": "Germany"},
    ],
    "credits": {
        "cast": [
            {"id": 287, "name": "Brad Pitt", "character": "Tyler Durden", "order": 1},
            {"id": 819, "name": "Edward Norton", "character": "The Narrator", "order": 0},
        ],
        "crew": [
            {"id": 7467, "name": "David Fincher", "job": "Director"},
            {"id": 7468, "name": "Jim Uhls", "job": "Screenplay"},
            {"id": 7469, "name": "Chuck Palahniuk", "job": "Novel"},
            {"id": 1060, "name": "Dust Brothers", "job": "Original Music Composer"},
            {"id": 7474, "name": "Art Linson", "job": "Producer"},
            {"id": 7475, "name": "Arnon Milchan", "job": "Executive Producer"},
        ],
    },
    "release_dates": {
        "results": [
            {"iso_3166_1": "US", "release_dates": [{"certification": "R", "type": 3}]},
            {
                "iso_3166_1": "RU",
                "release_dates": [
                    {"certification": "", "type": 4},
                    {"certification": "18+", "type": 3},
                ],
            },
        ]
    },
    "translations": {
        "translations": [
            {"iso_3166_1": "US", "iso_639_1": "en", "data": {"title": "", "overview": "A ticking-time-bomb..."}},
            {"iso_3166_1": "RU", "iso_639_1": "ru", "data": {"title": "Бойцовский клуб", "overview": "..."}},
        ]
    },
}


def movie_details(movie_id: int = 550, **overrides: Any) -> dict[str, Any]:
    details = copy.deepcopy(FIGHT_CLUB)
    details["id"] = movie_id
    details.update(overrides)
    return details


def discover_item(movie_id: int, **overrides: Any) -> dict[str, Any]:
    item = {"id": movie_id, "adult": False, "overview": "Описание", "poster_path": f"/{movie_id}.jpg"}
    item.update(overrides)
    return item
