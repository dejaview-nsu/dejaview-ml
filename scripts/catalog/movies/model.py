"""Запись каталога: строка таблицы movies, dejaview-docs/contracts/db-schema.md, раздел «Каталог фильмов».

Имена полей совпадают с колонками и ключами элементов JSONB, в JSON запись переводит dataclasses.asdict.
"""

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum


class CrewRole(StrEnum):
    DIRECTOR = "director"
    WRITER = "writer"
    COMPOSER = "composer"
    PRODUCER = "producer"


@dataclass(frozen=True)
class Genre:
    id: int
    name: str


@dataclass(frozen=True)
class Country:
    code: str
    name: str


@dataclass(frozen=True)
class CrewMember:
    person_id: int
    name: str
    role: CrewRole


@dataclass(frozen=True)
class Actor:
    person_id: int
    name: str
    character: str | None
    profile_path: str | None


@dataclass(frozen=True)
class Movie:
    movie_id: int
    title: str
    original_title: str
    release_date: date | None
    age_rating: str | None
    poster_path: str
    runtime_min: int | None
    overview: str
    genres: list[Genre]
    countries: list[Country]
    crew: list[CrewMember]
    actors: list[Actor]
    cached_at: datetime
