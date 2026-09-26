"""Запись каталога: поля карточки фильма из #17193."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Person:
    person_id: int
    name: str


@dataclass(frozen=True)
class Actor(Person):
    character: str | None


@dataclass(frozen=True)
class Movie:
    movie_id: int
    title_ru: str
    original_title: str
    release_date: date | None
    age_rating: str | None
    poster_url: str
    overview_ru: str
    runtime_minutes: int | None
    country_codes: list[str]
    genres: list[str]
    directors: list[Person]
    writers: list[Person]
    composers: list[Person]
    producers: list[Person]
    actors: list[Actor]
