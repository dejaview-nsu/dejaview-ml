class TmdbError(Exception):
    pass


class TmdbAuthError(TmdbError):
    """Ключ неверный или отозван: повторять запросы бессмысленно."""


class TmdbNotFoundError(TmdbError):
    pass


class TmdbUnavailableError(TmdbError):
    """TMDB не ответил успешно ни на одну из попыток."""


class TmdbRequestError(TmdbError):
    """TMDB отклонил запрос или прислал ответ неожиданного формата."""
