class IndexingError(Exception):
    pass


class ConfigError(IndexingError):
    pass


class UnsuitableFileError(IndexingError):
    pass
