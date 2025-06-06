class DataframeCreateException(Exception):
    """Exception raised when a dataframe cannot be created."""

    pass


class StorageReadException(Exception):
    """Exception raised when storage cannot be read."""

    pass


class InvalidSchemaException(Exception):
    """Exception raised when a schema is invalid."""

    pass
