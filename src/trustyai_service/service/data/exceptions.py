"""Custom exceptions for data storage and manipulation operations."""


class DataframeCreateError(Exception):
    """Exception raised when a dataframe cannot be created."""


class StorageReadError(Exception):
    """Exception raised when storage cannot be read."""


class InvalidSchemaError(Exception):
    """Exception raised when a schema is invalid."""
