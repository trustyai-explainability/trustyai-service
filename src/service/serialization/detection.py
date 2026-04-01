"""
Serialization format detection via magic bytes.

Identifies the format of serialized data to enable safe deserialization.
"""

# Gzip magic bytes (RFC 1952)
GZIP_MAGIC = b"\x1f\x8b"


def detect_format(data: bytes) -> str:
    """
    Detect serialization format from magic bytes.

    Args:
        data: Serialized data bytes

    Returns:
        Format identifier: "gzip" or "json"

    Raises:
        ValueError: If format cannot be determined

    Examples:
        >>> detect_format(b'\\x1f\\x8b...')  # gzip header
        'gzip'
        >>> detect_format(b'{"key": "value"}')
        'json'
    """
    if not data:
        raise ValueError("Cannot detect format of empty data")

    # Check for gzip format
    if data.startswith(GZIP_MAGIC):
        return "gzip"

    # Check if it looks like JSON
    if is_json(data):
        return "json"

    raise ValueError(f"Unknown serialization format (first bytes: {data[:10]!r})")


def is_gzip(data: bytes) -> bool:
    """Check if data is gzip format."""
    return data.startswith(GZIP_MAGIC)


def is_json(data: bytes) -> bool:
    """
    Check if data looks like JSON format.

    Checks for common JSON starting characters:
    - Objects: {
    - Arrays: [
    - Strings: "
    - Numbers: digits or -
    - Booleans: t (true) or f (false)
    - Null: n (null)
    """
    if not data:
        return False

    first_char = data[0:1]
    return (
        first_char in (b"{", b"[", b'"')
        or first_char.isdigit()
        or first_char == b"-"
        or data.startswith((b"true", b"false", b"null"))
    )
