"""Serialization format detection and safe decompression.

Identifies the format of serialized data to enable safe deserialization,
and provides bounded decompression to prevent decompression bombs.
"""

import gzip
import io

# Gzip magic bytes (RFC 1952)
GZIP_MAGIC = b"\x1f\x8b"

# 128 MiB decompressed ceiling — generous for legitimate payloads,
# prevents a small gzip bomb from exhausting memory.
MAX_DECOMPRESSED_SIZE = 128 * 1024 * 1024


def detect_format(data: bytes) -> str:
    r"""Detect serialization format from magic bytes.

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
        msg = "Cannot detect format of empty data"
        raise ValueError(msg)

    # Check for gzip format
    if data.startswith(GZIP_MAGIC):
        return "gzip"

    # Check if it looks like JSON
    if is_json(data):
        return "json"

    msg = f"Unknown serialization format (first bytes: {data[:10]!r})"
    raise ValueError(msg)


def is_gzip(data: bytes) -> bool:
    """Check if data is gzip format."""
    return data.startswith(GZIP_MAGIC)


def is_json(data: bytes) -> bool:
    """Check if data looks like JSON format.

    Checks for common JSON starting characters:
    - Objects: {
    - Arrays: [
    - Strings: "
    - Numbers: digits or -
    - Booleans: t (true) or f (false)
    - Null: n (null)

    Leading whitespace is skipped as per JSON specification.
    """
    if not data:
        return False

    # Skip leading ASCII whitespace (space, tab, newline, carriage return)
    stripped = data.lstrip(b" \t\n\r")
    if not stripped:
        return False

    first_char = stripped[0:1]
    return (
        first_char in (b"{", b"[", b'"')
        or first_char.isdigit()
        or first_char == b"-"
        or stripped.startswith((b"true", b"false", b"null"))
    )


def safe_gzip_decompress(
    data: bytes,
    max_size: int = MAX_DECOMPRESSED_SIZE,
) -> bytes:
    """Decompress gzip data with a size limit to prevent decompression bombs.

    Args:
        data: Gzip-compressed bytes
        max_size: Maximum allowed decompressed size in bytes

    Returns:
        Decompressed bytes

    Raises:
        ValueError: If decompressed data exceeds max_size

    """
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
        result = f.read(max_size + 1)
    if len(result) > max_size:
        msg = (
            f"Decompressed data exceeds maximum allowed size "
            f"({max_size} bytes). Possible decompression bomb."
        )
        raise ValueError(msg)
    return result
