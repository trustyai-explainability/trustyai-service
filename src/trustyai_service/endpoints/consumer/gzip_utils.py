"""Gzip magic-byte detection for CloudEvent payloads.

Knative Eventing reconstructs HTTP requests via the CloudEvents SDK,
which drops transport headers like Content-Encoding while leaving the
body gzip-compressed. This module detects gzip by magic bytes (0x1F 0x8B)
and decompresses at the application layer — the same fix applied to the
Java service's CloudEventConsumer.decompressIfGzip().
"""

import gzip
import logging
from io import BytesIO

from trustyai_service.middleware.gzip_middleware import GzipRequestMiddleware

logger = logging.getLogger(__name__)

_GZIP_MAGIC = b"\x1f\x8b"
_GZIP_MAGIC_LEN = len(_GZIP_MAGIC)
_CHUNK_SIZE = 64 * 1024  # 64KB streaming chunks
DEFAULT_MAX_DECOMPRESSED_SIZE = GzipRequestMiddleware.DEFAULT_MAX_SIZE


def decompress_if_gzip(
    data: bytes,
    max_size: int = DEFAULT_MAX_DECOMPRESSED_SIZE,
) -> bytes:
    """Decompress data if it starts with gzip magic bytes.

    Returns the original data unchanged if it is not gzip-compressed
    or if decompression fails.

    :param data: Raw bytes to check and potentially decompress
    :param max_size: Maximum allowed decompressed size in bytes
    :return: Decompressed bytes, or original data if not gzip
    """
    if len(data) < _GZIP_MAGIC_LEN or data[:_GZIP_MAGIC_LEN] != _GZIP_MAGIC:
        return data

    try:
        decompressed = bytearray()
        with BytesIO(data) as bio, gzip.GzipFile(fileobj=bio) as gz:
            while True:
                chunk = gz.read(_CHUNK_SIZE)
                if not chunk:
                    break
                if len(decompressed) + len(chunk) > max_size:
                    msg = f"Decompressed CloudEvent payload exceeds {max_size} bytes"
                    raise ValueError(msg)
                decompressed.extend(chunk)

        logger.debug("Decompressed gzip CloudEvent payload")
        return bytes(decompressed)

    except (gzip.BadGzipFile, OSError):
        logger.warning(
            "CloudEvent payload starts with gzip magic bytes but failed to decompress, using raw bytes",
        )
        return data
