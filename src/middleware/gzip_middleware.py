"""
Gzip decompression middleware for KServe agent uploads.

KServe agent's Go HTTP client automatically compresses requests, but FastAPI
doesn't decompress them by default. This middleware handles gzip-encoded
request bodies with decompression bomb protection.
"""

import gzip
import logging
from fnmatch import fnmatch
from http import HTTPStatus
from io import BytesIO

from prometheus_client import Counter, Histogram
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Chunk size for streaming decompression (64KB)
_DECOMPRESSION_CHUNK_SIZE = 64 * 1024

# HTTP status code 413 - use modern name (Python 3.13+) with fallback to legacy name
# Python 3.13+ uses CONTENT_TOO_LARGE (RFC 9110)
# Python 3.11-3.12 only has REQUEST_ENTITY_TOO_LARGE (deprecated but available)
_HTTP_413 = getattr(HTTPStatus, "CONTENT_TOO_LARGE", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)


class GzipRequestMiddleware:
    """
    ASGI middleware to decompress gzip-encoded request bodies.

    Processes requests with 'Content-Encoding: gzip' by decompressing the body,
    removing the Content-Encoding header, and updating Content-Length.
    Includes protection against decompression bombs via max_size limit.

    Defaults: paths=["/data/upload"], max_size=16MB, fail_on_error=True
    """

    # Default configuration constants
    DEFAULT_PATHS = ("/data/upload",)  # Tuple to avoid mutable default
    DEFAULT_ALLOWED_CONTENT_TYPES = (
        "application/json",
        "application/cloudevents+json",
    )
    DEFAULT_MAX_SIZE = 16 * 1024 * 1024  # 16MB

    # Prometheus metrics (class-level, shared across instances)
    REQUESTS_DECOMPRESSED = Counter(
        "gzip_requests_decompressed_total",
        "Total gzip-compressed requests decompressed",
        ["endpoint"],
    )

    DECOMPRESSION_ERRORS = Counter(
        "gzip_decompression_errors_total",
        "Total gzip decompression errors",
        ["endpoint", "error_type"],
    )

    COMPRESSION_RATIO = Histogram(
        "gzip_compression_ratio",
        "Compression ratio distribution",
        ["endpoint"],
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 5.0, 10.0],
    )

    DECOMPRESSION_ERROR_MESSAGE = "Request body could not be decompressed as gzip: invalid or corrupted content."

    def __init__(
        self,
        app: ASGIApp,
        paths: tuple[str, ...] | list[str] = DEFAULT_PATHS,
        max_size: int = DEFAULT_MAX_SIZE,
        fail_on_error: bool = True,
        *,
        allowed_content_types: tuple[str, ...] | list[str] = DEFAULT_ALLOWED_CONTENT_TYPES,
        enable_metrics: bool = True,
    ) -> None:
        """
        Initialize gzip decompression middleware.

        Args:
            app: ASGI application
            paths: Path patterns to apply (supports wildcards, default: ["/data/upload"])
            max_size: Max decompressed bytes (default: 16MB)
            fail_on_error: Return error on failure vs pass through (default: True)
            allowed_content_types: Eligible content types
            enable_metrics: Enable Prometheus metrics (default: True)
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")

        self.app = app
        self.paths = tuple(paths)
        self.max_size = max_size
        self.fail_on_error = fail_on_error
        self.allowed_content_types = tuple(allowed_content_types)
        self.enable_metrics = enable_metrics

    def _should_process_path(self, path: str) -> bool:
        """Check if path matches any configured pattern."""
        return any(fnmatch(path, pattern) for pattern in self.paths)

    def _should_process_content_type(self, content_type: str) -> bool:
        """Check if content type is eligible for decompression."""
        if "*/*" in self.allowed_content_types:
            return True

        base_type = content_type.split(";")[0].strip()
        return any(fnmatch(base_type, allowed) for allowed in self.allowed_content_types)

    def _decompress_body(
        self,
        body_parts: list[bytes],
        max_size: int,
    ) -> tuple[bytes, bytes]:
        """
        Decompress gzip body with size limit protection.

        Returns: (decompressed_body, compressed_body)
        Raises: gzip.BadGzipFile or ValueError
        """
        compressed = b"".join(body_parts)

        if not compressed:
            raise ValueError("Request body is empty")

        decompressed = bytearray()

        with BytesIO(compressed) as bio:
            with gzip.GzipFile(fileobj=bio) as decompressor:
                while True:
                    chunk = decompressor.read(_DECOMPRESSION_CHUNK_SIZE)
                    if not chunk:
                        break

                    if len(decompressed) + len(chunk) > max_size:
                        raise ValueError(
                            f"Decompressed size exceeds limit of {max_size} bytes (potential decompression bomb)"
                        )

                    decompressed.extend(chunk)

        return bytes(decompressed), compressed

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process ASGI request, decompressing gzip if applicable."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]

        if not self._should_process_path(path):
            await self.app(scope, receive, send)
            return

        headers_list = scope["headers"]
        content_encoding = next((v.decode("latin1") for k, v in headers_list if k.lower() == b"content-encoding"), None)

        if not content_encoding:
            await self.app(scope, receive, send)
            return

        encodings = [e.strip().lower() for e in content_encoding.split(",")]
        if "gzip" not in encodings:
            await self.app(scope, receive, send)
            return

        # RFC 7230/9110: HTTP headers are Latin-1 encoded
        content_type = next((v.decode("latin1") for k, v in headers_list if k.lower() == b"content-type"), None)

        if not content_type:
            logger.debug(f"Skipping gzip decompression for {path}: no content-type header")
            await self.app(scope, receive, send)
            return

        if not self._should_process_content_type(content_type):
            logger.debug(f"Skipping gzip decompression for {path}: content-type {content_type} not in allowed list")
            await self.app(scope, receive, send)
            return

        logger.debug(f"Processing gzip-compressed request to {path}")

        body_parts = []
        try:
            while True:
                message = await receive()
                if message["type"] == "http.request":
                    body_parts.append(message.get("body", b""))
                    if not message.get("more_body", False):
                        break
        except (OSError, EOFError, RuntimeError) as e:
            logger.exception(f"Failed to read request body for {path}")
            await self._send_error_response(send, HTTPStatus.INTERNAL_SERVER_ERROR, "Failed to read request body")
            return

        compressed_body = None

        try:
            decompressed_body, compressed_body = self._decompress_body(body_parts, self.max_size)

            compressed_size = len(compressed_body)
            decompressed_size = len(decompressed_body)
            ratio = compressed_size / decompressed_size if decompressed_size > 0 else 0

            if self.enable_metrics:
                self.REQUESTS_DECOMPRESSED.labels(endpoint=path).inc()
                self.COMPRESSION_RATIO.labels(endpoint=path).observe(ratio)

            logger.debug(f"Decompressed {path}: {compressed_size} → {decompressed_size} bytes (ratio: {ratio:.2%})")

            new_headers = []
            has_content_length = False

            for name, value in scope["headers"]:
                name_lower = name.lower()
                if name_lower == b"content-encoding":
                    continue
                elif name_lower == b"content-length":
                    has_content_length = True
                    new_headers.append((b"content-length", str(len(decompressed_body)).encode("utf-8")))
                else:
                    new_headers.append((name, value))

            if not has_content_length:
                new_headers.append((b"content-length", str(len(decompressed_body)).encode("utf-8")))

            scope["headers"] = new_headers

            body_sent = False

            async def receive_decompressed() -> Message:
                nonlocal body_sent
                if not body_sent:
                    body_sent = True
                    return {
                        "type": "http.request",
                        "body": decompressed_body,
                        "more_body": False,
                    }
                return {"type": "http.request", "body": b"", "more_body": False}

            await self.app(scope, receive_decompressed, send)

        except ValueError as e:
            error_message = str(e)
            is_empty_body = "empty" in error_message.lower()

            if is_empty_body:
                if self.enable_metrics:
                    self.DECOMPRESSION_ERRORS.labels(endpoint=path, error_type="empty_body").inc()
                logger.error(f"Empty request body for {path}")

                if self.fail_on_error:
                    await self._send_error_response(send, HTTPStatus.BAD_REQUEST, error_message)
                else:
                    raw_body = compressed_body if compressed_body is not None else b"".join(body_parts)
                    await self._passthrough_with_body(scope, raw_body, send)
            else:
                if self.enable_metrics:
                    self.DECOMPRESSION_ERRORS.labels(endpoint=path, error_type="size_limit").inc()
                logger.error(f"Decompression size limit exceeded for {path}: {e}")

                if self.fail_on_error:
                    await self._send_error_response(send, _HTTP_413, error_message)
                else:
                    raw_body = compressed_body if compressed_body is not None else b"".join(body_parts)
                    await self._passthrough_with_body(scope, raw_body, send)

        except gzip.BadGzipFile as e:
            if self.enable_metrics:
                self.DECOMPRESSION_ERRORS.labels(endpoint=path, error_type="bad_gzip").inc()
            logger.exception(f"Invalid gzip data for {path}")

            if self.fail_on_error:
                await self._send_error_response(send, HTTPStatus.BAD_REQUEST, self.DECOMPRESSION_ERROR_MESSAGE)
            else:
                raw_body = compressed_body if compressed_body is not None else b"".join(body_parts)
                await self._passthrough_with_body(scope, raw_body, send)

        except (OSError, EOFError, RuntimeError) as e:
            if self.enable_metrics:
                self.DECOMPRESSION_ERRORS.labels(endpoint=path, error_type="unknown").inc()
            logger.exception(f"Unexpected error decompressing {path}")

            if self.fail_on_error:
                await self._send_error_response(send, HTTPStatus.INTERNAL_SERVER_ERROR, "Internal Server Error")
            else:
                raw_body = compressed_body if compressed_body is not None else b"".join(body_parts)
                await self._passthrough_with_body(scope, raw_body, send)

    async def _send_error_response(self, send: Send, status_code: HTTPStatus | int, message: str) -> None:
        """Send HTTP error response."""
        body = message.encode("utf-8")
        status = status_code.value if isinstance(status_code, HTTPStatus) else status_code

        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"text/plain"),
                (b"content-length", str(len(body)).encode("utf-8")),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })

    async def _passthrough_with_body(self, scope: Scope, body: bytes, send: Send) -> None:
        """Pass request through with original body when fail_on_error=False."""
        body_sent = False

        async def receive_raw() -> Message:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, receive_raw, send)
