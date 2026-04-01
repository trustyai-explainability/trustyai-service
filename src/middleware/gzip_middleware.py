"""
Gzip decompression middleware for handling gzip-compressed request bodies.

This middleware addresses the issue where KServe agent in RawDeployment mode
sends gzip-compressed CloudEvent payloads. The Go HTTP client automatically
enables compression, but FastAPI doesn't decompress request bodies by default.

Configuration:
    paths: List of path patterns to apply decompression (supports wildcards)
    max_size: Maximum decompressed size (protection against decompression bombs)
    fail_on_error: Return HTTP error on decompression error (True) or pass through (False)
    allowed_content_types: List of content types eligible for decompression
    enable_metrics: Enable Prometheus metrics collection

Example:
    app.add_middleware(
        GzipRequestMiddleware,
        paths=["/data/*", "/consumer/*"],
        max_size=32 * 1024 * 1024,  # 32MB
    )
"""

import gzip
import logging
from fnmatch import fnmatch
from io import BytesIO

from prometheus_client import Counter, Histogram
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


class GzipRequestMiddleware:
    """
    ASGI middleware to transparently decompress gzip-encoded request bodies.

    This middleware addresses scenarios where clients automatically compress
    requests (e.g., KServe agent's Go HTTP client) but the web framework
    doesn't decompress them by default.

    When a request contains 'Content-Encoding: gzip' header, this middleware:
    1. Decompresses the request body using gzip
    2. Removes only "gzip" from the Content-Encoding header (preserving other encodings)
    3. Updates the Content-Length header
    4. Passes the decompressed body to the endpoint handler

    The middleware handles multiple/stacked encodings (e.g., "gzip, br") by:
    1. Detecting if "gzip" is present in the Content-Encoding header
    2. Decompressing the gzip layer
    3. Removing only "gzip" from the header, preserving other encodings for downstream processing

    Configuration:
        paths: List of path patterns to apply decompression (supports wildcards)
        max_size: Maximum decompressed size (protection against decompression bombs)
        fail_on_error: Return HTTP error on decompression error (True) or pass through (False)
        allowed_content_types: List of content types eligible for decompression
        enable_metrics: Enable Prometheus metrics collection

    Example:
        app.add_middleware(
            GzipRequestMiddleware,
            paths=["/data/*", "/consumer/*"],
            max_size=32 * 1024 * 1024,  # 32MB
        )
    """

    # Default configuration constants
    DATA_UPLOAD_PATH = "/data/upload"
    DEFAULT_PATHS = (DATA_UPLOAD_PATH,)  # Tuple to avoid mutable default
    DEFAULT_ALLOWED_CONTENT_TYPES = (
        "application/json",
        "application/cloudevents+json",
    )

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
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )

    DECOMPRESSION_ERROR_MESSAGE = (
        "Request body could not be decompressed as gzip: "
        "invalid or corrupted content."
    )

    def __init__(
        self,
        app: ASGIApp,
        paths: tuple[str, ...] | list[str] = DEFAULT_PATHS,
        max_size: int = 16 * 1024 * 1024,  # 16MB default
        fail_on_error: bool = True,
        *,
        allowed_content_types: tuple[str, ...] | list[str] = DEFAULT_ALLOWED_CONTENT_TYPES,
        enable_metrics: bool = True,
    ) -> None:
        """
        Initialize the middleware with configuration.

        Args:
            app: The ASGI application
            paths: Path patterns to apply decompression (supports wildcards like /data/*)
            max_size: Maximum decompressed size in bytes (default: 16MB)
            fail_on_error: Return HTTP error on decompression failure (default: True)
            allowed_content_types: Content types eligible for decompression
            enable_metrics: Enable Prometheus metrics (default: True)
        """
        self.app = app
        self.paths = list(paths)
        self.max_size = max_size
        self.fail_on_error = fail_on_error
        self.allowed_content_types = list(allowed_content_types)
        self.enable_metrics = enable_metrics

    def _should_process_path(self, path: str) -> bool:
        """Check if path matches any configured pattern."""
        return any(fnmatch(path, pattern) for pattern in self.paths)

    def _should_process_content_type(self, content_type: str) -> bool:
        """Check if content type is eligible for decompression."""
        if "*/*" in self.allowed_content_types:
            return True

        # Extract base content type (ignore charset, etc.)
        base_type = content_type.split(";")[0].strip()

        return any(
            fnmatch(base_type, allowed)
            for allowed in self.allowed_content_types
        )

    async def _decompress_body(
        self,
        body_parts: list[bytes],
        max_size: int,
    ) -> bytes:
        """
        Decompress gzip body with size limit protection.

        Args:
            body_parts: List of body chunks
            max_size: Maximum decompressed size

        Returns:
            Decompressed body bytes

        Raises:
            gzip.BadGzipFile: Invalid gzip data
            ValueError: Decompressed size exceeds max_size
        """
        compressed = b"".join(body_parts)

        # Stream decompress with size checking
        decompressor = gzip.GzipFile(fileobj=BytesIO(compressed))
        decompressed = bytearray()
        chunk_size = 64 * 1024  # 64KB chunks

        while True:
            chunk = decompressor.read(chunk_size)
            if not chunk:
                break

            decompressed.extend(chunk)

            # Protection against decompression bombs
            if len(decompressed) > max_size:
                raise ValueError(
                    f"Decompressed size exceeds limit of {max_size} bytes "
                    f"(potential decompression bomb)"
                )

        return bytes(decompressed)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        Process the ASGI request.

        Args:
            scope: The ASGI connection scope
            receive: The receive channel
            send: The send channel
        """
        if scope["type"] != "http":
            # Not an HTTP request, pass through
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Check if path should be processed
        if not self._should_process_path(path):
            await self.app(scope, receive, send)
            return

        # Check for gzip content encoding
        # Support multiple/stacked encodings (e.g., "gzip, br") by checking if gzip is present
        headers = Headers(scope=scope)
        content_encoding = headers.get("content-encoding", "")

        if not content_encoding or "gzip" not in content_encoding.lower():
            # No gzip encoding, pass through unchanged
            await self.app(scope, receive, send)
            return

        # Check content type
        content_type = headers.get("content-type", "application/octet-stream")
        if not self._should_process_content_type(content_type):
            logger.debug(
                f"Skipping gzip decompression for {path}: "
                f"content-type {content_type} not in allowed list"
            )
            await self.app(scope, receive, send)
            return

        # Gzip-encoded request detected
        logger.debug(f"Processing gzip-compressed request to {path}")

        try:
            # Collect the entire request body
            body_parts = []
            while True:
                message = await receive()
                if message["type"] == "http.request":
                    body = message.get("body", b"")
                    if body:
                        body_parts.append(body)
                    if not message.get("more_body", False):
                        break

            # Decompress with size limit protection
            compressed_body = b"".join(body_parts)
            decompressed_body = await self._decompress_body(
                body_parts,
                self.max_size,
            )

            # Calculate compression ratio
            ratio = len(compressed_body) / len(decompressed_body) if decompressed_body else 0

            # Record metrics
            if self.enable_metrics:
                self.REQUESTS_DECOMPRESSED.labels(endpoint=path).inc()
                self.COMPRESSION_RATIO.labels(endpoint=path).observe(ratio)

            logger.debug(
                f"Decompressed {path}: {len(compressed_body)} → "
                f"{len(decompressed_body)} bytes "
                f"(ratio: {ratio:.2%})"
            )

            # Update headers in scope - remove only "gzip" from Content-Encoding
            new_headers = []

            for name, value in scope["headers"]:
                if name.lower() == b"content-encoding":
                    # Remove only "gzip" from Content-Encoding, preserving other encodings
                    remaining_encodings = self._remove_gzip_encoding(value.decode())
                    if remaining_encodings:
                        new_headers.append((b"content-encoding", remaining_encodings.encode()))
                # Update Content-Length header
                elif name.lower() == b"content-length":
                    new_headers.append((b"content-length", str(len(decompressed_body)).encode()))
                else:
                    new_headers.append((name, value))

            # If Content-Length wasn't present, add it
            if not any(name.lower() == b"content-length" for name, _ in scope["headers"]):
                new_headers.append((b"content-length", str(len(decompressed_body)).encode()))

            scope["headers"] = new_headers

            # Create a new receive function that returns the decompressed body
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
                # Subsequent calls return empty body
                return {"type": "http.request", "body": b"", "more_body": False}

            # Call the app with the modified scope and receive function
            await self.app(scope, receive_decompressed, send)

        except ValueError as e:
            # Decompression bomb or size limit exceeded
            if self.enable_metrics:
                self.DECOMPRESSION_ERRORS.labels(
                    endpoint=path,
                    error_type="size_limit",
                ).inc()

            logger.error(f"Decompression size limit exceeded for {path}: {e}")

            if self.fail_on_error:
                await self._send_error_response(send, 413, str(e))
            else:
                await self.app(scope, receive, send)

        except gzip.BadGzipFile as e:
            if self.enable_metrics:
                self.DECOMPRESSION_ERRORS.labels(
                    endpoint=path,
                    error_type="bad_gzip",
                ).inc()

            logger.exception(f"Invalid gzip data for {path}")

            if self.fail_on_error:
                await self._send_error_response(
                    send,
                    400,
                    self.DECOMPRESSION_ERROR_MESSAGE,
                )
            else:
                await self.app(scope, receive, send)

        except Exception as e:
            if self.enable_metrics:
                self.DECOMPRESSION_ERRORS.labels(
                    endpoint=path,
                    error_type="unknown",
                ).inc()

            logger.exception(f"Unexpected error decompressing {path}")

            if self.fail_on_error:
                await self._send_error_response(send, 500, "Internal server error")
            else:
                await self.app(scope, receive, send)

    def _remove_gzip_encoding(self, content_encoding: str) -> str:
        """
        Removes "gzip" from the Content-Encoding header while preserving other encodings.
        For example: "gzip, br" becomes "br", and "gzip" is removed entirely.

        Args:
            content_encoding: The original Content-Encoding header value

        Returns:
            The updated Content-Encoding value without "gzip", or empty string if no encodings remain
        """
        encodings = [enc.strip() for enc in content_encoding.split(",")]
        remaining = [enc for enc in encodings if enc.lower() != "gzip"]
        return ", ".join(remaining)

    async def _send_error_response(self, send: Send, status_code: int, message: str) -> None:
        """
        Send an HTTP error response.

        Args:
            send: The ASGI send channel
            status_code: HTTP status code
            message: Error message to send in the response body
        """
        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"text/plain"),
                (b"content-length", str(len(message.encode())).encode()),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": message.encode(),
        })
