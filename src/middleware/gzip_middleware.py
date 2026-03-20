"""
Gzip decompression middleware for handling gzip-compressed request bodies.

This middleware addresses the issue where KServe agent in RawDeployment mode
sends gzip-compressed CloudEvent payloads. The Go HTTP client automatically
enables compression, but FastAPI doesn't decompress request bodies by default.
"""

import gzip
import logging
from typing import Awaitable, Callable

from fastapi import Response
from fastapi.responses import PlainTextResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


class GzipRequestMiddleware:
    """
    ASGI middleware to transparently decompress gzip-encoded request bodies for data upload endpoints.

    This middleware is scoped to the /data/upload endpoint to avoid unexpected behavior for other consumers.

    When a request contains 'Content-Encoding: gzip' header, this middleware:
    1. Decompresses the request body using gzip
    2. Removes only "gzip" from the Content-Encoding header (preserving other encodings)
    3. Updates the Content-Length header
    4. Passes the decompressed body to the endpoint handler

    The middleware handles multiple/stacked encodings (e.g., "gzip, br") by:
    1. Detecting if "gzip" is present in the Content-Encoding header
    2. Decompressing the gzip layer
    3. Removing only "gzip" from the header, preserving other encodings for downstream processing

    If decompression fails, returns HTTP 400 (Bad Request) with a clear error message rather than
    allowing the error to surface as a generic 500 error.

    This ensures compatibility with KServe agent's automatic compression while
    maintaining backward compatibility with uncompressed requests.
    """

    DATA_UPLOAD_PATH = "/data/upload"
    DECOMPRESSION_ERROR_MESSAGE = "Request body could not be decompressed as gzip: invalid or corrupted content."

    def __init__(self, app: ASGIApp) -> None:
        """
        Initialize the middleware.

        Args:
            app: The ASGI application
        """
        self.app = app

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

        # Only apply filter to /data/upload endpoint to avoid unexpected behavior for other consumers
        path = scope.get("path", "")
        if not path.endswith(self.DATA_UPLOAD_PATH):
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

        # Gzip-encoded request detected
        logger.debug(f"Detected gzip-compressed request to {path}")

        try:
            # Collect the entire request body
            # Note: This implementation buffers the full compressed body into memory before decompression.
            # For most CloudEvent payloads from KServe, this is acceptable. If very large uploads (>100MB)
            # become common, consider implementing a streaming/chunked decompression approach.
            body_parts = []
            while True:
                message = await receive()
                if message["type"] == "http.request":
                    body = message.get("body", b"")
                    if body:
                        body_parts.append(body)
                    if not message.get("more_body", False):
                        break

            # Decompress the body
            compressed_body = b"".join(body_parts)
            decompressed_body = gzip.decompress(compressed_body)

            logger.debug(
                f"Successfully decompressed gzip request: "
                f"{len(compressed_body)} bytes -> {len(decompressed_body)} bytes"
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

        except gzip.BadGzipFile as e:
            logger.error(f"Failed to decompress gzip request: {e}", exc_info=True)
            # Return 400 Bad Request with clear message instead of letting it surface as 500
            await self._send_error_response(send, 400, self.DECOMPRESSION_ERROR_MESSAGE)
        except Exception as e:
            logger.error(f"Failed to decompress gzip request: {e}", exc_info=True)
            # Return 400 Bad Request for any decompression errors
            await self._send_error_response(send, 400, self.DECOMPRESSION_ERROR_MESSAGE)

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
