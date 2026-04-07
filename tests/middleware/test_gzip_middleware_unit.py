"""
Unit tests for GzipRequestMiddleware.

These tests directly mock ASGI components to test middleware logic in isolation,
complementing the integration tests in test_upload_endpoint_pvc.py.
"""

import gzip
from unittest.mock import AsyncMock, patch

import pytest

from src.middleware.gzip_middleware import GzipRequestMiddleware


class TestGzipMiddlewareUnit:
    """Direct unit tests for GzipRequestMiddleware (non-integration)."""

    def make_scope(self, path="/data/upload", headers=None):
        """Create mock ASGI scope."""
        return {
            "type": "http",
            "path": path,
            "headers": headers or [],
        }

    def make_receive_with_body(self, body: bytes, chunks: int = 1):
        """Create ASGI-compliant mock receive callable that returns body in chunks."""
        chunk_size = max(1, len(body) // chunks) if body else 0
        sent = [0]  # Use list to avoid closure issues
        complete = [False]

        async def receive():
            # After body is complete, return disconnect (ASGI-compliant)
            if complete[0]:
                return {"type": "http.disconnect"}

            if sent[0] >= len(body):
                complete[0] = True
                return {"type": "http.request", "body": b"", "more_body": False}

            end = min(sent[0] + chunk_size, len(body))
            chunk = body[sent[0] : end]
            sent[0] = end
            more_body = sent[0] < len(body)

            if not more_body:
                complete[0] = True

            return {"type": "http.request", "body": chunk, "more_body": more_body}

        return receive

    # === fail_on_error=False Tests ===

    @pytest.mark.asyncio
    async def test_bad_gzip_passthrough_when_fail_on_error_false(self):
        """Bad gzip with fail_on_error=False passes through original body."""
        app = AsyncMock()
        middleware = GzipRequestMiddleware(app, fail_on_error=False)

        # Invalid gzip data - just plain text that's not gzip
        bad_gzip = b"not gzip data at all"
        scope = self.make_scope(
            headers=[
                (b"content-encoding", b"gzip"),
                (b"content-type", b"application/json"),
            ]
        )
        receive = self.make_receive_with_body(bad_gzip)
        send = AsyncMock()

        await middleware(scope, receive, send)

        # App should be called (not error response sent)
        app.assert_called_once()
        # send should not have been called with error response
        assert send.call_count == 0

        # Verify original body passed through
        receive_arg = app.call_args[0][1]
        msg = await receive_arg()
        assert msg["body"] == bad_gzip
        assert msg["more_body"] is False

    @pytest.mark.asyncio
    async def test_size_limit_passthrough_when_fail_on_error_false(self):
        """Size limit exceeded with fail_on_error=False passes through."""
        app = AsyncMock()
        # Set very small limit
        middleware = GzipRequestMiddleware(app, max_size=10, fail_on_error=False)

        # Compress data that will exceed 10 bytes when decompressed
        large_data = b"x" * 100
        compressed = gzip.compress(large_data)

        scope = self.make_scope(
            headers=[
                (b"content-encoding", b"gzip"),
                (b"content-type", b"application/json"),
            ]
        )
        receive = self.make_receive_with_body(compressed)
        send = AsyncMock()

        await middleware(scope, receive, send)

        # App should be called (not error response)
        app.assert_called_once()
        assert send.call_count == 0

        # Verify compressed body passed through
        receive_arg = app.call_args[0][1]
        msg = await receive_arg()
        assert msg["body"] == compressed

    @pytest.mark.asyncio
    async def test_empty_body_passthrough_when_fail_on_error_false(self):
        """Empty body with fail_on_error=False passes through."""
        app = AsyncMock()
        middleware = GzipRequestMiddleware(app, fail_on_error=False)

        scope = self.make_scope(
            headers=[
                (b"content-encoding", b"gzip"),
                (b"content-type", b"application/json"),
            ]
        )
        receive = self.make_receive_with_body(b"")
        send = AsyncMock()

        await middleware(scope, receive, send)

        # App should be called
        app.assert_called_once()
        assert send.call_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_during_body_read(self):
        """Client disconnect during body read sends 400 Bad Request."""
        app = AsyncMock()
        middleware = GzipRequestMiddleware(app)

        scope = self.make_scope(
            headers=[
                (b"content-encoding", b"gzip"),
                (b"content-type", b"application/json"),
            ]
        )

        # Create receive that simulates disconnect
        async def receive_with_disconnect():
            return {"type": "http.disconnect"}

        send = AsyncMock()

        await middleware(scope, receive_with_disconnect, send)

        # Should send 400 error response for disconnect
        assert send.call_count == 2  # start + body
        start_call = send.call_args_list[0][0][0]
        assert start_call["status"] == 400

    @pytest.mark.asyncio
    async def test_unexpected_error_during_body_read(self):
        """Unexpected errors during body read send 500 error."""
        app = AsyncMock()
        middleware = GzipRequestMiddleware(app)

        scope = self.make_scope(
            headers=[
                (b"content-encoding", b"gzip"),
                (b"content-type", b"application/json"),
            ]
        )

        # Create receive that raises an unexpected error during body read
        async def receive_with_error():
            raise OSError("Simulated I/O error")

        send = AsyncMock()

        await middleware(scope, receive_with_error, send)

        # Should send error response
        assert send.call_count == 2  # start + body
        start_call = send.call_args_list[0][0][0]
        assert start_call["status"] == 500

    @pytest.mark.asyncio
    async def test_unexpected_error_passthrough_when_fail_on_error_false(self):
        """Unexpected errors during decompression with fail_on_error=False pass through."""
        app = AsyncMock()
        middleware = GzipRequestMiddleware(app, fail_on_error=False)

        # Mock _decompress_body to raise OSError (unexpected error during decompression)
        def failing_decompress(body_parts, max_size):
            raise OSError("Simulated decompression I/O error")

        middleware._decompress_body = failing_decompress

        data = b'{"test": "data"}'
        compressed = gzip.compress(data)

        scope = self.make_scope(
            headers=[
                (b"content-encoding", b"gzip"),
                (b"content-type", b"application/json"),
            ]
        )
        receive = self.make_receive_with_body(compressed)
        send = AsyncMock()

        await middleware(scope, receive, send)

        # Should call app with passthrough (not send error)
        app.assert_called_once()
        assert send.call_count == 0

    @pytest.mark.asyncio
    async def test_unexpected_error_response_when_fail_on_error_true(self):
        """Unexpected errors during decompression with fail_on_error=True send error response."""
        app = AsyncMock()
        middleware = GzipRequestMiddleware(app, fail_on_error=True)

        # Create data that will cause RuntimeError during decompression
        # Mock _decompress_body to raise RuntimeError
        def failing_decompress(body_parts, max_size):
            raise RuntimeError("Simulated unexpected decompression error")

        middleware._decompress_body = failing_decompress

        data = b'{"test": "data"}'
        compressed = gzip.compress(data)

        scope = self.make_scope(
            headers=[
                (b"content-encoding", b"gzip"),
                (b"content-type", b"application/json"),
            ]
        )
        receive = self.make_receive_with_body(compressed)
        send = AsyncMock()

        await middleware(scope, receive, send)

        # Should send 500 error response
        assert send.call_count == 2  # start + body
        start_call = send.call_args_list[0][0][0]
        assert start_call["status"] == 500

    # === Path Matching Tests ===

    def test_should_process_path(self):
        """Path matching with exact, wildcard, and multiple patterns."""
        # Exact match
        m1 = GzipRequestMiddleware(None, paths=["/data/upload"])
        assert m1._should_process_path("/data/upload")
        assert not m1._should_process_path("/data/upload/sub")
        assert not m1._should_process_path("/other")

        # Wildcards
        m2 = GzipRequestMiddleware(None, paths=["/data/*", "/api/v*/upload"])
        assert m2._should_process_path("/data/upload")
        assert m2._should_process_path("/data/upload/subpath")
        assert m2._should_process_path("/api/v1/upload")
        assert not m2._should_process_path("/other/path")

        # Multiple patterns
        m3 = GzipRequestMiddleware(None, paths=["/data/upload", "/consumer/data"])
        assert m3._should_process_path("/data/upload")
        assert m3._should_process_path("/consumer/data")
        assert not m3._should_process_path("/other")

    # === Content-Type Matching Tests ===

    def test_should_process_content_type_exact_and_parameters(self):
        """Content type matching with exact matches and parameters."""
        middleware = GzipRequestMiddleware(None)
        # Exact matches
        assert middleware._should_process_content_type("application/json")
        assert middleware._should_process_content_type("application/cloudevents+json")
        assert not middleware._should_process_content_type("text/plain")
        # With parameters (charset, etc.)
        assert middleware._should_process_content_type("application/json; charset=utf-8")
        assert middleware._should_process_content_type("application/cloudevents+json; version=1.0")

    def test_should_process_content_type_patterns(self):
        """Content type wildcard and pattern matching."""
        # Wildcard allows everything
        m1 = GzipRequestMiddleware(None, allowed_content_types=["*/*"])
        assert m1._should_process_content_type("application/json")
        assert m1._should_process_content_type("text/plain")
        assert m1._should_process_content_type("image/png")

        # Pattern matching
        m2 = GzipRequestMiddleware(None, allowed_content_types=["application/*"])
        assert m2._should_process_content_type("application/json")
        assert m2._should_process_content_type("application/xml")
        assert not m2._should_process_content_type("text/plain")

    # === Decompression Tests ===

    def test_decompress_body_success(self):
        """Decompression with single chunk or multiple chunks."""
        middleware = GzipRequestMiddleware(None)

        # Single chunk
        data1 = b'{"test": "data"}'
        compressed1 = gzip.compress(data1)
        decompressed1, original1 = middleware._decompress_body([compressed1], max_size=1024)
        assert decompressed1 == data1
        assert original1 == compressed1

        # Multiple chunks
        data2 = b'{"test": "data with multiple chunks"}'
        compressed2 = gzip.compress(data2)
        chunks = [compressed2[:10], compressed2[10:20], compressed2[20:]]
        decompressed2, original2 = middleware._decompress_body(chunks, max_size=1024)
        assert decompressed2 == data2
        assert original2 == compressed2

    def test_decompress_body_empty_raises(self):
        """Empty body raises ValueError."""
        middleware = GzipRequestMiddleware(None)

        with pytest.raises(ValueError, match="empty"):
            middleware._decompress_body([], max_size=1024)

    def test_decompress_body_size_limit_raises(self):
        """Exceeding size limit raises ValueError."""
        middleware = GzipRequestMiddleware(None)
        large_data = b"x" * 1000
        compressed = gzip.compress(large_data)

        with pytest.raises(ValueError, match="exceeds limit"):
            middleware._decompress_body([compressed], max_size=100)

    def test_decompress_body_invalid_gzip_raises(self):
        """Invalid gzip data raises BadGzipFile."""
        middleware = GzipRequestMiddleware(None)
        invalid = b"not gzip data"

        with pytest.raises(gzip.BadGzipFile):
            middleware._decompress_body([invalid], max_size=1024)

    # === Metrics Tests ===

    @pytest.mark.asyncio
    async def test_metrics_disabled(self):
        """No metric calls when enable_metrics=False."""
        app = AsyncMock()

        with patch.object(GzipRequestMiddleware, 'REQUESTS_DECOMPRESSED') as mock_counter, \
             patch.object(GzipRequestMiddleware, 'COMPRESSION_RATIO') as mock_histogram:

            middleware = GzipRequestMiddleware(app, enable_metrics=False)

            data = b'{"test": "data"}'
            compressed = gzip.compress(data)

            scope = self.make_scope(
                headers=[
                    (b"content-encoding", b"gzip"),
                    (b"content-type", b"application/json"),
                ]
            )
            receive = self.make_receive_with_body(compressed)
            send = AsyncMock()

            # Should not raise even though metrics are disabled
            await middleware(scope, receive, send)
            app.assert_called_once()

            # Explicitly assert that no metric methods were invoked
            mock_counter.labels.assert_not_called()
            mock_histogram.labels.assert_not_called()

    @pytest.mark.asyncio
    async def test_metrics_error_disabled(self):
        """No error metric calls when enable_metrics=False."""
        app = AsyncMock()

        with patch.object(GzipRequestMiddleware, 'DECOMPRESSION_ERRORS') as mock_error_counter:

            middleware = GzipRequestMiddleware(app, enable_metrics=False, fail_on_error=True)

            # Invalid gzip data
            bad_gzip = b"not gzip data"

            scope = self.make_scope(
                headers=[
                    (b"content-encoding", b"gzip"),
                    (b"content-type", b"application/json"),
                ]
            )
            receive = self.make_receive_with_body(bad_gzip)
            send = AsyncMock()

            await middleware(scope, receive, send)

            # Should send error response
            assert send.call_count == 2  # start + body

            # Explicitly assert that no error metric methods were invoked
            mock_error_counter.labels.assert_not_called()

    # === ASGI Edge Cases ===

    @pytest.mark.asyncio
    async def test_non_http_request_passthrough(self):
        """Non-HTTP requests (e.g., websocket) pass through unchanged."""
        app = AsyncMock()
        middleware = GzipRequestMiddleware(app)

        scope = {"type": "websocket", "path": "/ws"}
        receive = AsyncMock()
        send = AsyncMock()

        await middleware(scope, receive, send)

        app.assert_called_once_with(scope, receive, send)
        send.assert_not_called()

    @pytest.mark.asyncio
    async def test_receive_decompressed_multiple_calls(self):
        """Subsequent calls to receive_decompressed return disconnect (ASGI-compliant)."""
        app = AsyncMock()
        middleware = GzipRequestMiddleware(app)

        data = b'{"test": "data"}'
        compressed = gzip.compress(data)

        scope = self.make_scope(
            headers=[
                (b"content-encoding", b"gzip"),
                (b"content-type", b"application/json"),
            ]
        )
        receive = self.make_receive_with_body(compressed)
        send = AsyncMock()

        await middleware(scope, receive, send)

        # Get the receive callable passed to app
        receive_arg = app.call_args[0][1]

        msg1 = await receive_arg()
        msg2 = await receive_arg()
        msg3 = await receive_arg()

        # First call returns the decompressed body
        assert len(msg1["body"]) > 0
        assert msg1["body"] == data
        assert msg1["more_body"] is False

        # Subsequent calls return disconnect (ASGI-compliant behavior)
        assert msg2["type"] == "http.disconnect"
        assert msg3["type"] == "http.disconnect"

    @pytest.mark.asyncio
    async def test_passthrough_receive_multiple_calls(self):
        """Multiple calls to passthrough receive return disconnect (ASGI-compliant)."""
        app = AsyncMock()
        middleware = GzipRequestMiddleware(app, fail_on_error=False)

        bad_gzip = b"\x1f\x8b\x08\x00"

        scope = self.make_scope(
            headers=[
                (b"content-encoding", b"gzip"),
                (b"content-type", b"application/json"),
            ]
        )
        receive = self.make_receive_with_body(bad_gzip)
        send = AsyncMock()

        await middleware(scope, receive, send)

        # Get the receive callable passed to app
        receive_arg = app.call_args[0][1]

        msg1 = await receive_arg()
        msg2 = await receive_arg()

        # First call returns the passthrough body
        assert msg1["body"] == bad_gzip
        assert msg1["more_body"] is False

        # Subsequent calls return disconnect (ASGI-compliant behavior)
        assert msg2["type"] == "http.disconnect"

    # === Header Handling Tests ===

    @pytest.mark.asyncio
    async def test_content_encoding_preserved_when_other_encodings_remain(self):
        """Content-Encoding header preserves remaining encodings after removing gzip."""
        app = AsyncMock()
        middleware = GzipRequestMiddleware(app)

        data = b'{"test": "data"}'
        compressed = gzip.compress(data)

        # Simulate "br, gzip" - gzip is outermost (last), br is innermost
        scope = self.make_scope(
            headers=[
                (b"content-encoding", b"br, gzip"),
                (b"content-type", b"application/json"),
            ]
        )
        receive = self.make_receive_with_body(compressed)
        send = AsyncMock()

        await middleware(scope, receive, send)

        # Check that scope was modified to preserve "br"
        modified_scope = app.call_args[0][0]
        headers_dict = dict(modified_scope["headers"])

        assert b"content-encoding" in headers_dict
        assert headers_dict[b"content-encoding"] == b"br"

    @pytest.mark.asyncio
    async def test_content_length_added_if_missing(self):
        """Content-Length header is added if not present."""
        app = AsyncMock()
        middleware = GzipRequestMiddleware(app)

        data = b'{"test": "data"}'
        compressed = gzip.compress(data)

        scope = self.make_scope(
            headers=[
                (b"content-encoding", b"gzip"),
                (b"content-type", b"application/json"),
                # No Content-Length header
            ]
        )
        receive = self.make_receive_with_body(compressed)
        send = AsyncMock()

        await middleware(scope, receive, send)

        # Check that scope was modified with Content-Length
        modified_scope = app.call_args[0][0]
        headers = dict(modified_scope["headers"])

        assert b"content-length" in headers
        assert headers[b"content-length"] == str(len(data)).encode("utf-8")

    @pytest.mark.asyncio
    async def test_content_length_updated_if_present(self):
        """Content-Length header is updated if already present."""
        app = AsyncMock()
        middleware = GzipRequestMiddleware(app)

        data = b'{"test": "data"}'
        compressed = gzip.compress(data)

        scope = self.make_scope(
            headers=[
                (b"content-encoding", b"gzip"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(compressed)).encode("utf-8")),
            ]
        )
        receive = self.make_receive_with_body(compressed)
        send = AsyncMock()

        await middleware(scope, receive, send)

        # Check that Content-Length was updated to decompressed size
        modified_scope = app.call_args[0][0]
        headers = dict(modified_scope["headers"])

        assert headers[b"content-length"] == str(len(data)).encode("utf-8")

    @pytest.mark.asyncio
    async def test_content_encoding_removed(self):
        """Content-Encoding header is removed after decompression."""
        app = AsyncMock()
        middleware = GzipRequestMiddleware(app)

        data = b'{"test": "data"}'
        compressed = gzip.compress(data)

        scope = self.make_scope(
            headers=[
                (b"content-encoding", b"gzip"),
                (b"content-type", b"application/json"),
            ]
        )
        receive = self.make_receive_with_body(compressed)
        send = AsyncMock()

        await middleware(scope, receive, send)

        # Check that Content-Encoding was removed
        modified_scope = app.call_args[0][0]
        headers = dict(modified_scope["headers"])

        assert b"content-encoding" not in headers

    @pytest.mark.asyncio
    async def test_other_headers_preserved(self):
        """Other headers are preserved during decompression."""
        app = AsyncMock()
        middleware = GzipRequestMiddleware(app)

        data = b'{"test": "data"}'
        compressed = gzip.compress(data)

        scope = self.make_scope(
            headers=[
                (b"content-encoding", b"gzip"),
                (b"content-type", b"application/json"),
                (b"authorization", b"Bearer token123"),
                (b"x-custom-header", b"custom-value"),
            ]
        )
        receive = self.make_receive_with_body(compressed)
        send = AsyncMock()

        await middleware(scope, receive, send)

        # Check that other headers are preserved
        modified_scope = app.call_args[0][0]
        headers = dict(modified_scope["headers"])

        assert headers[b"authorization"] == b"Bearer token123"
        assert headers[b"x-custom-header"] == b"custom-value"
        assert headers[b"content-type"] == b"application/json"

    # === Initialization Tests ===

    def test_init_validates_max_size(self):
        """Initialization validates max_size is positive."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            GzipRequestMiddleware(None, max_size=0)

        with pytest.raises(ValueError, match="max_size must be positive"):
            GzipRequestMiddleware(None, max_size=-1)

    def test_init_defaults_and_immutability(self):
        """Initialization defaults and list-to-tuple conversion."""
        # Default values
        m1 = GzipRequestMiddleware(None)
        assert m1.paths == ("/data/upload",)
        assert m1.max_size == 16 * 1024 * 1024
        assert m1.fail_on_error is True
        assert m1.enable_metrics is True
        assert "application/json" in m1.allowed_content_types
        assert "application/cloudevents+json" in m1.allowed_content_types

        # Lists converted to tuples for immutability
        m2 = GzipRequestMiddleware(None, paths=["/data/upload"], allowed_content_types=["application/json"])
        assert isinstance(m2.paths, tuple)
        assert isinstance(m2.allowed_content_types, tuple)
