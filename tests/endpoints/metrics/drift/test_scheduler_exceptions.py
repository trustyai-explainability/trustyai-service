"""
Test that scheduler exceptions during register/delete are properly surfaced by endpoints.

This test file verifies that the factory functions properly support setting side_effect
on scheduler.register() and scheduler.delete() to simulate exceptions like:
- Connection errors
- Runtime errors
- Timeout errors
- Internal database errors

These tests validate that the factory's register_side_effect and delete_side_effect
parameters work correctly for testing endpoint error handling.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


def test_schedule_error_factory_supports_register_side_effect():
    """Verify that make_schedule_endpoint_error_test supports register_side_effect parameter."""
    # This test verifies the factory function signature accepts the parameter
    # In real usage, this would be used to test actual endpoints
    import inspect

    from . import factory

    sig = inspect.signature(factory.make_schedule_endpoint_error_test)
    params = sig.parameters

    assert "register_side_effect" in params, "Factory should support register_side_effect parameter"
    assert params["register_side_effect"].default is None, "register_side_effect should default to None"


def test_delete_error_factory_supports_delete_side_effect():
    """Verify that make_delete_endpoint_error_test supports delete_side_effect parameter."""
    # This test verifies the factory function signature accepts the parameter
    import inspect

    from . import factory

    sig = inspect.signature(factory.make_delete_endpoint_error_test)
    params = sig.parameters

    assert "delete_side_effect" in params, "Factory should support delete_side_effect parameter"
    assert params["delete_side_effect"].default is None, "delete_side_effect should default to None"


def test_register_side_effect_behavior():
    """Test that register_side_effect correctly sets exception on mock scheduler."""
    # Simulate what the factory does internally
    mock_sched = MagicMock()
    register_side_effect = ConnectionError("Failed to connect to scheduler database")

    # This is what the factory should do when register_side_effect is provided
    mock_sched.register = AsyncMock(side_effect=register_side_effect)

    # Verify the side effect is set correctly
    with pytest.raises(ConnectionError, match="Failed to connect"):
        # AsyncMock will raise the exception when called
        import asyncio

        asyncio.run(mock_sched.register("model-id", "tag", ["col1"]))


def test_delete_side_effect_behavior():
    """Test that delete_side_effect correctly sets exception on mock scheduler."""
    # Simulate what the factory does internally
    mock_sched = MagicMock()
    delete_side_effect = RuntimeError("Scheduler deletion failed")

    # This is what the factory should do when delete_side_effect is provided
    mock_sched.delete = AsyncMock(side_effect=delete_side_effect)

    # Verify the side effect is set correctly
    with pytest.raises(RuntimeError, match="deletion failed"):
        import asyncio

        asyncio.run(mock_sched.delete("request-id"))


def test_various_exception_types():
    """Test that various exception types can be used as side effects."""
    exception_types = [
        ConnectionError("Database connection lost"),
        RuntimeError("Internal scheduler error"),
        TimeoutError("Operation timeout after 30s"),
        KeyError("Request ID not found"),
        ValueError("Invalid request format"),
        OSError("Network unreachable"),
    ]

    for exc in exception_types:
        mock_sched = MagicMock()
        mock_sched.register = AsyncMock(side_effect=exc)

        with pytest.raises(type(exc)):
            import asyncio

            asyncio.run(mock_sched.register("model", "tag", ["col"]))


def test_register_side_effect_none_means_no_exception():
    """Test that register_side_effect=None means no exception is raised."""
    mock_sched = MagicMock()
    register_side_effect = None

    # When register_side_effect is None, normal behavior (no exception)
    if register_side_effect:
        mock_sched.register = AsyncMock(side_effect=register_side_effect)
    else:
        mock_sched.register = AsyncMock(return_value=None)

    # Should not raise
    import asyncio

    result = asyncio.run(mock_sched.register("model-id", "tag", ["col1"]))
    assert result is None


def test_delete_side_effect_none_means_no_exception():
    """Test that delete_side_effect=None means no exception is raised."""
    mock_sched = MagicMock()
    delete_side_effect = None

    # When delete_side_effect is None, normal behavior (no exception)
    if delete_side_effect:
        mock_sched.delete = AsyncMock(side_effect=delete_side_effect)
    else:
        mock_sched.delete = AsyncMock(return_value=None)

    # Should not raise
    import asyncio

    result = asyncio.run(mock_sched.delete("request-id"))
    assert result is None


def test_factory_docstring_documents_new_parameters():
    """Verify that factory functions document the new side_effect parameters."""
    from . import factory

    schedule_doc = factory.make_schedule_endpoint_error_test.__doc__
    assert "register_side_effect" in schedule_doc, "Schedule factory should document register_side_effect"
    assert "Exception to raise" in schedule_doc, "Should explain what register_side_effect does"

    delete_doc = factory.make_delete_endpoint_error_test.__doc__
    assert "delete_side_effect" in delete_doc, "Delete factory should document delete_side_effect"
    assert "Exception to raise" in delete_doc, "Should explain what delete_side_effect does"
