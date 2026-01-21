"""
Utility functions for logging, including deprecation warnings.
"""
import logging


def log_deprecated_endpoint(
    logger: logging.Logger,
    deprecated_name: str,
    replacement_name: str | None = None,
    additional_context: str | None = None,
) -> None:
    """
    Log a deprecation warning for a deprecated endpoint.

    Args:
        logger: The logger instance to use for logging
        deprecated_name: The name of the deprecated endpoint/metric
        replacement_name: Optional name of the replacement endpoint/metric. If None,
            the message will not include replacement information.
        additional_context: Optional additional context to include in the message

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_deprecated_endpoint(logger, "Meanshift", "CompareMeans")
        # Logs: "Deprecated Meanshift endpoint called. Use CompareMeans endpoint instead."

        >>> log_deprecated_endpoint(logger, "OldEndpoint")
        # Logs: "Deprecated OldEndpoint endpoint called."
    """
    if replacement_name:
        message = f"Deprecated {deprecated_name} endpoint called. Use {replacement_name} endpoint instead."
    else:
        message = f"Deprecated {deprecated_name} endpoint called."
    
    if additional_context:
        message = f"{message} {additional_context}"
    logger.warning(message)


def log_deprecation(
    logger: logging.Logger,
    deprecated_item: str,
    replacement_item: str | None = None,
    message: str | None = None,
) -> None:
    """
    Log a general deprecation warning.

    This is a more flexible function that can be used for any deprecation warning,
    not just endpoints.

    Args:
        logger: The logger instance to use for logging
        deprecated_item: The name/identifier of the deprecated item
        replacement_item: Optional name/identifier of the replacement item
        message: Optional custom message. If provided, other parameters are ignored

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_deprecation(logger, "old_function", "new_function")
        # Logs: "Deprecated: old_function. Use new_function instead."

        >>> log_deprecation(logger, "old_config", message="This config will be removed in v2.0")
        # Logs: "Deprecated: old_config. This config will be removed in v2.0"
    """
    if message:
        logger.warning(f"Deprecated: {deprecated_item}. {message}")
    elif replacement_item:
        logger.warning(f"Deprecated: {deprecated_item}. Use {replacement_item} instead.")
    else:
        logger.warning(f"Deprecated: {deprecated_item}")
