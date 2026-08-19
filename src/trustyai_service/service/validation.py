"""Validation utilities for TrustyAI service."""

from trustyai_service.service.constants import TRUSTYAI_TAG_PREFIX


def validate_data_tag(tag: str | None) -> str | None:
    """Validate data tag format and content.

    Args:
        tag: Tag value to validate

    Returns:
        Error message if invalid, None if valid.

    """
    if tag is None:
        return None
    if not tag or not tag.strip():
        return "Tag name cannot be empty or whitespace-only"
    if tag.startswith(TRUSTYAI_TAG_PREFIX):
        return (
            f"The tag prefix '{TRUSTYAI_TAG_PREFIX}' is reserved for internal TrustyAI use only. "
            f"Provided tag '{tag}' violates this restriction."
        )
    return None
