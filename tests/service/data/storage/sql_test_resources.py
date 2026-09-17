"""Unique names and ownership-based cleanup for tests sharing a SQL database."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from trustyai_service.service.data.storage.sql.base import SQLStorage


class SQLTestResources:
    """Register only resources belonging to one test, including failed writes."""

    def __init__(self) -> None:
        """Allocate a namespace independent of other tests and existing data."""
        self.prefix = f"test_{uuid4().hex}_"
        self.datasets: set[str] = set()
        self.payloads: set[tuple[str, bool]] = set()

    def dataset(self, name: str) -> str:
        """Reserve a dataset name in this test's namespace."""
        result = self.prefix + name
        self.datasets.add(result)
        return result

    def payload(self, name: str, *, is_input: bool = True) -> str:
        """Reserve a partial-payload ID and its direction for cleanup."""
        result = self.prefix + name
        self.payloads.add((result, is_input))
        return result

    async def cleanup(self, storage: SQLStorage) -> None:
        """Delete registered resources without touching other database users."""
        for payload_id, is_input in self.payloads:
            await storage.delete_partial_payload(payload_id, is_input=is_input)
        for dataset_name in self.datasets:
            if await storage.dataset_exists(dataset_name):
                await storage.delete_dataset(dataset_name)
