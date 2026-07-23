"""Schema definition for model input and output data structures."""

from trustyai_service.service.payloads.service.schema_item import SchemaItem


class Schema:
    """Schema definition for model data with column types and name mappings."""

    items: dict[str, SchemaItem]
    name_mapping: dict[str, str]

    def __init__(
        self,
        items: dict[str, SchemaItem] | None = None,
        name_mapping: dict[str, str] | None = None,
    ) -> None:
        """Initialize schema with items and optional name mappings.

        :param items: Dictionary mapping column names to SchemaItem definitions
        :param name_mapping: Optional dictionary mapping original names to aliases
        """
        self.items = items if items is not None else {}
        self.name_mapping = name_mapping if name_mapping is not None else {}

    def get_name_mapped_items(self) -> dict[str, SchemaItem]:
        """Get items with name mappings applied."""
        mapped_items = dict(self.items)
        if self.name_mapping:
            for original_name, mapped_name in self.name_mapping.items():
                if original_name in self.items:
                    mapped_items.pop(original_name, None)
                    mapped_items[mapped_name] = self.items[original_name]
        return mapped_items

    def get_name_mapped_key_set(self) -> set[str]:
        """Get the set of mapped column names."""
        return set(self.get_name_mapped_items().keys())
