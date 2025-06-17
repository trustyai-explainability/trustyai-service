from typing import Dict, Optional, Set

from src.service.payloads.service.schema_item import SchemaItem


class Schema:
    items: Dict[str, SchemaItem]
    name_mapping: Dict[str, str]

    def __init__(
        self,
        items: Optional[Dict[str, SchemaItem]] = None,
        name_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        self.items = items
        self.name_mapping = name_mapping

    def get_name_mapped_items(self) -> Dict[str, SchemaItem]:
        """Get items with name mappings applied."""
        mapped_items = dict(self.items)
        if self.name_mapping:
            for original_name, mapped_name in self.name_mapping.items():
                if original_name in self.items:
                    mapped_items.pop(original_name, None)
                    mapped_items[mapped_name] = self.items[original_name]
        return mapped_items

    def get_name_mapped_key_set(self) -> Set[str]:
        """Get the set of mapped column names."""
        return set(self.get_name_mapped_items().keys())
