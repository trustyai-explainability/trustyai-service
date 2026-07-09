"""Tests for PVC storage path traversal prevention."""

import pytest

from src.service.data.storage.pvc import PVCStorage


class TestAllocateValidDatasetName:
    """Verify allocate_valid_dataset_name rejects path traversal."""

    @pytest.mark.parametrize(
        "name",
        [
            "../../etc/passwd",
            "../secret",
            "foo/bar",
            "foo\\bar",
            "model/../../../etc/shadow",
        ],
    )
    def test_rejects_path_traversal(self, name: str) -> None:
        """Dataset names with path separators or parent refs are rejected."""
        with pytest.raises(ValueError, match="Invalid dataset name"):
            PVCStorage.allocate_valid_dataset_name(name)

    @pytest.mark.parametrize(
        "name",
        [
            "my-model_inputs",
            "model-123_outputs",
            "test_metadata",
            "simple",
        ],
    )
    def test_accepts_valid_names(self, name: str) -> None:
        """Normal dataset names pass through unchanged."""
        result = PVCStorage.allocate_valid_dataset_name(name)
        assert result == name
