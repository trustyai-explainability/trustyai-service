from unittest.mock import Mock, patch

import pytest
from src.service.data.datasources.data_source import DataSource
from src.service.data.metadata.storage_metadata import StorageMetadata
from src.service.payloads.metrics.base_metric_request import BaseMetricRequest
from src.service.payloads.metrics.request_reconciler import (
    IllegalArgumentError,
    RequestReconciler,
)
from src.service.payloads.service.schema import Schema
from src.service.payloads.service.schema_item import SchemaItem
from src.service.payloads.values.data_type import DataType
from src.service.payloads.values.reconcilable_feature import ReconcilableFeature
from src.service.payloads.values.reconcilable_output import ReconcilableOutput
from src.service.payloads.values.reconciler_matcher import ReconcilerMatcher
from src.service.payloads.values.typed_value import TypedValue


class MockReconcilableRequest(BaseMetricRequest):
    """Mock request class with reconcilable fields."""

    def __init__(self, model_id: str = "test_model"):
        super().__init__(model_id, "test_metric", "test_request", 100)

        # Create reconcilable feature field
        self.protected_attribute = ReconcilableFeature(
            {"type": "STRING", "value": "gender"}
        )
        self.favorable_outcome = ReconcilableOutput({"type": "DOUBLE", "value": 1.0})

        # Add reconciler matcher annotations
        self.__class__.protected_attribute = Mock()
        self.__class__.protected_attribute._reconciler_matcher = ReconcilerMatcher(
            "get_protected_attribute"
        )

        self.__class__.favorable_outcome = Mock()
        self.__class__.favorable_outcome._reconciler_matcher = ReconcilerMatcher(
            "get_outcome_name"
        )

    def get_protected_attribute(self):
        return "gender"

    def get_outcome_name(self):
        return "income"

    def retrieve_tags(self):
        return {"test": "value"}


class TestRequestReconciler:
    """Test RequestReconciler functionality."""

    @pytest.fixture
    def mock_data_source(self) -> Mock:
        """Create mock DataSource."""
        return Mock(spec=DataSource)

    @pytest.fixture
    def mock_storage_metadata(self) -> Mock:
        """Create mock StorageMetadata."""

        input_items = {
            "gender": SchemaItem(DataType.STRING, "gender", 0),
            "age": SchemaItem(DataType.INT32, "age", 1),
        }
        input_schema = Schema(input_items)

        output_items = {"income": SchemaItem(DataType.DOUBLE, "income", 0)}
        output_schema = Schema(output_items)

        metadata = Mock(spec=StorageMetadata)
        metadata.get_input_schema.return_value = input_schema
        metadata.get_output_schema.return_value = output_schema

        return metadata

    @pytest.fixture
    def mock_request(self) -> MockReconcilableRequest:
        """Create mock reconcilable request."""
        return MockReconcilableRequest()

    def test_reconcile_calls_get_metadata(
        self, mock_data_source: Mock, mock_storage_metadata: Mock
    ) -> None:
        """Test that reconcile calls get_metadata on data source."""

        mock_data_source.get_metadata.return_value = mock_storage_metadata
        request = MockReconcilableRequest("test_model")

        RequestReconciler.reconcile(request=request, data_source=mock_data_source)

        mock_data_source.get_metadata.assert_called_once_with("test_model")

    def test_reconcile_with_metadata_reconciles_feature(
        self, mock_storage_metadata: Mock, mock_request: MockReconcilableRequest
    ) -> None:
        """Test reconciling a ReconcilableFeature field."""

        # Ensure the feature was not already reconciled
        assert mock_request.protected_attribute.get_reconciled_type() is None

        RequestReconciler.reconcile_with_metadata(
            request=mock_request, storage_metadata=mock_storage_metadata
        )

        # Verify the feature is reconciled
        reconciled = mock_request.protected_attribute.get_reconciled_type()
        assert reconciled is not None
        assert len(reconciled) == 1
        assert reconciled[0].get_type() == DataType.STRING

    def test_reconcile_with_metadata_reconciles_output(
        self, mock_storage_metadata: Mock, mock_request: MockReconcilableRequest
    ) -> None:
        """Test reconciling a ReconcilableOutput field."""

        # Ensure the output was not already reconciled
        assert mock_request.favorable_outcome.get_reconciled_type() is None

        RequestReconciler.reconcile_with_metadata(
            request=mock_request, storage_metadata=mock_storage_metadata
        )

        # Verify the output is reconciled
        reconciled = mock_request.favorable_outcome.get_reconciled_type()
        assert reconciled is not None
        assert len(reconciled) == 1
        assert reconciled[0].get_type() == DataType.DOUBLE

    def test_reconcile_skips_already_reconciled_fields(
        self, mock_storage_metadata: Mock, mock_request: MockReconcilableRequest
    ) -> None:
        """Test that already reconciled fields are skipped."""

        # Pre-reconcile the feature
        typed_value = TypedValue()
        typed_value.set_type(DataType.STRING)
        mock_request.protected_attribute.set_reconciled_type([typed_value])

        initial_reconciled = mock_request.protected_attribute.get_reconciled_type()

        RequestReconciler.reconcile_with_metadata(
            request=mock_request, storage_metadata=mock_storage_metadata
        )

        final_reconciled = mock_request.protected_attribute.get_reconciled_type()
        assert final_reconciled == initial_reconciled

    def test_reconcile_with_multiple_value_nodes(
        self, mock_storage_metadata: Mock
    ) -> None:
        """Test reconciling field with multiple value nodes."""

        request = MockReconcilableRequest()

        # Create feature with multiple values
        multiple_values = [
            {"type": "STRING", "value": "male"},
            {"type": "STRING", "value": "female"},
        ]
        request.protected_attribute = ReconcilableFeature(multiple_values)

        RequestReconciler.reconcile_with_metadata(request, mock_storage_metadata)

        # Should reconcile all values
        reconciled = request.protected_attribute.get_reconciled_type()
        assert len(reconciled) == 2
        assert all(tv.get_type() == DataType.STRING for tv in reconciled)

    def test_reconcile_with_non_callable_name_provider(
        self, mock_storage_metadata: Mock
    ) -> None:
        """Test error when name provider is not callable."""

        request = MockReconcilableRequest()

        # Set name provider to a non-callable attribute
        request.non_callable_attribute = "not_a_method"
        request.__class__.protected_attribute._reconciler_matcher = ReconcilerMatcher(
            "non_callable_attribute"
        )

        with pytest.raises(
            IllegalArgumentError, match="name-providing-method that does not exist"
        ):
            RequestReconciler.reconcile_with_metadata(request, mock_storage_metadata)

    def test_reconcile_ignores_private_fields(
        self, mock_storage_metadata: Mock
    ) -> None:
        """Test that private fields (starting with _) are ignored."""

        request = MockReconcilableRequest()

        # Add a private field with reconciler matcher
        request._private_field = ReconcilableFeature(
            {"type": "STRING", "value": "test"}
        )
        request.__class__._private_field = Mock()
        request.__class__._private_field._reconciler_matcher = ReconcilerMatcher(
            "get_protected_attribute"
        )

        # Should not raise any errors
        RequestReconciler.reconcile_with_metadata(request, mock_storage_metadata)

        # Private field should not be reconciled
        assert request._private_field.get_reconciled_type() is None

    def test_reconcile_ignores_fields_without_reconciler_matcher(
        self, mock_storage_metadata: Mock
    ) -> None:
        """Test that fields without reconciler matcher annotation are ignored."""

        request = MockReconcilableRequest()

        # Add a field without reconciler matcher
        request.normal_field = ReconcilableFeature({"type": "STRING", "value": "test"})

        # Should not raise any errors
        RequestReconciler.reconcile_with_metadata(request, mock_storage_metadata)

        # Field without matcher should not be reconciled
        assert request.normal_field.get_reconciled_type() is None

    def test_reconcile_ignores_none_field_values(
        self, mock_storage_metadata: Mock
    ) -> None:
        """Test that None field values are ignored."""

        request = MockReconcilableRequest()

        # Set field to None
        request.protected_attribute = None

        # Should not raise any errors
        RequestReconciler.reconcile_with_metadata(request, mock_storage_metadata)

    def test_reconcile_with_missing_schema_field_raises_error(
        self, mock_request: MockReconcilableRequest
    ) -> None:
        """Test error when field name not found in schema."""

        # Create metadata with different field names
        input_items = {
            "different_field": SchemaItem(DataType.STRING, "different_field", 0)
        }
        input_schema = Schema(input_items)

        output_items = {
            "different_output": SchemaItem(DataType.DOUBLE, "different_output", 0)
        }
        output_schema = Schema(output_items)

        metadata = Mock(spec=StorageMetadata)
        metadata.get_input_schema.return_value = input_schema
        metadata.get_output_schema.return_value = output_schema

        # Should raise an error when field name not found
        with pytest.raises(Exception):  # KeyError or similar
            RequestReconciler.reconcile_with_metadata(mock_request, metadata)

    def test_reconcile_logs_success(
        self,
        mock_data_source: Mock,
        mock_storage_metadata: Mock,
        mock_request: MockReconcilableRequest,
    ) -> None:
        """Test that successful reconciliation is logged."""

        mock_data_source.get_metadata.return_value = mock_storage_metadata

        with patch(
            "src.service.payloads.metrics.request_reconciler.logger"
        ) as mock_logger:
            RequestReconciler.reconcile(mock_request, mock_data_source)

            mock_logger.info.assert_called_with(
                f"Reconciled request for model {mock_request.model_id}"
            )
