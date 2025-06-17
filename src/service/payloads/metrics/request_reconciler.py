import logging

from src.service.data.datasources.data_source import DataSource
from src.service.data.metadata.storage_metadata import StorageMetadata

from src.service.payloads.metrics.base_metric_request import BaseMetricRequest
from src.service.payloads.values.data_type import DataType
from src.service.payloads.values.reconcilable_feature import ReconcilableFeature
from src.service.payloads.values.reconcilable_output import ReconcilableOutput
from src.service.payloads.values.reconciler_matcher import ReconcilerMatcher
from src.service.payloads.values.typed_value import TypedValue
from src.service.utils.exceptions import IllegalArgumentError

logger: logging.Logger = logging.getLogger(__name__)


class RequestReconciler:
    @staticmethod
    def reconcile(request: BaseMetricRequest, data_source: DataSource) -> None:
        """
        Reconcile a metric request with the data source.

        Args:
            request: The metric request to reconcile
            data_source: The data source to use for reconciliation
        """
        storage_metadata: StorageMetadata = data_source.get_metadata(request.model_id)
        RequestReconciler.reconcile_with_metadata(request, storage_metadata)

    @staticmethod
    def reconcile_with_metadata(
        request: BaseMetricRequest, storage_metadata: StorageMetadata
    ) -> None:
        """
        Reconcile a metric request with the provided storage metadata.

        Args:
            request: The metric request to reconcile
            storage_metadata: The storage metadata to use for reconciliation
        """
        try:
            for name in dir(request.__class__):
                if name.startswith("_"):
                    continue

                field_descriptor = getattr(request.__class__, name, None)
                if field_descriptor is None:
                    continue

                # Check if field has reconciler matcher annotation
                if hasattr(field_descriptor, "_reconciler_matcher"):
                    matcher: ReconcilerMatcher = field_descriptor._reconciler_matcher

                    field_value = getattr(request, name, None)
                    if field_value is None:
                        continue

                    # Check if it's a ReconcilableFeature
                    if isinstance(field_value, ReconcilableFeature):
                        # Skip if already reconciled
                        if field_value.get_reconciled_type() is not None:
                            continue

                        name_provider_method = getattr(request, matcher.name_provider)
                        if not callable(name_provider_method):
                            raise IllegalArgumentError(
                                f"Reconcilable matcher for field {name} gave a "
                                f"name-providing-method that does not exist: {matcher.name_provider}"
                            )
                        provided_name = name_provider_method()

                        # Get the data type from input schema
                        field_data_type: DataType = (
                            storage_metadata.get_input_schema()
                            .get_name_mapped_items()
                            .get(provided_name)
                            .get_type()
                        )
                        tvs = []

                        for sub_node in field_value.get_raw_value_nodes():
                            tv = TypedValue()
                            tv.set_type(field_data_type)
                            tv.set_value(sub_node)
                            tvs.append(tv)

                        field_value.set_reconciled_type(tvs)

                    # Check if it's a ReconcilableOutput
                    elif isinstance(field_value, ReconcilableOutput):
                        if field_value.get_reconciled_type() is not None:
                            continue

                        name_provider_method = getattr(request, matcher.name_provider)
                        if not callable(name_provider_method):
                            raise IllegalArgumentError(
                                f"Reconcilable matcher for field {name} gave a "
                                f"name-providing-method that is not callable: "
                                f"{matcher.name_provider}"
                            )

                        provided_name = name_provider_method()

                        # Get the data type from output schema
                        field_data_type = (
                            storage_metadata.get_output_schema()
                            .get_name_mapped_items()[provided_name]
                            .get_type()
                        )
                        tvs = []

                        for sub_node in field_value.get_raw_value_nodes():
                            tv = TypedValue()
                            tv.set_type(field_data_type)
                            tv.set_value(sub_node)
                            tvs.append(tv)

                        field_value.set_reconciled_type(tvs)
            logger.info(f"Reconciled request for model {request.model_id}")
        except Exception as e:
            logger.error(f"Error reconciling request: {e}")
            raise
