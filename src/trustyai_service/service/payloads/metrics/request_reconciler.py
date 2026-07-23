"""Request reconciler for matching metric requests with stored data schemas."""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from trustyai_service.service.data.datasources.data_source import DataSource
from trustyai_service.service.data.metadata.storage_metadata import StorageMetadata
from trustyai_service.service.payloads.metrics.base_metric_request import (
    BaseMetricRequest,
)
from trustyai_service.service.payloads.service.schema import Schema
from trustyai_service.service.payloads.values.reconcilable_feature import (
    ReconcilableFeature,
)
from trustyai_service.service.payloads.values.reconcilable_output import (
    ReconcilableOutput,
)
from trustyai_service.service.payloads.values.typed_value import TypedValue
from trustyai_service.service.utils.exceptions import IllegalArgumentError

if TYPE_CHECKING:
    from trustyai_service.service.payloads.values.reconciler_matcher import (
        ReconcilerMatcher,
    )

logger: logging.Logger = logging.getLogger(__name__)


def _validate_name_provider_callable(
    name_provider_method: object, field_name: str, provider_name: str
) -> None:
    """Validate that name provider method is callable.

    :param name_provider_method: The method to validate
    :param field_name: Name of the field being reconciled
    :param provider_name: Name of the provider attribute
    :raises IllegalArgumentError: If provider is not callable
    """
    if not callable(name_provider_method):
        msg = (
            f"Reconcilable matcher for field {field_name} gave a "
            f"name-providing-method that is not callable: {provider_name}"
        )
        raise IllegalArgumentError(msg)


def _validate_provided_name(provided_name: object) -> None:
    """Validate that provided name is a string.

    :param provided_name: The name to validate
    :raises TypeError: If provided_name is not a string
    """
    if not isinstance(provided_name, str):
        msg = "provided_name must be str"
        raise TypeError(msg)


def _validate_schema_item_exists(schema_item: object, provided_name: str) -> None:
    """Validate that schema item exists.

    :param schema_item: The schema item to validate
    :param provided_name: Name that was looked up
    :raises IllegalArgumentError: If schema item is None
    """
    if schema_item is None:
        msg = f"Schema item not found for field name: {provided_name}"
        raise IllegalArgumentError(msg)


def _reconcile_field_with_schema(
    field_value: ReconcilableFeature | ReconcilableOutput,
    request: BaseMetricRequest,
    matcher: "ReconcilerMatcher",
    field_name: str,
    schema_getter: Callable[[], Schema],
) -> None:
    """Reconcile a field value with schema.

    :param field_value: The reconcilable field value
    :param request: The metric request
    :param matcher: The reconciler matcher
    :param field_name: Name of the field
    :param schema_getter: Bound method to get schema (input or output)
    """
    if field_value.get_reconciled_type() is not None:
        return

    name_provider_method = getattr(request, matcher.name_provider)
    _validate_name_provider_callable(
        name_provider_method, field_name, matcher.name_provider
    )
    provided_name = name_provider_method()
    _validate_provided_name(provided_name)

    schema_item = schema_getter().get_name_mapped_items().get(provided_name)
    _validate_schema_item_exists(schema_item, provided_name)

    field_data_type = schema_item.get_type()
    tvs = []
    for sub_node in field_value.get_raw_value_nodes():
        tv = TypedValue()
        tv.set_type(field_data_type)
        tv.set_value(sub_node)
        tvs.append(tv)

    field_value.set_reconciled_type(tvs)


class RequestReconciler:
    """Reconciles metric requests with stored data schemas."""

    @staticmethod
    async def reconcile(request: BaseMetricRequest, data_source: DataSource) -> None:
        """Reconcile a metric request with the data source.

        Args:
            request: The metric request to reconcile
            data_source: The data source to use for reconciliation

        """
        storage_metadata: StorageMetadata = await data_source.get_metadata(
            request.model_id
        )
        RequestReconciler.reconcile_with_metadata(request, storage_metadata)

    @staticmethod
    def reconcile_with_metadata(
        request: BaseMetricRequest, storage_metadata: StorageMetadata
    ) -> None:
        """Reconcile a metric request with the provided storage metadata.

        Args:
            request: The metric request to reconcile
            storage_metadata: The storage metadata to use for reconciliation

        """
        try:
            # Get both model fields and dynamically added instance attributes
            model_field_names = set(request.__class__.model_fields.keys())
            instance_attribute_names = set(request.__dict__.keys())
            all_field_names = model_field_names.union(instance_attribute_names)

            for name in all_field_names:
                if name.startswith("_"):
                    continue

                field_descriptor = getattr(request.__class__, name, None)
                if field_descriptor is None:
                    continue

                # Check if field has reconciler matcher annotation
                if hasattr(field_descriptor, "_reconciler_matcher"):
                    # Access private annotation set by reconciler_matcher decorator
                    matcher: ReconcilerMatcher = field_descriptor._reconciler_matcher

                    field_value = getattr(request, name, None)
                    if field_value is None:
                        continue

                    if isinstance(field_value, ReconcilableFeature):
                        _reconcile_field_with_schema(
                            field_value,
                            request,
                            matcher,
                            name,
                            storage_metadata.get_input_schema,
                        )
                    elif isinstance(field_value, ReconcilableOutput):
                        _reconcile_field_with_schema(
                            field_value,
                            request,
                            matcher,
                            name,
                            storage_metadata.get_output_schema,
                        )
            logger.info("Reconciled request for model %s", request.model_id)
        except Exception:  # Broad catch intentional: log context before re-raising any reconciliation error
            logger.exception("Error reconciling request")
            raise
