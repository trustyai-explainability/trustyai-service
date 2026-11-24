"""Contract constraints for drift request bodies.

The handlers answer HTTP 400 for a missing referenceTag and for an explicit
empty fitColumns list. The schema must declare that, without moving the
rejection into Pydantic validation (which would answer 422 instead).
"""

import pytest

from trustyai_service.endpoints.metrics.drift.compare_means import (
    CompareMeansMetricRequest,
)
from trustyai_service.endpoints.metrics.drift.jensen_shannon import (
    JensenShannonMetricRequest,
)
from trustyai_service.endpoints.metrics.drift.kolmogorov_smirnov import (
    KSTestMetricRequest,
)
from trustyai_service.endpoints.metrics.drift.kolmogorov_smirnov_streaming import (
    ApproxKSTestMetricRequest,
)
from trustyai_service.endpoints.metrics.drift.mmd import MMDMetricRequest

DRIFT_REQUESTS = [
    CompareMeansMetricRequest,
    KSTestMetricRequest,
    ApproxKSTestMetricRequest,
    JensenShannonMetricRequest,
    MMDMetricRequest,
]
DRIFT_REQUEST_IDS = [model.__name__ for model in DRIFT_REQUESTS]


@pytest.mark.parametrize("model", DRIFT_REQUESTS, ids=DRIFT_REQUEST_IDS)
def test_reference_tag_is_declared_required(model: type) -> None:
    """A body without referenceTag is rejected with 400, so it is required."""
    schema = model.model_json_schema(by_alias=True)

    assert "referenceTag" in schema["required"]
    # Leaving the default in place would suggest the field can be omitted.
    assert "default" not in schema["properties"]["referenceTag"]


@pytest.mark.parametrize("model", DRIFT_REQUESTS, ids=DRIFT_REQUEST_IDS)
def test_fit_columns_array_requires_one_item(model: type) -> None:
    """An explicit empty list is rejected, omission and null are not."""
    fit_columns = model.model_json_schema(by_alias=True)["properties"]["fitColumns"]
    branches = {branch["type"]: branch for branch in fit_columns["anyOf"]}

    assert branches["array"]["minItems"] == 1
    assert "null" in branches


def test_runtime_still_accepts_what_the_handler_rejects() -> None:
    """The constraint is declared in the schema only; HTTP 400 is unchanged."""
    omitted = CompareMeansMetricRequest.model_validate({"modelId": "model"})
    assert omitted.reference_tag is None
    assert omitted.fit_columns is None

    explicit_empty = CompareMeansMetricRequest.model_validate(
        {"modelId": "model", "referenceTag": "baseline", "fitColumns": []}
    )
    assert explicit_empty.fit_columns == []
