"""Contract tests for the local LIME endpoint."""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from http import HTTPStatus
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from trustyai_service.core.explainers.local.lime import (
    LIMEAllocationError,
    LIMEConfidenceIntervalResult,
    LIMEExplanationResult,
)
from trustyai_service.endpoints.explainers import local_lime
from trustyai_service.service.data import local_explanation as data_module
from trustyai_service.service.data.local_explanation import (
    LocalExplanationData,
)
from trustyai_service.service.explainers.local.error_mapping import (
    LocalDataError,
    LocalDataNotFoundError,
)
from trustyai_service.service.explainers.local.model_provider import (
    ProviderDeadlineError,
    ProviderInvalidRequestError,
    ProviderInvalidResponseError,
    ProviderUnavailableError,
    ProviderUnsupportedModelError,
)

from .integration_helpers import LocalStorage

if TYPE_CHECKING:
    from collections.abc import Callable


class _FakeExecution:
    """Provider-backed execution double that records every prediction batch."""

    def __init__(
        self,
        *,
        classification: bool = False,
        classification_width: int = 2,
    ) -> None:
        self.provider = _ProviderObservability()
        self.resolved_output_name = "score"
        self.classification = classification
        self.classification_width = classification_width
        self.calls: list[np.ndarray] = []
        self.close_calls = 0

    def predict_fn(self, values: np.ndarray) -> np.ndarray:
        """Return deterministic regression, binary, or multiclass scores."""
        self.calls.append(np.array(values, copy=True))
        if self.classification:
            if self.classification_width == 2:
                scores = np.asarray([[0.8, 0.2]])
            else:
                scores = np.asarray([[0.2, 0.3, 0.5]])
            return np.tile(scores, (len(values), 1))
        return values.sum(axis=1)[:, None] / 4.0

    def close(self) -> None:
        """Record worker-owned cleanup."""
        self.close_calls += 1


class _ProviderObservability:
    """Existing provider counters exposed to the endpoint completion log."""

    provider_latency = 0.125
    inference_batch_count = 3


class _FakeExplainer:
    """Minimal LIME boundary used to isolate endpoint orchestration."""

    feature_names: ClassVar[list[str]] = ["f0", "f1"]


def _data() -> LocalExplanationData:
    """Build one target and two organic background rows."""
    return LocalExplanationData(
        model_id="model",
        prediction_id="target",
        instance=np.asarray([2.0, 2.0]),
        feature_names=["f0", "f1"],
        background=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )


def _app() -> FastAPI:
    """Create an application with only the LIME router."""
    app = FastAPI()
    app.include_router(local_lime.router)
    return app


def _payload(
    *,
    task: str = "REGRESSION",
    explainer: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a canonical local LIME request."""
    model: dict[str, object] = {
        "model_name": "model",
        "task": task,
        "base_url": "http://model.example",
    }
    return {
        "predictionId": "target",
        "config": {
            "model": model,
            **({"explainer": explainer} if explainer is not None else {}),
        },
    }


def _install_successful_worker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    execution: _FakeExecution,
    data: LocalExplanationData | None = None,
    captured: dict[str, object] | None = None,
) -> None:
    """Install real endpoint orchestration around deterministic LIME doubles."""
    if data is None:
        data = _data()
    if captured is None:
        captured = {}

    async def load_data(
        model_id: str, prediction_id: str, max_background_rows: int
    ) -> LocalExplanationData:
        """Return the shared storage context from the worker event loop."""
        captured["loader_args"] = (model_id, prediction_id, max_background_rows)
        return data

    def create_execution(*args: object, **kwargs: object) -> _FakeExecution:
        """Record the shared execution factory call."""
        captured["execution_args"] = args
        captured["execution_kwargs"] = kwargs
        return execution

    def create_explainer(*args: object, **kwargs: object) -> _FakeExplainer:
        """Record training data and explicit LIME options."""
        captured["explainer_args"] = args
        captured["explainer_kwargs"] = kwargs
        return _FakeExplainer()

    def compute_explanation(
        _explainer: object,
        _instance: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        **kwargs: object,
    ) -> LIMEExplanationResult:
        """Exercise the endpoint's selected label and shared callable."""
        captured["primary_label"] = kwargs["label"]
        captured["primary_predict_fn"] = predict_fn
        return LIMEExplanationResult(
            feature_weights=[("f0", 0.4), ("f1", -0.1)],
            score=0.91,
            local_prediction=0.37,
            intercept=0.12,
        )

    def compute_confidence(
        _explainer: object,
        _instance: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        **kwargs: object,
    ) -> LIMEConfidenceIntervalResult:
        """Exercise the endpoint's shared callable for bootstrap work."""
        captured["bootstrap_called"] = True
        captured["bootstrap_label"] = kwargs["label"]
        captured["bootstrap_predict_fn"] = predict_fn
        captured["bootstrap_n_bootstrap"] = kwargs["n_bootstrap"]
        captured["bootstrap_seed"] = kwargs["seed"]
        return LIMEConfidenceIntervalResult(
            lower_bounds={"f0": 0.1, "f1": -0.2},
            upper_bounds={"f0": 0.7, "f1": 0.0},
        )

    async def run_worker(function: Callable[[], object], _duration: float) -> object:
        """Run the synchronous worker body on a thread like production."""
        return await asyncio.to_thread(function)

    monkeypatch.setattr(local_lime, "_LIME_AVAILABLE", True)
    monkeypatch.setattr(local_lime, "load_local_explanation_data", load_data)
    monkeypatch.setattr(local_lime, "create_prediction_execution", create_execution)
    monkeypatch.setattr(local_lime, "create_lime_explainer", create_explainer)
    monkeypatch.setattr(local_lime, "compute_lime_explanation", compute_explanation)
    monkeypatch.setattr(
        local_lime,
        "compute_lime_confidence_intervals",
        compute_confidence,
    )
    monkeypatch.setattr(local_lime, "run_local_worker", run_worker)


def test_lime_options_are_only_the_supported_algorithm_fields() -> None:
    """Reject historical placeholder knobs from the LIME request."""
    fields = set(local_lime.LimeExplainerConfig.model_fields)
    assert fields == {
        "num_samples",
        "n_training_rows",
        "kernel_width",
        "num_features",
        "timeout",
        "confidence",
        "n_bootstrap",
        "seed",
        "class_index",
    }
    defaults = local_lime.LimeExplainerConfig()
    assert defaults.confidence is None
    assert defaults.n_bootstrap == 10
    explicit = local_lime.LimeExplainerConfig(confidence=0.9, n_bootstrap=7)
    assert explicit.confidence == pytest.approx(0.9)
    assert explicit.n_bootstrap == 7

    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(explainer={"num_samples": 10, "track_counterfactuals": True}),
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert isinstance(response.json()["detail"], list)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence": 0.0},
        {"confidence": 1.1},
        {"confidence": math.inf},
        {"confidence": math.nan},
        {"n_bootstrap": 0},
        {"n_bootstrap": 51},
        {"n_bootstrap": True},
    ],
)
def test_lime_confidence_and_bootstrap_limits_are_request_validated(
    kwargs: dict[str, object],
) -> None:
    """Reject non-finite confidence and out-of-range bootstrap requests."""
    with pytest.raises(ValidationError):
        local_lime.LimeExplainerConfig(**kwargs)


@pytest.mark.parametrize("kernel_width", [0.0, -1.0, math.inf, math.nan])
def test_lime_kernel_width_must_be_finite_and_positive(kernel_width: float) -> None:
    """Reject kernel widths that cannot produce a valid LIME kernel."""
    with pytest.raises(ValidationError):
        local_lime.LimeExplainerConfig(kernel_width=kernel_width)


@pytest.mark.parametrize("seed", [-1, 2**32, True])
def test_lime_seed_is_validated_for_numpy_random_state(seed: object) -> None:
    """Reject seeds that LIME cannot pass to NumPy's RandomState."""
    with pytest.raises(ValidationError):
        local_lime.LimeExplainerConfig(seed=seed)


def test_lime_rejects_one_sample_neighborhoods_at_request_validation() -> None:
    """Reject a neighborhood too small to produce a finite local fit."""
    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(explainer={"num_samples": 1}),
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_non_finite_core_result_is_not_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map core numerical rejection before an invalid response can be returned."""
    execution = _FakeExecution()
    _install_successful_worker(monkeypatch, execution=execution)

    def reject_non_finite(*_args: object, **_kwargs: object) -> LIMEExplanationResult:
        """Represent the core's non-finite numerical-result rejection."""
        msg = "LIME returned non-finite numerical results"
        raise ValueError(msg)

    monkeypatch.setattr(local_lime, "compute_lime_explanation", reject_non_finite)
    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(explainer={"confidence": 1.0}),
    )

    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert response.json()["detail"]["code"] == "execution_failed"


def test_lime_allocation_guard_maps_to_resource_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a bounded resource error instead of an opaque HTTP 500."""
    execution = _FakeExecution()
    _install_successful_worker(monkeypatch, execution=execution)

    def reject_allocation(*_args: object, **_kwargs: object) -> LIMEExplanationResult:
        """Represent the core's pre-allocation guard."""
        raise LIMEAllocationError

    monkeypatch.setattr(local_lime, "compute_lime_explanation", reject_allocation)
    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(explainer={"confidence": 1.0}),
    )

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["detail"]["code"] == "resource_exhausted"


def test_model_uses_one_worker_execution_and_preserves_provider_counters(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Use one model execution and retain response values and provider counters."""
    execution = _FakeExecution()
    captured: dict[str, object] = {}
    _install_successful_worker(
        monkeypatch,
        execution=execution,
        captured=captured,
        data=_data(),
    )
    caplog.set_level(logging.INFO, logger=local_lime.__name__)
    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(
            explainer={
                "num_samples": 12,
                "n_training_rows": 2,
                "num_features": 2,
                "confidence": 1.0,
            }
        ),
    )

    assert response.status_code == HTTPStatus.OK, response.text
    payload = response.json()
    assert payload["model"] == "model"
    assert payload["prediction_output"] == pytest.approx(1.0)
    assert payload["local_prediction"] == pytest.approx(0.37)
    assert execution.calls[0].shape == (1, 2)
    assert captured["loader_args"] == ("model", "target", 2)
    assert captured["execution_kwargs"] == {}
    completion = next(
        record
        for record in caplog.records
        if record.getMessage() == "local_explanation_complete"
    )
    assert completion.explainer == "LIME"
    assert completion.model_name == "model"
    assert completion.model_version is None
    assert completion.prediction_id == "target"
    assert completion.task == "REGRESSION"
    assert completion.latency_seconds >= 0
    assert completion.provider_latency == pytest.approx(0.125)
    assert completion.inference_batch_count == 3
    assert completion.final_status == HTTPStatus.OK


def test_binary_classification_defaults_to_class_one_and_preserves_prediction_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default binary classification to class one instead of the argmax."""
    execution = _FakeExecution(classification=True)
    captured: dict[str, object] = {}
    _install_successful_worker(
        monkeypatch,
        execution=execution,
        captured=captured,
        data=_data(),
    )

    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(
            task="CLASSIFICATION",
            explainer={
                "num_samples": 10,
                "confidence": 0.9,
                "n_bootstrap": 7,
                "seed": 23,
            },
        ),
    )

    assert response.status_code == HTTPStatus.OK, response.text
    payload = response.json()
    assert payload["class_index"] == 1
    assert payload["prediction_output"] == pytest.approx([0.8, 0.2])
    assert payload["local_prediction"] == pytest.approx(0.37)
    assert captured["primary_label"] == 1
    assert captured["bootstrap_label"] == 1
    assert captured["primary_predict_fn"] is captured["bootstrap_predict_fn"]
    assert captured["bootstrap_n_bootstrap"] == 7
    assert captured["bootstrap_seed"] == 23
    assert len(execution.calls) == 1


def test_omitted_confidence_skips_bootstrap_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave confidence bounds absent when confidence is not requested."""
    execution = _FakeExecution()
    captured: dict[str, object] = {}
    _install_successful_worker(
        monkeypatch,
        execution=execution,
        captured=captured,
    )

    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(explainer={"num_samples": 10}),
    )

    assert response.status_code == HTTPStatus.OK, response.text
    assert captured.get("bootstrap_called") is None
    assert all(
        attribution["confidence_lower"] is None
        and attribution["confidence_upper"] is None
        for attribution in response.json()["attributions"]
    )


def test_multiclass_classification_requires_explicit_class_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject wider classification output when no class was requested."""
    execution = _FakeExecution(classification=True, classification_width=3)
    _install_successful_worker(monkeypatch, execution=execution)

    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(
            task="CLASSIFICATION",
            explainer={"num_samples": 10, "confidence": 1.0},
        ),
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["detail"]["code"] == "invalid_request"


def test_multiclass_classification_uses_explicit_class_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward a valid explicit class index for wider classification output."""
    execution = _FakeExecution(classification=True, classification_width=3)
    captured: dict[str, object] = {}
    _install_successful_worker(
        monkeypatch,
        execution=execution,
        captured=captured,
    )

    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(
            task="CLASSIFICATION",
            explainer={"num_samples": 10, "confidence": 1.0, "class_index": 2},
        ),
    )

    assert response.status_code == HTTPStatus.OK, response.text
    assert response.json()["class_index"] == 2
    assert response.json()["prediction_output"] == pytest.approx([0.2, 0.3, 0.5])
    assert captured["primary_label"] == 2


def test_regression_class_index_is_rejected_before_worker_or_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject regression class selectors before any worker or prediction work."""
    calls: list[str] = []

    async def run_worker(*_args: object, **_kwargs: object) -> object:
        """Record an invalid worker entry."""
        calls.append("worker")
        msg = "worker must not start"
        raise AssertionError(msg)

    monkeypatch.setattr(local_lime, "_LIME_AVAILABLE", True)
    monkeypatch.setattr(local_lime, "run_local_worker", run_worker)
    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(
            task="REGRESSION",
            explainer={"class_index": 0},
        ),
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["detail"]["code"] == "invalid_request"
    assert calls == []


def test_lime_class_index_boolean_is_request_validation_error() -> None:
    """Reject JSON booleans instead of coercing them to integer class indices."""
    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(
            task="CLASSIFICATION",
            explainer={"class_index": True},
        ),
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_explicit_class_index_is_validated_as_a_semantic_request_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a class selector outside the returned class-score width."""
    execution = _FakeExecution(classification=True)
    _install_successful_worker(monkeypatch, execution=execution, data=_data())

    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(
            task="CLASSIFICATION",
            explainer={"class_index": 2, "confidence": 1.0},
        ),
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    detail = response.json()["detail"]
    assert detail == {
        "code": "invalid_request",
        "message": "Invalid local explanation request",
    }


@pytest.mark.parametrize(
    ("error", "status", "code", "message"),
    [
        (
            LocalDataNotFoundError("dataset password=do-not-return"),
            HTTPStatus.NOT_FOUND,
            "data_missing",
            "Local explanation data was not found",
        ),
        (
            LocalDataError("storage service=internal-database"),
            HTTPStatus.BAD_REQUEST,
            "data_invalid",
            "Local explanation data is invalid",
        ),
        (
            ProviderInvalidRequestError("provider service=internal-model"),
            HTTPStatus.BAD_REQUEST,
            "invalid_request",
            "Invalid local explanation request",
        ),
        (
            ProviderInvalidResponseError("private malformed upstream body"),
            HTTPStatus.BAD_GATEWAY,
            "invalid_response",
            "Model provider returned an invalid response",
        ),
        (
            ProviderUnsupportedModelError("ambiguous upstream output"),
            HTTPStatus.BAD_GATEWAY,
            "unsupported_model",
            "Model provider returned an unsupported model",
        ),
        (
            ProviderUnavailableError("secret upstream outage"),
            HTTPStatus.SERVICE_UNAVAILABLE,
            "unavailable",
            "Model provider is unavailable",
        ),
        (
            ProviderDeadlineError("secret timeout detail"),
            HTTPStatus.GATEWAY_TIMEOUT,
            "deadline_exceeded",
            "Explanation exceeded its deadline",
        ),
    ],
)
def test_typed_failures_use_shared_safe_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status: HTTPStatus,
    code: str,
    message: str,
) -> None:
    """Map provider/data failures without exposing exception details."""
    monkeypatch.setattr(local_lime, "_LIME_AVAILABLE", True)

    async def load_data(*_args: object, **_kwargs: object) -> LocalExplanationData:
        """Raise the configured data error before worker execution."""
        if isinstance(error, LocalDataNotFoundError):
            raise error
        return _data()

    def fail_execution(*_args: object, **_kwargs: object) -> object:
        """Raise the configured provider error inside the worker."""
        raise error

    monkeypatch.setattr(local_lime, "load_local_explanation_data", load_data)
    monkeypatch.setattr(local_lime, "create_prediction_execution", fail_execution)

    async def run_worker(function: Callable[[], object], _duration: float) -> object:
        """Execute the worker body on a thread so typed errors reach the mapper."""
        return await asyncio.to_thread(function)

    monkeypatch.setattr(local_lime, "run_local_worker", run_worker)
    response = TestClient(_app()).post(
        "/explainers/local/lime", json=_payload(explainer={"confidence": 1.0})
    )

    assert response.status_code == status
    assert response.json()["detail"] == {"code": code, "message": message}
    assert str(error) not in response.text


def test_missing_background_is_not_found_and_unavailable_lime_is_service_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify expected loader data failures and missing optional LIME safely."""
    monkeypatch.setattr(local_lime, "_LIME_AVAILABLE", True)

    async def no_background(*_args: object, **_kwargs: object) -> LocalExplanationData:
        """Represent the shared loader's missing organic background failure."""
        message = "No organic background data is available"
        raise LocalDataNotFoundError(message)

    monkeypatch.setattr(local_lime, "load_local_explanation_data", no_background)
    response = TestClient(_app()).post(
        "/explainers/local/lime", json=_payload(explainer={"confidence": 1.0})
    )
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["detail"]["code"] == "data_missing"

    monkeypatch.setattr(local_lime, "_LIME_AVAILABLE", False)
    unavailable = TestClient(_app()).post("/explainers/local/lime", json=_payload())
    assert unavailable.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert unavailable.json()["detail"]["code"] == "dependency_unavailable"
    assert "unavailable" in unavailable.json()["detail"]["message"].lower()


def test_missing_output_dataset_does_not_prevent_model_explanation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load real stored inputs and explain them when stored outputs are absent."""
    storage = LocalStorage()
    del storage.outputs
    execution = _FakeExecution()
    _install_successful_worker(monkeypatch, execution=execution)

    async def dataset_exists(name: str) -> bool:
        """Expose only the input and metadata datasets in this storage."""
        return name in {"model_inputs", "model_metadata"}

    monkeypatch.setattr(storage, "dataset_exists", dataset_exists)
    monkeypatch.setattr(data_module, "get_global_storage_interface", lambda: storage)
    monkeypatch.setattr(
        local_lime,
        "load_local_explanation_data",
        data_module.load_local_explanation_data,
    )
    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(explainer={"n_training_rows": 2, "confidence": 1.0}),
    )

    assert response.status_code == HTTPStatus.OK, response.text
    assert response.json()["prediction_output"] == pytest.approx(1.0)
    assert response.json()["output_name"] == "score"
    np.testing.assert_array_equal(execution.calls[0], [[2.0, 2.0]])
    assert execution.close_calls == 1


def test_lime_failure_log_is_structured_and_safe(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Capture provider fields from the worker execution on failure."""
    error = ProviderUnavailableError("secret-token raw features")
    execution = _FakeExecution()

    async def load_data(*_args: object, **_kwargs: object) -> LocalExplanationData:
        """Return data so production worker setup reaches the compute stage."""
        return _data()

    def create_execution(*_args: object, **_kwargs: object) -> _FakeExecution:
        """Return the provider-backed execution whose counters are observed."""
        return execution

    def create_explainer(*_args: object, **_kwargs: object) -> _FakeExplainer:
        """Keep the failure test independent of the optional LIME package."""
        return _FakeExplainer()

    def fail_explanation(*_args: object, **_kwargs: object) -> LIMEExplanationResult:
        """Represent a provider failure after execution creation."""
        raise error

    monkeypatch.setattr(local_lime, "_LIME_AVAILABLE", True)
    monkeypatch.setattr(local_lime, "load_local_explanation_data", load_data)
    monkeypatch.setattr(local_lime, "create_prediction_execution", create_execution)
    monkeypatch.setattr(local_lime, "create_lime_explainer", create_explainer)
    monkeypatch.setattr(local_lime, "compute_lime_explanation", fail_explanation)
    caplog.set_level(logging.WARNING, logger=local_lime.__name__)
    payload = _payload(explainer={"confidence": 1.0})
    payload["config"]["model"]["model_version"] = "v1"  # type: ignore[index]

    response = TestClient(_app()).post("/explainers/local/lime", json=payload)

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert str(error) not in caplog.text
    failure = next(
        record
        for record in caplog.records
        if record.getMessage() == "local_explanation_failed"
    )
    assert failure.explainer == "LIME"
    assert failure.model_name == "model"
    assert failure.model_version == "v1"
    assert failure.prediction_id == "target"
    assert failure.task == "REGRESSION"
    assert failure.final_status == HTTPStatus.SERVICE_UNAVAILABLE
    assert failure.error_code == "unavailable"
    assert failure.latency_seconds >= 0
    assert failure.provider_latency == pytest.approx(0.125)
    assert failure.inference_batch_count == 3
    assert execution.close_calls == 1


def test_untyped_loader_failure_remains_an_internal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not reinterpret an unrelated backend ValueError as client data."""
    monkeypatch.setattr(local_lime, "_LIME_AVAILABLE", True)

    async def backend_failure(
        *_args: object, **_kwargs: object
    ) -> LocalExplanationData:
        """Represent an unexpected storage implementation failure."""
        message = "private backend failure"
        raise ValueError(message)

    monkeypatch.setattr(local_lime, "load_local_explanation_data", backend_failure)
    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(explainer={"confidence": 1.0}),
    )

    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert response.json()["detail"]["code"] == "execution_failed"
    assert "private backend failure" not in response.text


@pytest.mark.parametrize(
    "error", [KeyError("backend key"), IndexError("backend index")]
)
def test_unexpected_storage_lookup_failure_remains_an_internal_error(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    """Do not reinterpret storage lookup failures as missing model data."""
    monkeypatch.setattr(local_lime, "_LIME_AVAILABLE", True)

    async def backend_failure(
        *_args: object, **_kwargs: object
    ) -> LocalExplanationData:
        """Represent an unexpected lookup failure from a storage backend."""
        raise error

    monkeypatch.setattr(local_lime, "load_local_explanation_data", backend_failure)
    response = TestClient(_app()).post(
        "/explainers/local/lime", json=_payload(explainer={"confidence": 1.0})
    )

    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert response.json()["detail"]["code"] == "execution_failed"
    assert "backend" not in response.text


def test_slow_loader_is_cancelled_by_the_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel a loader coroutine at the request deadline and finish the worker."""
    started = threading.Event()
    stopped = threading.Event()
    monkeypatch.setattr(local_lime, "_LIME_AVAILABLE", True)

    async def slow_loader(*_args: object, **_kwargs: object) -> LocalExplanationData:
        """Represent storage work that cooperates with asyncio cancellation."""
        started.set()
        try:
            await asyncio.sleep(1.0)
        finally:
            stopped.set()
        return _data()

    monkeypatch.setattr(local_lime, "load_local_explanation_data", slow_loader)
    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(explainer={"timeout": 0.05, "confidence": 1.0}),
    )

    assert response.status_code == HTTPStatus.GATEWAY_TIMEOUT
    assert response.json()["detail"]["code"] == "deadline_exceeded"
    assert started.is_set()
    assert stopped.wait(1.0)


def test_deadline_guard_stops_slow_cpu_work_and_worker_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop LIME before bootstrap work after a slow CPU stage crosses expiry."""
    execution = _FakeExecution()
    started = threading.Event()
    cleaned = threading.Event()
    confidence_called = threading.Event()
    original_close = execution.close

    def close() -> None:
        """Record cleanup completion from the worker thread."""
        original_close()
        cleaned.set()

    execution.close = close  # type: ignore[method-assign]

    async def load_data(*_args: object, **_kwargs: object) -> LocalExplanationData:
        """Return a small context before the CPU-bound stage."""
        return _data()

    def create_execution(*_args: object, **_kwargs: object) -> _FakeExecution:
        """Return the execution whose prediction callable is deadline guarded."""
        return execution

    def slow_explanation(
        _explainer: object,
        _instance: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        **_kwargs: object,
    ) -> LIMEExplanationResult:
        """Cross the deadline before the shared prediction callable is reused."""
        started.set()
        time.sleep(0.08)
        predict_fn(np.ones((1, 2)))
        return LIMEExplanationResult(
            feature_weights=[], score=0.0, local_prediction=0.0, intercept=0.0
        )

    def confidence(*_args: object, **_kwargs: object) -> LIMEConfidenceIntervalResult:
        """Detect an incorrect continuation into bootstrap work."""
        confidence_called.set()
        return LIMEConfidenceIntervalResult(None, None)

    monkeypatch.setattr(local_lime, "_LIME_AVAILABLE", True)
    monkeypatch.setattr(local_lime, "load_local_explanation_data", load_data)
    monkeypatch.setattr(local_lime, "create_prediction_execution", create_execution)
    monkeypatch.setattr(
        local_lime, "create_lime_explainer", lambda *_a, **_k: _FakeExplainer()
    )
    monkeypatch.setattr(local_lime, "compute_lime_explanation", slow_explanation)
    monkeypatch.setattr(local_lime, "compute_lime_confidence_intervals", confidence)

    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(explainer={"timeout": 0.02, "confidence": 0.9}),
    )

    assert response.status_code == HTTPStatus.GATEWAY_TIMEOUT
    assert response.json()["detail"]["code"] == "deadline_exceeded"
    assert started.wait(1.0)
    assert cleaned.wait(1.0)
    assert not confidence_called.is_set()


def test_worker_cleanup_runs_when_lime_core_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close the provider from the synchronous worker's finally block."""
    execution = _FakeExecution()
    _install_successful_worker(monkeypatch, execution=execution, data=_data())

    def fail_core(*_args: object, **_kwargs: object) -> object:
        """Simulate an unexpected numerical failure after provider creation."""
        message = "private LIME failure"
        raise RuntimeError(message)

    monkeypatch.setattr(local_lime, "compute_lime_explanation", fail_core)
    response = TestClient(_app()).post(
        "/explainers/local/lime", json=_payload(explainer={"confidence": 1.0})
    )

    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert "private LIME failure" not in response.text
    assert execution.close_calls == 1
