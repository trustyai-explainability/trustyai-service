"""Contract tests for the local LIME endpoint."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from http import HTTPStatus
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trustyai_service.core.explainers.local.lime import (
    LIMEConfidenceIntervalResult,
    LIMEExplanationResult,
)
from trustyai_service.endpoints.explainers import local_lime
from trustyai_service.service.data.local_explanation import (
    LocalExplanationData,
    LocalExplanationMissingOutputError,
)
from trustyai_service.service.explainers.local import execution as execution_module
from trustyai_service.service.explainers.local.error_mapping import (
    LocalDataError,
    LocalDataNotFoundError,
)
from trustyai_service.service.explainers.local.model_provider import (
    ProviderDeadlineError,
    ProviderInvalidRequestError,
    ProviderUnavailableError,
    ProviderUnsupportedModelError,
)
from trustyai_service.service.explainers.local.types import PredictionSource

if TYPE_CHECKING:
    from collections.abc import Callable


class _FakeExecution:
    """Provider-backed execution double that records every prediction batch."""

    def __init__(self, *, classification: bool = False) -> None:
        self.source = PredictionSource.MODEL
        self.provider = _ProviderObservability()
        self.resolved_output_name = "score"
        self.classification = classification
        self.calls: list[np.ndarray] = []
        self.close_calls = 0

    def predict_fn(self, values: np.ndarray) -> np.ndarray:
        """Return deterministic regression or binary class scores."""
        self.calls.append(np.array(values, copy=True))
        if self.classification:
            return np.tile(np.asarray([[0.2, 0.8]]), (len(values), 1))
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


def _data(*, with_output: bool = False) -> LocalExplanationData:
    """Build one target and two organic background rows."""
    return LocalExplanationData(
        model_id="model",
        prediction_id="target",
        instance=np.asarray([2.0, 2.0]),
        feature_names=["f0", "f1"],
        background=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        background_output=(np.asarray([[0.25], [0.5]]) if with_output else None),
        output_names=["score"] if with_output else [],
        input_tensor_name="input",
        output_tensor_name="score",
    )


def _app() -> FastAPI:
    """Create an application with only the LIME router."""
    app = FastAPI()
    app.include_router(local_lime.router)
    return app


def _payload(
    *,
    source: str = "MODEL",
    task: str = "REGRESSION",
    explainer: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a canonical local LIME request."""
    model: dict[str, object] = {
        "model_name": "model",
        "task": task,
        "prediction_source": source,
    }
    if source == "MODEL":
        model["base_url"] = "http://model.example"
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
        data = _data(with_output=True)
    if captured is None:
        captured = {}

    async def load_data(*_args: object, **_kwargs: object) -> LocalExplanationData:
        """Return the shared storage context from the worker event loop."""
        captured["include_stored_output"] = _kwargs["include_stored_output"]
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
        captured["bootstrap_label"] = kwargs["label"]
        captured["bootstrap_predict_fn"] = predict_fn
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
        "seed",
        "class_index",
    }

    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(explainer={"num_samples": 10, "track_counterfactuals": True}),
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert isinstance(response.json()["detail"], list)


def test_default_model_uses_one_worker_execution_and_never_builds_rf(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Use the MODEL execution by default and pass generated rows to it."""
    execution = _FakeExecution()
    captured: dict[str, object] = {}
    _install_successful_worker(
        monkeypatch,
        execution=execution,
        captured=captured,
        data=_data(with_output=False),
    )

    def fail_surrogate() -> object:
        """Fail if the default MODEL request tries an RF fallback."""
        pytest.fail("MODEL request attempted surrogate construction")

    monkeypatch.setattr(execution_module, "_load_surrogate_builder", fail_surrogate)
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
    assert payload["prediction_source"] == "MODEL"
    assert payload["model"] == "model"
    assert payload["prediction_output"] == pytest.approx(1.0)
    assert payload["local_prediction"] == pytest.approx(0.37)
    assert execution.calls[0].shape == (1, 2)
    assert captured["include_stored_output"] is False
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
    assert completion.source == "MODEL"
    assert completion.prediction_source == "MODEL"
    assert completion.provider_latency == pytest.approx(0.125)
    assert completion.inference_batch_count == 3
    assert completion.final_status == HTTPStatus.OK


def test_classification_selects_argmax_once_and_preserves_prediction_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return full class scores while LIME reports the selected local value."""
    execution = _FakeExecution(classification=True)
    captured: dict[str, object] = {}
    _install_successful_worker(
        monkeypatch,
        execution=execution,
        captured=captured,
        data=_data(with_output=False),
    )

    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(
            task="CLASSIFICATION",
            explainer={"num_samples": 10, "confidence": 0.9},
        ),
    )

    assert response.status_code == HTTPStatus.OK, response.text
    payload = response.json()
    assert payload["class_index"] == 1
    assert payload["prediction_output"] == pytest.approx([0.2, 0.8])
    assert payload["local_prediction"] == pytest.approx(0.37)
    assert captured["primary_label"] == 1
    assert captured["bootstrap_label"] == 1
    assert captured["primary_predict_fn"] is captured["bootstrap_predict_fn"]
    assert len(execution.calls) == 1


@pytest.mark.parametrize("source", ["MODEL", "SURROGATE"])
def test_regression_class_index_is_rejected_before_worker_or_provider(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
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
            source=source,
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


def test_missing_background_is_a_bad_request_and_unavailable_lime_is_service_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify expected loader data failures and missing optional LIME safely."""
    monkeypatch.setattr(local_lime, "_LIME_AVAILABLE", True)

    async def no_background(*_args: object, **_kwargs: object) -> LocalExplanationData:
        """Represent the shared loader's missing organic background failure."""
        message = "No organic background data is available"
        raise ValueError(message)

    monkeypatch.setattr(local_lime, "load_local_explanation_data", no_background)
    response = TestClient(_app()).post(
        "/explainers/local/lime", json=_payload(explainer={"confidence": 1.0})
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["detail"]["code"] == "data_invalid"

    monkeypatch.setattr(local_lime, "_LIME_AVAILABLE", False)
    unavailable = TestClient(_app()).post("/explainers/local/lime", json=_payload())
    assert unavailable.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert unavailable.json()["detail"]["code"] == "dependency_unavailable"
    assert "unavailable" in unavailable.json()["detail"]["message"].lower()


def test_missing_surrogate_output_is_data_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map the known absent surrogate-label dataset to a not-found response."""
    monkeypatch.setattr(local_lime, "_LIME_AVAILABLE", True)

    async def missing_output(*_args: object, **_kwargs: object) -> LocalExplanationData:
        """Represent the loader's known missing surrogate output condition."""
        msg = "Stored output labels are unavailable"
        raise LocalExplanationMissingOutputError(msg)

    monkeypatch.setattr(local_lime, "load_local_explanation_data", missing_output)
    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(source="SURROGATE", explainer={"confidence": 1.0}),
    )

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["detail"]["code"] == "data_missing"


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
    assert failure.source == "MODEL"
    assert failure.final_status == HTTPStatus.SERVICE_UNAVAILABLE
    assert failure.error_code == "unavailable"
    assert failure.latency_seconds >= 0
    assert failure.provider_latency == pytest.approx(0.125)
    assert failure.inference_batch_count == 3
    assert execution.close_calls == 1


@pytest.mark.parametrize("source", ["MODEL", "SURROGATE"])
def test_untyped_same_message_loader_failure_remains_an_internal_error(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    """Do not reinterpret an unrelated same-message ValueError as client data."""
    monkeypatch.setattr(local_lime, "_LIME_AVAILABLE", True)

    async def backend_failure(
        *_args: object, **_kwargs: object
    ) -> LocalExplanationData:
        """Represent an unexpected storage implementation failure."""
        message = "Stored output labels are unavailable"
        raise ValueError(message)

    monkeypatch.setattr(local_lime, "load_local_explanation_data", backend_failure)
    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(source=source, explainer={"confidence": 1.0}),
    )

    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert response.json()["detail"]["code"] == "execution_failed"
    assert "Stored output labels are unavailable" not in response.text


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
        """Return the execution whose provider-free callable is deadline guarded."""
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


def test_surrogate_source_is_provider_free_and_reports_surrogate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not resolve transport or construct a provider for SURROGATE."""
    execution = _FakeExecution()
    execution.source = PredictionSource.SURROGATE
    captured: dict[str, object] = {}
    _install_successful_worker(
        monkeypatch,
        execution=execution,
        captured=captured,
        data=_data(with_output=True),
    )

    monkeypatch.setattr(
        execution_module,
        "get_transport_config",
        lambda _source: pytest.fail("SURROGATE resolved transport settings"),
    )
    response = TestClient(_app()).post(
        "/explainers/local/lime",
        json=_payload(
            source="SURROGATE",
            explainer={"confidence": 1.0},
        ),
    )

    assert response.status_code == HTTPStatus.OK, response.text
    assert response.json()["prediction_source"] == "SURROGATE"
    assert captured["include_stored_output"] is True
