"""Contract tests for the local KernelSHAP endpoint."""

from __future__ import annotations

import asyncio
import logging
import threading
from http import HTTPStatus
from typing import TYPE_CHECKING

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trustyai_service.core.explainers.local.shap import ShapExplanationResult
from trustyai_service.endpoints.explainers import local_shap
from trustyai_service.service.data.local_explanation import (
    LocalExplanationData,
    LocalExplanationMissingOutputError,
)
from trustyai_service.service.explainers.local import execution as execution_module
from trustyai_service.service.explainers.local.error_mapping import (
    LocalDataNotFoundError,
)
from trustyai_service.service.explainers.local.model_provider import (
    ProviderDeadlineError,
    ProviderUnavailableError,
    ProviderUnsupportedModelError,
)
from trustyai_service.service.explainers.local.types import PredictionSource
from trustyai_service.service.explainers.local.worker import (
    run_local_worker as actual_run_local_worker,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class _Provider:
    """Raw provider double that records every model batch."""

    def __init__(self, output: np.ndarray) -> None:
        """Configure deterministic raw output rows."""
        self.output = output
        self.calls: list[np.ndarray] = []
        self.timeouts: list[float | None] = []
        self.provider_latency = 0.25
        self.inference_batch_count = 0

    def predict(
        self,
        values: np.ndarray,
        *,
        timeout_seconds: float | None = None,
    ) -> np.ndarray:
        """Record a batch and return one configured output per row."""
        self.calls.append(np.array(values, copy=True))
        self.timeouts.append(timeout_seconds)
        self.inference_batch_count += 1
        return np.tile(self.output, (len(values), 1))


class _Execution:
    """Execution double matching the shared factory result contract."""

    def __init__(
        self,
        *,
        source: PredictionSource = PredictionSource.MODEL,
        output: np.ndarray | None = None,
    ) -> None:
        """Create a model or provider-free execution."""
        self.source = source
        self.provider = (
            _Provider(
                np.asarray(
                    [[0.75]] if output is None else output,
                    dtype=float,
                )
            )
            if source is PredictionSource.MODEL
            else None
        )
        self.resolved_output_name = "score"
        self.close_calls = 0

    def predict_fn(self, values: np.ndarray) -> np.ndarray:
        """Return the normalized provider-free regression or class output."""
        if self.provider is not None:
            return self.provider.predict(values)
        return values.sum(axis=1) / 4.0

    def close(self) -> None:
        """Record worker-owned cleanup."""
        self.close_calls += 1


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
    """Create an application with only the SHAP router."""
    app = FastAPI()
    app.include_router(local_shap.router)
    return app


def _payload(
    *,
    source: str = "MODEL",
    task: str = "REGRESSION",
    explainer: dict[str, object] | None = None,
    base_url: str | None = "http://model.example",
) -> dict[str, object]:
    """Build a canonical local SHAP request."""
    model: dict[str, object] = {
        "model_name": "model",
        "task": task,
        "prediction_source": source,
    }
    if base_url is not None:
        model["base_url"] = base_url
    return {
        "predictionId": "target",
        "config": {
            "model": model,
            **({"explainer": explainer} if explainer is not None else {}),
        },
    }


def _install_worker_doubles(  # noqa: PLR0913
    monkeypatch: pytest.MonkeyPatch,
    *,
    execution: _Execution,
    data: LocalExplanationData | None = None,
    result: ShapExplanationResult | None = None,
    confidence: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
    captured: dict[str, object] | None = None,
) -> None:
    """Install deterministic shared-runtime and SHAP-core doubles."""
    if data is None:
        data = _data(with_output=True)
    if result is None:
        result = ShapExplanationResult(
            values=np.asarray([0.4, -0.1]),
            base_value=0.7,
            linked_prediction=1.0,
        )
    if captured is None:
        captured = {}

    async def load_data(*_args: object, **kwargs: object) -> LocalExplanationData:
        """Return the shared storage context and record output loading."""
        captured["include_stored_output"] = kwargs["include_stored_output"]
        return data

    def create_execution(*args: object, **kwargs: object) -> _Execution:
        """Record the canonical execution-factory invocation."""
        captured["execution_args"] = args
        captured["execution_kwargs"] = kwargs
        return execution

    def compute_result(
        _background: np.ndarray,
        _instance: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        **kwargs: object,
    ) -> ShapExplanationResult:
        """Exercise the endpoint's selected callable and explicit core options."""
        captured["primary_predict_fn"] = predict_fn
        captured["primary_kwargs"] = kwargs
        return result

    def compute_confidence(
        _background: np.ndarray,
        _instance: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        **kwargs: object,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Exercise the same selected callable for confidence intervals."""
        captured["confidence_predict_fn"] = predict_fn
        captured["confidence_kwargs"] = kwargs
        return confidence

    async def run_worker(function: Callable[[], object], _duration: float) -> object:
        """Run the synchronous worker body in a thread."""
        return await asyncio.to_thread(function)

    monkeypatch.setattr(local_shap, "_SHAP_AVAILABLE", True)
    monkeypatch.setattr(local_shap, "load_local_explanation_data", load_data)
    monkeypatch.setattr(local_shap, "create_prediction_execution", create_execution)
    monkeypatch.setattr(local_shap, "compute_shap_result", compute_result)
    monkeypatch.setattr(local_shap, "compute_confidence_intervals", compute_confidence)
    monkeypatch.setattr(local_shap, "run_local_worker", run_worker)


def test_shap_options_reject_unused_track_counterfactuals() -> None:
    """Reject the removed placeholder option and expose only SHAP fields."""
    fields = set(local_shap.SHAPExplainerConfig.model_fields)
    assert fields == {
        "n_samples",
        "n_training_rows",
        "timeout",
        "link",
        "regularizer",
        "confidence",
        "class_index",
        "single_probability",
    }

    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(explainer={"track_counterfactuals": True}),
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_regression_class_index_is_rejected_before_worker_or_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a regression class selector before provider or surrogate setup."""
    execution = _Execution()
    calls: list[str] = []

    async def run_worker(function: Callable[[], object], _duration: float) -> object:
        """Record worker entry while preserving the current execution path."""
        calls.append("worker")
        return await asyncio.to_thread(function)

    def create_execution(*_args: object, **_kwargs: object) -> _Execution:
        """Record execution creation and return a provider-backed double."""
        calls.append("execution")
        return execution

    async def load_data(*_args: object, **_kwargs: object) -> LocalExplanationData:
        """Provide stored data if the request incorrectly enters the worker."""
        return _data()

    monkeypatch.setattr(local_shap, "_SHAP_AVAILABLE", True)
    monkeypatch.setattr(local_shap, "run_local_worker", run_worker)
    monkeypatch.setattr(local_shap, "create_prediction_execution", create_execution)
    monkeypatch.setattr(local_shap, "load_local_explanation_data", load_data)

    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(
            task="REGRESSION",
            explainer={"class_index": 0, "confidence": 1.0},
        ),
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["detail"]["code"] == "invalid_request"
    assert calls == []
    assert execution.provider.calls == []  # type: ignore[union-attr]
    assert execution.close_calls == 0


def test_regression_class_index_is_rejected_before_surrogate_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a regression class selector before the surrogate branch runs."""
    calls: list[str] = []

    async def run_worker(function: Callable[[], object], _duration: float) -> object:
        """Record worker entry while preserving the current execution path."""
        calls.append("worker")
        return await asyncio.to_thread(function)

    async def load_data(*_args: object, **_kwargs: object) -> LocalExplanationData:
        """Provide stored labels if the request incorrectly enters the worker."""
        return _data(with_output=True)

    def fail_builder() -> object:
        """Record an attempted random-forest surrogate construction."""
        calls.append("surrogate")
        raise AssertionError

    monkeypatch.setattr(local_shap, "_SHAP_AVAILABLE", True)
    monkeypatch.setattr(local_shap, "run_local_worker", run_worker)
    monkeypatch.setattr(local_shap, "load_local_explanation_data", load_data)
    monkeypatch.setattr(execution_module, "_load_surrogate_builder", fail_builder)

    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(
            source="SURROGATE",
            base_url=None,
            task="REGRESSION",
            explainer={"class_index": 0, "confidence": 1.0},
        ),
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["detail"]["code"] == "invalid_request"
    assert calls == []


def test_class_index_boolean_uses_normal_request_validation() -> None:
    """Reject JSON booleans instead of coercing them to integer class indices."""
    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(
            task="CLASSIFICATION",
            explainer={"class_index": True},
        ),
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_shap_response_uses_shared_worker_and_keeps_link_space_fields_distinct(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Return attributions and confidence bounds without duplicate base fields."""
    execution = _Execution()
    captured: dict[str, object] = {}
    _install_worker_doubles(
        monkeypatch,
        execution=execution,
        confidence=(np.asarray([0.1, -0.2]), np.asarray([0.7, 0.0])),
        captured=captured,
    )
    caplog.set_level(logging.INFO, logger=local_shap.__name__)

    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(
            explainer={
                "n_samples": 12,
                "n_training_rows": 2,
                "confidence": 0.9,
                "regularizer": "BIC",
            }
        ),
    )

    assert response.status_code == HTTPStatus.OK, response.text
    payload = response.json()
    assert payload["prediction_source"] == "MODEL"
    assert payload["prediction_output"] == pytest.approx(0.75)
    assert payload["shap_base_value"] == pytest.approx(0.7)
    assert payload["linked_prediction_output"] == pytest.approx(1.0)
    assert "base_value" not in payload
    assert payload["output_name"] == "score"
    assert payload["attributions"] == [
        {
            "feature_name": "f0",
            "importance": 0.4,
            "confidence_lower": 0.1,
            "confidence_upper": 0.7,
        },
        {
            "feature_name": "f1",
            "importance": -0.1,
            "confidence_lower": -0.2,
            "confidence_upper": 0.0,
        },
    ]
    assert captured["include_stored_output"] is False
    assert captured["confidence_predict_fn"] is captured["primary_predict_fn"]
    assert captured["primary_kwargs"]["link"] == "identity"  # type: ignore[index]
    assert captured["primary_kwargs"]["instance_prediction"] == pytest.approx(0.75)  # type: ignore[index]
    assert execution.close_calls == 1
    completion = next(
        record
        for record in caplog.records
        if record.getMessage() == "local_explanation_complete"
    )
    assert completion.explainer == "SHAP"
    assert completion.model_name == "model"
    assert completion.model_version is None
    assert completion.prediction_id == "target"
    assert completion.source == "MODEL"
    assert completion.prediction_source == "MODEL"
    assert completion.provider_latency == pytest.approx(0.25)
    assert completion.inference_batch_count == 1
    assert completion.final_status == HTTPStatus.OK


def test_default_model_uses_provider_batches_and_never_builds_rf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the default MODEL execution and pass the selected callable to SHAP."""
    execution = _Execution(output=np.asarray([[0.25, 0.75]]))
    captured: dict[str, object] = {}
    _install_worker_doubles(
        monkeypatch,
        execution=execution,
        data=_data(with_output=False),
        captured=captured,
    )

    monkeypatch.setattr(
        execution_module,
        "_load_surrogate_builder",
        lambda: pytest.fail("MODEL attempted surrogate construction"),
    )

    def invoke_selected(predict_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        """Exercise one generated coalition batch through the selected callable."""
        np.testing.assert_allclose(
            predict_fn(np.asarray([[1.0, 0.0], [0.0, 1.0]])),
            [0.75, 0.75],
        )

    primary = captured.get("primary_predict_fn")
    del primary
    monkeypatch.setattr(
        local_shap,
        "compute_shap_result",
        lambda _b, _i, predict_fn, **_k: (
            invoke_selected(predict_fn),
            ShapExplanationResult(np.asarray([0.0, 0.0]), 0.75, 0.75),
        )[1],
    )

    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(task="CLASSIFICATION", explainer={"confidence": 1.0}),
    )

    assert response.status_code == HTTPStatus.OK, response.text
    assert response.json()["class_index"] == 1
    assert len(execution.provider.calls) == 2  # type: ignore[union-attr]
    assert execution.provider.calls[0].shape == (1, 2)  # type: ignore[union-attr]
    assert execution.provider.calls[1].shape == (2, 2)  # type: ignore[union-attr]


def test_explicit_surrogate_is_provider_free_and_does_not_resolve_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep explicit SURROGATE execution independent of network configuration."""
    execution = _Execution(source=PredictionSource.SURROGATE)
    captured: dict[str, object] = {}
    _install_worker_doubles(
        monkeypatch,
        execution=execution,
        data=_data(with_output=True),
        captured=captured,
    )
    monkeypatch.setattr(
        execution_module,
        "get_transport_config",
        lambda _source: pytest.fail("SURROGATE resolved transport settings"),
    )

    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(source="SURROGATE", base_url=None),
    )

    assert response.status_code == HTTPStatus.OK, response.text
    assert response.json()["prediction_source"] == "SURROGATE"
    assert captured["include_stored_output"] is True
    assert execution.provider is None


def test_binary_classification_defaults_to_class_one_and_reuses_instance_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select class one and do not issue a second singleton provider call."""
    execution = _Execution(output=np.asarray([[0.25, 0.75]]))
    captured: dict[str, object] = {}
    _install_worker_doubles(
        monkeypatch,
        execution=execution,
        data=_data(with_output=False),
        captured=captured,
    )

    def compute(
        _background: np.ndarray,
        _instance: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        **kwargs: object,
    ) -> ShapExplanationResult:
        """Call the selected callable for a coalition and the cached instance."""
        captured["primary_kwargs"] = kwargs
        assert kwargs["instance_prediction"] == pytest.approx(0.75)
        np.testing.assert_allclose(predict_fn(np.asarray([[2.0, 2.0]])), [0.75])
        return ShapExplanationResult(np.zeros(2), 0.75, 0.75)

    monkeypatch.setattr(local_shap, "compute_shap_result", compute)
    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(task="CLASSIFICATION", explainer={"confidence": 1.0}),
    )

    assert response.status_code == HTTPStatus.OK, response.text
    payload = response.json()
    assert payload["class_index"] == 1
    assert payload["prediction_output"] == pytest.approx(0.75)
    assert len(execution.provider.calls) == 1  # type: ignore[union-attr]
    assert captured["primary_kwargs"]["instance_prediction"] == pytest.approx(0.75)  # type: ignore[index]


def test_wider_classification_requires_class_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject wider class outputs without guessing a class."""
    execution = _Execution(output=np.asarray([[0.2, 0.3, 0.5]]))
    _install_worker_doubles(
        monkeypatch,
        execution=execution,
        data=_data(with_output=False),
    )

    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(task="CLASSIFICATION", explainer={"confidence": 1.0}),
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["detail"]["code"] == "invalid_request"
    assert execution.close_calls == 1


@pytest.mark.parametrize(
    ("single_probability", "class_index", "status"),
    [
        (False, None, HTTPStatus.BAD_REQUEST),
        (True, 0, HTTPStatus.BAD_REQUEST),
        (True, 1, HTTPStatus.OK),
    ],
)
def test_one_column_positive_probability_rules(
    monkeypatch: pytest.MonkeyPatch,
    single_probability: bool,  # noqa: FBT001
    class_index: int | None,
    status: HTTPStatus,
) -> None:
    """Require explicit positive-class opt-in for one-column classification."""
    execution = _Execution(output=np.asarray([[0.75]]))
    _install_worker_doubles(
        monkeypatch,
        execution=execution,
        data=_data(with_output=False),
    )
    explainer: dict[str, object] = {
        "confidence": 1.0,
        "single_probability": single_probability,
    }
    if class_index is not None:
        explainer["class_index"] = class_index

    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(task="CLASSIFICATION", explainer=explainer),
    )

    assert response.status_code == status, response.text
    if status == HTTPStatus.OK:
        assert response.json()["class_index"] == 1
        assert response.json()["prediction_output"] == pytest.approx(0.75)


def test_logit_rejects_probability_endpoints_before_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject raw LOGIT probabilities at zero or one without partial output."""
    execution = _Execution(output=np.asarray([[1.0, 0.0]]))
    _install_worker_doubles(
        monkeypatch,
        execution=execution,
        data=_data(with_output=False),
    )

    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(
            task="CLASSIFICATION",
            explainer={"link": "LOGIT", "confidence": 1.0},
        ),
    )

    assert response.status_code == HTTPStatus.BAD_GATEWAY
    assert response.json()["detail"]["code"] == "invalid_response"
    assert execution.close_calls == 1


def test_identity_regression_preserves_model_output_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep regression prediction and linked SHAP values in identity units."""
    execution = _Execution(output=np.asarray([[1.0]]))
    captured: dict[str, object] = {}
    _install_worker_doubles(
        monkeypatch,
        execution=execution,
        data=_data(with_output=False),
        result=ShapExplanationResult(np.zeros(2), 0.25, 1.0),
        captured=captured,
    )

    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(
            task="REGRESSION",
            explainer={"link": "IDENTITY", "confidence": 1.0},
        ),
    )

    assert response.status_code == HTTPStatus.OK, response.text
    payload = response.json()
    assert payload["prediction_output"] == pytest.approx(1.0)
    assert payload["shap_base_value"] == pytest.approx(0.25)
    assert payload["linked_prediction_output"] == pytest.approx(1.0)
    assert captured["primary_kwargs"]["link"] == "identity"  # type: ignore[index]


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (
            ProviderUnavailableError("upstream outage"),
            HTTPStatus.SERVICE_UNAVAILABLE,
            "unavailable",
        ),
        (
            ProviderDeadlineError("deadline"),
            HTTPStatus.GATEWAY_TIMEOUT,
            "deadline_exceeded",
        ),
        (
            ProviderUnsupportedModelError("ambiguous output"),
            HTTPStatus.BAD_GATEWAY,
            "unsupported_model",
        ),
    ],
)
def test_provider_failures_use_shared_mapper_without_partial_explanation(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status: HTTPStatus,
    code: str,
) -> None:
    """Map provider failures safely and never return partial SHAP values."""
    monkeypatch.setattr(local_shap, "_SHAP_AVAILABLE", True)

    async def load_data(*_args: object, **_kwargs: object) -> LocalExplanationData:
        """Return data before the provider fails."""
        return _data()

    def fail_execution(*_args: object, **_kwargs: object) -> object:
        """Raise the configured typed provider failure."""
        raise error

    async def run_worker(function: Callable[[], object], _duration: float) -> object:
        """Execute the worker so the shared mapper sees the original error."""
        return await asyncio.to_thread(function)

    monkeypatch.setattr(local_shap, "load_local_explanation_data", load_data)
    monkeypatch.setattr(local_shap, "create_prediction_execution", fail_execution)
    monkeypatch.setattr(local_shap, "run_local_worker", run_worker)

    response = TestClient(_app()).post(
        "/explainers/local/shap", json=_payload(explainer={"confidence": 1.0})
    )

    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert "upstream outage" not in response.text
    assert "partial" not in response.text


def test_missing_background_and_missing_storage_use_shared_mapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map missing organic background and stored data to stable errors."""
    monkeypatch.setattr(local_shap, "_SHAP_AVAILABLE", True)

    async def no_background(*_args: object, **_kwargs: object) -> LocalExplanationData:
        """Represent the loader's missing organic-background failure."""
        message = "No organic background data is available"
        raise ValueError(message)

    monkeypatch.setattr(local_shap, "load_local_explanation_data", no_background)
    response = TestClient(_app()).post(
        "/explainers/local/shap", json=_payload(explainer={"confidence": 1.0})
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["detail"]["code"] == "data_invalid"

    async def missing_data(*_args: object, **_kwargs: object) -> LocalExplanationData:
        """Represent the shared loader's missing dataset failure."""
        message = "private storage detail"
        raise LocalDataNotFoundError(message)

    monkeypatch.setattr(local_shap, "load_local_explanation_data", missing_data)
    response = TestClient(_app()).post(
        "/explainers/local/shap", json=_payload(explainer={"confidence": 1.0})
    )
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["detail"]["code"] == "data_missing"


def test_missing_surrogate_output_is_data_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map the known absent surrogate-label dataset to a not-found response."""
    monkeypatch.setattr(local_shap, "_SHAP_AVAILABLE", True)

    async def missing_output(*_args: object, **_kwargs: object) -> LocalExplanationData:
        """Represent the loader's known missing surrogate output condition."""
        msg = "Stored output labels are unavailable"
        raise LocalExplanationMissingOutputError(msg)

    monkeypatch.setattr(local_shap, "load_local_explanation_data", missing_output)
    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(source="SURROGATE", explainer={"confidence": 1.0}),
    )

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["detail"]["code"] == "data_missing"


@pytest.mark.parametrize("source", ["MODEL", "SURROGATE"])
def test_untyped_same_message_loader_failure_remains_an_internal_error(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    """Do not reinterpret an unrelated same-message ValueError as client data."""
    monkeypatch.setattr(local_shap, "_SHAP_AVAILABLE", True)

    async def backend_failure(
        *_args: object, **_kwargs: object
    ) -> LocalExplanationData:
        """Represent an unexpected backend failure with the loader's message."""
        message = "Stored output labels are unavailable"
        raise ValueError(message)

    monkeypatch.setattr(local_shap, "load_local_explanation_data", backend_failure)
    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(source=source, explainer={"confidence": 1.0}),
    )

    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert response.json()["detail"]["code"] == "execution_failed"
    assert "Stored output labels are unavailable" not in response.text


def test_shap_failure_log_is_structured_and_safe(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Capture provider fields from the worker execution on failure."""
    error = ProviderUnavailableError("secret-token raw features")
    execution = _Execution()

    async def load_data(*_args: object, **_kwargs: object) -> LocalExplanationData:
        """Return data so production worker setup reaches the compute stage."""
        return _data()

    def create_execution(*_args: object, **_kwargs: object) -> _Execution:
        """Return the provider-backed execution whose counters are observed."""
        return execution

    def fail_result(*_args: object, **_kwargs: object) -> ShapExplanationResult:
        """Represent a provider failure after execution creation."""
        raise error

    monkeypatch.setattr(local_shap, "_SHAP_AVAILABLE", True)
    monkeypatch.setattr(local_shap, "load_local_explanation_data", load_data)
    monkeypatch.setattr(local_shap, "create_prediction_execution", create_execution)
    monkeypatch.setattr(local_shap, "compute_shap_result", fail_result)
    caplog.set_level(logging.WARNING, logger=local_shap.__name__)
    payload = _payload(explainer={"confidence": 1.0})
    payload["config"]["model"]["model_version"] = "v1"  # type: ignore[index]

    response = TestClient(_app()).post("/explainers/local/shap", json=payload)

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert str(error) not in caplog.text
    failure = next(
        record
        for record in caplog.records
        if record.getMessage() == "local_explanation_failed"
    )
    assert failure.explainer == "SHAP"
    assert failure.model_name == "model"
    assert failure.model_version == "v1"
    assert failure.prediction_id == "target"
    assert failure.source == "MODEL"
    assert failure.final_status == HTTPStatus.SERVICE_UNAVAILABLE
    assert failure.error_code == "unavailable"
    assert failure.latency_seconds >= 0
    assert failure.provider_latency == pytest.approx(0.25)
    assert failure.inference_batch_count == 1
    assert execution.close_calls == 1


def test_worker_deadline_stops_cpu_stage_and_closes_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use real worker timeout cancellation and observe provider cleanup."""
    execution = _Execution()
    worker_started = threading.Event()
    release_worker = threading.Event()
    cleaned = threading.Event()
    original_close = execution.close

    def close() -> None:
        """Record cleanup after the synchronous worker unwinds."""
        original_close()
        cleaned.set()

    execution.close = close  # type: ignore[method-assign]
    _install_worker_doubles(monkeypatch, execution=execution, data=_data())
    monkeypatch.setattr(local_shap, "run_local_worker", actual_run_local_worker)

    def slow_result(*_args: object, **_kwargs: object) -> ShapExplanationResult:
        """Hold the real worker past the endpoint deadline until released."""
        worker_started.set()
        release_worker.wait(timeout=1.0)
        raise ProviderDeadlineError

    monkeypatch.setattr(local_shap, "compute_shap_result", slow_result)
    try:
        with TestClient(_app()) as client:
            response = client.post(
                "/explainers/local/shap",
                json=_payload(explainer={"timeout": 0.01, "confidence": 1.0}),
            )

            assert response.status_code == HTTPStatus.GATEWAY_TIMEOUT
            assert response.json()["detail"]["code"] == "deadline_exceeded"
            assert worker_started.wait(timeout=1.0)
            assert not cleaned.is_set()
    finally:
        release_worker.set()

    assert cleaned.wait(timeout=1.0)
    assert execution.close_calls == 1
