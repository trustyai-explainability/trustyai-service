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

from trustyai_service.core.explainers.local.shap import (
    SHAPAllocationError,
    ShapExplanationResult,
)
from trustyai_service.endpoints.explainers import local_shap
from trustyai_service.service.data.local_explanation import (
    LocalExplanationData,
)
from trustyai_service.service.explainers.local.error_mapping import (
    LocalDataNotFoundError,
)
from trustyai_service.service.explainers.local.model_provider import (
    ProviderDeadlineError,
    ProviderInvalidResponseError,
    ProviderUnavailableError,
    ProviderUnsupportedModelError,
)
from trustyai_service.service.explainers.local.worker import (
    run_local_worker as actual_run_local_worker,
)

from .integration_helpers import LocalStorage

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
        output: np.ndarray | None = None,
    ) -> None:
        """Create a model-backed execution."""
        self.provider = _Provider(
            np.asarray(
                [[0.75]] if output is None else output,
                dtype=float,
            )
        )
        self.resolved_output_name = "score"
        self.close_calls = 0

    def predict_fn(self, values: np.ndarray) -> np.ndarray:
        """Return the configured model output."""
        return self.provider.predict(values)

    def close(self) -> None:
        """Record worker-owned cleanup."""
        self.close_calls += 1


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
    """Create an application with only the SHAP router."""
    app = FastAPI()
    app.include_router(local_shap.router)
    return app


def _payload(
    *,
    task: str = "REGRESSION",
    explainer: dict[str, object] | None = None,
    base_url: str | None = "http://model.example",
) -> dict[str, object]:
    """Build a canonical local SHAP request."""
    model: dict[str, object] = {
        "model_name": "model",
        "task": task,
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
        data = _data()
    if result is None:
        result = ShapExplanationResult(
            values=np.asarray([0.4, -0.1]),
            base_value=0.7,
            linked_prediction=1.0,
        )
    if captured is None:
        captured = {}

    async def load_data(
        model_id: str, prediction_id: str, max_background_rows: int
    ) -> LocalExplanationData:
        """Require the model-only loader contract and record its arguments."""
        captured["loader_args"] = (model_id, prediction_id, max_background_rows)
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
        "n_bootstrap",
        "seed",
        "class_index",
        "single_probability",
    }
    defaults = local_shap.SHAPExplainerConfig()
    assert defaults.confidence is None
    assert defaults.n_bootstrap == 10
    assert defaults.seed is None

    configured = local_shap.SHAPExplainerConfig(
        confidence=0.9,
        n_bootstrap=50,
        seed=17,
    )
    assert configured.confidence == pytest.approx(0.9)
    assert configured.n_bootstrap == 50
    assert configured.seed == 17

    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(explainer={"track_counterfactuals": True}),
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


@pytest.mark.parametrize(
    "confidence",
    [0.0, -0.1, 1.1, float("nan"), float("inf"), -float("inf")],
)
def test_confidence_must_be_finite_and_bounded(confidence: float) -> None:
    """Reject confidence values outside the finite open/closed interval."""
    with pytest.raises(ValueError, match="confidence"):
        local_shap.SHAPExplainerConfig(confidence=confidence)


@pytest.mark.parametrize("n_bootstrap", [0, 51, True])
def test_n_bootstrap_is_bounded_at_request_validation(
    n_bootstrap: int,
) -> None:
    """Reject bootstrap counts outside the request's bounded range."""
    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(explainer={"n_bootstrap": n_bootstrap}),
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_seed_must_be_non_negative_at_request_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject negative bootstrap seeds before the worker starts."""
    execution = _Execution()
    actual_compute_confidence = local_shap.compute_confidence_intervals
    _install_worker_doubles(monkeypatch, execution=execution)
    monkeypatch.setattr(
        local_shap,
        "compute_confidence_intervals",
        actual_compute_confidence,
    )

    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(
            explainer={
                "confidence": 0.9,
                "n_bootstrap": 1,
                "seed": -1,
            }
        ),
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert any(
        error.get("loc", [])[-1] == "seed" for error in response.json()["detail"]
    )


def test_omitted_confidence_skips_confidence_interval_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not perform optional bootstrap work when confidence is omitted."""
    execution = _Execution()
    _install_worker_doubles(monkeypatch, execution=execution)

    def unexpected_confidence_call(*_args: object, **_kwargs: object) -> object:
        """Fail if the worker computes an interval without opt-in."""
        pytest.fail("confidence intervals must be opt-in")

    monkeypatch.setattr(
        local_shap,
        "compute_confidence_intervals",
        unexpected_confidence_call,
    )
    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(),
    )

    assert response.status_code == HTTPStatus.OK, response.text
    assert all(
        attribution["confidence_lower"] is None
        and attribution["confidence_upper"] is None
        for attribution in response.json()["attributions"]
    )


@pytest.mark.parametrize("regularizer", [True, None, [], {}])
def test_unsupported_json_regularizers_use_request_validation(
    regularizer: object,
) -> None:
    """Reject JSON values that are not supported SHAP regularizers with 422."""
    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(explainer={"regularizer": regularizer}),
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert any(
        error.get("loc", [])[-1] == "regularizer" for error in response.json()["detail"]
    )


@pytest.mark.parametrize(
    "regularizer",
    ["num_features(100001)", f"num_features({'9' * 5_000})"],
)
def test_oversized_num_features_regularizers_use_request_validation(
    regularizer: str,
) -> None:
    """Reject unbounded SHAP regularizer text before starting the worker."""
    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(explainer={"regularizer": regularizer}),
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert any(
        error.get("loc", [])[-1] == "regularizer" for error in response.json()["detail"]
    )


def test_shap_allocation_guard_maps_to_invalid_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return an actionable client error for deterministic allocation limits."""
    execution = _Execution()
    _install_worker_doubles(monkeypatch, execution=execution)

    def reject_allocation(*_args: object, **_kwargs: object) -> ShapExplanationResult:
        """Represent the core's pre-allocation guard."""
        raise SHAPAllocationError

    monkeypatch.setattr(local_shap, "compute_shap_result", reject_allocation)
    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(explainer={"confidence": 1.0}),
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["detail"] == {
        "code": "invalid_request",
        "message": (
            "SHAP request exceeds the memory budget; reduce n_samples or n_training_rows"
        ),
    }


def test_regression_class_index_is_rejected_before_worker_or_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a regression class selector before provider setup."""
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
    assert execution.provider.calls == []
    assert execution.close_calls == 0


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
    assert set(payload) == {
        "prediction_id",
        "model",
        "task",
        "output_name",
        "class_index",
        "prediction_output",
        "shap_base_value",
        "linked_prediction_output",
        "attributions",
    }
    assert payload["model"] == "model"
    assert payload["task"] == "REGRESSION"
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
    assert captured["loader_args"] == ("model", "target", 2)
    spec, data, deadline, transport = captured["execution_args"]
    assert spec.base_url == "http://model.example/"
    assert spec.model_name == "model"
    assert spec.output_name is None
    assert data.instance.tolist() == [2.0, 2.0]
    assert isinstance(deadline, float)
    assert transport is None
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
    assert completion.task == "REGRESSION"
    assert completion.output_name == "score"
    assert completion.class_index is None
    assert completion.provider_latency == pytest.approx(0.25)
    assert completion.inference_batch_count == 1
    assert completion.final_status == HTTPStatus.OK


def test_model_uses_provider_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass the selected model callable to SHAP for generated batches."""
    execution = _Execution(output=np.asarray([[0.25, 0.75]]))
    captured: dict[str, object] = {}
    _install_worker_doubles(
        monkeypatch,
        execution=execution,
        data=_data(),
        captured=captured,
    )

    def invoke_selected(predict_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        """Exercise one generated coalition batch through the selected callable."""
        np.testing.assert_allclose(
            predict_fn(np.asarray([[1.0, 0.0], [0.0, 1.0]])),
            [0.75, 0.75],
        )

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
    assert len(execution.provider.calls) == 2
    assert execution.provider.calls[0].shape == (1, 2)
    assert execution.provider.calls[1].shape == (2, 2)


def test_missing_output_dataset_does_not_prevent_model_explanation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the real loader when storage has only inputs and metadata."""
    execution = _Execution()
    actual_load_data = local_shap.load_local_explanation_data
    _install_worker_doubles(monkeypatch, execution=execution)
    storage = LocalStorage()

    async def dataset_exists(name: str) -> bool:
        """Expose only the datasets needed for model-backed SHAP."""
        return name.endswith(("_inputs", "_metadata"))

    monkeypatch.setattr(storage, "dataset_exists", dataset_exists)
    monkeypatch.setattr(
        "trustyai_service.service.data.local_explanation.get_global_storage_interface",
        lambda: storage,
    )
    monkeypatch.setattr(local_shap, "load_local_explanation_data", actual_load_data)

    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(explainer={"confidence": 1.0}),
    )

    assert response.status_code == HTTPStatus.OK, response.text
    assert response.json()["prediction_output"] == pytest.approx(0.75)
    assert response.json()["attributions"]
    assert execution.provider.inference_batch_count == 1
    assert execution.close_calls == 1


def test_confidence_intervals_share_model_batches_and_request_deadline(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Count primary and confidence batches under the same shrinking deadline."""
    execution = _Execution()
    captured: dict[str, object] = {}
    _install_worker_doubles(monkeypatch, execution=execution)
    caplog.set_level(logging.INFO, logger=local_shap.__name__)

    def compute_result(
        background: np.ndarray,
        _instance: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        **_kwargs: object,
    ) -> ShapExplanationResult:
        """Evaluate the background through the primary selected callable."""
        np.testing.assert_allclose(predict_fn(background), [0.75, 0.75])
        return ShapExplanationResult(np.zeros(2), 0.75, 0.75)

    def compute_confidence(
        background: np.ndarray,
        _instance: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a resampled batch through the confidence callable."""
        captured["confidence_kwargs"] = kwargs
        np.testing.assert_allclose(predict_fn(background[::-1]), [0.75, 0.75])
        return np.zeros(2), np.zeros(2)

    monkeypatch.setattr(local_shap, "compute_shap_result", compute_result)
    monkeypatch.setattr(local_shap, "compute_confidence_intervals", compute_confidence)
    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(
            explainer={
                "timeout": 30,
                "confidence": 0.9,
                "n_bootstrap": 7,
                "seed": 123,
            }
        ),
    )

    assert response.status_code == HTTPStatus.OK, response.text
    assert [call.shape for call in execution.provider.calls] == [(1, 2), (2, 2), (2, 2)]
    timeouts = execution.provider.timeouts
    assert all(timeout is not None and 0 < timeout <= 30 for timeout in timeouts)
    present_timeouts = [timeout for timeout in timeouts if timeout is not None]
    assert present_timeouts == sorted(present_timeouts, reverse=True)
    confidence_kwargs = captured["confidence_kwargs"]
    assert confidence_kwargs["confidence"] == pytest.approx(0.9)  # type: ignore[index]
    assert confidence_kwargs["n_bootstrap"] == 7  # type: ignore[index]
    assert confidence_kwargs["seed"] == 123  # type: ignore[index]
    completion = next(
        record
        for record in caplog.records
        if record.getMessage() == "local_explanation_complete"
    )
    assert completion.inference_batch_count == 3
    assert completion.provider_latency == pytest.approx(0.25)
    assert execution.close_calls == 1


def test_binary_classification_defaults_to_class_one_and_reuses_instance_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select class one and do not issue a second singleton provider call."""
    execution = _Execution(output=np.asarray([[0.25, 0.75]]))
    captured: dict[str, object] = {}
    _install_worker_doubles(
        monkeypatch,
        execution=execution,
        data=_data(),
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
    assert len(execution.provider.calls) == 1
    assert captured["primary_kwargs"]["instance_prediction"] == pytest.approx(0.75)  # type: ignore[index]


def test_wider_classification_requires_class_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject wider class outputs without guessing a class."""
    execution = _Execution(output=np.asarray([[0.2, 0.3, 0.5]]))
    _install_worker_doubles(
        monkeypatch,
        execution=execution,
        data=_data(),
    )

    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(task="CLASSIFICATION", explainer={"confidence": 1.0}),
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["detail"]["code"] == "invalid_request"
    assert execution.close_calls == 1


def test_wider_classification_uses_explicit_class_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the requested class rather than guessing for wider output."""
    execution = _Execution(output=np.asarray([[0.2, 0.3, 0.5]]))
    _install_worker_doubles(
        monkeypatch,
        execution=execution,
        data=_data(),
    )

    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(
            task="CLASSIFICATION",
            explainer={"class_index": 2, "confidence": 1.0},
        ),
    )

    assert response.status_code == HTTPStatus.OK, response.text
    assert response.json()["class_index"] == 2
    assert response.json()["prediction_output"] == pytest.approx(0.5)


def test_wider_classification_rejects_out_of_range_class_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a class selector at or above the returned output width."""
    execution = _Execution(output=np.asarray([[0.2, 0.3, 0.5]]))
    _install_worker_doubles(
        monkeypatch,
        execution=execution,
        data=_data(),
    )

    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(
            task="CLASSIFICATION",
            explainer={"class_index": 3, "confidence": 1.0},
        ),
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["detail"]["code"] == "invalid_request"
    assert execution.close_calls == 1


@pytest.mark.parametrize(
    ("single_probability", "class_index", "status"),
    [
        (False, None, HTTPStatus.BAD_REQUEST),
        (True, 0, HTTPStatus.BAD_REQUEST),
        (True, None, HTTPStatus.OK),
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
        data=_data(),
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
        data=_data(),
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
        data=_data(),
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
        (
            ProviderInvalidResponseError("malformed output"),
            HTTPStatus.BAD_GATEWAY,
            "invalid_response",
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
    assert "attributions" not in response.json()


def test_missing_background_and_missing_storage_use_shared_mapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map missing organic background and stored data to stable errors."""
    monkeypatch.setattr(local_shap, "_SHAP_AVAILABLE", True)

    async def no_background(*_args: object, **_kwargs: object) -> LocalExplanationData:
        """Represent the loader's missing organic-background failure."""
        message = "No organic background data is available"
        raise LocalDataNotFoundError(message)

    monkeypatch.setattr(local_shap, "load_local_explanation_data", no_background)
    response = TestClient(_app()).post(
        "/explainers/local/shap", json=_payload(explainer={"confidence": 1.0})
    )
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["detail"]["code"] == "data_missing"

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


def test_unexpected_loader_failure_remains_an_internal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep unexpected backend failures out of client-facing data errors."""
    monkeypatch.setattr(local_shap, "_SHAP_AVAILABLE", True)

    async def backend_failure(
        *_args: object, **_kwargs: object
    ) -> LocalExplanationData:
        """Represent an unexpected backend failure with private details."""
        message = "private backend failure"
        raise ValueError(message)

    monkeypatch.setattr(local_shap, "load_local_explanation_data", backend_failure)
    response = TestClient(_app()).post(
        "/explainers/local/shap",
        json=_payload(explainer={"confidence": 1.0}),
    )

    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert response.json()["detail"]["code"] == "execution_failed"
    assert "private backend failure" not in response.text


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
    assert failure.task == "REGRESSION"
    assert failure.output_name is None
    assert failure.class_index is None
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
