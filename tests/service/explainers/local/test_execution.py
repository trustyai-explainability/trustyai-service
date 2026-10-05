"""Tests for shared local-explainer prediction execution."""

from __future__ import annotations

import importlib
import sys
import time
from types import ModuleType

import numpy as np
import pytest

from trustyai_service.service.data.local_explanation import LocalExplanationData
from trustyai_service.service.explainers.local import execution as execution_module
from trustyai_service.service.explainers.local.error_mapping import (
    LocalDataError,
    map_error,
)
from trustyai_service.service.explainers.local.model_provider import (
    HttpTransportConfig,
    PredictionMetadata,
    ProviderDeadlineError,
    ProviderInvalidRequestError,
    ProviderInvalidResponseError,
    ProviderUnavailableError,
)
from trustyai_service.service.explainers.local.types import TaskType


def _data(*, width: int = 2) -> LocalExplanationData:
    """Build a small storage context for execution tests."""
    return LocalExplanationData(
        model_id="model",
        prediction_id="prediction",
        instance=np.ones(width),
        feature_names=[f"feature-{index}" for index in range(width)],
        background=np.ones((3, width)),
    )


def _spec(
    *,
    task: TaskType = TaskType.REGRESSION,
    input_name: str | None = None,
    output_name: str | None = None,
) -> execution_module.LocalExecutionSpec:
    """Build a canonical execution specification."""
    return execution_module.LocalExecutionSpec(
        base_url="http://model.example",
        model_name="credit-model",
        model_version="v1",
        input_name=input_name,
        output_name=output_name,
        task=task,
    )


def _transport() -> HttpTransportConfig:
    """Build an injected deployment transport policy."""
    return HttpTransportConfig(
        headers={},
        allowed_hosts=frozenset({"model.example"}),
    )


class _Provider:
    """Small synchronous provider recording calls and cleanup."""

    def __init__(self, width: int = 2) -> None:
        """Initialize metadata and call tracking."""
        self._metadata = PredictionMetadata(
            input_name="server-input",
            output_name="server-output",
            input_datatype="FP32",
            output_datatype="FP32",
            input_shape=(-1, width),
            output_shape=(-1, 1),
        )
        self.predict_calls = 0
        self.timeouts: list[float | None] = []
        self.close_calls = 0

    @property
    def metadata(self) -> PredictionMetadata:
        """Return negotiated provider metadata."""
        return self._metadata

    def predict(
        self,
        inputs: np.ndarray,
        *,
        timeout_seconds: float | None = None,
    ) -> np.ndarray:
        """Return one deterministic regression value per input row."""
        self.predict_calls += 1
        self.timeouts.append(timeout_seconds)
        return inputs.sum(axis=1, keepdims=True)

    def close(self) -> None:
        """Record idempotent cleanup."""
        self.close_calls += 1


def _install_provider_import(
    monkeypatch: pytest.MonkeyPatch,
    provider: _Provider,
    *,
    captured: dict[str, object] | None = None,
    connect_error: Exception | None = None,
) -> None:
    """Patch the dynamic module loader while retaining real imports."""
    real_import = importlib.import_module

    class ProviderType:
        """Provider class exposed by the fake KServe module."""

        @classmethod
        def connect(
            cls,
            spec: object,
            transport: object,
            *,
            timeout_seconds: float,
        ) -> _Provider:
            del cls
            if connect_error is not None:
                raise connect_error
            if captured is not None:
                captured.update(
                    spec=spec,
                    transport=transport,
                    timeout_seconds=timeout_seconds,
                )
            return provider

    class ModelSpec:
        """Minimal KServe model specification used by the fake module."""

        def __init__(
            self,
            *,
            base_url: str,
            model_name: str,
            model_version: str | None,
            input_name: str | None,
            output_name: str | None,
        ) -> None:
            """Store the provider identity and tensor selectors."""
            self.base_url = base_url
            self.model_name = model_name
            self.model_version = model_version
            self.input_name = input_name
            self.output_name = output_name

    def fake_import(name: str, package: str | None = None) -> ModuleType:
        """Return the fake model-provider module and retain other imports."""
        del package
        if name == "trustyai_service.service.explainers.local.kserve_v2_http":
            module = ModuleType(name)
            module.KServeModelSpec = ModelSpec  # type: ignore[attr-defined]
            module.KServeV2HttpPredictionProvider = ProviderType  # type: ignore[attr-defined]
            return module
        return real_import(name)

    monkeypatch.setattr(execution_module.importlib, "import_module", fake_import)


def test_model_execution_adapts_predictions_and_closes_idempotently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create a model provider and expose its adapted prediction callable."""
    provider = _Provider()
    captured: dict[str, object] = {}
    _install_provider_import(monkeypatch, provider, captured=captured)
    transport = _transport()
    monkeypatch.setattr(
        execution_module,
        "get_transport_config",
        lambda: pytest.fail("Injected transport was ignored"),
    )

    execution = execution_module.create_prediction_execution(
        _spec(),
        _data(),
        time.monotonic() + 5,
        transport,
    )

    result = execution.predict_fn(np.ones((2, 2)))

    assert execution.provider is provider
    assert execution.metadata is provider.metadata
    assert execution.resolved_input_name == "server-input"
    assert execution.resolved_output_name == "server-output"
    np.testing.assert_array_equal(result, [2.0, 2.0])
    assert captured["transport"] is transport
    execution.close()
    execution.close()
    assert provider.close_calls == 1


def test_model_propagates_identity_and_tensor_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass model identity and explicit tensor selectors to provider connect."""
    provider = _Provider()
    captured: dict[str, object] = {}
    _install_provider_import(monkeypatch, provider, captured=captured)

    execution_module.create_prediction_execution(
        _spec(input_name="features", output_name="probability"),
        _data(),
        time.monotonic() + 5,
        _transport(),
    )

    provider_spec = captured["spec"]
    assert provider_spec.base_url == "http://model.example"
    assert provider_spec.model_name == "credit-model"
    assert provider_spec.model_version == "v1"
    assert provider_spec.input_name == "features"
    assert provider_spec.output_name == "probability"


@pytest.mark.parametrize(
    "data_width",
    [
        2,
        3,
    ],
)
def test_model_width_mismatch_closes_provider_before_inference(
    monkeypatch: pytest.MonkeyPatch,
    data_width: int,
) -> None:
    """Reject instance/background width mismatches before any prediction call."""
    provider = _Provider(width=3)
    _install_provider_import(monkeypatch, provider)
    data = _data(width=data_width)
    if data_width == 3:
        data = LocalExplanationData(
            model_id=data.model_id,
            prediction_id=data.prediction_id,
            instance=data.instance,
            feature_names=data.feature_names,
            background=np.ones((3, 2)),
        )

    with pytest.raises(LocalDataError, match="feature width"):
        execution_module.create_prediction_execution(
            _spec(), data, time.monotonic() + 5, _transport()
        )

    assert provider.predict_calls == 0
    assert provider.close_calls == 1


def test_model_deadline_is_checked_before_connect_and_before_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject expired execution deadlines without contacting the provider."""
    provider = _Provider()
    _install_provider_import(monkeypatch, provider)
    transport_calls = 0

    def resolve_transport() -> HttpTransportConfig:
        nonlocal transport_calls
        transport_calls += 1
        return _transport()

    monkeypatch.setattr(execution_module, "get_transport_config", resolve_transport)
    with pytest.raises(ProviderDeadlineError):
        execution_module.create_prediction_execution(
            _spec(), _data(), time.monotonic() - 1, None
        )
    assert transport_calls == 0

    captured: dict[str, object] = {}
    _install_provider_import(monkeypatch, provider, captured=captured)
    execution = execution_module.create_prediction_execution(
        _spec(), _data(), time.monotonic() + 5, _transport()
    )
    expired = time.monotonic() + 10
    monkeypatch.setattr(execution_module.time, "monotonic", lambda: expired)
    with pytest.raises(ProviderDeadlineError):
        execution.predict_fn(np.ones((1, 2)))
    assert provider.predict_calls == 0


def test_model_connect_uses_remaining_deadline_after_delayed_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recompute the provider budget after setup has consumed deadline time."""
    provider = _Provider()
    captured: dict[str, float] = {}
    clock = 0.0
    deadline = 10.0

    def monotonic() -> float:
        """Return the deterministic monotonic clock used by this regression."""
        return clock

    def advance() -> None:
        """Consume one second of the execution deadline."""
        nonlocal clock
        clock += 1.0

    class ProviderType:
        """Provider class exposed after delayed dynamic loading."""

        @classmethod
        def connect(
            cls,
            spec: object,
            transport: object,
            *,
            timeout_seconds: float,
        ) -> _Provider:
            """Record the timeout supplied after model setup."""
            del cls, spec, transport
            captured["timeout_seconds"] = timeout_seconds
            return provider

    class ModelSpec:
        """Model specification whose construction consumes setup time."""

        def __init__(self, **_kwargs: object) -> None:
            """Consume the model-spec construction arguments."""
            advance()

    def load_provider() -> tuple[type, type]:
        """Simulate delayed dynamic provider import."""
        advance()
        return ProviderType, ModelSpec

    def resolve_transport() -> HttpTransportConfig:
        """Simulate delayed transport configuration resolution."""
        advance()
        return _transport()

    monkeypatch.setattr(execution_module.time, "monotonic", monotonic)
    monkeypatch.setattr(execution_module, "get_transport_config", resolve_transport)
    monkeypatch.setattr(execution_module, "_load_model_provider", load_provider)

    execution = execution_module.create_prediction_execution(
        _spec(), _data(), deadline, None
    )

    assert captured["timeout_seconds"] == pytest.approx(7.0)
    execution.close()


def test_model_prediction_uses_remaining_deadline_as_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cap each provider call by the remaining monotonic execution deadline."""
    provider = _Provider()
    _install_provider_import(monkeypatch, provider)
    deadline = time.monotonic() + 5
    execution = execution_module.create_prediction_execution(
        _spec(), _data(), deadline, _transport()
    )
    monkeypatch.setattr(execution_module.time, "monotonic", lambda: deadline - 1.25)

    execution.predict_fn(np.ones((1, 2)))

    assert provider.timeouts == [pytest.approx(1.25)]


@pytest.mark.parametrize("phase", ["connect", "predict"])
def test_model_provider_failure_never_falls_back_to_surrogate(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    """Map provider failures without importing or fitting a local fallback."""
    error = ProviderUnavailableError("provider offline")

    class FailingProvider(_Provider):
        """Provider whose inference always raises the original failure."""

        def predict(
            self,
            inputs: np.ndarray,
            *,
            timeout_seconds: float | None = None,
        ) -> np.ndarray:
            """Fail inference after provider construction has succeeded."""
            del inputs, timeout_seconds
            raise error

    provider = FailingProvider()
    _install_provider_import(
        monkeypatch,
        provider,
        connect_error=error if phase == "connect" else None,
    )
    real_import = execution_module.importlib.import_module
    module_name = "trustyai_service.core.explainers.local.surrogate"
    assert module_name not in sys.modules

    def reject_surrogate(name: str, package: str | None = None) -> ModuleType:
        """Reject any attempt to load the removed fallback module."""
        if name == module_name:
            message = "surrogate fallback was attempted"
            raise AssertionError(message)
        return real_import(name, package)

    monkeypatch.setattr(execution_module.importlib, "import_module", reject_surrogate)
    execution = None

    def run_prediction() -> None:
        """Exercise construction and inference through the shared model path."""
        nonlocal execution
        execution = execution_module.create_prediction_execution(
            _spec(), _data(), time.monotonic() + 5, _transport()
        )
        execution.predict_fn(np.ones((1, 2)))

    try:
        with pytest.raises(
            ProviderUnavailableError, match="provider offline"
        ) as raised:
            run_prediction()
    finally:
        if execution is not None:
            execution.close()
            execution.close()

    assert raised.value is error
    response = map_error(raised.value)
    assert response.status_code == 503
    assert response.code == "unavailable"
    assert response.detail == "Model provider is unavailable"
    assert module_name not in sys.modules
    assert provider.close_calls == (1 if phase == "predict" else 0)


def test_model_classification_preserves_probability_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose model class probabilities unchanged to the explainers."""

    class ClassificationProvider(_Provider):
        """Provider returning a complete three-class probability matrix."""

        def predict(
            self,
            inputs: np.ndarray,
            *,
            timeout_seconds: float | None = None,
        ) -> np.ndarray:
            """Return deterministic class probabilities for each row."""
            del timeout_seconds
            return np.tile([[0.2, 0.3, 0.5]], (len(inputs), 1))

    provider = ClassificationProvider()
    _install_provider_import(monkeypatch, provider)
    execution = execution_module.create_prediction_execution(
        _spec(task=TaskType.CLASSIFICATION),
        _data(),
        time.monotonic() + 5,
        _transport(),
    )
    try:
        assert execution.task is TaskType.CLASSIFICATION
        np.testing.assert_allclose(
            execution.predict_fn(np.ones((2, 2))),
            [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]],
        )
    finally:
        execution.close()
    assert provider.close_calls == 1


def test_model_invalid_prediction_preserves_typed_error_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a wider regression output at the shared adaptation boundary."""

    class InvalidProvider(_Provider):
        """Provider returning two outputs for a scalar regression task."""

        def predict(
            self,
            inputs: np.ndarray,
            *,
            timeout_seconds: float | None = None,
        ) -> np.ndarray:
            """Return raw outputs that the adapter must reject."""
            del timeout_seconds
            return inputs

    provider = InvalidProvider()
    _install_provider_import(monkeypatch, provider)
    execution = execution_module.create_prediction_execution(
        _spec(), _data(), time.monotonic() + 5, _transport()
    )
    try:
        with pytest.raises(ProviderInvalidResponseError, match="one numeric output"):
            execution.predict_fn(np.ones((1, 2)))
    finally:
        execution.close()
        execution.close()
    assert provider.close_calls == 1


def test_model_requires_a_base_url_before_provider_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a missing model URL before resolving transport or loading a provider."""
    spec = execution_module.LocalExecutionSpec(
        base_url=None,
        model_name="credit-model",
        model_version=None,
        input_name=None,
        output_name=None,
        task=TaskType.REGRESSION,
    )
    monkeypatch.setattr(
        execution_module,
        "_load_model_provider",
        lambda: pytest.fail("Provider was loaded without a model URL"),
    )
    monkeypatch.setattr(
        execution_module,
        "get_transport_config",
        lambda: pytest.fail("Transport was resolved without a model URL"),
    )
    with pytest.raises(ProviderInvalidRequestError, match="base URL"):
        execution_module.create_prediction_execution(
            spec, _data(), time.monotonic() + 5, None
        )
