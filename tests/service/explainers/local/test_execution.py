"""Tests for shared local-explainer prediction execution."""

from __future__ import annotations

import importlib
import time
from types import ModuleType

import numpy as np
import pytest

from trustyai_service.service.data.local_explanation import LocalExplanationData
from trustyai_service.service.explainers.local import execution as execution_module
from trustyai_service.service.explainers.local.error_mapping import LocalDataError
from trustyai_service.service.explainers.local.model_provider import (
    HttpTransportConfig,
    PredictionMetadata,
    ProviderDeadlineError,
    ProviderUnavailableError,
)
from trustyai_service.service.explainers.local.types import PredictionSource, TaskType


def _data(
    *,
    width: int = 2,
    background_output: np.ndarray | None = None,
    output_names: list[str] | None = None,
) -> LocalExplanationData:
    """Build a small storage context for execution tests."""
    return LocalExplanationData(
        model_id="model",
        prediction_id="prediction",
        instance=np.ones(width),
        feature_names=[f"feature-{index}" for index in range(width)],
        background=np.ones((3, width)),
        background_output=background_output,
        output_names=[] if output_names is None else output_names,
        input_tensor_name="stored-input",
        output_tensor_name="stored-output",
    )


def _spec(
    source: PredictionSource = PredictionSource.MODEL,
    *,
    task: TaskType = TaskType.REGRESSION,
    input_name: str | None = None,
    output_name: str | None = None,
) -> execution_module.LocalExecutionSpec:
    """Build a canonical execution specification."""
    return execution_module.LocalExecutionSpec(
        prediction_source=source,
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
    surrogate_builder: object | None = None,
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
        """Return fake optional modules only at the requested branch."""
        del package
        if name == "trustyai_service.service.explainers.local.kserve_v2_http":
            module = ModuleType(name)
            module.KServeModelSpec = ModelSpec  # type: ignore[attr-defined]
            module.KServeV2HttpPredictionProvider = ProviderType  # type: ignore[attr-defined]
            return module
        if (
            name == "trustyai_service.core.explainers.local.surrogate"
            and surrogate_builder is not None
        ):
            module = ModuleType(name)
            module.build_surrogate = surrogate_builder  # type: ignore[attr-defined]
            return module
        return real_import(name)

    monkeypatch.setattr(execution_module.importlib, "import_module", fake_import)


def test_model_is_the_default_execution_branch_and_closes_idempotently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create a provider for MODEL and expose its adapted prediction callable."""
    provider = _Provider()
    captured: dict[str, object] = {}
    _install_provider_import(monkeypatch, provider, captured=captured)

    execution = execution_module.create_prediction_execution(
        _spec(),
        _data(),
        time.monotonic() + 5,
        _transport(),
    )

    result = execution.predict_fn(np.ones((2, 2)))

    assert execution.source is PredictionSource.MODEL
    assert execution.provider is provider
    assert execution.metadata is provider.metadata
    assert execution.resolved_input_name == "server-input"
    assert execution.resolved_output_name == "server-output"
    assert result.shape == (2,)
    assert captured["transport"] is not None
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
            background_output=data.background_output,
            output_names=data.output_names,
            input_tensor_name=data.input_tensor_name,
            output_tensor_name=data.output_tensor_name,
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

    def resolve_transport(source: PredictionSource) -> HttpTransportConfig:
        nonlocal transport_calls
        transport_calls += 1
        assert source is PredictionSource.MODEL
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

    def resolve_transport(source: PredictionSource) -> HttpTransportConfig:
        """Simulate delayed transport configuration resolution."""
        assert source is PredictionSource.MODEL
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


def test_model_provider_failure_never_falls_back_to_surrogate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate provider failures without importing or fitting a surrogate."""
    provider = _Provider()
    _install_provider_import(
        monkeypatch,
        provider,
        connect_error=ProviderUnavailableError("provider offline"),
    )
    real_import = execution_module.importlib.import_module

    def reject_surrogate(name: str, package: str | None = None) -> ModuleType:
        if name == "trustyai_service.core.explainers.local.surrogate":
            message = "surrogate fallback was attempted"
            raise AssertionError(message)
        return real_import(name, package)

    monkeypatch.setattr(execution_module.importlib, "import_module", reject_surrogate)
    with pytest.raises(ProviderUnavailableError, match="provider offline"):
        execution_module.create_prediction_execution(
            _spec(), _data(), time.monotonic() + 5, _transport()
        )


def test_surrogate_is_explicit_provider_free_and_selects_one_output_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fit only the explicit surrogate branch without resolving transport."""
    captured: dict[str, object] = {}

    class Estimator:
        """Minimal fitted surrogate returned by the fake builder."""

        def predict(self, values: np.ndarray) -> np.ndarray:
            return values[:, 0]

    def build_surrogate(
        inputs: np.ndarray,
        outputs: np.ndarray,
        task: str,
    ) -> Estimator:
        captured.update(inputs=inputs, outputs=outputs, task=task)
        return Estimator()

    _install_provider_import(
        monkeypatch,
        _Provider(),
        surrogate_builder=build_surrogate,
    )
    monkeypatch.setattr(
        execution_module,
        "get_transport_config",
        lambda _source: pytest.fail("SURROGATE resolved transport settings"),
    )

    data = _data(
        background_output=np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        output_names=["first", "second"],
    )
    execution = execution_module.create_prediction_execution(
        _spec(PredictionSource.SURROGATE, output_name="second"),
        data,
        time.monotonic() + 5,
        None,
    )

    np.testing.assert_array_equal(captured["inputs"], data.background)
    np.testing.assert_array_equal(captured["outputs"], [10.0, 20.0, 30.0])
    assert captured["task"] == TaskType.REGRESSION.value
    assert execution.provider is None
    assert execution.resolved_output_name == "second"
    np.testing.assert_array_equal(execution.predict_fn(np.ones((2, 2))), [1.0, 1.0])
    execution.close()


def test_surrogate_requires_output_selection_for_multiple_stored_columns() -> None:
    """Reject ambiguous stored labels instead of guessing an output column."""
    data = _data(
        background_output=np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        output_names=["first", "second"],
    )

    with pytest.raises(LocalDataError, match="output_name"):
        execution_module.create_prediction_execution(
            _spec(PredictionSource.SURROGATE),
            data,
            time.monotonic() + 5,
            None,
        )


@pytest.mark.parametrize(
    "background_output",
    [
        np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]]),
    ],
)
def test_classification_surrogate_rejects_probability_matrix_before_selection(
    monkeypatch: pytest.MonkeyPatch,
    background_output: np.ndarray,
) -> None:
    """Do not turn stored class-score columns into explicit surrogate labels."""
    builder_calls = 0

    class Estimator:
        """Minimal estimator proving whether surrogate fitting was reached."""

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            return np.tile(np.array([[0.4, 0.6]]), (len(values), 1))

    def build_surrogate(
        inputs: np.ndarray,
        outputs: np.ndarray,
        task: str,
    ) -> Estimator:
        nonlocal builder_calls
        del inputs, outputs, task
        builder_calls += 1
        return Estimator()

    _install_provider_import(
        monkeypatch,
        _Provider(),
        surrogate_builder=build_surrogate,
    )
    data = _data(
        background_output=background_output,
        output_names=["class-zero", "class-one"],
    )

    with pytest.raises(LocalDataError, match=r"probability|score"):
        execution_module.create_prediction_execution(
            _spec(
                PredictionSource.SURROGATE,
                task=TaskType.CLASSIFICATION,
                output_name="class-one",
            ),
            data,
            time.monotonic() + 5,
            None,
        )

    assert builder_calls == 0


def test_classification_surrogate_accepts_selected_discrete_multi_output_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve a discrete label column when its output is explicitly selected."""
    captured: dict[str, object] = {}

    class Estimator:
        """Minimal classification estimator returned by the fake builder."""

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            return np.tile(np.array([[0.25, 0.75]]), (len(values), 1))

    def build_surrogate(
        inputs: np.ndarray,
        outputs: np.ndarray,
        task: str,
    ) -> Estimator:
        captured.update(inputs=inputs, outputs=outputs, task=task)
        return Estimator()

    _install_provider_import(
        monkeypatch,
        _Provider(),
        surrogate_builder=build_surrogate,
    )
    data = _data(
        background_output=np.array(
            [[0.0, 10.0], [1.0, 20.0], [0.0, 30.0]],
        ),
        output_names=["label", "other-output"],
    )

    execution = execution_module.create_prediction_execution(
        _spec(
            PredictionSource.SURROGATE,
            task=TaskType.CLASSIFICATION,
            output_name="label",
        ),
        data,
        time.monotonic() + 5,
        None,
    )

    np.testing.assert_array_equal(captured["outputs"], [0.0, 1.0, 0.0])
    assert captured["task"] == TaskType.CLASSIFICATION.value
    assert execution.resolved_output_name == "label"
    np.testing.assert_allclose(
        execution.predict_fn(np.ones((2, 2))), [[0.25, 0.75]] * 2
    )


def test_classification_surrogate_accepts_discrete_label_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep ordinary stored classification labels valid without output selection."""
    captured: dict[str, object] = {}

    class Estimator:
        """Minimal classification estimator returned by the fake builder."""

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            return np.tile(np.array([[0.25, 0.75]]), (len(values), 1))

    def build_surrogate(
        inputs: np.ndarray,
        outputs: np.ndarray,
        task: str,
    ) -> Estimator:
        del inputs
        captured["outputs"] = outputs
        captured["task"] = task
        return Estimator()

    _install_provider_import(
        monkeypatch,
        _Provider(),
        surrogate_builder=build_surrogate,
    )
    data = _data(
        background_output=np.array([0.0, 1.0, 0.0]),
        output_names=["label"],
    )

    execution = execution_module.create_prediction_execution(
        _spec(PredictionSource.SURROGATE, task=TaskType.CLASSIFICATION),
        data,
        time.monotonic() + 5,
        None,
    )

    np.testing.assert_array_equal(captured["outputs"], [0.0, 1.0, 0.0])
    assert captured["task"] == TaskType.CLASSIFICATION.value
    assert execution.resolved_output_name == "label"
