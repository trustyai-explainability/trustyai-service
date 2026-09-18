"""Tests for shared MODEL/SURROGATE execution selection."""

from types import SimpleNamespace

import numpy as np
import pytest

from trustyai_service.service.data.local_explanation import LocalExplanationData
from trustyai_service.service.explainers.local import execution as execution_module
from trustyai_service.service.explainers.local.kserve_v2_http import (
    KServeV2HttpPredictionProvider,
)
from trustyai_service.service.explainers.local.model_provider import (
    HttpTransportConfig,
    PredictionMetadata,
)
from trustyai_service.service.explainers.local.types import PredictionSource, TaskType


def _data(
    *,
    targets: np.ndarray | None = None,
    width: int = 2,
    output_names: list[str] | None = None,
) -> LocalExplanationData:
    return LocalExplanationData(
        model_id="m",
        prediction_id="p",
        instance=np.ones(width),
        feature_names=[f"f{i}" for i in range(width)],
        background=np.ones((3, width)),
        background_output=targets,
        output_names=[] if output_names is None else output_names,
    )


def _config(source: PredictionSource) -> SimpleNamespace:
    return SimpleNamespace(
        prediction_source=source,
        task=TaskType.REGRESSION,
        base_url="http://model.example",
        model_name="credit-model",
        model_version="v1",
        input_name=None,
        output_name=None,
    )


class _Provider:
    metadata = PredictionMetadata("input", "output", "FP32", "FP32", (-1, 2), (-1, 1))

    def __init__(self) -> None:
        self.closed = False

    def predict(
        self, values: np.ndarray, *, timeout_seconds: float | None = None
    ) -> np.ndarray:
        assert timeout_seconds is not None
        assert timeout_seconds > 0
        return values.sum(axis=1, keepdims=True)

    def close(self) -> None:
        self.closed = True


def test_model_provider_is_lazy_and_receives_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create the real-model provider lazily with deployment transport settings."""
    provider = _Provider()
    captured: dict[str, object] = {}

    def connect(spec: object, timeout: float, transport: object) -> _Provider:
        captured.update(spec=spec, timeout=timeout, transport=transport)
        return provider

    monkeypatch.setattr(KServeV2HttpPredictionProvider, "connect", connect)
    transport = HttpTransportConfig(
        headers={}, allowed_hosts=frozenset({"model.example"})
    )
    execution = execution_module.create_execution(
        _config(PredictionSource.MODEL), _data(), 5, transport=transport
    )
    result = execution.predict_fn(np.ones((2, 2)))

    assert result.shape == (2,)
    assert captured["timeout"] == pytest.approx(5, abs=0.01)
    assert captured["transport"] is transport
    execution.close()
    execution.close()
    assert provider.closed


def test_model_rejects_stored_feature_width_before_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a stored target whose feature width disagrees with model metadata."""
    provider = _Provider()

    def connect(*_args: object, **_kwargs: object) -> _Provider:
        return provider

    monkeypatch.setattr(KServeV2HttpPredictionProvider, "connect", connect)
    with pytest.raises(ValueError, match="feature width"):
        execution_module.create_execution(
            _config(PredictionSource.MODEL),
            _data(width=3),
            5,
            transport=HttpTransportConfig(
                headers={}, allowed_hosts=frozenset({"model.example"})
            ),
        )
    assert provider.closed


def test_model_rejects_dynamic_scalar_width_before_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat KServe's dynamic scalar shape as a one-feature input contract."""
    provider = _Provider()
    provider.metadata = PredictionMetadata(
        "input", "output", "FP32", "FP32", (-1,), (-1, 1)
    )

    def connect(*_args: object, **_kwargs: object) -> _Provider:
        return provider

    monkeypatch.setattr(KServeV2HttpPredictionProvider, "connect", connect)
    with pytest.raises(ValueError, match="feature width"):
        execution_module.create_execution(
            _config(PredictionSource.MODEL),
            _data(width=2),
            5,
            transport=HttpTransportConfig(
                headers={}, allowed_hosts=frozenset({"model.example"})
            ),
        )
    assert provider.closed


def test_surrogate_does_not_construct_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep explicit surrogate execution independent from HTTP provider creation."""

    def fail_connect(*_args: object, **_kwargs: object) -> None:
        raise AssertionError

    monkeypatch.setattr(KServeV2HttpPredictionProvider, "connect", fail_connect)
    execution = execution_module.create_execution(
        _config(PredictionSource.SURROGATE),
        _data(targets=np.array([0.0, 1.0, 0.0])),
        5,
    )
    assert execution.provider is None
    assert execution.predict_fn(np.ones((1, 2))).shape == (1,)


def test_surrogate_requires_stored_targets() -> None:
    """Require stored output labels before constructing a surrogate execution."""
    with pytest.raises(ValueError, match="stored organic labels"):
        execution_module.create_execution(
            _config(PredictionSource.SURROGATE), _data(), 5
        )


def test_surrogate_rejects_output_alias_width_mismatch() -> None:
    """Reject a stored output matrix whose aliases cannot select its columns."""
    with pytest.raises(ValueError, match="columns"):
        execution_module.create_execution(
            _config(PredictionSource.SURROGATE),
            _data(
                targets=np.array([[0.0], [1.0], [0.0]]),
                output_names=["score", "other"],
            ),
            5,
        )
