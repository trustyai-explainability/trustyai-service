"""Real FastAPI-to-KServe integration tests for local LIME."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from .integration_helpers import FakeKServe, LocalStorage, enabled_test_client


def _request(
    base_url: str,
    *,
    task: str = "REGRESSION",
    class_index: int | None = None,
) -> dict[str, object]:
    """Build one canonical deployed-model request."""
    model: dict[str, object] = {
        "model_name": "m",
        "model_version": "v1",
        "task": task,
        "base_url": base_url,
        "input_name": "input",
        "output_name": "output",
    }
    explainer: dict[str, object] = {
        "num_samples": 24,
        "n_training_rows": 2,
        "num_features": 2,
        "confidence": 1.0,
        "timeout": 30,
        "seed": 7,
    }
    if class_index is not None:
        explainer["class_index"] = class_index
    return {
        "predictionId": "target",
        "config": {
            "model": model,
            "explainer": explainer,
        },
    }


@pytest.mark.skipif(
    not importlib.util.find_spec("lime"),
    reason="optional LIME dependency is unavailable",
)
def test_model_lime_calls_loopback_provider_and_excludes_target_and_synthetic_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the actual FastAPI route and provider with organic-only training data."""
    storage = LocalStorage()
    captured: dict[str, object] = {}

    with FakeKServe(model_name="m", model_version="v1") as fake:
        monkeypatch.setenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "127.0.0.1")
        data_module = importlib.import_module(
            "trustyai_service.service.data.local_explanation"
        )
        monkeypatch.setattr(
            data_module,
            "get_global_storage_interface",
            lambda: storage,
        )

        async def load_data(*args: object, **kwargs: object) -> object:
            """Record the shared loader result while retaining real storage behavior."""
            module = importlib.import_module(
                "trustyai_service.service.data.local_explanation"
            )
            data = await module.load_local_explanation_data(*args, **kwargs)
            captured["data"] = data
            return data

        with enabled_test_client() as client:
            lime_module = importlib.import_module(
                "trustyai_service.endpoints.explainers.local_lime"
            )
            monkeypatch.setattr(lime_module, "load_local_explanation_data", load_data)

            response = client.post(
                "/explainers/local/lime", json=_request(fake.base_url)
            )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model"] == "m"
    assert payload["output_name"] == "output"
    assert payload["prediction_output"] == pytest.approx(1.0)

    data = captured["data"]
    np.testing.assert_array_equal(data.instance, [2.0, 2.0])  # type: ignore[union-attr]
    np.testing.assert_array_equal(
        data.background,  # type: ignore[union-attr]
        [[1.0, 0.0], [0.0, 1.0]],
    )
    assert fake.metadata_calls == 1
    assert fake.infer_calls[0]["inputs"][0]["shape"] == [1, 2]
    perturbation_batches = [call["inputs"][0] for call in fake.infer_calls[1:]]
    assert perturbation_batches
    assert sum(batch["shape"][0] for batch in perturbation_batches) == 24
    assert any(batch["shape"][0] > 1 for batch in perturbation_batches)
    perturbations = np.concatenate(
        [
            np.asarray(batch["data"]).reshape(batch["shape"])
            for batch in perturbation_batches
        ]
    )
    assert np.any(perturbations != np.asarray([2.0, 2.0]))
    assert fake.metadata_paths == ["/v2/models/m/versions/v1"]
    assert all(path == "/v2/models/m/versions/v1/infer" for path in fake.infer_paths)
    assert all(
        call["inputs"][0]["name"] == "input"  # type: ignore[index]
        for call in fake.infer_calls
    )


@pytest.mark.skipif(
    not importlib.util.find_spec("lime"),
    reason="optional LIME dependency is unavailable",
)
def test_classification_model_lime_preserves_selected_class_and_full_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real classification calls and LIME class selection semantics."""
    storage = LocalStorage()

    with FakeKServe(classification=True, model_name="m", model_version="v1") as fake:
        monkeypatch.setenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "127.0.0.1")
        data_module = importlib.import_module(
            "trustyai_service.service.data.local_explanation"
        )
        monkeypatch.setattr(
            data_module,
            "get_global_storage_interface",
            lambda: storage,
        )

        with enabled_test_client() as client:
            response = client.post(
                "/explainers/local/lime",
                json=_request(
                    fake.base_url,
                    task="CLASSIFICATION",
                    class_index=0,
                ),
            )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["task"] == "CLASSIFICATION"
    assert payload["class_index"] == 0
    assert payload["prediction_output"] == pytest.approx([0.25, 0.75])
    assert payload["local_prediction"] == pytest.approx(0.25, abs=0.15)
    assert fake.metadata_calls == 1
    assert fake.infer_calls


@pytest.mark.skipif(
    not importlib.util.find_spec("lime"),
    reason="optional LIME dependency is unavailable",
)
def test_model_lime_maps_real_loopback_provider_outage_with_stable_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the mapped upstream outage without a fallback explanation."""
    storage = LocalStorage()

    with FakeKServe(metadata_status=503) as fake:
        monkeypatch.setenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "127.0.0.1")
        data_module = importlib.import_module(
            "trustyai_service.service.data.local_explanation"
        )
        monkeypatch.setattr(
            data_module,
            "get_global_storage_interface",
            lambda: storage,
        )

        with enabled_test_client() as client:
            response = client.post(
                "/explainers/local/lime", json=_request(fake.base_url)
            )

    assert response.status_code == 503, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "unavailable"
    assert detail["message"] == "Model provider is unavailable"
    assert set(response.json()) == {"detail"}
    assert "invalid KServe" not in response.text
    assert fake.metadata_calls == 1
    assert not fake.infer_calls
