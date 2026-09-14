"""Unified test factories for local explainer endpoints (LIME, SHAP, etc.).

Local explainer endpoints differ from global ones:
- Request key is ``predictionId`` (instance lookup), not ``modelId``
- Storage uses a module-level ``storage_interface`` singleton, not an injected
  ``get_data_source`` function
- Error paths include a 400 for unknown predictionId (not present in global)

For global explainer factories (PDP, global LIME) see
``tests/endpoints/explainers/global/factory.py`` (not yet created — no global
explainer has a real implementation).

All local explainer endpoints share:
- POST /explainers/local/{explainer} — compute explanation for a stored prediction
- A module-level ``storage_interface`` singleton (patch it directly, not via a getter)
- ``get_shared_data_source`` function call inside the handler (patch the function)
- An availability flag (e.g. ``_LIME_AVAILABLE``) checked at handler entry

Patch targets for every factory function (``MODULE`` defined below):
- ``{MODULE}.storage_interface``                  — module-level singleton
- ``{MODULE}.get_shared_data_source``             — called inside the handler
- ``{MODULE}.{_LIME_AVAILABLE|_SHAP_AVAILABLE}``  — availability guard
- ``MODEL_DATA_MODULE.get_global_storage_interface`` — used by ModelData.data()
"""

from collections.abc import Callable
from http import HTTPStatus
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

MODULE = "trustyai_service.endpoints.explainers.local_explainer"
MODEL_DATA_MODULE = "trustyai_service.service.data.model_data"


def _make_storage_mocks(
    pred_id: str,
    feature_names: list[str],
    output_names: list[str],
    n_organic_rows: int = 30,
) -> tuple[MagicMock, MagicMock]:
    """Return ``(mock_storage_interface, mock_data_source)`` for happy-path tests.

    ``mock_storage_interface`` simulates:
    - ``dataset_exists`` → True
    - ``read_data`` → metadata array (col-0 = pred_id) for metadata datasets,
      input row array for input datasets
    - ``get_aliased_column_names`` → ``feature_names`` for input datasets,
      ``output_names`` for output datasets

    ``mock_data_source`` simulates:
    - ``get_organic_dataframe`` → DataFrame with feature + output columns
    """
    rng = np.random.default_rng(0)
    n_features = len(feature_names)

    # Metadata: col-0 is prediction ID (other cols are timestamp, score, tags)
    metadata = np.array([[pred_id, "2025-01-01T00:00:00", 1.0, []]], dtype="O")
    # One input row matching feature_names
    input_row = rng.standard_normal((1, n_features))
    # Organic training data
    organic_data: dict[str, np.ndarray] = {
        name: rng.standard_normal(n_organic_rows) for name in feature_names
    }
    for out in output_names:
        organic_data[out] = rng.standard_normal(n_organic_rows)
    organic_df = pd.DataFrame(organic_data)

    mock_si = MagicMock()
    mock_si.dataset_exists = AsyncMock(return_value=True)
    mock_si.read_data = AsyncMock(
        side_effect=lambda name, *_a, **_kw: (
            metadata if "metadata" in name else input_row
        )
    )
    mock_si.get_aliased_column_names = AsyncMock(
        side_effect=lambda name: feature_names if "input" in name else output_names
    )

    mock_ds = MagicMock()
    mock_ds.get_organic_dataframe = AsyncMock(return_value=organic_df)

    return mock_si, mock_ds


def make_explainer_unavailable_test(
    explainer_name: str,
    availability_flag: str,
    endpoint_path: str,
    client: TestClient,
    request_payload: dict[str, Any],
) -> Callable[[object], None]:
    """503 when the explainer package is not installed.

    Args:
        explainer_name: Human-readable name for assertion messages.
        availability_flag: Full dotted path to the availability bool, e.g.
            ``"trustyai_service.endpoints.explainers.local_explainer._LIME_AVAILABLE"``.
        endpoint_path: HTTP path, e.g. ``"/explainers/local/lime"``.
        client: ``TestClient`` wired to the explainer router.
        request_payload: Valid JSON body for the POST request.

    """

    @patch(availability_flag, new=False)
    def test_impl(_: object) -> None:
        response = client.post(endpoint_path, json=request_payload)
        assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE, (
            f"{explainer_name}: expected 503, got {response.status_code}: {response.text}"
        )
        assert "not installed" in response.json()["detail"].lower(), (
            f"{explainer_name}: expected 'not installed' in detail"
        )

    return test_impl


def make_model_not_found_test(
    explainer_name: str,
    endpoint_path: str,
    client: TestClient,
    request_payload: dict[str, Any],
    availability_flag: str | None = None,
) -> Callable[[object], None]:
    """404 when the model has no stored data (dataset does not exist).

    Args:
        explainer_name: Human-readable name for assertion messages.
        endpoint_path: HTTP path, e.g. ``"/explainers/local/lime"``.
        client: ``TestClient`` wired to the explainer router.
        request_payload: Valid JSON body for the POST request.
        availability_flag: Full dotted path to the availability bool (e.g.
            ``"...local_explainer._SHAP_AVAILABLE"``).  When provided, the flag
            is patched to ``True`` so that the handler reaches the storage check
            even when the explainer package is not installed.

    """

    def _check_not_found_404(mock_si: MagicMock) -> None:
        mock_si.dataset_exists = AsyncMock(return_value=False)
        response = client.post(endpoint_path, json=request_payload)
        assert response.status_code == HTTPStatus.NOT_FOUND, (
            f"{explainer_name}: expected 404, got {response.status_code}: {response.text}"
        )
        assert "no data found" in response.json()["detail"].lower(), (
            f"{explainer_name}: expected 'no data found' in detail"
        )

    def _make_test(avail_flag: str | None) -> Callable[[object], None]:
        @patch(f"{MODULE}.storage_interface")
        def test_impl_no_avail(_: object, mock_si: MagicMock) -> None:
            _check_not_found_404(mock_si)

        if avail_flag is None:
            return test_impl_no_avail

        @patch(avail_flag, new=True)
        @patch(f"{MODULE}.storage_interface")
        def test_impl_with_avail(_: object, mock_si: MagicMock) -> None:
            _check_not_found_404(mock_si)

        return test_impl_with_avail

    return _make_test(availability_flag)


def make_prediction_id_not_found_test(
    explainer_name: str,
    endpoint_path: str,
    client: TestClient,
    request_payload: dict[str, Any],
    availability_flag: str | None = None,
) -> Callable[[object], None]:
    """400 when the predictionId is absent from stored metadata.

    The metadata dataset exists but contains a *different* prediction ID,
    so the scan exhausts all rows and raises 400.

    Args:
        explainer_name: Human-readable name for assertion messages.
        endpoint_path: HTTP path.
        client: ``TestClient`` wired to the explainer router.
        request_payload: JSON body whose ``predictionId`` does NOT match
            the ID stored in the mock metadata.  The stored ID is derived
            from ``request_payload["predictionId"]`` to guarantee a mismatch.
        availability_flag: Full dotted path to the availability bool.  When
            provided, patched to ``True`` so the handler reaches the storage
            check even when the explainer package is not installed.

    """

    def _make_test(avail_flag: str | None) -> Callable[[object], None]:
        requested_pred_id = request_payload["predictionId"]
        stored_other = f"other-{requested_pred_id}"
        wrong_metadata = np.array(
            [[stored_other, "2025-01-01T00:00:00", 1.0, []]], dtype="O"
        )

        def _check_pred_id_not_found_400(mock_si: MagicMock) -> None:
            mock_si.dataset_exists = AsyncMock(return_value=True)
            mock_si.read_data = AsyncMock(return_value=wrong_metadata)
            response = client.post(endpoint_path, json=request_payload)
            assert response.status_code == HTTPStatus.BAD_REQUEST, (
                f"{explainer_name}: expected 400, got {response.status_code}: {response.text}"
            )
            assert "not found" in response.json()["detail"].lower(), (
                f"{explainer_name}: expected 'not found' in detail"
            )

        @patch(f"{MODULE}.storage_interface")
        def test_impl_no_avail(_: object, mock_si: MagicMock) -> None:
            _check_pred_id_not_found_400(mock_si)

        if avail_flag is None:
            return test_impl_no_avail

        @patch(avail_flag, new=True)
        @patch(f"{MODEL_DATA_MODULE}.get_global_storage_interface")
        @patch(f"{MODULE}.storage_interface")
        def test_impl_with_avail(
            _: object, mock_si: MagicMock, mock_global: MagicMock
        ) -> None:
            mock_global.return_value = mock_si
            _check_pred_id_not_found_400(mock_si)

        return test_impl_with_avail

    return _make_test(availability_flag)


def make_missing_field_validation_test(
    explainer_name: str,
    endpoint_path: str,
    client: TestClient,
    invalid_payload: dict[str, Any],
    expected_field: str,
) -> Callable[[object], None]:
    """422 when a required field is absent from the request body.

    Args:
        explainer_name: Human-readable name for assertion messages.
        endpoint_path: HTTP path.
        client: ``TestClient`` wired to the explainer router.
        invalid_payload: JSON body with a required field omitted
            (e.g. missing ``predictionId``).
        expected_field: Field name expected to appear somewhere in the
            422 error detail (e.g. ``"predictionId"``).

    """

    def test_impl(_: object) -> None:
        response = client.post(endpoint_path, json=invalid_payload)
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY, (
            f"{explainer_name}: expected 422, got {response.status_code}: {response.text}"
        )
        detail = response.json()["detail"]
        if isinstance(detail, list):
            detail_text = " ".join(
                " ".join(str(s) for s in err.get("loc", []))
                + " "
                + str(err.get("msg", ""))
                for err in detail
                if isinstance(err, dict)
            )
        else:
            detail_text = str(detail)
        assert expected_field.lower() in detail_text.lower(), (
            f"{explainer_name}: expected '{expected_field}' in 422 detail, got: {detail}"
        )

    return test_impl


def make_compute_endpoint_test(
    explainer_name: str,
    endpoint_path: str,
    client: TestClient,
    request_payload: dict[str, Any],
    expected_response_keys: list[str],
    feature_names: list[str],
    compute_thread_return: object,
    availability_flag: str,
    output_names: list[str] | None = None,
    pred_id: str = "pred-123",
) -> Callable[[object], None]:
    """200 happy-path: valid request returns all expected response keys.

    The explainer computation (LIME/SHAP) is bypassed by patching
    ``asyncio.to_thread`` to return ``compute_thread_return`` directly.

    Args:
        explainer_name: Human-readable name for assertion messages.
        endpoint_path: HTTP path, e.g. ``"/explainers/local/lime"``.
        client: ``TestClient`` wired to the explainer router.
        request_payload: Valid JSON body (must use the same ``predictionId``
            as ``pred_id``).
        expected_response_keys: Top-level keys that must appear in the 200
            response JSON (e.g. ``["prediction_id", "model", "attributions"]``).
        feature_names: Input feature column names used to build mock storage.
        compute_thread_return: Return value from mocked ``asyncio.to_thread``.
            Must match the shape unpacked by the handler (e.g., LIME:
            ``(weights, r2, local_pred, intercept, lower, upper)``).
        availability_flag: Full dotted path to the availability bool, e.g.
            ``"trustyai_service.endpoints.explainers.local_explainer._LIME_AVAILABLE"``.
            Patched to ``True`` so the guard passes.
        output_names: Output column names for mock storage.  Defaults to
            ``["output"]`` when omitted.
        pred_id: Prediction ID stored in mock metadata; must match the
            ``predictionId`` field in ``request_payload``.

    """
    if output_names is None:
        output_names = ["output"]

    @patch(availability_flag, new=True)
    @patch(f"{MODULE}.asyncio.to_thread", new_callable=AsyncMock)
    @patch(f"{MODULE}.get_shared_data_source")
    @patch(f"{MODEL_DATA_MODULE}.get_global_storage_interface")
    @patch(f"{MODULE}.storage_interface")
    def test_impl(
        _: object,
        mock_si: MagicMock,
        mock_global: MagicMock,
        mock_ds_factory: MagicMock,
        mock_to_thread: AsyncMock,
    ) -> None:
        mock_si_inst, mock_ds = _make_storage_mocks(
            pred_id, feature_names, output_names
        )
        mock_si.dataset_exists = mock_si_inst.dataset_exists
        mock_si.read_data = mock_si_inst.read_data
        mock_si.get_aliased_column_names = mock_si_inst.get_aliased_column_names
        mock_global.return_value = mock_si
        mock_ds_factory.return_value = mock_ds
        mock_to_thread.return_value = compute_thread_return

        response = client.post(endpoint_path, json=request_payload)

        assert response.status_code == HTTPStatus.OK, (
            f"{explainer_name}: expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        for key in expected_response_keys:
            assert key in data, (
                f"{explainer_name}: missing key '{key}' in response. Got: {list(data.keys())}"
            )

    return test_impl
