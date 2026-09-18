"""Contract tests for the optional KServe HTTP transport boundary."""

from types import SimpleNamespace

import pytest

from trustyai_service.service.explainers.local.kserve_v2_http import (
    KServeV2HttpPredictionProvider,
    normalize_base_url,
)
from trustyai_service.service.explainers.local.model_provider import (
    HttpTransportConfig,
    KServeModelSpec,
    PredictionMetadata,
)
from trustyai_service.service.explainers.local.types import TaskType


def test_transport_headers_are_copied() -> None:
    """Copy deployment headers so later caller mutation cannot alter transport."""
    headers = {"Authorization": "Bearer secret"}
    config = HttpTransportConfig(headers=headers, allowed_hosts=frozenset({"m"}))
    headers["Authorization"] = "changed"
    assert config.headers["Authorization"] == "Bearer secret"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("HTTP://example.test/prefix/", "http://example.test/prefix"),
        ("https://example.test", "https://example.test"),
    ],
)
def test_base_url_normalization_preserves_safe_prefix(
    value: str, expected: str
) -> None:
    """Normalize schemes and prefixes without changing safe URL components."""
    assert normalize_base_url(value) == expected


def test_metadata_url_uses_encoded_path_segments() -> None:
    """Encode model and version path segments when requesting metadata."""
    requests: list[str] = []

    class Client:
        def get(self, url: str, **_kwargs: object) -> object:
            requests.append(url)
            return SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "name": "model name",
                    "inputs": [{"name": "input", "datatype": "FP32", "shape": [-1, 2]}],
                    "outputs": [
                        {"name": "output", "datatype": "FP32", "shape": [-1, 1]}
                    ],
                },
            )

    provider = KServeV2HttpPredictionProvider._from_metadata(
        Client(),
        KServeModelSpec(
            "https://example.test/prefix",
            "model name",
            "v1",
            None,
            None,
            TaskType.REGRESSION,
        ),
        4,
        "https://example.test/prefix",
        1,
    )
    assert requests == [
        "https://example.test/prefix/v2/models/model%20name/versions/v1"
    ]
    assert provider.metadata == PredictionMetadata(
        "input", "output", "FP32", "FP32", (-1, 2), (-1, 1)
    )
