"""Blocking KServe V2 HTTP/JSON prediction provider."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import numpy as np

from .model_provider import (
    DependencyUnavailableError,
    HttpTransportConfig,
    KServeModelSpec,
    PredictionMetadata,
    ProviderConfigurationError,
    ProviderDeadlineError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderInvalidResponseError,
    ProviderUnavailableError,
    ProviderUnsupportedModelError,
)

_NUMERIC = frozenset(
    {
        "BOOL",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "FP16",
        "FP32",
        "FP64",
    }
)
_MAX_RESPONSE_ELEMENTS = 10_000_000
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def _validate_input_datatype(values: np.ndarray, datatype: str) -> None:
    if datatype == "BOOL":
        if not np.all(np.isin(values, [0, 1])):
            raise ProviderInvalidRequestError(
                "Boolean model inputs must be zero or one"
            )
        return
    if datatype.startswith(("INT", "UINT")):
        if not np.equal(values, np.floor(values)).all():
            raise ProviderInvalidRequestError("Integer model inputs must be integral")
        bits = int(datatype.removeprefix("UINT").removeprefix("INT"))
        lower = 0 if datatype.startswith("UINT") else -(2 ** (bits - 1))
        upper = 2**bits - 1 if datatype.startswith("UINT") else 2 ** (bits - 1) - 1
        if np.any(values < lower) or np.any(values > upper):
            raise ProviderInvalidRequestError("Integer model inputs are out of range")


def _validate_output_datatype(values: np.ndarray, datatype: str) -> None:
    if datatype == "BOOL":
        if not np.all(np.isin(values, [0, 1])):
            raise ValueError("Boolean model outputs are invalid")
        return
    if datatype.startswith(("INT", "UINT")):
        if not np.equal(values, np.floor(values)).all():
            raise ValueError("Integer model outputs are not integral")
        bits = int(datatype.removeprefix("UINT").removeprefix("INT"))
        lower = 0 if datatype.startswith("UINT") else -(2 ** (bits - 1))
        upper = 2**bits - 1 if datatype.startswith("UINT") else 2 ** (bits - 1) - 1
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError("Integer model outputs are out of range")


def _timeout_error(error: Exception) -> bool:
    return isinstance(error, TimeoutError) or error.__class__.__name__.endswith(
        "TimeoutException"
    )


def normalize_base_url(value: str) -> str:
    """Validate and normalize a KServe HTTP server root."""
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ProviderInvalidRequestError("base_url must be an HTTP(S) URL")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ProviderInvalidRequestError("base_url has an invalid port") from exc
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ProviderInvalidRequestError(
            "base_url must not contain credentials, query, or fragment"
        )
    if any(ord(char) < 32 for char in parsed.path) or "%" in parsed.path:
        raise ProviderInvalidRequestError("base_url contains an unsafe path")
    if any(part in {".", ".."} for part in parsed.path.split("/")):
        raise ProviderInvalidRequestError("base_url contains path traversal")
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, parsed.path.rstrip("/"), "", "")
    )


def _segment(value: str, label: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or any(char in value for char in "/\\%")
        or any(ord(char) < 32 for char in value)
    ):
        raise ProviderInvalidRequestError(f"Invalid {label}")
    return quote(value, safe="-_.~")


def _allowlist_entry(value: str) -> str:
    """Normalize one host[:port] allowlist entry for exact authority checks."""
    raw = str(value).strip().lower().rstrip(".")
    if not raw:
        raise ValueError("empty allowlist entry")
    parsed = urlsplit(f"//{raw}")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("allowlist entry must be a host and optional port")
    host = parsed.hostname
    if not host or parsed.username or parsed.password:
        raise ValueError("invalid allowlist host")
    port = parsed.port
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("invalid allowlist port")
    normalized_host = f"[{host}]" if ":" in host else host
    return f"{normalized_host}:{port}" if port is not None else normalized_host


class KServeV2HttpPredictionProvider:
    """One metadata-negotiated KServe V2 client for one explanation."""

    def __init__(
        self,
        client: Any,
        metadata: PredictionMetadata,
        spec: KServeModelSpec,
        max_batch_size: int,
        default_timeout: float = 30.0,
    ) -> None:
        self._client = client
        self._metadata = metadata
        self._spec = spec
        self._max_batch_size = max_batch_size
        self._default_timeout = default_timeout
        self._closed = False
        self._inference_batch_count = 0
        self._provider_latency = 0.0

    @property
    def metadata(self) -> PredictionMetadata:
        return self._metadata

    @property
    def inference_batch_count(self) -> int:
        return self._inference_batch_count

    @property
    def provider_latency(self) -> float:
        return self._provider_latency

    def record_metadata_latency(self, elapsed: float) -> None:
        self._provider_latency += elapsed

    @classmethod
    def connect(
        cls,
        spec: KServeModelSpec,
        timeout_seconds: float,
        transport: HttpTransportConfig,
    ) -> KServeV2HttpPredictionProvider:
        base = normalize_base_url(spec.base_url)
        parsed_base = urlsplit(base)
        host = (parsed_base.hostname or "").lower().rstrip(".")
        try:
            effective_port = parsed_base.port or (
                443 if parsed_base.scheme == "https" else 80
            )
        except ValueError as exc:
            raise ProviderInvalidRequestError("base_url has an invalid port") from exc
        if not transport.allowed_hosts:
            raise ProviderConfigurationError(
                "Outbound model host allowlist is required"
            )
        if transport.follow_redirects or transport.trust_env:
            raise ProviderConfigurationError(
                "Redirects and ambient proxy settings must remain disabled"
            )
        try:
            entries = {_allowlist_entry(entry) for entry in transport.allowed_hosts}
        except (TypeError, ValueError) as exc:
            raise ProviderConfigurationError("Invalid outbound host allowlist") from exc
        normalized_host = f"[{host}]" if ":" in host else host
        host_ok = normalized_host in entries or host in entries
        host_ok = host_ok or f"{normalized_host}:{effective_port}" in entries
        if not host_ok:
            raise ProviderInvalidRequestError(
                "Model host is not in the outbound allowlist"
            )
        try:
            import httpx2
        except ImportError as exc:
            raise DependencyUnavailableError from exc
        try:
            client = httpx2.Client(
                headers=dict(transport.headers),
                verify=transport.verify,
                cert=transport.cert,
                follow_redirects=transport.follow_redirects,
                trust_env=transport.trust_env,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            raise ProviderUnavailableError from exc
        provider = None
        metadata_started = time.monotonic()
        try:
            provider = cls._from_metadata(
                client, spec, transport.max_batch_size, base, timeout_seconds
            )
            provider.record_metadata_latency(time.monotonic() - metadata_started)
            return provider
        except Exception:
            client.close()
            raise

    @classmethod
    def _from_metadata(
        cls,
        client: Any,
        spec: KServeModelSpec,
        max_batch_size: int,
        base: str,
        timeout_seconds: float,
    ) -> KServeV2HttpPredictionProvider:
        model = _segment(spec.model_name, "model name")
        version = (
            f"/versions/{_segment(spec.model_version, 'model version')}"
            if spec.model_version
            else ""
        )
        url = f"{base}/v2/models/{model}{version}"
        try:
            response = client.get(url, timeout=max(0.001, timeout_seconds))
        except Exception as exc:
            if _timeout_error(exc):
                raise ProviderDeadlineError from exc
            raise ProviderUnavailableError from exc
        if (
            response.status_code in {401, 403, 404, 408, 429}
            or response.status_code >= 500
        ):
            raise ProviderUnavailableError
        if response.status_code >= 400:
            raise ProviderInvalidRequestError("Model metadata request was rejected")
        try:
            payload = response.json()
            if payload.get("name") != spec.model_name:
                raise ProviderUnsupportedModelError(
                    "Model metadata name does not match the request"
                )
            inputs = payload.get("inputs") or []
            outputs = payload.get("outputs") or []
            if spec.model_version is not None and "versions" in payload:
                versions = payload["versions"]
                if not isinstance(versions, list) or spec.model_version not in versions:
                    raise ProviderInvalidRequestError(
                        "Requested model version was not found"
                    )
            if len(inputs) != 1:
                raise ProviderUnsupportedModelError(
                    "Model input metadata is ambiguous or has multiple required inputs"
                )
            selected_inputs = [
                item
                for item in inputs
                if spec.input_name is None or item.get("name") == spec.input_name
            ]
            selected_outputs = [
                item
                for item in outputs
                if spec.output_name is None or item.get("name") == spec.output_name
            ]
            if not selected_inputs:
                raise ProviderInvalidRequestError(
                    "Requested model input tensor was not found"
                )
            if not selected_outputs:
                raise ProviderInvalidRequestError(
                    "Requested model output tensor was not found"
                )
            if len(selected_inputs) != 1:
                raise ProviderUnsupportedModelError(
                    "Model input metadata is ambiguous or has multiple required inputs"
                )
            if len(selected_outputs) != 1:
                raise ProviderUnsupportedModelError(
                    "Model output metadata is ambiguous"
                )
            inp, out = selected_inputs[0], selected_outputs[0]
            if (
                inp.get("datatype") not in _NUMERIC
                or out.get("datatype") not in _NUMERIC
            ):
                raise ProviderUnsupportedModelError(
                    "Only numeric tensors are supported"
                )
            in_shape, out_shape = (
                tuple(int(x) for x in inp.get("shape", [])),
                tuple(int(x) for x in out.get("shape", [])),
            )
            if not in_shape or not out_shape or len(in_shape) > 2 or len(out_shape) > 2:
                raise ProviderUnsupportedModelError(
                    "Only flat rank-one or rank-two tensors are supported"
                )
            if any(x == 0 or x < -1 for x in in_shape + out_shape):
                raise ProviderUnsupportedModelError(
                    "Tensor dimensions must be positive or -1"
                )
            if any(x == -1 for x in in_shape[1:]) or any(
                x == -1 for x in out_shape[1:]
            ):
                raise ProviderUnsupportedModelError(
                    "Only a leading dynamic batch dimension is supported"
                )
            if len(in_shape) == 2 and in_shape[0] not in {-1, 1}:
                raise ProviderUnsupportedModelError(
                    "Fixed model batches other than one are unsupported"
                )
            if len(out_shape) == 2 and out_shape[0] not in {-1, 1}:
                raise ProviderUnsupportedModelError(
                    "Fixed model batches other than one are unsupported"
                )
            return cls(
                client,
                PredictionMetadata(
                    inp["name"],
                    out["name"],
                    inp["datatype"],
                    out["datatype"],
                    in_shape,
                    out_shape,
                ),
                spec,
                max_batch_size,
                timeout_seconds,
            )
        except ProviderError:
            raise
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise ProviderInvalidResponseError from exc

    def _infer_url(self) -> str:
        model = _segment(self._spec.model_name, "model name")
        version = (
            f"/versions/{_segment(self._spec.model_version, 'model version')}"
            if self._spec.model_version
            else ""
        )
        return f"{normalize_base_url(self._spec.base_url)}/v2/models/{model}{version}/infer"

    def predict(
        self, inputs: np.ndarray, *, timeout_seconds: float | None = None
    ) -> np.ndarray:
        if self._closed:
            raise ProviderUnavailableError("Model provider is closed")
        values = np.asarray(inputs)
        if (
            values.ndim != 2
            or values.dtype.kind not in "bfiu"
            or not np.isfinite(values).all()
        ):
            raise ProviderInvalidRequestError(
                "Model inputs must be a finite numeric matrix"
            )
        _validate_input_datatype(values, self._metadata.input_datatype)
        row_shape = (
            self._metadata.input_shape[1:]
            if len(self._metadata.input_shape) == 2
            else (
                ()
                if self._metadata.input_shape == (-1,)
                else self._metadata.input_shape
            )
        )
        if row_shape and row_shape[0] > 0 and values.shape[1] != row_shape[-1]:
            raise ProviderInvalidRequestError(
                "Input feature width does not match model metadata"
            )
        if self._metadata.input_shape == (-1,) and values.shape[1] != 1:
            raise ProviderInvalidRequestError(
                "Dynamic scalar input metadata requires one feature"
            )
        effective_timeout = (
            self._default_timeout if timeout_seconds is None else timeout_seconds
        )
        deadline = time.monotonic() + effective_timeout
        chunks: list[np.ndarray] = []
        batch_limit = (
            1
            if (
                (
                    len(self._metadata.input_shape) == 2
                    and self._metadata.input_shape[0] == 1
                )
                or (
                    len(self._metadata.output_shape) == 2
                    and self._metadata.output_shape[0] == 1
                )
            )
            else self._max_batch_size
        )
        for start in range(0, len(values), batch_limit):
            chunk = values[start : start + batch_limit]
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderDeadlineError
            shape = [len(chunk), *row_shape] if row_shape else [len(chunk)]
            body = {
                "inputs": [
                    {
                        "name": self._metadata.input_name,
                        "shape": shape,
                        "datatype": self._metadata.input_datatype,
                        "data": chunk.reshape(-1).tolist(),
                    }
                ]
            }
            started = time.monotonic()
            try:
                response = self._client.post(
                    self._infer_url(),
                    json=body,
                    headers={"Content-Type": "application/json"},
                    timeout=remaining,
                )
                self._inference_batch_count += 1
            except Exception as exc:
                if _timeout_error(exc):
                    raise ProviderDeadlineError from exc
                raise ProviderUnavailableError from exc
            finally:
                self._provider_latency += time.monotonic() - started
            if response.status_code >= 500:
                raise ProviderUnavailableError
            if response.status_code in {401, 403, 404, 408, 429}:
                raise ProviderUnavailableError
            if response.status_code >= 400:
                raise ProviderInvalidResponseError(
                    "Model inference request was rejected"
                )
            try:
                content = getattr(response, "content", None)
                if content is not None and len(content) > _MAX_RESPONSE_BYTES:
                    raise ValueError("model response is too large")
                payload = response.json()
                if payload.get("model_name") != self._spec.model_name:
                    raise ValueError("response model name mismatch")
                if self._spec.model_version is not None and payload.get(
                    "model_version"
                ) not in {
                    None,
                    self._spec.model_version,
                }:
                    raise ValueError("response model version mismatch")
                outputs = payload.get("outputs") or []
                selected_outputs = [
                    item
                    for item in outputs
                    if item.get("name") == self._metadata.output_name
                ]
                if len(selected_outputs) != 1:
                    raise ValueError("selected output is missing or duplicated")
                selected = selected_outputs[0]
                if selected.get("datatype") != self._metadata.output_datatype:
                    raise ValueError("response datatype mismatch")
                out_shape = tuple(int(x) for x in selected["shape"])
                data = np.asarray(selected["data"])
                expected = int(np.prod(out_shape))
                metadata_shape = self._metadata.output_shape
                if len(out_shape) not in {1, 2}:
                    raise ValueError("unsupported output rank")
                if len(metadata_shape) == 1 and metadata_shape[0] == -1:
                    valid_shape = out_shape == (len(chunk),)
                elif len(metadata_shape) == 1:
                    valid_shape = out_shape == (len(chunk), metadata_shape[0]) or (
                        metadata_shape[0] == 1 and out_shape == (len(chunk),)
                    )
                else:
                    valid_shape = out_shape == (len(chunk), *metadata_shape[1:]) or (
                        metadata_shape[1:] == (1,) and out_shape == (len(chunk),)
                    )
                scalar_output = (
                    len(metadata_shape) == 1 and metadata_shape[0] in {-1, 1}
                ) or (len(metadata_shape) == 2 and metadata_shape[1] == 1)
                representation_ok = data.shape == out_shape or (
                    scalar_output and data.ndim == 1 and data.size == expected
                )
                if (
                    not out_shape
                    or data.size != expected
                    or expected > _MAX_RESPONSE_ELEMENTS
                    or out_shape[0] != len(chunk)
                    or not valid_shape
                    or not representation_ok
                    or data.dtype.kind not in "bfiu"
                    or not np.isfinite(data.astype(float)).all()
                ):
                    raise ValueError("output shape mismatch")
                _validate_output_datatype(
                    data.astype(float), self._metadata.output_datatype
                )
                chunks.append(
                    data.reshape(len(chunk), 1)
                    if scalar_output
                    else data.reshape(out_shape)
                )
            except (
                AttributeError,
                StopIteration,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                raise ProviderInvalidResponseError from exc
        if not chunks:
            return np.empty((0, 1), dtype=float)
        return np.concatenate(chunks, axis=0)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._client.close()
