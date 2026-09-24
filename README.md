# TrustyAI Service

[![CI](https://img.shields.io/github/actions/workflow/status/trustyai-explainability/trustyai-service/python-tests.yaml?branch=main&label=CI&logo=github)](https://github.com/trustyai-explainability/trustyai-service/actions/workflows/python-tests.yaml)
[![Python](https://img.shields.io/badge/python-3.12--3.14-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Coverage](https://img.shields.io/codecov/c/github/trustyai-explainability/trustyai-service?logo=codecov&label=Coverage)](https://codecov.io/gh/trustyai-explainability/trustyai-service)
[![License](https://img.shields.io/github/license/trustyai-explainability/trustyai-service?label=License)](https://github.com/trustyai-explainability/trustyai-service/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)
[![uv](https://img.shields.io/badge/uv-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Types - Pyrefly](https://img.shields.io/badge/types-pyrefly-blue?logo=python&logoColor=white)](https://pyrefly.org/)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)
[![Contributing](https://img.shields.io/badge/Contributing-guide-blue)](https://github.com/trustyai-explainability/trustyai-service/blob/main/CONTRIBUTING.md)
[![CodeRabbit](https://img.shields.io/badge/CodeRabbit-AI%20Reviews-orange?logo=coderabbit)](https://coderabbit.ai)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/trustyai-explainability/trustyai-service/badge)](https://scorecard.dev/viewer/?uri=github.com/trustyai-explainability/trustyai-service)

The TrustyAI Service is a REST API for Responsible AI workflows:
drift detection, fairness monitoring, and model explainability.
Built on FastAPI + Hypercorn, it consumes inference data from
KServe, stores it, and computes metrics on a schedule via
Prometheus.

Part of [Red Hat OpenShift AI](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai)
and [Open Data Hub](https://opendatahub.io/).

**[Documentation](https://trustyai.org/docs/main/main)**
· **[API Reference](https://trustyai.org/docs/main/trustyai-service-api-reference)**

---

## Key Features

- **Real-time drift detection** on live inference streams
- **Automatic Prometheus metric publishing** on a configurable
  schedule
- **KServe-native** — consumes inference payloads directly via
  CloudEvents
- **Dual storage backends** — PVC (HDF5) or MariaDB
- **Runs anywhere** — locally, in Jupyter, or on Kubernetes

---

## Local KernelSHAP explainer

KernelSHAP is an experimental route. It is feature-flagged and disabled by
default. The SHAP and outbound HTTP client dependencies are optional. Install
the SHAP extra before calling the route:

```bash
uv sync --all-groups --extra shap
```

Enable both flags before the process starts:

```bash
TRUSTYAI_ENABLE_EXPLAINER=true \
TRUSTYAI_ENABLE_EXPLAINER_LOCAL=true \
uv run python -m trustyai_service.main
```

The base install (`uv sync --all-groups`) remains import-safe without this
extra. If an enabled route is requested without the extra, it returns a
`dependency_unavailable` error at request time.

The route is:

- `POST /explainers/local/shap`

The flags are evaluated at import time. Changing their environment variables
after startup does not register the routes.

### KServe direction and protocol

TrustyAI has two separate KServe V2 flows:

- The inbound KServe consumer receives inference events at
  `/consumer/kserve/v2` and stores the input, output, and metadata rows used to
  identify a prediction.
- The outbound model provider is used by the local SHAP explainer. It calls the
  deployed model's KServe V2 HTTP/REST metadata and inference endpoints for
  generated SHAP coalitions.

The word `V2` describes the KServe wire protocol, not the direction of a
request. `model_version` identifies the deployed model version and is separate
from the protocol version. Local explainers use outbound KServe V2 HTTP/REST;
they do not add a V1, gRPC, ModelMesh, or automatic transport fallback.

### Model configuration

The explainer uses these `config.model` fields:

- `base_url` is the HTTP(S) KServe server root, for example
  `https://inference.example/ingress`. It is required for `MODEL` and is not a
  model URL or a credential container. The provider appends the KServe V2 model
  and `/infer` paths.
- `model_name` selects the served model and the stored TrustyAI model data.
  `model_version` is optional and is sent unchanged when supplied.
- `input_name` and `output_name` are optional exact tensor selectors. An
  explicitly requested selector that is absent from upstream metadata is
  rejected. An omitted selector follows the metadata ambiguity rules: a unique
  tensor may be selected, while missing metadata or ambiguous outputs are
  rejected; metadata must still advertise exactly one required input.
- `task` is required and must be `REGRESSION` or `CLASSIFICATION`. It is never
  inferred from output dtype or shape.
- `prediction_source` defaults to `MODEL`. `MODEL` evaluates generated inputs
  against the deployed model. `SURROGATE` is an explicit opt-in that trains the
  stored-data random-forest surrogate and does not contact a model endpoint.

### Numeric tensor contract

The outbound codec requires one numeric input tensor and one selected numeric
output tensor. Supported datatypes are `BOOL`, `UINT8`, `UINT16`, `UINT32`,
`UINT64`, `INT8`, `INT16`, `INT32`, `INT64`, `FP16`, `FP32`, and `FP64`.
Metadata shapes must be rank one or rank two; each dimension is positive or
`-1`, and only a leading `-1` is supported. The accepted forms use `width` for
a positive feature/output width and `batch` for the number of request rows:

- `[-1]` emits `[batch]`, with one scalar feature per row, and accepts scalar
  responses shaped `[batch]` or `[batch, 1]`.
- `[width]` (including `[1]`) emits `[batch, width]`. For `width=1`, scalar
  responses may be `[batch]` or `[batch, 1]`; for wider outputs, the response
  must be `[batch, width]`.
- `[-1, width]` emits `[batch, width]` with a dynamic batch. Its scalar and
  wider response shapes follow the `[width]` rules above.
- `[1, width]` emits `[1, width]` and accepts one row only. For `width=1`,
  scalar responses may be `[1]` or `[1, 1]`; for wider outputs, the response
  must be `[1, width]`.

The rank-one `[-1]` form is a special dynamic scalar form, not a dynamic
feature width. Positive rank-one shapes describe the per-row feature/output
width. Rank-two fixed batches are supported only when the batch dimension is
`1`; dynamic rank-two batches use `-1`. Scalar responses are normalized to a
`[batch, 1]` result. Request data is a finite numeric matrix serialized as a
flattened row-major list. Response data may be a flattened row-major list or a
nested list matching its declared shape, but a wider output must declare
`[batch, width]` rather than a flattened shape such as `[batch * width]`.

`BOOL` values must be zero or one, integer values must be integral and within
the declared range, and input/output datatypes must match the negotiated
metadata. Rank-zero and rank-three-or-higher tensors, non-leading dynamic
dimensions, fixed rank-two batches other than one, categorical or string
tensors, and multiple required inputs are unsupported. If metadata advertises
multiple outputs, `output_name` must select one by exact name; an omitted
selector is rejected when that output metadata is ambiguous. `MODEL` mode does
not require stored output rows; `SURROGATE` mode requires stored labels and an
unambiguous output selection.

Minimal request examples:

```json
{
  "predictionId": "prediction-123",
  "config": {
    "model": {
      "base_url": "https://inference.example/ingress",
      "model_name": "credit-model",
      "model_version": "v1",
      "input_name": "input",
      "output_name": "output",
      "task": "REGRESSION"
    },
    "explainer": {
      "n_samples": 300,
      "n_training_rows": 1000,
      "timeout": 120,
      "link": "IDENTITY",
      "confidence": 0.95,
      "single_probability": false
    }
  }
}
```

The example above is sent to `/explainers/local/shap`.

### Operational controls

Deployment-level model transport settings are not accepted in the explanation
body:

- `TRUSTYAI_EXPLAINER_ALLOWED_HOSTS`: required for `MODEL`; exact host match;
  fails closed.
- `TRUSTYAI_EXPLAINER_CA_BUNDLE`: optional CA path. Normal TLS verification is
  used when it is unset.
- `TRUSTYAI_EXPLAINER_CLIENT_CERT` and `TRUSTYAI_EXPLAINER_CLIENT_KEY`: optional
  readable certificate and key pair.
- `TRUSTYAI_EXPLAINER_AUTH_TOKEN`: optional deployment Bearer token.
- `TRUSTYAI_EXPLAINER_MAX_BATCH_SIZE`: outbound batch cap, default `1024`, valid
  range 1 to 100,000.

The allowlist matches exact outbound host authorities, with an optional port;
wildcards are not accepted. `SURROGATE` does not read the allowlist or any
other model transport setting. The client certificate and key must be supplied
as a readable pair. The auth token is sent only as
`Authorization: Bearer ...`; it is never read from the request body, returned,
or logged.

Each request has a single monotonic `timeout`. SHAP defaults to 300 seconds and
accepts at most 3600 seconds. The timeout covers data loading, provider
metadata and inference calls, explanation work, and confidence intervals.
Organic background rows are capped by `n_training_rows`, which defaults to
10,000 and has a maximum of 100,000. SHAP `n_samples` defaults to 300, with a
maximum of 100,000.

The provider disables redirects and ambient proxy/environment settings. Request
and response tensors are capped at 10,000,000 elements, and materialized model
responses are capped at 64 MiB. These limits apply before large tensor
conversion or reshaping.

Provider, metadata, decoding, and timeout failures in `MODEL` mode are returned
as errors. They never trigger an implicit `SURROGATE` fallback.

### Response semantics

The response includes `prediction_source`, model identity, task, and output
metadata. The prediction fields have these meanings:

- SHAP `prediction_output` is the raw selected scalar output. `shap_base_value`
  and `linked_prediction_output` are in the requested `link` space. With
  `IDENTITY` they remain in model output units. `LOGIT` requires the selected
  classification values to be strict probabilities in `(0, 1)`; for regression,
  the raw scalar values must also be in `(0, 1)` for the link, but they are model
  outputs rather than inherently probabilities. `prediction_output` remains
  raw in either case. SHAP attributions add to `linked_prediction_output` in
  that same link space.

Successful local explanations emit the existing `local_explanation_complete`
structured log event. Its fields include the explainer, model name and version,
prediction ID, prediction source, overall latency, provider latency,
inference batch count, and final status. Provider latency and batch count are
not applicable to `SURROGATE`. Logs never include credentials, raw feature
values, or complete request and response payloads.

---

## Metrics

### Drift Detection

- Compare Means
  ([Welch's t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html))
- Kolmogorov–Smirnov Test
  ([scipy.stats.ks_2samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html))
- Streaming Kolmogorov–Smirnov Test
  ([Lall 2015](https://ieeexplore.ieee.org/document/7363746/))
  using the Greenwald–Khanna quantile sketch
  ([Greenwald & Khanna 2001](https://dl.acm.org/doi/10.1145/375663.375670))
- Jensen–Shannon Divergence
  ([scipy.spatial.distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html))
- Maximum Mean Discrepancy
  ([Domingo-Enrich et al. 2023](https://proceedings.mlr.press/v206/domingo-enrich23a/domingo-enrich23a.pdf))
  with CTT, RFF, and ACTT methods via
  [goodpoints](https://github.com/microsoft/goodpoints)

### Fairness

- Statistical Parity Difference (SPD)
- Disparate Impact Ratio (DIR)

---

## Quickstart

```bash
# Install
uv sync --all-groups

# Run
uv run python -m src.main
```

Once running, the API is available at `http://localhost:8080`.
Interactive OpenAPI documentation is at
`http://localhost:8080/docs`.

---

## Container

```bash
# Minimal (PVC storage only)
podman build -t trustyai:latest .

# With MariaDB support
podman build -t trustyai:latest --build-arg EXTRAS="mariadb" .

# Run
podman run -p 8080:8080 trustyai:latest
```

---

## Configuration

| Environment Variable | Default | Description |
| -------- | ------- | ----------- |
| `SERVICE_STORAGE_FORMAT` | `PVC` | Storage backend (`PVC` or `MARIA`) |
| `SERVICE_METRICS_SCHEDULE` | `30` | Seconds between scheduled computations |
| `HTTP_PORT` | `8080` | HTTP listener port |
| `SSL_PORT` | `4443` | HTTPS listener port |
| `TLS_CERT_FILE` | `/etc/tls/internal/tls.crt` | TLS certificate path |
| `TLS_KEY_FILE` | `/etc/tls/internal/tls.key` | TLS private key path |
| `DATABASE_HOST` | — | MariaDB hostname |
| `DATABASE_PORT` | `3306` | MariaDB port |
| `DATABASE_USERNAME` | — | MariaDB username |
| `DATABASE_PASSWORD` | — | MariaDB password |
| `DATABASE_DATABASE` | — | MariaDB database name |

TLS is enabled automatically when both the certificate and key
files are present.

---

## Testing

```bash
uv run pytest tests/ -v
uv run pytest tests/ -v --cov=src --cov-report=xml  # with coverage
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup,
coding standards, and the pull request process.

Please report security vulnerabilities via
[GitHub Security Advisories](https://github.com/trustyai-explainability/trustyai-service/security/advisories/new),
not public issues.
See [SECURITY.md](SECURITY.md) for details.

---

## License

[Apache License 2.0](LICENSE)
