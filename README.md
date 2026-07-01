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
