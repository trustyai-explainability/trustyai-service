# TrustyAI Service

👋 The TrustyAI Service is intended to be a hub for all kinds of Responsible AI workflows, such as
explainability, drift, and Large Language Model (LLM) evaluation. Designed as a REST server wrapping
a core Python library, the TrustyAI service is intended to be a tool that can operate in a local
environment, a Jupyter Notebook, or in Kubernetes.

---
## Native Algorithms
### 📈Drift  📉
- Fourier Maximum Mean Discrepancy (FourierMMD)
- Jensen-Shannon
- Approximate Kolmogorov–Smirnov Test
- Kolmogorov–Smirnov Test (KS-Test)
- Meanshift

### ⚖️ Fairness ⚖️
- Statistical Parity Difference
- Disparate Impact Ratio
- Average Odds Ratio (WIP)
- Average Predictive Value Difference (WIP)
- Individual Consistency (WIP)

---
## Imported Algorithms/Libraries
### 🔬Explainability 🔬
- [LIME](https://github.com/marcotcr/lime) (WIP)
- [SHAP](https://github.com/shap/shap) (WIP)

### 📋 LLM Evaluation  📋
- [LM-Evaluation-Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main)

---
## 📦 Building 📦
### Locally
```bash
uv pip install ".[$EXTRAS]"
```

### Container
```bash
podman build -t $IMAGE_NAME --build-arg EXTRAS="$EXTRAS" .
```

### Available Extras
Pass these extras as a comma separated list, e.g., `"mariadb,protobuf"`
* `protobuf`: To process model inference data from ModelMesh models, you can install with `protobuf` support. Otherwise, only KServe models will be supported.
* `eval`: To enable the Language Model Evaluation servers, install with `eval` support.
* `mariadb` (If installing locally, install the [MariaDB Connector/C](https://mariadb.com/docs/server/connect/programming-languages/c/install/) first.)

### Examples
```bash
uv pip install ".[mariadb,protobuf,eval]"
podman build -t $IMAGE_NAME --build-arg EXTRAS="mariadb,protobuf,eval" .
```

## 🏃Running 🏃‍♀️
### Locally
```bash
uv run uvicorn src.main --host 0.0.0.0 --port 8080
```

### Container
```bash
podman run -t $IMAGE_NAME -p 8080:8080 .
```

### 🔐 TLS Support
The service supports TLS encryption and automatically detects certificates at startup:

- **With TLS certificates**: Runs on port 4443 (HTTPS)
- **Without TLS certificates**: Runs on port 8080 (HTTP)

**Certificate locations** (configurable via environment variables):
- Certificate: `/etc/tls/internal/tls.crt` (or `TLS_CERT_FILE`)
- Private key: `/etc/tls/internal/tls.key` (or `TLS_KEY_FILE`)

**Environment variables**:
- `TLS_CERT_FILE`: Path to TLS certificate file
- `TLS_KEY_FILE`: Path to TLS private key file
- `SSL_PORT`: HTTPS port (default: 4443)
- `HTTP_PORT`: HTTP port (default: 8080)

The TLS implementation is fully compatible with the TrustyAI operator for seamless Kubernetes deployment.

## 🧪 Testing 🧪
### Running All Tests
To run all tests in the project:
```bash
python -m pytest
```

Or with more verbose output:
```bash
python -m pytest -v
```

### Running with Coverage
To run tests with coverage reporting:
```bash
python -m pytest --cov=src
```

---
## 🔄 Protobuf Support 🔄
To process model inference data from ModelMesh models, you can install protobuf support. Otherwise, only KServe models will be supported.

### Generating Protobuf Code
After installing dependencies, generate Python code from the protobuf definitions:

```bash
# From the project root
bash scripts/generate_protos.sh
```

### Testing Protobuf Functionality
Run the tests for the protobuf implementation:

```bash
# From the project root
python -m pytest tests/service/data/test_modelmesh_parser.py -v
```

---
## ☎️ API ☎️
When the service is running, visit `localhost:8080/docs` to see the OpenAPI documentation!
