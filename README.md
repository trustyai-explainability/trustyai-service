# TrustyAI Service

👋 The TrustyAI Service is intended to a hub for all kinds of Responsible AI workflows, such as 
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
### Locally (Without Eval Support)
```bash
uv pip install .
````

### Locally (With Eval Support)
```bash
uv pip install .[eval]
````

### Container (Without Eval Support)
```bash
podman build -t $IMAGE_NAME .
````

### Container (With Eval Support)
```bash
podman build -t $IMAGE_NAME --build-arg EXTRAS=eval .
````


## 🏃Running 🏃‍♀️
### Locally
```bash
uv run uvicorn src.main --host 0.0.0.0 --port 8080
```

### Container
```bash
podman run -t $IMAGE_NAME -p 8080:8080 .
```

---
## ☎️ API ☎️
When the service is running, visit [localhost:8080](localhost:8080/docs) to see the OpenAPI documentation!

