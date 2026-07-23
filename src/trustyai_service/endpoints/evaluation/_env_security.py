"""Environment variable validation for LM-Eval subprocess isolation.

Provides a two-layer defense against CWE-426/427 (Untrusted Search Path):
  1. API-boundary validation — reject dangerous or unrecognised env var names
  2. Minimal environment construction — subprocess inherits only ML-related vars

This module is deliberately free of lm-eval and FastAPI dependencies so that
its security logic can be tested independently.
"""

import os

from pydantic import BaseModel

# Dangerous vars that enable code injection or traffic interception.
BLOCKED_ENV_VARS: frozenset[str] = frozenset(
    {
        # Dynamic linker injection
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",
        "LD_DEBUG",
        "LD_PROFILE",
        # Python import/startup hijack
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONHOME",
        # Executable search path
        "PATH",
        # Shell injection vectors
        "BASH_ENV",
        "ENV",
        "CDPATH",
        "HOSTALIASES",
        # Proxy hijack (data exfiltration via MITM)
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    }
)

# Prefixes covering the ML ecosystem env vars (lm-eval, torch, transformers,
# accelerate, huggingface_hub, experiment trackers, GPU runtimes).
ALLOWED_ENV_PREFIXES: tuple[str, ...] = (
    "HF_",
    "HUGGING",
    "TRANSFORMERS_",
    "ACCELERATE_",
    "FSDP_",
    "CUDA_",
    "NCCL_",
    "ROCR_",
    "HIP_",
    "WANDB_",
    "MLFLOW_",
    "CLEARML_",
    "TOKENIZERS_",
    "FLASH_ATTENTION_",
    "LM_HARNESS_",
    "TORCH_",
    "INDUCTOR_",
)

# Individual vars that don't match a prefix but are legitimate for lm-eval jobs.
ALLOWED_ENV_EXACT: frozenset[str] = frozenset(
    {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "TEXTSYNTH_API_SECRET_KEY",
        "PERSPECTIVE_API_KEY",
        "PERSPECTIVE_API_QPS",
        "MASTER_ADDR",
        "MASTER_PORT",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "NODE_RANK",
        "NPROC",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "TOKENIZERS_PARALLELISM",
    }
)


def is_env_var_allowed(name: str) -> bool:
    """Check whether an environment variable name is permitted."""
    if name in BLOCKED_ENV_VARS:
        return False
    if name in ALLOWED_ENV_EXACT:
        return True
    return any(name.startswith(p) for p in ALLOWED_ENV_PREFIXES)


def validate_env_var_names(env_vars: dict[str, str]) -> None:
    """Validate that all env var names are allowed.

    Raises ValueError listing any rejected names.
    """
    if not env_vars:
        return
    blocked = {k for k in env_vars if k in BLOCKED_ENV_VARS}
    if blocked:
        msg = f"Blocked environment variables (security risk): {', '.join(sorted(blocked))}"
        raise ValueError(msg)
    rejected = {k for k in env_vars if not is_env_var_allowed(k)}
    if rejected:
        msg = (
            f"Unrecognised environment variables: {', '.join(sorted(rejected))}. "
            f"Allowed prefixes: {', '.join(ALLOWED_ENV_PREFIXES)}"
        )
        raise ValueError(msg)


def validate_env_vars_model(self: BaseModel) -> BaseModel:
    """Pydantic model_validator for env_vars on the dynamic LMEvalRequest model."""
    env_vars = getattr(self, "env_vars", None)
    if env_vars:
        validate_env_var_names(env_vars)
    return self


def build_subprocess_env(user_env_vars: dict[str, str]) -> dict[str, str]:
    """Build a minimal subprocess environment.

    Starts with essential base vars, selectively inherits ML-related vars
    from the server environment, then overlays validated user-supplied vars.
    Server secrets (DATABASE_PASSWORD, TLS_KEY_FILE, etc.) are never passed.

    ``user_env_vars`` must contain only allowed names (see
    :func:`validate_env_var_names`).  A defensive re-validation is performed
    here so that callers who bypass the Pydantic model layer are still safe.
    """
    validate_env_var_names(user_env_vars)
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),  # noqa: S108
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }
    env.update(
        {
            key: val
            for key, val in os.environ.items()
            if any(key.startswith(p) for p in ALLOWED_ENV_PREFIXES)
        }
    )
    env.update(user_env_vars)
    return env
