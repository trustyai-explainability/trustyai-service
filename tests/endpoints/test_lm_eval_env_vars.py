"""Tests for LM-Eval environment variable validation and subprocess env construction.

The security functions are in ``_env_security.py`` which has no lm-eval
dependency, so these tests run regardless of whether lm-eval is installed.
"""

from unittest.mock import patch

import pytest

from trustyai_service.endpoints.evaluation._env_security import (
    ALLOWED_ENV_EXACT,
    ALLOWED_ENV_PREFIXES,
    BLOCKED_ENV_VARS,
    build_subprocess_env,
    is_env_var_allowed,
    validate_env_var_names,
)


class TestIsEnvVarAllowed:
    """Test the is_env_var_allowed predicate."""

    @pytest.mark.parametrize("var", sorted(BLOCKED_ENV_VARS))
    def test_blocked_vars_rejected(self, var: str) -> None:
        """Ensure every blocked variable is rejected."""
        assert is_env_var_allowed(var) is False

    @pytest.mark.parametrize(
        "var",
        [
            "HF_TOKEN",
            "HF_HOME",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "ACCELERATE_USE_FSDP",
            "CUDA_VISIBLE_DEVICES",
            "NCCL_DEBUG",
            "WANDB_PROJECT",
            "MLFLOW_EXPERIMENT_NAME",
            "TOKENIZERS_PARALLELISM",
            "LM_HARNESS_CACHE_PATH",
            "TORCH_DTYPE",
            "FLASH_ATTENTION_DETERMINISTIC",
        ],
    )
    def test_prefix_vars_allowed(self, var: str) -> None:
        """Ensure ML ecosystem prefix vars are accepted."""
        assert is_env_var_allowed(var) is True

    @pytest.mark.parametrize("var", sorted(ALLOWED_ENV_EXACT))
    def test_exact_vars_allowed(self, var: str) -> None:
        """Ensure every exact-match var is accepted."""
        assert is_env_var_allowed(var) is True

    @pytest.mark.parametrize(
        "var",
        [
            "DATABASE_PASSWORD",
            "DATABASE_HOST",
            "TLS_KEY_FILE",
            "MY_CUSTOM_VAR",
            "SECRET_TOKEN",
        ],
    )
    def test_unknown_vars_rejected(self, var: str) -> None:
        """Ensure server-internal and arbitrary vars are rejected."""
        assert is_env_var_allowed(var) is False


class TestValidateEnvVarNames:
    """Test the validate_env_var_names validation function."""

    def test_empty_dict_passes(self) -> None:
        """Empty env_vars should not raise."""
        validate_env_var_names({})

    def test_allowed_vars_pass(self) -> None:
        """Mix of prefix and exact-match vars should pass."""
        validate_env_var_names(
            {
                "HF_TOKEN": "hf_abc123",
                "CUDA_VISIBLE_DEVICES": "0,1",
                "OPENAI_API_KEY": "sk-test",  # pragma: allowlist secret
            }
        )

    def test_blocked_var_raises(self) -> None:
        """A blocked variable should raise with 'Blocked' in the message."""
        with pytest.raises(ValueError, match="Blocked environment variables"):
            validate_env_var_names({"LD_PRELOAD": "evil.so"})

    def test_blocked_var_message_lists_names(self) -> None:
        """Error message should list all blocked variable names."""
        with pytest.raises(ValueError, match="LD_PRELOAD") as exc_info:
            validate_env_var_names(
                {
                    "LD_PRELOAD": "evil.so",
                    "PYTHONPATH": "evil",
                }
            )
        assert "PYTHONPATH" in str(exc_info.value)

    def test_blocked_takes_precedence_over_unknown(self) -> None:
        """When both blocked and unknown vars present, report blocked first."""
        with pytest.raises(ValueError, match="Blocked"):
            validate_env_var_names(
                {
                    "LD_PRELOAD": "evil",
                    "MY_UNKNOWN": "value",
                }
            )

    def test_unknown_var_raises(self) -> None:
        """An unrecognised variable should raise with 'Unrecognised' in the message."""
        with pytest.raises(ValueError, match="Unrecognised environment variables"):
            validate_env_var_names({"MY_CUSTOM_VAR": "value"})

    def test_unknown_var_message_lists_prefixes(self) -> None:
        """Error for unknown vars should list allowed prefixes."""
        with pytest.raises(ValueError, match="Allowed prefixes") as exc_info:
            validate_env_var_names({"UNKNOWN": "value"})
        assert "HF_" in str(exc_info.value)

    def test_mixed_allowed_and_blocked_raises(self) -> None:
        """A request with both allowed and blocked vars should still reject."""
        with pytest.raises(ValueError, match="Blocked"):
            validate_env_var_names(
                {
                    "HF_TOKEN": "valid",
                    "PATH": "/evil/bin",
                }
            )

    def test_proxy_vars_blocked(self) -> None:
        """All proxy-related vars (upper and lowercase) should be blocked."""
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            with pytest.raises(ValueError, match="Blocked"):
                validate_env_var_names({var: "http://evil.proxy:8080"})

    def test_all_prefixes_have_coverage(self) -> None:
        """Every allowed prefix should actually match when used."""
        for prefix in ALLOWED_ENV_PREFIXES:
            var = f"{prefix}TEST_COVERAGE"
            assert is_env_var_allowed(var), f"Prefix {prefix} not matching"


class TestBuildSubprocessEnv:
    """Test the build_subprocess_env function."""

    @patch.dict(
        "os.environ",
        {
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/test",
            "LANG": "en_US.UTF-8",
            "HF_TOKEN": "hf_server_token",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "DATABASE_PASSWORD": "supersecret",  # pragma: allowlist secret
            "DATABASE_HOST": "db.internal",
            "TLS_KEY_FILE": "/etc/tls/key.pem",
            "SERVICE_STORAGE_FORMAT": "PVC",
        },
        clear=True,
    )
    def test_base_vars_present(self) -> None:
        """Subprocess env should always include PATH, HOME, LANG."""
        env = build_subprocess_env({})
        assert "PATH" in env
        assert "HOME" in env
        assert "LANG" in env

    @patch.dict(
        "os.environ",
        {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "HF_TOKEN": "hf_server",
            "CUDA_VISIBLE_DEVICES": "0",
            "NCCL_DEBUG": "INFO",
        },
        clear=True,
    )
    def test_ml_vars_inherited(self) -> None:
        """ML ecosystem vars from the server env should pass through."""
        env = build_subprocess_env({})
        assert env["HF_TOKEN"] == "hf_server"  # noqa: S105
        assert env["CUDA_VISIBLE_DEVICES"] == "0"
        assert env["NCCL_DEBUG"] == "INFO"

    @patch.dict(
        "os.environ",
        {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "DATABASE_PASSWORD": "supersecret",  # pragma: allowlist secret
            "DATABASE_HOST": "db.internal",
            "TLS_KEY_FILE": "/etc/tls/key.pem",
            "SERVICE_STORAGE_FORMAT": "PVC",
        },
        clear=True,
    )
    def test_server_secrets_excluded(self) -> None:
        """Server-internal secrets must never reach the subprocess."""
        env = build_subprocess_env({})
        assert "DATABASE_PASSWORD" not in env
        assert "DATABASE_HOST" not in env
        assert "TLS_KEY_FILE" not in env
        assert "SERVICE_STORAGE_FORMAT" not in env

    @patch.dict(
        "os.environ",
        {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "HF_TOKEN": "server_token",
        },
        clear=True,
    )
    def test_user_vars_override_server(self) -> None:
        """User-supplied vars should override same-named server vars."""
        env = build_subprocess_env({"HF_TOKEN": "user_token"})
        assert env["HF_TOKEN"] == "user_token"  # noqa: S105

    @patch.dict(
        "os.environ",
        {"PATH": "/usr/bin", "HOME": "/home/test"},
        clear=True,
    )
    def test_user_vars_added(self) -> None:
        """User-supplied vars should appear in the subprocess env."""
        env = build_subprocess_env({"WANDB_PROJECT": "my-eval"})
        assert env["WANDB_PROJECT"] == "my-eval"

    @patch.dict("os.environ", {}, clear=True)
    def test_defaults_when_env_empty(self) -> None:
        """When host env is empty, sensible defaults should be used."""
        env = build_subprocess_env({})
        assert env["PATH"] == "/usr/local/bin:/usr/bin:/bin"
        assert env["HOME"] == "/tmp"  # noqa: S108
        assert env["LANG"] == "C.UTF-8"
