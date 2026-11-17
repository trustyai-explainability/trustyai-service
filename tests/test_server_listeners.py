"""Listener layout tests driven by Hypercorn's own socket creation.

Hypercorn serves plaintext from ``bind`` and only opens ``insecure_bind`` when
TLS is enabled. A layout that declares the loopback listener on ``insecure_bind``
without serving material therefore binds Hypercorn's default ``127.0.0.1:8000``
instead, leaving the forwarding proxy with a closed port. These tests assert the
addresses the process would really listen on, not the attributes we happened to
set, because that is the difference the 502 in RHOAIENG-102565 came down to.
"""

import socket
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from hypercorn.config import Config

from trustyai_service import main
from trustyai_service.main import (
    build_api_config,
    build_health_config,
    plaintext_listeners,
    run_server,
)

HYPERCORN_DEFAULT_PORT = 8000


def _free_port() -> int:
    """Return a port that was free on loopback a moment ago."""
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _use_plaintext_tls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Select optional TLS with no serving material, i.e. plaintext mode."""
    monkeypatch.setenv("TLS_REQUIRED", "false")
    monkeypatch.setenv("TLS_POLICY", "system")
    monkeypatch.setenv("TLS_CERT_FILE", str(tmp_path / "missing.crt"))
    monkeypatch.setenv("TLS_KEY_FILE", str(tmp_path / "missing.key"))


def _bound_ports(config: Config) -> dict[str, list[int]]:
    """Create real sockets for a config and report the ports per listener class."""
    sockets = config.create_sockets()
    try:
        return {
            "secure": [sock.getsockname()[1] for sock in sockets.secure_sockets],
            "insecure": [sock.getsockname()[1] for sock in sockets.insecure_sockets],
        }
    finally:
        for sock in sockets.secure_sockets + sockets.insecure_sockets:
            sock.close()


def test_plaintext_api_binds_http_port_not_hypercorns_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Without TLS the API listens on HTTP_PORT, never on Hypercorn's default."""
    _use_plaintext_tls(monkeypatch, tmp_path)
    http_port = _free_port()

    config = build_api_config(None, http_port=http_port, ssl_port=_free_port())

    assert _bound_ports(config) == {"secure": [], "insecure": [http_port]}
    assert HYPERCORN_DEFAULT_PORT not in _bound_ports(config)["insecure"]


def test_tls_api_keeps_a_plaintext_loopback_listener_for_the_proxy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With serving material the TLS bind and the loopback proxy listener coexist."""
    _use_plaintext_tls(monkeypatch, tmp_path)
    http_port = _free_port()
    ssl_port = _free_port()
    tls_config: dict[str, Any] = {
        "ssl_certfile": "/etc/tls/internal/tls.crt",
        "ssl_keyfile": "/etc/tls/internal/tls.key",
    }

    config = build_api_config(tls_config, http_port=http_port, ssl_port=ssl_port)

    assert config.bind == [f"0.0.0.0:{ssl_port}"]
    assert config.insecure_bind == [f"127.0.0.1:{http_port}"]
    assert plaintext_listeners(config) == [f"127.0.0.1:{http_port}"]


def test_health_listener_binds_the_health_port_on_all_interfaces() -> None:
    """The kubelet probe listener is reachable from the node network."""
    health_port = _free_port()

    config = build_health_config(health_port)

    assert config.bind == [f"0.0.0.0:{health_port}"]
    assert _bound_ports(config) == {"secure": [], "insecure": [health_port]}


def test_insecure_bind_alone_would_fall_back_to_hypercorns_default() -> None:
    """Pin the mechanism behind the 502: insecure_bind is ignored without TLS."""
    config = Config()
    config.insecure_bind = ["127.0.0.1:8081"]

    assert plaintext_listeners(config) == ["127.0.0.1:8000"]
    assert _bound_ports(config) == {"secure": [], "insecure": [HYPERCORN_DEFAULT_PORT]}


def test_api_config_refuses_a_layout_without_a_plaintext_listener(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A layout the proxy cannot reach fails instead of binding the default port."""
    _use_plaintext_tls(monkeypatch, tmp_path)
    with (
        patch.object(
            main,
            "plaintext_listeners",
            return_value=[f"127.0.0.1:{HYPERCORN_DEFAULT_PORT}"],
        ),
        pytest.raises(ValueError, match="requires a plaintext listener"),
    ):
        build_api_config(None, http_port=_free_port(), ssl_port=_free_port())


@pytest.mark.asyncio
async def test_run_server_binds_the_configured_ports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The server startup path itself produces the expected listener layout."""
    _use_plaintext_tls(monkeypatch, tmp_path)
    http_port = _free_port()
    health_port = _free_port()
    monkeypatch.setenv("HTTP_PORT", str(http_port))
    monkeypatch.setenv("HEALTH_PORT", str(health_port))
    monkeypatch.setenv("SSL_PORT", str(_free_port()))

    with patch("trustyai_service.main.serve") as serve_mock:
        await run_server()

    api_config = serve_mock.call_args_list[0].args[1]
    health_config = serve_mock.call_args_list[1].args[1]
    assert _bound_ports(api_config) == {"secure": [], "insecure": [http_port]}
    assert _bound_ports(health_config) == {"secure": [], "insecure": [health_port]}
