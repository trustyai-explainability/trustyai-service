"""Listener layout tests driven by Hypercorn's own socket creation.

Hypercorn serves plaintext from ``bind`` and only opens ``insecure_bind`` when
TLS is enabled. A layout that declares the loopback listener on ``insecure_bind``
without serving material therefore binds Hypercorn's default ``127.0.0.1:8000``
instead, leaving the forwarding proxy with a closed port. These tests assert the
addresses the process would really listen on, not the attributes we happened to
set, because that is the difference the 502 in RHOAIENG-102565 came down to.
"""

import asyncio
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


@pytest.fixture(autouse=True)
def _isolated_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep the developer's PV out of every startup path in this module.

    ``run_server()`` runs the Java PVC migration when ``STORAGE_DATA_FOLDER``
    points at a folder holding ``*-internal_data.csv`` files. Pointing PVC
    storage at a fresh temporary folder keeps the tests from converting (and
    writing to) a developer's local data.
    """
    storage = tmp_path / "storage"
    storage.mkdir()
    monkeypatch.setenv("SERVICE_STORAGE_FORMAT", "PVC")
    monkeypatch.setenv("STORAGE_DATA_FOLDER", str(storage))


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

    assert config.bind == [f"127.0.0.1:{HYPERCORN_DEFAULT_PORT}"]
    assert plaintext_listeners(config) == [f"127.0.0.1:{HYPERCORN_DEFAULT_PORT}"]


def test_insecure_bind_alone_is_never_bound_without_tls() -> None:
    """The unopened listener is the declared one, independent of port 8000.

    The insecure port is held by a listening socket for the duration, so
    bind-on-port-0 cannot be assigned it and Hypercorn would fail with
    EADDRINUSE if it tried to open the declared listener.
    """
    config = Config()
    config.bind = ["127.0.0.1:0"]

    with closing(socket.socket()) as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        held_port = held.getsockname()[1]
        config.insecure_bind = [f"127.0.0.1:{held_port}"]

        bound = _bound_ports(config)

    assert bound["secure"] == []
    assert len(bound["insecure"]) == 1
    assert bound["insecure"][0] != held_port


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

    configs = {call.args[0]: call.args[1] for call in serve_mock.call_args_list}
    api_config = configs[main.app]
    health_config = configs[main.health_app]
    assert _bound_ports(api_config) == {"secure": [], "insecure": [http_port]}
    assert _bound_ports(health_config) == {"secure": [], "insecure": [health_port]}


@pytest.mark.asyncio
async def test_run_server_migrates_data_after_probes_and_before_the_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Probes are up while data migrates, and the API starts only afterwards."""
    _use_plaintext_tls(monkeypatch, tmp_path)
    monkeypatch.setenv("HTTP_PORT", str(_free_port()))
    monkeypatch.setenv("HEALTH_PORT", str(_free_port()))
    monkeypatch.setenv("SSL_PORT", str(_free_port()))
    events: list[str] = []

    async def fake_serve(app: object, _config: Config) -> None:
        events.append("health serving" if app is main.health_app else "api serving")

    async def fake_migration() -> None:
        # A real migration yields to the event loop while it works.
        await asyncio.sleep(0)
        events.append("migration")

    with (
        patch("trustyai_service.main.serve", fake_serve),
        patch("trustyai_service.main.run_startup_migrations", fake_migration),
    ):
        await run_server()

    assert events == ["health serving", "migration", "api serving"]


@pytest.mark.asyncio
async def test_run_server_does_not_start_the_api_when_migration_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed migration ends startup instead of serving without data."""
    _use_plaintext_tls(monkeypatch, tmp_path)
    monkeypatch.setenv("HTTP_PORT", str(_free_port()))
    monkeypatch.setenv("HEALTH_PORT", str(_free_port()))
    monkeypatch.setenv("SSL_PORT", str(_free_port()))
    served: list[object] = []

    async def fake_serve(app: object, _config: Config) -> None:
        served.append(app)

    async def failing_migration() -> None:
        msg = "migration failed"
        raise RuntimeError(msg)

    with (
        patch("trustyai_service.main.serve", fake_serve),
        patch("trustyai_service.main.run_startup_migrations", failing_migration),
        pytest.raises(RuntimeError, match="migration failed"),
    ):
        await run_server()

    assert main.app not in served


@pytest.mark.asyncio
async def test_run_server_still_starts_when_the_migration_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Without the optional dependencies the service starts and says so."""
    _use_plaintext_tls(monkeypatch, tmp_path)
    monkeypatch.setenv("HTTP_PORT", str(_free_port()))
    monkeypatch.setenv("HEALTH_PORT", str(_free_port()))
    monkeypatch.setenv("SSL_PORT", str(_free_port()))

    with (
        patch("trustyai_service.main.serve") as serve_mock,
        patch("trustyai_service.main.JAVA_PVC_MIGRATION_AVAILABLE", new=False),
        patch.object(main.logger, "warning") as warning_mock,
    ):
        await run_server()

    assert serve_mock.call_count == 2
    assert "unavailable" in warning_mock.call_args.args[0]
