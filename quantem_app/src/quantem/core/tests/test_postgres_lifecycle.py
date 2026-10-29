from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from quantem.core.postgres import (
    POSTGRES_RUNTIME_OPTIONS,
    EmbeddedPostgres,
    PostgresEndpoint,
    PostgresError,
    _cluster_requires_recovery,
    _postgres_server_arguments,
)


def _endpoint() -> PostgresEndpoint:
    return PostgresEndpoint("127.0.0.1", 54321, "quantem", "quantem", "secret")


def test_readiness_waits_through_recovery(monkeypatch):
    attempts = iter(
        [
            subprocess.CompletedProcess([], 1, "rejecting connections", ""),
            subprocess.CompletedProcess([], 0, "accepting connections", ""),
        ]
    )
    monkeypatch.setattr("quantem.core.postgres._run", lambda *args, **kwargs: next(attempts))
    monkeypatch.setattr("quantem.core.postgres._tool", lambda name: name)
    monkeypatch.setattr("quantem.core.postgres.time.sleep", lambda _: None)

    EmbeddedPostgres()._wait_ready(_endpoint(), recovery_expected=True)


def test_readiness_allows_recovery_to_exceed_normal_startup_timeout(monkeypatch, capsys):
    attempts = iter(
        [
            subprocess.CompletedProcess([], 1, "rejecting connections", ""),
            subprocess.CompletedProcess([], 0, "accepting connections", ""),
        ]
    )
    clock = iter([0.0, 0.0, 61.0])
    monkeypatch.setattr("quantem.core.postgres._run", lambda *args, **kwargs: next(attempts))
    monkeypatch.setattr("quantem.core.postgres._tool", lambda name: name)
    monkeypatch.setattr("quantem.core.postgres.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("quantem.core.postgres.time.sleep", lambda _: None)

    EmbeddedPostgres()._wait_ready(_endpoint(), recovery_expected=True)

    assert "database recovery:" in capsys.readouterr().out


def test_readiness_keeps_short_timeout_when_server_never_answers(monkeypatch, capsys):
    clock = iter([0.0, 0.0, 61.0])
    monkeypatch.setattr(
        "quantem.core.postgres._run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 2, "no response", ""),
    )
    monkeypatch.setattr("quantem.core.postgres._tool", lambda name: name)
    monkeypatch.setattr("quantem.core.postgres.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("quantem.core.postgres.time.sleep", lambda _: None)

    with pytest.raises(PostgresError, match="within 60 seconds: no response"):
        EmbeddedPostgres()._wait_ready(_endpoint())

    assert "database recovery:" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("shut down", False),
        ("shut down in recovery", True),
        ("in production", True),
    ],
)
def test_cluster_recovery_detection_uses_control_state(monkeypatch, state, expected):
    output = f"Database cluster state:               {state}\n"
    monkeypatch.setattr(
        "quantem.core.postgres._run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, output, ""),
    )
    monkeypatch.setattr("quantem.core.postgres._tool", lambda name: name)

    assert _cluster_requires_recovery() is expected


def test_server_arguments_apply_cutover_tuning_to_existing_clusters():
    arguments = _postgres_server_arguments(54321)

    assert arguments[0] == "-D"
    assert Path(arguments[1]).name == "cluster"
    assert arguments[2:6] == ["-p", "54321", "-h", "127.0.0.1"]
    assert list(zip(arguments[6::2], arguments[7::2], strict=True)) == [
        ("-c", option) for option in POSTGRES_RUNTIME_OPTIONS
    ]
