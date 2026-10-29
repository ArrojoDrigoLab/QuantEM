from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from quantem.core.postgres import (
    DATABASE_RECOVERY_PROGRESS_PREFIX,
    POSTGRES_RECOVERY_STALL_TIMEOUT_SECONDS,
    POSTGRES_RUNTIME_OPTIONS,
    EmbeddedPostgres,
    PostgresEndpoint,
    PostgresError,
    _cluster_requires_recovery,
    _postgres_server_arguments,
    _RecoveryLogMonitor,
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
    monkeypatch.setattr("quantem.core.postgres._RecoveryLogMonitor.poll", lambda self: False)
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
    monkeypatch.setattr("quantem.core.postgres._RecoveryLogMonitor.poll", lambda self: False)
    monkeypatch.setattr("quantem.core.postgres.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("quantem.core.postgres.time.sleep", lambda _: None)

    EmbeddedPostgres()._wait_ready(_endpoint(), recovery_expected=True)

    assert DATABASE_RECOVERY_PROGRESS_PREFIX in capsys.readouterr().out


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

    assert DATABASE_RECOVERY_PROGRESS_PREFIX not in capsys.readouterr().out


def test_recovery_fails_only_after_measurable_progress_stalls(monkeypatch):
    stalled_at = float(POSTGRES_RECOVERY_STALL_TIMEOUT_SECONDS + 1)
    clock = iter([0.0, 0.0, stalled_at])
    monkeypatch.setattr(
        "quantem.core.postgres._run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 1, "rejecting connections", ""),
    )
    monkeypatch.setattr("quantem.core.postgres._tool", lambda name: name)
    monkeypatch.setattr("quantem.core.postgres._RecoveryLogMonitor.poll", lambda self: False)
    monkeypatch.setattr("quantem.core.postgres.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("quantem.core.postgres.time.sleep", lambda _: None)

    with pytest.raises(PostgresError, match="no measurable recovery progress"):
        EmbeddedPostgres()._wait_ready(_endpoint(), recovery_expected=True)


def test_recovery_progress_resets_the_stall_window(monkeypatch):
    almost_stalled = float(POSTGRES_RECOVERY_STALL_TIMEOUT_SECONDS - 1)
    past_original_window = float(POSTGRES_RECOVERY_STALL_TIMEOUT_SECONDS + 1)
    clock = iter([0.0, 0.0, almost_stalled, past_original_window])
    attempts = iter(
        [
            subprocess.CompletedProcess([], 1, "rejecting connections", ""),
            subprocess.CompletedProcess([], 1, "rejecting connections", ""),
            subprocess.CompletedProcess([], 0, "accepting connections", ""),
        ]
    )
    progress = iter([False, True, False])
    monkeypatch.setattr("quantem.core.postgres._run", lambda *args, **kwargs: next(attempts))
    monkeypatch.setattr("quantem.core.postgres._tool", lambda name: name)
    monkeypatch.setattr(
        "quantem.core.postgres._RecoveryLogMonitor.poll", lambda self: next(progress)
    )
    monkeypatch.setattr("quantem.core.postgres.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("quantem.core.postgres.time.sleep", lambda _: None)

    EmbeddedPostgres()._wait_ready(_endpoint(), recovery_expected=True)


def test_recovery_log_monitor_tracks_wal_and_checkpoint_progress(monkeypatch, tmp_path):
    log_path = tmp_path / "postgres.log"
    log_path.write_text("", encoding="utf-8")
    monkeypatch.setattr("quantem.core.postgres.POSTGRES_LOG_PATH", log_path)
    monitor = _RecoveryLogMonitor(0)

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("LOG: redo in progress, elapsed time: 20.62 s, current LSN: 3/4A481230\n")
    assert monitor.poll() is True
    assert monitor.phase == "redo"
    assert monitor.current_lsn == "3/4A481230"

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("LOG: redo done at 3/5C5103F8\n")
        handle.write("LOG: checkpoint starting: end-of-recovery immediate wait\n")
    assert monitor.poll() is True
    assert monitor.phase == "checkpoint"
    assert monitor.current_lsn == "3/5C5103F8"

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("LOG: checkpoint complete: wrote 3749 buffers\n")
    assert monitor.poll() is True
    assert monitor.phase == "finalizing"


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
