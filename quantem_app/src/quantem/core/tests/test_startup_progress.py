from quantem.cli import _DjangoMigrationProgressStream
from quantem.core.startup_progress import StartupProgress


def test_general_startup_protocol_tracks_stages_and_failure_context():
    events: list[dict[str, object]] = []
    now = [10.0]
    progress = StartupProgress(events.append, heartbeat_interval=0, clock=lambda: now[0])

    progress.stage(
        "database-backup",
        "Creating a database recovery snapshot",
        category="database-upgrade",
        critical=True,
        completed=0,
        total=4,
        unit="changes",
    )
    now[0] = 15.0
    progress.update(completed=1)
    progress.fail(RuntimeError("snapshot disk is full"))
    progress.close()

    assert events[0]["category"] == "database-upgrade"
    assert events[0]["can_close"] is False
    assert events[1]["percent"] == 25.0
    assert events[1]["elapsed_seconds"] == 5
    assert events[1]["eta_seconds"] == 15
    assert events[2]["phase"] == "failed"
    assert events[2]["failed_phase"] == "database-backup"
    assert events[2]["detail"] == "snapshot disk is full"
    assert events[2]["can_close"] is True


def test_forwarded_heartbeats_do_not_claim_measurable_progress():
    events: list[dict[str, object]] = []
    now = [20.0]
    progress = StartupProgress(events.append, heartbeat_interval=0, clock=lambda: now[0])
    progress.stage("database", "Starting the private database")
    now[0] = 50.0

    progress.forward(
        {
            "phase": "redo",
            "message": "Recovering",
            "heartbeat": True,
            "elapsed_seconds": 4,
        },
        category="database-recovery",
        critical=True,
    )
    progress.close()

    assert events[-1]["last_progress_seconds"] == 30
    assert events[-1]["elapsed_seconds"] == 30
    assert events[-1]["heartbeat"] is True


def test_django_migrations_report_completed_changes_without_time_estimates():
    events: list[dict[str, object]] = []
    progress = StartupProgress(events.append, heartbeat_interval=0)
    progress.stage(
        "schema-migration",
        "Updating the database schema",
        category="database-upgrade",
        critical=True,
        completed=0,
        total=2,
        unit="changes",
    )
    stream = _DjangoMigrationProgressStream(progress, 2)

    stream.write("  Applying assets.0012_add_index...")
    stream.write(" OK\n")
    stream.write("  Applying jobs.0009_add_state...")
    stream.write(" OK\n")
    progress.close()

    assert events[-1]["completed"] == 2
    assert events[-1]["total"] == 2
    assert events[-1]["percent"] == 100.0
    assert events[-1]["detail"] == "Applied jobs.0009_add_state"
