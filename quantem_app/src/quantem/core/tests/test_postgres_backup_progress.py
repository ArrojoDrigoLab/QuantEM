from pathlib import Path

from quantem.core import postgres_backup


def test_pg_dump_has_no_wall_clock_deadline_and_reports_written_bytes(monkeypatch, tmp_path):
    events: list[dict[str, object]] = []
    captured: list[str] = []

    class Process:
        def __init__(self):
            self.returncode = 0
            self._polls = iter([None, 0])

        def poll(self):
            return next(self._polls)

        def wait(self):
            return 0

    def popen(command, **_kwargs):
        captured.extend(command)
        destination = Path(command[command.index("--file") + 1])
        destination.write_bytes(b"verified-backup")
        return Process()

    monkeypatch.setattr(postgres_backup.subprocess, "Popen", popen)
    monkeypatch.setattr(postgres_backup, "postgres_tool", lambda name: Path(name))
    monkeypatch.setattr(postgres_backup, "postgres_runtime_environment", lambda env: env)
    monkeypatch.setattr(postgres_backup, "_tool_environment", lambda: {})
    monkeypatch.setattr(postgres_backup.time, "sleep", lambda _seconds: None)

    destination = tmp_path / ".quantem.dump.tmp"
    postgres_backup._dump_database(destination, progress_callback=events.append)

    assert "--compress=zstd:1" in captured
    assert events[-1]["completed"] == len(b"verified-backup")
    assert events[-1]["total"] == len(b"verified-backup")
    assert events[-1]["phase"] == "database-backup"


def test_checksum_reports_real_progress(monkeypatch, tmp_path):
    archive = tmp_path / "quantem.dump"
    archive.write_bytes(b"backup-contents")
    events: list[dict[str, object]] = []

    digest = postgres_backup._sha256(archive, progress_callback=events.append)

    assert len(digest) == 64
    assert events[-1]["phase"] == "database-backup-verify"
    assert events[-1]["completed"] == archive.stat().st_size
    assert events[-1]["total"] == archive.stat().st_size


def test_backup_falls_back_when_runtime_does_not_recognize_zstd(monkeypatch, tmp_path):
    compressions: list[str] = []

    def dump(destination, *, progress_callback, compression="zstd:1"):
        del progress_callback
        compressions.append(compression)
        if compression.startswith("zstd"):
            raise postgres_backup.BackupError("compression algorithm zstd is unrecognized")
        destination.write_bytes(b"portable-backup")

    monkeypatch.setattr(postgres_backup, "_dump_database", dump)
    monkeypatch.setenv("QUANTEM_POSTGRES_DATABASE", "quantem")

    archive, manifest = postgres_backup.create_postgres_backup(tmp_path / "quantem.dump")

    assert compressions == ["zstd:1", "gzip:1"]
    assert archive.read_bytes() == b"portable-backup"
    assert manifest.is_file()
