"""Exercise the actual NSIS functions without installing the application.

Set QUANTEM_MAKENSIS to the Windows NSIS compiler to enable this integration
test. The harness is silent and writes exclusively beneath pytest's temp dir.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quantem.core.database_cache import cache_limits


def test_installer_cache_policy_and_upgrade_preservation(tmp_path):
    compiler = os.environ.get("QUANTEM_MAKENSIS")
    if sys.platform != "win32" or not compiler:
        pytest.skip("Set QUANTEM_MAKENSIS to run the Windows installer harness.")
    tests = Path(__file__).resolve().parent
    app = tests.parents[2]
    hook = app / "desktop/src-tauri/nsis/database-cache.nsh"
    environment = {
        **os.environ,
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
        "TMPDIR": str(tmp_path),
    }
    subprocess.run(
        [
            compiler,
            "/V2",
            f"/DCACHE_HOOK={hook}",
            f"/DTEST_ROOT={tmp_path}",
            str(tests / "cache_harness.nsi"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    subprocess.run(
        [str(tmp_path / "cache-harness.exe"), "/S"],
        check=True,
        env=environment,
        timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    results_dir = tmp_path / "nsis-tests"
    rows = [json.loads(line) for line in (results_dir / "results.jsonl").read_text().splitlines()]
    assert rows[0]["detected_ram_mb"] > 0
    for row in rows:
        if "ram_mb" in row:
            policy = cache_limits(row["ram_mb"] * 1024**2 or None)
            assert (row["recommended"], row["maximum"]) == (
                policy["recommended_mb"],
                policy["max_mb"],
            )
        if "input" in row:
            assert row["valid"] == (row["input"] in ("32", "1024", "4096"))
    assert (
        json.loads((results_dir / "fresh/data/settings.json").read_text())["database_cache_mb"]
        == 768
    )
    assert (results_dir / "existing-settings/data/settings.json").read_text() == (
        '{"database_cache_mb":1024,"pre_migration_backup_retention_count":3}'
    )
    for name in ("existing-postgres", "legacy-sqlite", "upgrade"):
        assert not (results_dir / name / "data/settings.json").exists()
    assert (
        results_dir / "existing-postgres/data/postgres/cluster/postgresql.auto.conf"
    ).read_text() == ("shared_buffers = '1GB'\n")
    assert (
        results_dir / "legacy-sqlite/data/quantem.sqlite3"
    ).read_text() == "legacy database sentinel"
    assert (
        json.loads((results_dir / "silent-fresh/data/settings.json").read_text())[
            "database_cache_mb"
        ]
        == rows[0]["suggestion"]
    )
