"""Database configuration.

QuantEM is a single-user desktop application. There is exactly one backend --
plain SQLite in the user data directory -- and no server to connect to, so
there are no credentials to configure.

WAL journalling lets the job scheduler read while a request writes; the busy
timeout absorbs the short write contention that produces on a local disk.
"""

import os

from quantem.core.config import DB_PATH


def _get_sqlite_timeout_seconds() -> float:
    raw = str(os.environ.get("SQLITE_TIMEOUT_SECONDS", "30")).strip()
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        timeout = 30.0
    return max(1.0, timeout)


def get_database_config() -> dict:
    """Return the ``DATABASES['default']`` mapping for the local SQLite file."""
    timeout_seconds = _get_sqlite_timeout_seconds()
    return {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": str(DB_PATH),
        "OPTIONS": {
            # Python-level wait for the GIL-side connection lock.
            "timeout": timeout_seconds,
            "init_command": (
                "PRAGMA journal_mode=WAL;"
                "PRAGMA synchronous=NORMAL;"
                f"PRAGMA busy_timeout={int(timeout_seconds * 1000)};"
                "PRAGMA foreign_keys=ON;"
            ),
            "transaction_mode": "IMMEDIATE",
        },
    }
