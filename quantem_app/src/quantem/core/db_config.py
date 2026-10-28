"""Database configuration for the local application database.

Installed builds use QuantEM's private loopback-only PostgreSQL server. SQLite
remains available only for tests, development comparisons, and the explicit
SQLite-to-PostgreSQL migration command.
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


def _sqlite_config() -> dict:
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


def _postgres_config() -> dict:
    required = {
        "NAME": "QUANTEM_POSTGRES_DATABASE",
        "USER": "QUANTEM_POSTGRES_USER",
        "PASSWORD": "QUANTEM_POSTGRES_PASSWORD",
        "HOST": "QUANTEM_POSTGRES_HOST",
        "PORT": "QUANTEM_POSTGRES_PORT",
    }
    missing = [environment for environment in required.values() if not os.environ.get(environment)]
    if missing:
        raise RuntimeError(
            "The private PostgreSQL server was not prepared before Django started; "
            f"missing {', '.join(missing)}. Launch QuantEM through quantem-app."
        )
    return {
        "ENGINE": "django.db.backends.postgresql",
        **{key: os.environ[environment] for key, environment in required.items()},
        "CONN_MAX_AGE": 60,
        "CONN_HEALTH_CHECKS": True,
        "OPTIONS": {
            "connect_timeout": 10,
            "application_name": "quantem-desktop",
            "sslmode": "disable",
        },
    }


def get_database_config() -> dict:
    """Return Django's default database mapping.

    ``postgresql`` is the product default. ``sqlite`` is deliberately explicit
    so an unavailable private server cannot silently redirect writes into an
    older SQLite snapshot.
    """

    engine = os.environ.get("QUANTEM_DATABASE_ENGINE", "postgresql").strip().lower()
    if engine == "postgresql":
        return _postgres_config()
    if engine == "sqlite":
        return _sqlite_config()
    raise RuntimeError(
        f"QUANTEM_DATABASE_ENGINE must be 'postgresql' or 'sqlite' (got {engine!r})."
    )
