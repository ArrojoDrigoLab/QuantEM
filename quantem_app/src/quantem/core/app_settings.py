"""Small persisted settings needed before the application database is migrated.

Database-migration backup policy cannot live in PostgreSQL: the launcher must
read it before it knows whether the installed database schema is current.  The
settings therefore live in one atomic JSON file under the QuantEM data root.
They contain no secrets and are independent of both SQLite and PostgreSQL.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any

from quantem.core.config import STORAGE_DIR
from quantem.core.database_cache import MAX_CACHE_MB, MIN_CACHE_MB, machine_cache_limits

logger = logging.getLogger(__name__)

APP_SETTINGS_PATH = STORAGE_DIR / "settings.json"
APP_SETTINGS_SCHEMA_VERSION = 1

DEFAULT_PRE_MIGRATION_BACKUP_RETENTION_COUNT = 1
DEFAULT_PRE_MIGRATION_BACKUP_RETENTION_DAYS = 30
MAX_PRE_MIGRATION_BACKUP_RETENTION_COUNT = 100
MAX_PRE_MIGRATION_BACKUP_RETENTION_DAYS = 3650

_SETTINGS_LOCK = threading.RLock()


class AppSettingsValidationError(ValueError):
    """A user-supplied application setting is outside its supported range."""


@dataclass(frozen=True)
class AppSettings:
    pre_migration_backup_retention_count: int = DEFAULT_PRE_MIGRATION_BACKUP_RETENTION_COUNT
    pre_migration_backup_retention_days: int = DEFAULT_PRE_MIGRATION_BACKUP_RETENTION_DAYS
    # None preserves the cluster's existing setting, including ALTER SYSTEM
    # customizations made before QuantEM exposed this preference.
    database_cache_mb: int | None = None

    def as_dict(self) -> dict[str, int | None]:
        return asdict(self)


def _bounded_integer(
    value: Any,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AppSettingsValidationError(f"{name} must be a whole number.")
    if not minimum <= value <= maximum:
        raise AppSettingsValidationError(f"{name} must be between {minimum} and {maximum}.")
    return value


def validate_app_settings(values: dict[str, Any]) -> AppSettings:
    """Validate a complete settings payload and return its typed form."""

    cache_mb = values.get("database_cache_mb")
    if cache_mb is not None:
        cache_mb = _bounded_integer(
            cache_mb, name="Database cache (MiB)", minimum=MIN_CACHE_MB, maximum=MAX_CACHE_MB
        )
    return AppSettings(
        database_cache_mb=cache_mb,
        pre_migration_backup_retention_count=_bounded_integer(
            values.get(
                "pre_migration_backup_retention_count",
                DEFAULT_PRE_MIGRATION_BACKUP_RETENTION_COUNT,
            ),
            name="Backup count",
            minimum=1,
            maximum=MAX_PRE_MIGRATION_BACKUP_RETENTION_COUNT,
        ),
        pre_migration_backup_retention_days=_bounded_integer(
            values.get(
                "pre_migration_backup_retention_days",
                DEFAULT_PRE_MIGRATION_BACKUP_RETENTION_DAYS,
            ),
            name="Backup age",
            minimum=1,
            maximum=MAX_PRE_MIGRATION_BACKUP_RETENTION_DAYS,
        ),
    )


def load_app_settings() -> AppSettings:
    """Read settings, using safe defaults for a missing or damaged file."""

    if not APP_SETTINGS_PATH.is_file():
        return AppSettings()
    try:
        payload = json.loads(APP_SETTINGS_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise AppSettingsValidationError("The settings document must be an object.")
        return validate_app_settings(payload)
    except (OSError, json.JSONDecodeError, AppSettingsValidationError) as exc:
        # A damaged preference file must not disable the safety backup that is
        # meant to make a database upgrade recoverable. Keep the conservative
        # defaults and leave a diagnostic in the application log.
        logger.warning(
            "Could not read %s; using the default application settings: %s",
            APP_SETTINGS_PATH,
            exc,
        )
        return AppSettings()


def save_app_settings(settings: AppSettings) -> AppSettings:
    """Persist settings atomically so an interrupted write cannot corrupt them."""

    payload = {
        "schema_version": APP_SETTINGS_SCHEMA_VERSION,
        **settings.as_dict(),
    }
    APP_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = APP_SETTINGS_PATH.with_name(f".{APP_SETTINGS_PATH.name}.{os.getpid()}.tmp")
    with _SETTINGS_LOCK:
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, APP_SETTINGS_PATH)
        finally:
            if temporary.exists():
                temporary.unlink()
    return settings


def update_app_settings(values: dict[str, Any]) -> AppSettings:
    """Merge and persist the supported subset of application settings."""

    unknown = set(values) - set(AppSettings.__dataclass_fields__)
    if unknown:
        raise AppSettingsValidationError("Unknown setting(s): " + ", ".join(sorted(unknown)) + ".")
    if values.get("database_cache_mb") is not None:
        _bounded_integer(
            values["database_cache_mb"],
            name="Database cache (MiB)",
            minimum=MIN_CACHE_MB,
            maximum=int(machine_cache_limits()["max_mb"]),
        )
    # Cache and backup controls can save concurrently. Lock the merge as well
    # as the replace so one preference cannot overwrite the other.
    with _SETTINGS_LOCK:
        merged = load_app_settings().as_dict()
        merged.update(values)
        return save_app_settings(validate_app_settings(merged))
