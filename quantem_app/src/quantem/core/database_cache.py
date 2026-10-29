"""Database RAM recommendations and the active cache shown in Settings."""

from __future__ import annotations

from quantem.core.machine import get_machine_profile

MIN_CACHE_MB = 32
MAX_CACHE_MB = 65536


def cache_limits(total_ram_bytes: int | None) -> dict[str, int | None]:
    """Reserve a small share for database pages, leaving room for image processing.

    The Windows installer uses this same policy: 1/32 of RAM rounded down to
    64 MiB, between 128 and 4096 MiB. Manual choices may use up to 1/4 of RAM
    (at most 64 GiB). Unknown hardware gets a conservative 256 MiB suggestion.
    These are suggestions only; upgrades never recalculate a saved choice.
    """
    ram_mb = total_ram_bytes // (1024**2) if total_ram_bytes else None
    maximum = max(MIN_CACHE_MB, min(MAX_CACHE_MB, ram_mb // 4)) if ram_mb else 4096
    recommended = min(4096, max(128, (ram_mb // 32 // 64) * 64)) if ram_mb else 256
    return {
        "total_ram_mb": ram_mb,
        "recommended_mb": min(recommended, maximum),
        "min_mb": MIN_CACHE_MB,
        "max_mb": maximum,
    }


def machine_cache_limits() -> dict[str, int | None]:
    return cache_limits(get_machine_profile().total_ram_bytes)


def database_cache_status(configured_mb: int | None) -> dict[str, object]:
    """Read the live value without changing or restarting the database."""
    from django.db import connection

    active_mb = None
    available = connection.vendor == "postgresql"
    if available:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_size_bytes(current_setting('shared_buffers')) / 1048576")
            active_mb = int(cursor.fetchone()[0])
    return {
        **machine_cache_limits(),
        "available": available,
        "active_mb": active_mb,
        "restart_required": available and configured_mb is not None and configured_mb != active_mb,
    }
