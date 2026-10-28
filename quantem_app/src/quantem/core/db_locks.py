"""Small cross-backend transaction-locking primitives.

SQLite compatibility mode begins explicit transactions with ``IMMEDIATE`` and
therefore serializes every writer. PostgreSQL deliberately allows independent
writers, so code that protects a logical key rather than an existing row uses
a transaction-scoped advisory lock.
"""

from __future__ import annotations

import hashlib

from django.db import connection


def lock_transaction_key(namespace: str, key: str) -> None:
    """Serialize this transaction with peers using the same logical key."""

    if not connection.in_atomic_block:
        raise RuntimeError("transaction key locks require transaction.atomic()")
    if connection.vendor != "postgresql":
        return
    payload = f"quantem:{namespace}:{key}".encode()
    unsigned = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    signed = unsigned if unsigned < (1 << 63) else unsigned - (1 << 64)
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", [signed])
