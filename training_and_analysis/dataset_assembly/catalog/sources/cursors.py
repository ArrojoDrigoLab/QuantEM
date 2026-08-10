"""Small helpers for scanner cursor payloads."""

from __future__ import annotations

from typing import Any

from ..models import Candidate
from .base import ScannerResult


def cursor_int(cursor: dict[str, Any] | None, key: str, default: int) -> int:
    if not isinstance(cursor, dict):
        return default
    try:
        return int(cursor.get(key, default))
    except (TypeError, ValueError):
        return default


def cursor_result(candidates: list[Candidate], *, complete: bool, cursor: dict[str, Any]) -> ScannerResult:
    if complete:
        return ScannerResult(candidates=candidates, cursor={"complete": True}, cursor_complete=True)
    payload = {key: value for key, value in cursor.items() if value is not None}
    payload["complete"] = False
    return ScannerResult(candidates=candidates, cursor=payload, cursor_complete=False)
