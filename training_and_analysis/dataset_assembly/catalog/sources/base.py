"""Common scanner types and error handling."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..http import HttpFetchError
from ..models import Candidate, ScannerError


@dataclass
class ScannerResult:
    candidates: list[Candidate] = field(default_factory=list)
    errors: list[ScannerError] = field(default_factory=list)
    cursor: dict[str, Any] | None = None
    cursor_complete: bool = True

    def extend(self, other: "ScannerResult") -> None:
        self.candidates.extend(other.candidates)
        self.errors.extend(other.errors)
        self.cursor = other.cursor
        self.cursor_complete = other.cursor_complete


def safe_collect(source_name: str, adapter: str, fn, *args, **kwargs) -> ScannerResult:
    try:
        value = fn(*args, **kwargs)
        if isinstance(value, ScannerResult):
            return value
        return ScannerResult(candidates=list(value))
    except HttpFetchError as exc:
        return ScannerResult(
            errors=[
                ScannerError(
                    source_name=source_name,
                    adapter=adapter,
                    url=exc.url,
                    response_status=exc.status,
                    response_excerpt=exc.excerpt,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    stack_trace=traceback.format_exc(),
                    reproduction_command=f"python -m catalog.scan --source {source_name}",
                )
            ]
        )
    except Exception as exc:  # noqa: BLE001 - scanner must convert all adapter failures to queueable errors.
        return ScannerResult(
            errors=[
                ScannerError(
                    source_name=source_name,
                    adapter=adapter,
                    url=None,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    stack_trace=traceback.format_exc(),
                    reproduction_command=f"python -m catalog.scan --source {source_name}",
                )
            ]
        )


def unique_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    seen: set[tuple[str, str]] = set()
    unique: list[Candidate] = []
    for candidate in candidates:
        key = (candidate.source_name, candidate.source_record_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique
