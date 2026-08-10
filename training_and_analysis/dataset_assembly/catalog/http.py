"""Small HTTP helpers for deterministic scanners."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


USER_AGENT = "public-intracellular-em-catalog/0.1 (+research dataset catalog)"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RATE_LIMIT_DELAY_SECONDS = 65.0
MAX_RETRY_DELAY_SECONDS = 120.0


class HttpFetchError(RuntimeError):
    def __init__(self, url: str, message: str, status: int | None = None, excerpt: str | None = None):
        super().__init__(message)
        self.url = url
        self.status = status
        self.excerpt = excerpt


def get_json(url: str, timeout: int = 30, max_retries: int = DEFAULT_MAX_RETRIES) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    return _open_json(request, url, timeout, max_retries=max_retries)


def get_text(url: str, timeout: int = 30, max_retries: int = DEFAULT_MAX_RETRIES) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/xml,text/xml,*/*"},
    )
    return _open_text(request, url, timeout, max_retries=max_retries)


def post_json(url: str, payload: dict[str, Any], timeout: int = 30, max_retries: int = DEFAULT_MAX_RETRIES) -> Any:
    raw_payload = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=raw_payload,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    return _open_json(request, url, timeout, max_retries=max_retries)


def _open_json(request: urllib.request.Request, url: str, timeout: int, *, max_retries: int) -> Any:
    for attempt in range(max(0, max_retries) + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            excerpt = exc.read(1000).decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < max_retries:
                time.sleep(_retry_delay_seconds(exc, excerpt, attempt))
                continue
            raise HttpFetchError(url, f"HTTP {exc.code}", exc.code, excerpt) from exc
        except urllib.error.URLError as exc:
            raise HttpFetchError(url, str(exc.reason)) from exc
        except json.JSONDecodeError as exc:
            raise HttpFetchError(url, f"invalid JSON: {exc}") from exc
    raise HttpFetchError(url, "HTTP retry loop exited unexpectedly")


def _open_text(request: urllib.request.Request, url: str, timeout: int, *, max_retries: int) -> str:
    for attempt in range(max(0, max_retries) + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                return raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            excerpt = exc.read(1000).decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < max_retries:
                time.sleep(_retry_delay_seconds(exc, excerpt, attempt))
                continue
            raise HttpFetchError(url, f"HTTP {exc.code}", exc.code, excerpt) from exc
        except urllib.error.URLError as exc:
            raise HttpFetchError(url, str(exc.reason)) from exc
    raise HttpFetchError(url, "HTTP retry loop exited unexpectedly")


def _retry_delay_seconds(exc: urllib.error.HTTPError, excerpt: str, attempt: int) -> float:
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    try:
        if retry_after:
            return min(MAX_RETRY_DELAY_SECONDS, max(1.0, float(retry_after)))
    except ValueError:
        pass
    match = re.search(r"(\d+)\s+per\s+1\s+minute", excerpt or "")
    if match:
        return DEFAULT_RATE_LIMIT_DELAY_SECONDS
    return min(MAX_RETRY_DELAY_SECONDS, DEFAULT_RATE_LIMIT_DELAY_SECONDS * (attempt + 1))


def urlencode(params: dict[str, Any]) -> str:
    return urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
