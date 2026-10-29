"""Local stage timings, shared by HTTP diagnostics and background job logs."""

from __future__ import annotations

import functools
import logging
import time
from contextvars import ContextVar
from uuid import uuid4

logger = logging.getLogger("quantem.performance")
_TIMINGS = ContextVar("quantem_performance_timings", default=None)


def timed_stage(name):
    def decorate(function):
        @functools.wraps(function)
        def run(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - started) * 1000
                timings = _TIMINGS.get()
                if timings is not None:
                    timings[name] = timings.get(name, 0.0) + elapsed
                elif elapsed >= 500:
                    logger.info("stage=%s duration_ms=%.1f", name, elapsed)

        return run

    return decorate


class PerformanceTimingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        trace = uuid4().hex
        timings = {}
        token = _TIMINGS.set(timings)
        started = time.perf_counter()
        try:
            response = self.get_response(request)
            timings["app"] = (time.perf_counter() - started) * 1000
            response["Server-Timing"] = ", ".join(
                f"{name};dur={ms:.1f}" for name, ms in timings.items()
            )
            response["X-QuantEM-Trace"] = trace
            if timings["app"] >= 500:
                logger.info(
                    "trace=%s method=%s path=%s status=%s timings_ms=%s",
                    trace,
                    request.method,
                    request.path,
                    response.status_code,
                    timings,
                )
            return response
        finally:
            _TIMINGS.reset(token)
