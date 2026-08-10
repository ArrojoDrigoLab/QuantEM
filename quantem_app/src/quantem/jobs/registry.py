"""Job-type -> handler registry.

Handlers register themselves at import time; ``jobs.apps.JobsConfig.ready``
autodiscovers every ``handlers`` module so the table is complete before the
scheduler dispatches anything.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from quantem.jobs.constants import ALLOWED_JOB_TYPES, LEGACY_JOB_TYPES

if TYPE_CHECKING:
    # Imported for typing only: `reporter` imports nothing from this module, but
    # keeping it out of the runtime import graph avoids any future cycle.
    from quantem.jobs.reporter import CancelToken, JobReporter

JobHandler = Callable[[dict, "JobReporter", "CancelToken"], dict]

_HANDLERS: dict[str, JobHandler] = {}


def job_handler(job_type: str) -> Callable[[JobHandler], JobHandler]:
    """Register ``func`` as the handler for ``job_type``.

    Only job types QuantEM actually ships may be registered: an unknown name is
    almost always a handler left behind by a dropped feature, and a queue that
    accepts work it cannot describe to the user is worse than one that refuses
    to start.
    """
    if job_type not in ALLOWED_JOB_TYPES:
        raise ValueError(
            f"Cannot register a handler for unknown job type '{job_type}'. "
            "Add it to ALLOWED_JOB_TYPES (and JOB_DEFAULTS) in "
            "quantem.jobs.constants, or delete the handler."
        )

    def decorator(func: JobHandler) -> JobHandler:
        _HANDLERS[job_type] = func
        return func

    return decorator


def get_handler(job_type: str) -> JobHandler:
    if job_type not in _HANDLERS:
        if job_type in LEGACY_JOB_TYPES:
            raise KeyError(
                f"Job type '{job_type}' was removed from QuantEM and has no "
                "handler. This row predates the current build."
            )
        raise KeyError(f"No job handler registered for '{job_type}'")
    return _HANDLERS[job_type]
