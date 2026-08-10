"""Import shim for the job registry.

``quantem.jobs.handlers`` reaches for ``quantem.analysis.run_job.run_analysis_job``
and that module is owned by the jobs package, so the name is matched here rather
than changed there. The implementation is :func:`quantem.analysis.job.run_job`;
this module adds nothing and must stay that way.
"""

from __future__ import annotations

from .job import run_job

#: The name ``quantem.jobs.handlers.handle_run_analysis`` imports.
run_analysis_job = run_job

__all__ = ["run_analysis_job", "run_job"]
