"""Job-queue entry point for a quantitative analysis run.

``quantem.jobs.handlers.handle_run_analysis`` validates the payload and hands
off here. The split exists so the analysis package never imports the jobs
package at module scope: the queue depends on the analysis, not the other way
round.
"""

from __future__ import annotations

from typing import Any

from quantem.jobs.reporter import CancelToken, JobCancelledError, JobReporter

from .models import AnalysisRun
from .service import run_for_segmentation


def run_job(
    payload: dict[str, Any], reporter: JobReporter, cancel: CancelToken
) -> dict[str, Any]:
    """Execute the ``AnalysisRun`` named in ``payload["analysis_run_id"]``.

    Cancellation is checked at every progress step rather than inside the
    numerics: one image's analysis is seconds of CPU per phase, so a per-phase
    check bounds the wait without threading a token through pure functions that
    are also used offline.
    """
    cancel.check_cancelled()

    run_id = str(payload.get("analysis_run_id") or "").strip()
    if not run_id:
        raise ValueError("payload.analysis_run_id is required")

    run = (
        AnalysisRun.objects.select_related(
            "segmentation", "segmentation__asset", "segmentation__segmentation_type"
        )
        .filter(id=run_id)
        .first()
    )
    if run is None:
        raise ValueError(f"Analysis run {run_id} not found")

    def progress(percent: float, message: str) -> None:
        cancel.check_cancelled()
        reporter.update(progress=percent, message=message)

    try:
        result = run_for_segmentation(run, progress=progress)
    except JobCancelledError:
        # The service has already marked the run FAILED with the cancellation
        # exception's text; replace it with something a user asked for.
        AnalysisRun.objects.filter(id=run.id).update(
            error="Cancelled before the analysis finished."
        )
        raise

    reporter.log(
        "info",
        f"Wrote the export bundle to {run.export_dir}",
    )
    return {
        "analysis_run_id": str(run.id),
        "segmentation_id": str(run.segmentation_id),
        "status": run.status,
        "export_dir": run.export_dir,
        "n_objects": result["objects"]["n"],
        "calibrated": result["calibrated"],
        "caveats": result["caveats"],
    }
