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


def run_job(payload: dict[str, Any], reporter: JobReporter, cancel: CancelToken) -> dict[str, Any]:
    """Execute the ``AnalysisRun`` named in ``payload["analysis_run_id"]``.

    Cancellation is checked while masks are read, between measurement phases,
    and inside every Monte-Carlo replicate. Progress is persisted at bounded
    intervals so a large null remains responsive without turning every draw
    into a database write.
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
        # This is both the per-chunk progress hook and the service's coarse
        # phase-boundary check, which is why the cancel read here is not rate
        # limited: see CancelToken's docstring for the version that was.
        cancel.check_cancelled()
        reporter.update(progress=percent, message=message)

    try:
        from quantem.analysis.global_runs import run_global_area
        from quantem.analysis.identity import input_fingerprint

        if (
            run.input_fingerprint
            and input_fingerprint(run.segmentation, run.params) != run.input_fingerprint
        ):
            raise ValueError("Inputs changed while this analysis was queued. Start a new batch.")
        runner = run_global_area if run.params.get("global_area") else run_for_segmentation
        result = runner(
            run,
            progress=progress,
            cancel_check=cancel.check_cancelled,
        )
        if (
            run.input_fingerprint
            and input_fingerprint(run.segmentation, run.params) != run.input_fingerprint
        ):
            raise ValueError(
                "Inputs changed during measurement. Re-run to measure one consistent version."
            )
    except JobCancelledError:
        # The service has already marked the run FAILED with the cancellation
        # exception's text; replace it with something a user asked for.
        AnalysisRun.objects.filter(id=run.id).update(
            error="Cancelled before the analysis finished."
        )
        raise
    except Exception as exc:
        AnalysisRun.objects.filter(id=run.id).update(status="FAILED", error=str(exc), export_dir="")
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
        "n_objects": (result.get("objects") or {}).get("n"),
        "calibrated": result["calibrated"],
        "caveats": result["caveats"],
    }
