"""The analysis job handler."""

from quantem.jobs.constants import JOB_TYPE_RUN_ANALYSIS
from quantem.jobs.registry import job_handler
from quantem.jobs.reporter import CancelToken, JobReporter


@job_handler(JOB_TYPE_RUN_ANALYSIS)
def handle_run_analysis(
    payload: dict, reporter: JobReporter, cancel: CancelToken
) -> dict:
    """Run a quantitative analysis and write its export bundle.

    The payload is passed through untouched; ``analysis_run_id`` identifies both
    the run record and its export directory.
    """
    cancel.check_cancelled()
    analysis_run_id = str(payload.get("analysis_run_id") or "").strip()
    if not analysis_run_id:
        raise ValueError("payload.analysis_run_id is required")

    # Imported lazily: the analysis suite reaches back into segmentation and
    # assets, and importing it here at module load would make the handler
    # registry depend on the whole graph.
    from quantem.analysis.run_job import run_analysis_job

    reporter.update(progress=1.0, message="running analysis")
    return run_analysis_job(
        payload=payload,
        reporter=reporter,
        cancel=cancel,
    )
