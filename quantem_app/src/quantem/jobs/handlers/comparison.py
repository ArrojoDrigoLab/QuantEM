"""Read-only postprocessing for detailed saved-result comparison displays."""

from quantem.jobs.constants import JOB_TYPE_BUILD_RESULT_COMPARISON
from quantem.jobs.registry import job_handler


@job_handler(JOB_TYPE_BUILD_RESULT_COMPARISON)
def handle_build_result_comparison(payload, reporter, cancel):
    from quantem.segmentation.result_overlays import build_comparison

    build_comparison(
        payload["result_id"],
        payload["comparison_key"],
        cancel_check=cancel.check_cancelled,
        progress=lambda progress, message: reporter.update(progress=progress, message=message),
    )
    return {"result_id": payload["result_id"], "comparison_key": payload["comparison_key"]}
