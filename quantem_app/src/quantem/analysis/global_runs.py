"""Save global foreground reports using the same run and export lifecycle."""

import csv
import json

from django.utils import timezone

from quantem.analysis.global_area import global_area_report
from quantem.analysis.service import export_dir_for_run


def run_global_area(run, *, progress, cancel_check):
    run.status = "RUNNING"
    run.started_at = timezone.now()
    run.save(update_fields=["status", "started_at"])
    try:
        cancel_check()
        progress(15, "Measuring foreground area")
        report = global_area_report(
            run.segmentation, analysis_mask_ids=run.params.get("analysis_mask_ids", [])
        )
        cancel_check()
        pixel = run.segmentation.asset.pixel_size_nm
        rows = [report["whole_image"], *report["analysis_masks"]]
        result = {
            "global_area": report,
            "pixel_size_nm": pixel,
            "calibrated": bool(pixel),
            "caveats": [],
            "identity": run.input_identity,
        }
        directory = export_dir_for_run(run.id)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "area.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": str(run.id),
                    "input_fingerprint": run.input_fingerprint,
                    "inputs": run.input_identity,
                    "settings": run.params,
                    "result": result,
                    "measurement": "Foreground pixels divided by measurement-area pixels; holes excluded.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        run.results = result
        run.export_dir = str(directory)
        run.status = "SUCCESS"
        run.finished_at = timezone.now()
        run.save()
        return result
    except Exception as exc:
        run.status = "FAILED"
        run.error = str(exc)
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "error", "finished_at"])
        raise
