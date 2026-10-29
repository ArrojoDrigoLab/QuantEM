"""Save global foreground reports using the same run and export lifecycle."""

import csv
import json
import shutil

from django.utils import timezone

from quantem.analysis.global_area import global_area_report
from quantem.analysis.notes import analysis_notes, settings_notes
from quantem.analysis.publication import input_guard, publish, staging_directory
from quantem.analysis.settings import global_settings


def run_global_area(run, *, progress, cancel_check, apply_output_selection=False):
    run.refresh_from_db()
    if run.status == "SUCCESS":
        return run.results
    run.status = "RUNNING"
    run.started_at = timezone.now()
    if (
        not type(run)
        .objects.filter(pk=run.pk)
        .exclude(status="SUCCESS")
        .update(status=run.status, started_at=run.started_at)
    ):
        run.refresh_from_db()
        return run.results
    directory = None
    try:
        from quantem.analysis.input_contract import require_complete_organelles

        require_complete_organelles(run.segmentation, run.params)
        expected = input_guard(run)
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
            "segmentation_settings": global_settings(run.segmentation),
        }
        result["notes"] = settings_notes(result["segmentation_settings"])
        result["caveats"] = analysis_notes(result)
        directory = staging_directory(run)
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
        publish(
            run,
            directory,
            expected=expected,
            results=result,
            cancel_check=cancel_check,
            apply_output_selection=apply_output_selection,
        )
        return result
    except Exception as exc:
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
        run.status = "FAILED"
        run.error = str(exc)
        run.finished_at = timezone.now()
        type(run).objects.filter(pk=run.pk).exclude(status="SUCCESS").update(
            status=run.status, error=run.error, finished_at=run.finished_at
        )
        raise
