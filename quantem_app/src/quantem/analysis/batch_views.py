"""Read, retry and export complete or partially completed measurement batches."""

import csv
import io
import json
import shutil
import tempfile
import zipfile

from django.db import transaction
from django.http import FileResponse
from django.shortcuts import get_object_or_404
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.analysis.batches import preview_batch, start_batch
from quantem.analysis.creation import enqueue_run
from quantem.analysis.export_files import open_export
from quantem.analysis.identity import input_fingerprint, run_is_current
from quantem.analysis.models import AnalysisBatch, AnalysisBatchItem, AnalysisRun
from quantem.analysis.notes import analysis_notes
from quantem.analysis.rollup import aggregate
from quantem.analysis.service import export_dir_for_run, image_summary_row
from quantem.jobs.models import Job
from quantem.library.models import Experiment
from quantem.library.sample_identity import sample_identity


def item_body(item):
    run = item.run
    status = run.status if run else "SKIPPED"
    if item.job and item.job.status in ("FAILED", "CANCELLED"):
        status = item.job.status
    body = {
        "id": str(item.id),
        "identity": {
            **item.identity,
            **sample_identity(
                item.identity.get("image"),
                item.identity.get("asset_id"),
                item.identity.get("sample_id"),
                explicit=item.identity.get("sample_is_explicit"),
            ),
        },
        "status": status,
        "reason": item.skip_reason or (run.error if run else ""),
        "run_id": str(run.id) if run else None,
        "job_id": str(item.job_id) if item.job_id else None,
        "reused": item.reused,
        "params": item.params,
        "current": run_is_current(run) if run and status == "SUCCESS" else None,
        "results": run.results if run and status == "SUCCESS" else None,
        "notes": analysis_notes(run.results) if run and status == "SUCCESS" else [],
    }
    body["metrics"] = numeric_metrics(body)
    return body


def numeric_metrics(item):
    if not item.get("results"):
        return {}
    rows = measurement_rows({"items": [item]})
    return {
        key: value
        for key, value in rows[0].items()
        if (
            key.startswith(("mean_", "median_", "area_fraction_", "foreground_"))
            or key in ("n_objects", "objects_per_um2")
        )
        and (value is None or isinstance(value, (int, float)))
    }


def batch_body(batch):
    items = [
        item_body(item)
        for item in batch.items.select_related(
            "run", "run__segmentation", "run__segmentation__asset", "job"
        ).order_by("created_at", "id")
    ]
    return {
        "id": str(batch.id),
        "name": batch.name,
        "experiment": batch.experiment_name,
        "created_at": batch.created_at,
        "settings": batch.settings,
        "items": items,
    }


class ExperimentBatchesView(APIView):
    def get(self, request, experiment_id):
        rows = AnalysisBatch.objects.filter(experiment_id=experiment_id).order_by("-created_at")
        return Response(
            [
                {
                    "id": str(b.id),
                    "name": b.name,
                    "created_at": b.created_at,
                    "count": b.items.count(),
                }
                for b in rows
            ]
        )

    def post(self, request, experiment_id):
        experiment = get_object_or_404(Experiment, id=experiment_id)
        if request.data.get("preview"):
            rows = preview_batch(experiment, request.data)
            return Response(
                {
                    "items": [
                        {
                            "identity": row["identity"],
                            "reason": row["reason"],
                            "ready": not row["reason"],
                        }
                        for row in rows
                    ]
                }
            )
        return Response(batch_body(start_batch(experiment, request.data)), status=202)


class BatchDetailView(APIView):
    def get(self, request, batch_id):
        return Response(batch_body(get_object_or_404(AnalysisBatch, id=batch_id)))

    def post(self, request, batch_id):
        batch = get_object_or_404(AnalysisBatch, id=batch_id)
        action = request.data.get("action")
        if action == "regenerate":
            if not batch.experiment_id:
                raise ValidationError("This experiment is no longer available.")
            return Response(
                batch_body(start_batch(batch.experiment, {**batch.settings, "reuse": True})),
                status=201,
            )
        if action not in ("retry_failed", "cancel"):
            raise ValidationError("Choose retry_failed or cancel.")
        with transaction.atomic():
            # PostgreSQL cannot lock the nullable side of an outer join. Load
            # related state after locking the items, so a waiting retry also
            # sees the replacement run committed by the preceding request.
            items = (
                batch.items.select_for_update()
                .order_by("id")
                .prefetch_related("run", "job", "segmentation__asset")
            )
            for item in items:
                if action == "cancel":
                    if item.job and item.job.status in Job.OPEN_STATUSES:
                        from django.utils import timezone

                        from quantem.jobs.failure_reconcile import (
                            reconcile_domain_objects_for_cancelled_job,
                        )

                        cancelled = Job.objects.filter(
                            id=item.job_id, status__in=("PENDING", "RETRY")
                        ).update(
                            status="CANCELLED",
                            cancel_requested=True,
                            finished_at=timezone.now(),
                            message="Cancelled before measurement started.",
                        )
                        if cancelled:
                            reconcile_domain_objects_for_cancelled_job(
                                item.job.type, item.job.payload_json
                            )
                        else:
                            Job.objects.filter(id=item.job_id, status="RUNNING").update(
                                cancel_requested=True
                            )
                    continue
                if (
                    not item.run
                    or item.run.status not in ("FAILED",)
                    and (not item.job or item.job.status not in ("FAILED", "CANCELLED"))
                ):
                    continue
                if item.job and item.job.status in Job.OPEN_STATUSES:
                    continue
                if not item.segmentation:
                    continue
                if input_fingerprint(item.segmentation, item.params) != item.fingerprint:
                    item.run.error = "Inputs changed. Create a new batch with the current inputs."
                    item.run.save(update_fields=["error"])
                    continue
                item.run = AnalysisRun.objects.create(
                    segmentation=item.segmentation,
                    params=item.params,
                    input_fingerprint=item.fingerprint,
                    input_identity=item.identity,
                    group=item.identity.get("group", ""),
                )
                item.job = enqueue_run(item.run, batch_id=f"analysis-batch:{batch.id}")
                item.reused = False
                item.save()
        return Response(batch_body(batch))


def measurement_rows(body):
    rows = []
    for item in body["items"]:
        identity = item["identity"]
        base = {
            **identity,
            "datasets": "; ".join(identity.get("datasets", [])),
            "provenance": json.dumps(identity.get("provenance", {}), sort_keys=True),
            "status": item["status"],
            "reason": item["reason"],
            "run_id": item["run_id"],
            "current": item["current"],
        }
        result = item["results"]
        if not result:
            rows.append(base)
            continue
        if result.get("global_area"):
            report = result["global_area"]
            selected = (
                report["analysis_masks"]
                if identity["measurement_area"] != "Whole image"
                else [report["whole_image"]]
            )
            for region in selected:
                rows.append(
                    {
                        **base,
                        "measurement_area": region["name"],
                        "foreground_pixels": region["foreground_pixels"],
                        "denominator_pixels": region["denominator_pixels"],
                        "foreground_percent": region["foreground_percent"],
                    }
                )
        else:
            values = image_summary_row(result)
            for metric, stats in (result.get("objects", {}).get("summary") or {}).items():
                for statistic in ("mean", "median", "sd", "n", "n_missing"):
                    values[f"{statistic}_{metric}"] = stats.get(statistic)
            rows.append({**values, **base})
    return rows


def sample_summary(rows):
    """Include each image, combining fields only when a shared sample ID is assigned."""
    cells = {}
    for index, row in enumerate(rows):
        if row.get("status") != "SUCCESS" or not row.get("current"):
            continue
        sample_info = sample_identity(
            row.get("image"),
            row.get("asset_id", index),
            row.get("sample_id"),
            explicit=row.get("sample_is_explicit"),
        )
        # Identical filenames must not accidentally combine independent images,
        # or collide with a user-assigned sample ID of the same name.
        asset_id = "" if sample_info["sample_is_explicit"] else row.get("asset_id", str(index))
        key = (
            row.get("group", ""),
            sample_info["sample_id"],
            asset_id,
            row["organelle"],
            row["measurement_area"],
        )
        for metric, value in row.items():
            if (
                (
                    metric.startswith(("area_fraction_", "foreground_percent", "mean_", "median_"))
                    or metric in ("objects_per_um2", "distance_median_nm", "n_objects")
                )
                and isinstance(value, (float, int))
                and not isinstance(value, bool)
            ):
                cells.setdefault((*key, metric), []).append(value)
    groups = {}
    samples = []
    for (group, sample, asset_id, organelle, area, metric), values in cells.items():
        mean = aggregate(values).mean
        samples.append(
            {
                "group": group,
                "sample_id": sample,
                "asset_id": asset_id,
                "organelle": organelle,
                "measurement_area": area,
                "metric": metric,
                "n_images": len(values),
                "mean": mean,
            }
        )
        groups.setdefault((group, organelle, area, metric), []).append(mean)
    summaries = [
        {
            "group": group,
            "organelle": organelle,
            "measurement_area": area,
            "metric": metric,
            **aggregate(values).as_dict(),
        }
        for (group, organelle, area, metric), values in groups.items()
    ]
    return samples, summaries


def csv_text(rows):
    from quantem.analysis.columns import column_title

    stream = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if fields:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writerow({key: column_title(key) for key in fields})
        writer.writerows(rows)
    return stream.getvalue()


class BatchExportView(APIView):
    def get(self, request, batch_id):
        body = batch_body(get_object_or_404(AnalysisBatch, id=batch_id))
        rows = measurement_rows(body)
        samples, groups = sample_summary(rows)
        directory = export_dir_for_run(batch_id)
        directory.mkdir(parents=True, exist_ok=True)
        stream = tempfile.TemporaryFile(dir=directory)  # noqa: SIM115 -- FileResponse owns and closes it.
        try:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                selection = body["settings"].get("export_tables", {})
                for filename, contents in (
                    ("measurements.csv", rows),
                    ("samples.csv", samples),
                    ("groups.csv", groups),
                ):
                    if selection.get(filename, True):
                        bundle.writestr(filename, csv_text(contents))
                manifest = {
                    **body,
                    "aggregation": (
                        "Each image is a sample by default, labelled with its image name. "
                        "Images with the same explicitly assigned sample ID are averaged first, "
                        "then samples are averaged with equal weight within each group. "
                        "Only successful, current measurements contribute; "
                        "missing sample IDs never exclude an image."
                    ),
                    "missing_values": "Blank values are unavailable; skipped images are not zero.",
                    "items": [
                        {key: val for key, val in item.items() if key != "results"}
                        for item in body["items"]
                    ],
                }
                bundle.writestr("manifest.json", json.dumps(manifest, indent=2, default=str))
                for item in AnalysisBatchItem.objects.filter(batch_id=batch_id).select_related(
                    "run"
                ):
                    if item.run and item.run.status == "SUCCESS" and item.run.export_path:
                        for name in (
                            "objects.csv",
                            "composition.csv",
                            "area.csv",
                            "image_summary.csv",
                            "region_summary.csv",
                            "cell_summary.csv",
                            "organelle_objects.csv",
                            "manifest.json",
                        ):
                            if (
                                name in {"objects.csv", "organelle_objects.csv"}
                                and request.query_params.get("objects", "true") != "true"
                            ):
                                continue
                            path = item.run.export_path / name
                            if path.is_file():
                                with (
                                    open_export(path, result=item.run.results) as source,
                                    bundle.open(f"runs/{item.run.id}/{name}", "w") as target,
                                ):
                                    shutil.copyfileobj(source, target)
            stream.seek(0)
            return FileResponse(stream, as_attachment=True, filename=f"measurements-{batch_id}.zip")
        except Exception:
            stream.close()
            raise
