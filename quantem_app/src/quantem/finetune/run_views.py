"""The Fine-Tune dialog's endpoints: choose images, preview, run, watch, apply.

Seven routes, all under ``/api/finetune/``. They are the round-3 flow and they
sit beside — not instead of — the five older ``adapt/`` routes in
:mod:`quantem.finetune.views`, which the labeling view's Improve panel still
uses.

What is different here is the unit of work. The old flow fits a model to
whatever the library happens to hold for one organelle. This one fits a
**named** model to an explicit set of images inside one experiment, so a run can
be repeated, overwritten, and quoted: "the model I fitted to the fasted cohort"
is a thing with a name rather than the most recent row in a table.

Three rules the whole surface turns on:

* **One experiment.** A selection spanning two experiments is refused outright
  — a hard 400,
  not a warning (owner R13). Mixing conditions inside one fitted model is the
  kind of mistake that is invisible in the output.
* **Nothing is automatic.** A finished run changes nothing until the user asks
  for it. Applying is its own request, over a subset of the scope the user
  chooses.
* **Overwriting cannot lose the old model.** The new weights are written beside
  the live ones and only moved over them once the run has finished; a failed
  overwrite says so.

Errors are ``{"detail": "..."}`` in the app's user-facing voice: no identifiers
in prose, nothing a person cannot act on.
"""

from __future__ import annotations

import uuid

from django.db import transaction
from django.db.models import Prefetch
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.assets.models import Asset
from quantem.finetune.adapt import (
    MAX_ADAPT_STEPS,
    MIN_ADAPT_STEPS,
    AdaptConfig,
    torch_available,
)
from quantem.finetune.models import (
    HOLDOUT_CV_MIN_ANNOTATIONS,
    HOLDOUT_MIN_ANNOTATIONS,
    TRAINING_MODE_HOLDOUT_1,
    TRAINING_MODE_USE_ALL,
    TRAINING_MODES,
    Adapter,
)
from quantem.finetune.scope import (
    ResolvedScope,
    default_training_mode,
    plan_folds,
    plan_step_counts,
    resolve_scope,
    tiles_for_crop,
)
from quantem.finetune.views import serialize_adapter
from quantem.inference.specs import MODEL_SPECS
from quantem.jobs.constants import (
    ACTIVE_SEGMENTATION_JOB_TYPES,
    JOB_DEFAULTS,
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
)
from quantem.jobs.models import Job
from quantem.library.models import Dataset, Experiment
from quantem.segmentation.api_views.shared import active_segmentation_job
from quantem.segmentation.models import ImageSegmentation, SegmentationType
from quantem.segmentation.services.adapt import CropSet, collect_crops_for_scope
from quantem.segmentation.services.adapt.extract_crops import count_annotations
from quantem.segmentation.source_models import (
    default_source_model_for_organelle,
    source_models_for_organelle,
)

__all__ = [
    "FineTuneAdaptersView",
    "FineTunePreviewView",
    "FineTuneRunApplyView",
    "FineTuneRunDetailView",
    "FineTuneRunProgressView",
    "FineTuneRunsView",
    "FineTuneScopeView",
]

#: A completed round, or this share of the steps, before an ETA is offered. A
#: number guessed from a standing start is worse than no number: the first steps
#: include the model load and are not representative of the rest.
ETA_MIN_FRACTION = 0.10


class _ApplyConflict(Exception):
    """A selected image cannot safely accept another segmentation run."""


_EMPTY_COUNTS = {"confirmed_areas": 0, "done_rois": 0, "annotation_count": 0}


def _detail(message: str, code: int = status.HTTP_400_BAD_REQUEST) -> Response:
    return Response({"detail": message}, status=code)


def _segmentation_type(request) -> SegmentationType | None:
    value = str(request.query_params.get("segmentation_type") or "").strip()
    if not value:
        return None
    return SegmentationType.objects.filter(id=value).first()


def _base_model_for(segmentation_type: SegmentationType) -> str:
    """The pack a fine-tune for this organelle starts from, by default."""
    return default_source_model_for_organelle(segmentation_type.internal_name)


def _base_models_for(segmentation_type: SegmentationType) -> set[str]:
    """Released packs whose output semantics match this segmentation type."""
    return {
        definition.value
        for definition in source_models_for_organelle(segmentation_type.internal_name)
    }


def _runnable_reason(base_model: str) -> str | None:
    """Why this pack cannot be fitted here, or None when it can."""
    if base_model not in MODEL_SPECS:
        return "There is no released model for this organelle, so there is nothing to fine-tune."
    if not torch_available():
        return (
            "This copy of QuantEM was installed without PyTorch, so no model can "
            "be fine-tuned on it."
        )
    try:
        from quantem.registry.catalogue import probe_runnable  # noqa: PLC0415

        verdict = probe_runnable(base_model)
    except Exception:  # pragma: no cover - registry unavailable in a bare install
        return None
    return None if verdict.ok else verdict.reason


# ---------------------------------------------------------------------------
# 1. Scope selection
# ---------------------------------------------------------------------------


class FineTuneScopeView(APIView):
    """``GET /api/finetune/scope/?segmentation_type=<id>``

    Everything the dialog's tree needs, in one call. The library is a desktop
    library — hundreds of images — so one call is the right shape and there is
    no pagination to keep in sync with a selection.

    Every active image belongs to an experiment. Images not placed in one of
    that experiment's datasets appear in its ``ungrouped_images`` list.
    """

    def get(self, request):
        segmentation_type = _segmentation_type(request)
        if segmentation_type is None:
            return _detail("Choose which organelle to fine-tune for.")

        counts = count_annotations(str(segmentation_type.id))
        assets = list(
            Asset.objects.filter(lifecycle_status=Asset.LIFECYCLE_ACTIVE)
            .prefetch_related("datasets")
            .order_by("display_name", "created_at")
        )

        by_dataset: dict[str, list[dict]] = {}
        by_experiment_ungrouped: dict[str, list[dict]] = {}
        for asset in assets:
            row = self._image(asset, counts)
            dataset_ids = [str(dataset.id) for dataset in asset.datasets.all()]
            if dataset_ids:
                for dataset_id in dataset_ids:
                    by_dataset.setdefault(dataset_id, []).append(row)
            else:
                by_experiment_ungrouped.setdefault(str(asset.experiment_id), []).append(row)

        experiments = []
        for experiment in Experiment.objects.prefetch_related(
            Prefetch("datasets", queryset=Dataset.objects.order_by("name", "created_at"))
        ).order_by("name", "created_at"):
            datasets = []
            for dataset in experiment.datasets.all():
                images = by_dataset.get(str(dataset.id), [])
                datasets.append(
                    {
                        "id": str(dataset.id),
                        "name": dataset.name,
                        **self._rollup(images),
                        "images": images,
                    }
                )
            experiments.append(
                {
                    "id": str(experiment.id),
                    "name": experiment.name,
                    "datasets": datasets,
                    "ungrouped_images": by_experiment_ungrouped.get(str(experiment.id), []),
                }
            )

        return Response({"experiments": experiments}, status=status.HTTP_200_OK)

    @staticmethod
    def _image(asset: Asset, counts: dict[str, dict[str, int]]) -> dict:
        entry = counts.get(str(asset.id), _EMPTY_COUNTS)
        return {
            "id": str(asset.id),
            "name": asset.display_name,
            "confirmed_areas": entry["confirmed_areas"],
            "done_rois": entry["done_rois"],
            "annotation_count": entry["annotation_count"],
        }

    @staticmethod
    def _rollup(images: list[dict]) -> dict:
        return {
            "image_count": len(images),
            "annotated_image_count": sum(1 for image in images if image["annotation_count"]),
            "annotation_count": sum(image["annotation_count"] for image in images),
        }


# ---------------------------------------------------------------------------
# 2. Preview
# ---------------------------------------------------------------------------


def _preview(
    segmentation_type: SegmentationType, data: dict
) -> tuple[dict, ResolvedScope, CropSet | None]:
    """The numbers behind the dialog's summary line, and whether it can run.

    Computed from the real crop set rather than from a count of rows, so the
    number shown and the areas trained on are the same set: an area struck out
    for overlapping another is neither counted here nor trained on there.

    Returns the body **and** the two things it cost to work out. Starting a run
    needs both, and reading the library twice would be both slower and a chance
    for the answer the user was shown and the answer the run uses to differ.
    ``CropSet`` is None when the scope was refused before it was read.
    """
    scope = resolve_scope(asset_ids=data.get("asset_ids"), dataset_ids=data.get("dataset_ids"))
    base_model = str(data.get("base_model") or "").strip() or _base_model_for(segmentation_type)
    body: dict = {
        "experiment": (
            {"id": scope.experiment_id, "name": scope.experiment_name}
            if scope.experiment_id
            else None
        ),
        "base_model": base_model,
        "asset_count": len(scope.asset_ids),
        "annotation_count": 0,
        "confirmed_areas": 0,
        "done_rois": 0,
        "tile_count": 0,
        "per_image": [],
        "default_mode": default_training_mode(0),
        "eligible": False,
        "blockers": list(scope.blockers),
    }
    if base_model not in _base_models_for(segmentation_type):
        body["blockers"].append(
            "That model is for a different organelle. Choose a QuantEM or OmniEM "
            f"model for {segmentation_type.short_name}."
        )
    if scope.blockers:
        return body, scope, None
    if body["blockers"]:
        return body, scope, None

    crop_set = collect_crops_for_scope(str(segmentation_type.id), scope.asset_ids)
    tiles_by_image: dict[str, int] = {}
    tile_count = 0
    if base_model in MODEL_SPECS:
        for crop in crop_set.crops:
            tiles = tiles_for_crop(crop, base_model)
            tiles_by_image[crop.image_key] = tiles_by_image.get(crop.image_key, 0) + tiles
            tile_count += tiles

    names = {
        str(asset_id): display_name
        for asset_id, display_name in Asset.objects.filter(id__in=scope.asset_ids).values_list(
            "id", "display_name"
        )
    }
    per_image_counts = crop_set.per_image_counts()
    per_image = [
        {
            "asset_id": asset_id,
            "name": names.get(asset_id, ""),
            "confirmed_areas": entry["confirmed_areas"],
            "done_rois": entry["done_rois"],
            "tiles": tiles_by_image.get(asset_id, 0),
        }
        for asset_id, entry in sorted(per_image_counts.items(), key=lambda kv: names.get(kv[0], ""))
    ]

    blockers = list(crop_set.blockers)
    reason = _runnable_reason(base_model)
    if reason:
        blockers.append(reason)
    if not ImageSegmentation.objects.filter(
        segmentation_type=segmentation_type, asset_id__in=scope.asset_ids
    ).exists():
        blockers.append(
            "None of the images you chose has been set up for this organelle "
            "yet. Run the model on one of them first, or annotate it."
        )

    body.update(
        {
            "annotation_count": crop_set.annotation_count,
            "confirmed_areas": crop_set.confirmed_areas,
            "done_rois": crop_set.done_rois,
            "tile_count": tile_count,
            "per_image": per_image,
            "default_mode": (
                default_training_mode(tile_count)
                if crop_set.annotation_count >= HOLDOUT_MIN_ANNOTATIONS
                else TRAINING_MODE_USE_ALL
            ),
            "eligible": not blockers,
            "blockers": blockers,
        }
    )
    return body, scope, crop_set


class FineTunePreviewView(APIView):
    """``POST /api/finetune/preview/``"""

    def post(self, request):
        data = request.data if isinstance(request.data, dict) else {}
        segmentation_type = SegmentationType.objects.filter(
            id=str(data.get("segmentation_type") or "").strip() or None
        ).first()
        if segmentation_type is None:
            return _detail("Choose which organelle to fine-tune for.")
        body, _scope, _crops = _preview(segmentation_type, data)
        return Response(body, status=status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# 3. Start a run
# ---------------------------------------------------------------------------


class FineTuneRunsView(APIView):
    """``POST /api/finetune/runs/``

    A name and a scope in, a queued job out. The name is what makes a fine-tune
    a thing rather than an event: it is how the user finds it again, and it is
    how they say "replace that one" instead of accumulating a list of runs whose
    only distinguishing feature is a timestamp.
    """

    def post(self, request):
        data = request.data if isinstance(request.data, dict) else {}
        name = str(data.get("name") or "").strip()
        if not name:
            return _detail("Give this fine-tune a name so you can find it again.")

        segmentation_type = SegmentationType.objects.filter(
            id=str(data.get("segmentation_type") or "").strip() or None
        ).first()
        if segmentation_type is None:
            return _detail("Choose which organelle to fine-tune for.")

        base_model = str(data.get("base_model") or "").strip() or _base_model_for(segmentation_type)
        if base_model not in MODEL_SPECS:
            return _detail(
                "There is no released model for this organelle, so there is nothing to fine-tune."
            )
        if base_model not in _base_models_for(segmentation_type):
            return _detail(
                "That model is for a different organelle. Choose a QuantEM or OmniEM "
                f"model for {segmentation_type.short_name}."
            )

        training_mode = str(data.get("mode") or data.get("training_mode") or "").strip()
        if training_mode and training_mode not in TRAINING_MODES:
            return _detail("Choose whether to train on every annotated area or to hold one back.")
        cv_benchmark = bool(data.get("cv_benchmark"))

        preview, scope, crop_set = _preview(segmentation_type, data)
        if not preview["eligible"] or crop_set is None:
            return _detail(
                preview["blockers"][0]
                if preview["blockers"]
                else "This selection cannot be fine-tuned yet."
            )
        training_mode = training_mode or preview["default_mode"]
        annotation_count = crop_set.annotation_count
        if cv_benchmark and training_mode != TRAINING_MODE_HOLDOUT_1:
            return _detail("Cross-validation requires the hold-out-one training mode.")
        if cv_benchmark and annotation_count < HOLDOUT_CV_MIN_ANNOTATIONS:
            return _detail("Hold-out-one cross-validation requires at least 3 annotations.")
        if training_mode == TRAINING_MODE_HOLDOUT_1 and annotation_count < HOLDOUT_MIN_ANNOTATIONS:
            return _detail("Hold out one requires at least 2 annotations.")

        overwrite_id = str(data.get("overwrite_adapter_id") or "").strip()
        existing = (
            Adapter.objects.filter(name=name, segmentation_type=segmentation_type).first()
            if name
            else None
        )
        if existing is not None and str(existing.id) != overwrite_id:
            return _detail(
                f"A fine-tune for this organelle is already called “{name}”. "
                "Choose it from the list to replace it, or pick another name.",
                status.HTTP_409_CONFLICT,
            )

        adapter = None
        if overwrite_id:
            adapter = Adapter.objects.filter(id=overwrite_id).first()
            if adapter is None:
                return _detail(
                    "The fine-tune you asked to replace is no longer here. Start a new one instead."
                )
            if adapter.segmentation_type_id != segmentation_type.id:
                return _detail(
                    "That fine-tune was made for a different organelle, so it "
                    "cannot be replaced by this one."
                )

        # Planned before the row is written, because the queue asks for the
        # denominator at enqueue and a worker does not exist yet to be asked.
        # Over the crop set the preview already read, so the number the dialog
        # showed and the number the bar counts to describe the same data.
        folds, _split_mode = plan_folds(
            crop_set.crops, training_mode=training_mode, cv_benchmark=cv_benchmark
        )
        requested_steps = data.get("steps")
        try:
            fixed_steps = (
                int(requested_steps)
                if requested_steps is not None and requested_steps != ""
                else None
            )
            if fixed_steps is not None and not MIN_ADAPT_STEPS <= fixed_steps <= MAX_ADAPT_STEPS:
                raise ValueError(
                    f"Fine-tuning steps must be between {MIN_ADAPT_STEPS} and {MAX_ADAPT_STEPS}."
                )
            planned_steps = plan_step_counts(
                folds,
                base_model,
                fixed_steps=fixed_steps,
            )
        except ValueError as exc:
            return _detail(str(exc))
        params = {
            # ``steps`` remains the number that produced the shipped head (the
            # last fold under CV), preserving the adapter/provenance contract.
            "steps": planned_steps[-1],
            "steps_by_round": planned_steps,
            "total_steps": sum(planned_steps),
            "step_policy": "fixed" if fixed_steps is not None else "20_per_tile_clamped_300_600",
            "lr": float(data.get("lr") or AdaptConfig.lr),
            "seed": int(data.get("seed") or AdaptConfig.seed),
        }

        # Kept populated when the dialog was opened from a labeling view, so
        # ``active_adapter_for`` still finds this run through the old
        # single-segmentation path as well as through the scope. Ignored when it
        # names a segmentation outside the scope or for another organelle: that
        # is a stale client, not an instruction.
        launched_from = ImageSegmentation.objects.filter(
            id=str(data.get("segmentation_id") or "").strip() or None,
            segmentation_type=segmentation_type,
            asset_id__in=scope.asset_ids,
        ).first()

        with transaction.atomic():
            preserves_live_version = False
            if adapter is None:
                adapter = Adapter.objects.create(
                    base_model=base_model,
                    name=name,
                    mode="head",
                    params=params,
                    segmentation=launched_from,
                    segmentation_type=segmentation_type,
                    experiment_id=scope.experiment_id,
                    training_mode=training_mode,
                    cv_benchmark=cv_benchmark,
                )
            else:
                # Serialize replacement with Dataset apply. If apply committed
                # first, its batch must finish before the adapter can change;
                # if replacement committed first, apply's own locked recheck
                # sees PENDING and refuses to enqueue stale work.
                adapter = Adapter.objects.select_for_update().get(id=adapter.id)
                applying = Job.objects.filter(
                    batch_id__startswith=f"finetune-apply:{adapter.id}:",
                    status__in=Job.OPEN_STATUSES,
                ).first()
                if applying is not None:
                    return _detail(
                        "This fine-tune is still running on selected images "
                        f"({applying.id}). Wait for that batch or cancel it before replacing it.",
                        status.HTTP_409_CONFLICT,
                    )

                preserves_live_version = bool(adapter.applied_at) and (
                    adapter.status == "SUCCESS" or adapter.preserves_live_version
                )
                if preserves_live_version and not adapter.applied_assets.exists():
                    # Materialize the v0.1.1 scope fallback before changing the
                    # training scope. Otherwise a legacy applied adapter would
                    # lose its targets during an otherwise safe overwrite.
                    adapter.applied_assets.add(*adapter.scope_assets.all())

                # The row reports the replacement job's state. If it was live,
                # keep its old inference fields intact until the worker has
                # verified and promoted the staged head. A failed replacement
                # therefore continues routing the previous version.
                if not preserves_live_version:
                    adapter.base_model = base_model
                    adapter.sweep = {}
                    adapter.cv_results = {}
                    adapter.calibrated_threshold = None
                    adapter.verified_reload = False
                    adapter.applied_at = None
                    adapter.applied_assets.clear()
                adapter.name = name
                adapter.mode = "head"
                adapter.params = params
                adapter.segmentation = launched_from or adapter.segmentation
                adapter.experiment_id = scope.experiment_id
                adapter.training_mode = training_mode
                adapter.cv_benchmark = cv_benchmark
                adapter.status = "PENDING"
                adapter.error = ""
                adapter.preserves_live_version = preserves_live_version
                adapter.save()

            adapter.scope_assets.set(scope.asset_ids)
            adapter.scope_datasets.set(scope.dataset_ids)

            defaults = JOB_DEFAULTS[JOB_TYPE_TRAIN_ORGANELLE_ADAPTER]
            job = Job.enqueue(
                job_type=JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
                payload={
                    "adapter_id": str(adapter.id),
                    "segmentation_type_id": str(segmentation_type.id),
                    "asset_ids": list(scope.asset_ids),
                    "dataset_ids": list(scope.dataset_ids),
                    "base_model": base_model,
                    "mode": "head",
                    "training_mode": training_mode,
                    "cv_benchmark": cv_benchmark,
                    "planned_rounds": len(folds),
                    "planned_steps": planned_steps,
                    "fixed_steps": fixed_steps,
                    "overwrite": bool(overwrite_id),
                    "preserves_live_version": preserves_live_version,
                    "name": name,
                    "lr": params["lr"],
                    "seed": params["seed"],
                },
                priority=defaults["priority"],
                resource_class=defaults["resource_class"],
                queue_name=defaults["queue_name"],
                max_attempts=1,
                tags=[f"adapter:{adapter.id}"],
            )
        return Response(
            {"adapter_id": str(adapter.id), "job_id": str(job.id)},
            status=status.HTTP_202_ACCEPTED,
        )


# ---------------------------------------------------------------------------
# 4. Progress
# ---------------------------------------------------------------------------


def _job_for(adapter: Adapter):
    """The most recent queue row for this fine-tune, if the queue still has one.

    Matched on the payload rather than on ``tags``: ``JSONField.__contains`` is
    unsupported on SQLite, which is the shipped database.
    """
    return (
        Job.objects.filter(
            type=JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
            payload_json__adapter_id=str(adapter.id),
        )
        .order_by("-created_at")
        .first()
    )


def _progress_body(adapter: Adapter, job) -> dict:
    """One answer for the bar and the text, so the two cannot disagree.

    ``percent`` is computed here from the step counts rather than taken from the
    job's own coarse percentage, and the ETA is derived from how long the steps
    that have actually happened took. Both are withheld rather than guessed:
    ``eta_seconds`` stays null until a round or a tenth of the steps has gone
    by, because the first steps of a run include loading the model and predict
    nothing about the rest of it.

    ``step / total_steps`` **is** the contract's round-weighted formula, not an
    approximation of it. ``total_steps`` is the grand total across every round
    and ``step`` counts through all of them, so
    ``step / (steps_per_round x rounds)`` expands to
    ``(completed_rounds + step_in_round / steps_per_round) / rounds`` -- the same
    number, from one monotone counter instead of two that can disagree.
    """
    detail = (job.progress_detail_json or {}) if job is not None else {}
    total_steps = int((job.progress_units_total if job else 0) or 0)
    step = int((job.progress_units_done if job else 0) or 0)
    total_rounds = int(detail.get("total_rounds") or 1)
    current_round = int(detail.get("round") or (1 if adapter.status == "RUNNING" else 0))

    percent = 0.0
    if adapter.status == "SUCCESS":
        percent = 100.0
    elif total_steps > 0:
        percent = round(100.0 * min(step, total_steps) / total_steps, 1)

    eta = None
    if (
        job is not None
        and adapter.status == "RUNNING"
        and total_steps > 0
        and step > 0
        and step < total_steps
        and job.started_at is not None
    ):
        done_enough = current_round > 1 or (step / total_steps) >= ETA_MIN_FRACTION
        if done_enough:
            elapsed = (timezone.now() - job.started_at).total_seconds()
            if elapsed > 0:
                eta = int(round(elapsed * (total_steps - step) / step))

    return {
        "status": adapter.status,
        "stage": (job.progress_stage if job is not None else "") or "",
        "step": step,
        "total_steps": total_steps,
        "round": current_round,
        "total_rounds": total_rounds,
        "percent": percent,
        "eta_seconds": eta,
        "message": (job.message if job is not None else "") or "",
        "error": adapter.error or "",
    }


class FineTuneRunProgressView(APIView):
    """``GET /api/finetune/runs/<adapter_id>/progress/``"""

    def get(self, request, adapter_id):
        adapter = get_object_or_404(Adapter, id=adapter_id)
        return Response(_progress_body(adapter, _job_for(adapter)), status=status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# 5. Result
# ---------------------------------------------------------------------------


def serialize_run(adapter: Adapter) -> dict:
    """The adapter, plus everything the fine-tune flow adds to it."""
    body = serialize_adapter(adapter)
    scope_assets = list(adapter.scope_assets.all())
    body.update(
        {
            "segmentation_type": (
                str(adapter.segmentation_type_id) if adapter.segmentation_type_id else None
            ),
            "experiment": (
                {
                    "id": str(adapter.experiment_id),
                    "name": adapter.experiment.name,
                }
                if adapter.experiment_id
                else None
            ),
            "training_mode": adapter.training_mode,
            "cv_benchmark": adapter.cv_benchmark,
            "cv_results": adapter.cv_results or {},
            "asset_count": len(scope_assets),
            "asset_ids": [str(asset.id) for asset in scope_assets],
            "dataset_ids": [str(dataset.id) for dataset in adapter.scope_datasets.all()],
        }
    )
    return body


class FineTuneRunDetailView(APIView):
    """``GET /api/finetune/runs/<adapter_id>/``"""

    def get(self, request, adapter_id):
        adapter = get_object_or_404(Adapter.objects.select_related("experiment"), id=adapter_id)
        return Response(serialize_run(adapter), status=status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# 6. Apply
# ---------------------------------------------------------------------------


class FineTuneRunApplyView(APIView):
    """``POST /api/finetune/runs/<adapter_id>/apply/``

    Never automatic (owner R13). A finished fine-tune sits there until the user
    says to run it, and then only over the images or datasets they pick from its
    experiment. Applying stamps the fine-tune as the one to use for those
    targets and queues one run per image; nothing already on screen is touched
    until that run finishes, and what it does to annotated areas is the
    preservation invariant's business.
    """

    def get(self, request, adapter_id):
        adapter = get_object_or_404(Adapter, id=adapter_id)
        prefix = f"finetune-apply:{adapter.id}:"
        batch_id = str(request.query_params.get("batch_id") or "").strip()
        if batch_id and not batch_id.startswith(prefix):
            return _detail("That apply batch does not belong to this fine-tune.")
        if not batch_id:
            batch_id = (
                Job.objects.filter(batch_id__startswith=prefix)
                .order_by("-created_at")
                .values_list("batch_id", flat=True)
                .first()
                or ""
            )
        jobs = list(Job.objects.filter(batch_id=batch_id).order_by("batch_seq")) if batch_id else []
        job_asset_ids = [
            str(job.payload_json.get("asset_id"))
            for job in jobs
            if isinstance(job.payload_json, dict) and job.payload_json.get("asset_id")
        ]
        asset_names = {
            str(asset_id): name
            for asset_id, name in Asset.objects.filter(id__in=job_asset_ids).values_list(
                "id", "display_name"
            )
        }
        items = []
        for job in jobs:
            payload = job.payload_json if isinstance(job.payload_json, dict) else {}
            legs = payload.get("legs") if isinstance(payload.get("legs"), list) else []
            leg = legs[0] if legs and isinstance(legs[0], dict) else {}
            total = job.progress_units_total
            done = job.progress_units_done
            items.append(
                {
                    "asset_id": str(payload.get("asset_id") or ""),
                    "asset_name": asset_names.get(str(payload.get("asset_id") or ""), ""),
                    "segmentation_id": str(leg.get("segmentation_id") or ""),
                    "job_id": str(job.id),
                    "status": job.status,
                    "stage": job.progress_stage or "",
                    "progress": float(job.progress),
                    "units_done": int(done) if done is not None else None,
                    "units_total": int(total) if total is not None else None,
                    "message": job.message or "",
                    "failure": job.message if job.status == "FAILED" else "",
                    "adapter_id": str(adapter.id),
                    "result": job.result_json if isinstance(job.result_json, dict) else None,
                }
            )
        terminal = {"SUCCESS", "FAILED", "CANCELLED"}
        return Response(
            {
                "batch_id": batch_id or None,
                "adapter_id": str(adapter.id),
                "total": len(items),
                "complete": sum(1 for item in items if item["status"] in terminal),
                "succeeded": sum(1 for item in items if item["status"] == "SUCCESS"),
                "failed": sum(1 for item in items if item["status"] == "FAILED"),
                "images": items,
            },
            status=status.HTTP_200_OK,
        )

    def post(self, request, adapter_id):
        adapter = get_object_or_404(Adapter, id=adapter_id)
        if adapter.status != "SUCCESS":
            return _detail(
                "This fine-tune has not finished yet, so there is nothing to run.",
                status.HTTP_409_CONFLICT,
            )
        if adapter.segmentation_type_id is None:
            return _detail(
                "This fine-tune predates named runs, so it can only be applied "
                "from the image it was made on."
            )

        data = request.data if isinstance(request.data, dict) else {}
        raw_asset_ids = data.get("asset_ids", [])
        raw_dataset_ids = data.get("dataset_ids", [])
        if not isinstance(raw_asset_ids, list) or not isinstance(raw_dataset_ids, list):
            return _detail("Images and datasets must be sent as lists.")

        def validated_ids(values) -> list[str] | None:
            result = []
            for value in values:
                try:
                    parsed = str(uuid.UUID(str(value)))
                except (TypeError, ValueError, AttributeError):
                    return None
                if parsed not in result:
                    result.append(parsed)
            return result

        requested_asset_ids = validated_ids(raw_asset_ids)
        requested_dataset_ids = validated_ids(raw_dataset_ids)
        if requested_asset_ids is None or requested_dataset_ids is None:
            return _detail("Every selected image and dataset must have a valid identifier.")
        if not requested_asset_ids and not requested_dataset_ids:
            return _detail("Choose at least one dataset or image to run this on.")

        datasets = list(
            Dataset.objects.filter(
                id__in=requested_dataset_ids,
                experiment_id=adapter.experiment_id,
            )
        )
        if len({str(dataset.id) for dataset in datasets}) != len(set(requested_dataset_ids)):
            return _detail("Every selected dataset must belong to this fine-tune's experiment.")
        assets = Asset.objects.filter(lifecycle_status=Asset.LIFECYCLE_ACTIVE)
        if adapter.experiment_id is None:
            experiment_assets = assets.filter(experiment_id__isnull=True)
        else:
            experiment_assets = assets.filter(experiment_id=adapter.experiment_id)
        selected_ids = {
            str(value)
            for value in experiment_assets.filter(datasets__in=datasets)
            .values_list("id", flat=True)
            .distinct()
        }
        if requested_asset_ids:
            explicit = experiment_assets.filter(id__in=requested_asset_ids)
            explicit_ids = {str(value) for value in explicit.values_list("id", flat=True)}
            if len(explicit_ids) != len(set(requested_asset_ids)):
                return _detail("Every selected image must belong to this fine-tune's experiment.")
            selected_ids.update(explicit_ids)

        if not selected_ids:
            return _detail("None of the selected datasets contains an active image.")

        batch_id = f"finetune-apply:{adapter.id}:{uuid.uuid4().hex[:12]}"
        queued = []
        try:
            with transaction.atomic():
                adapter = Adapter.objects.select_for_update().get(id=adapter.id)
                if adapter.status != "SUCCESS":
                    raise _ApplyConflict("This fine-tune is no longer ready to run.")

                targets = []
                for asset in Asset.objects.filter(id__in=selected_ids).order_by(
                    "display_name", "id"
                ):
                    segmentation, _created = ImageSegmentation.objects.get_or_create(
                        asset=asset, segmentation_type_id=adapter.segmentation_type_id
                    )
                    if segmentation.status_stage == "COMPLETED":
                        raise _ApplyConflict(
                            f"{asset.display_name} is marked complete. Unlock it before "
                            "running this fine-tune."
                        )
                    blocking = active_segmentation_job(
                        segmentation,
                        job_types=ACTIVE_SEGMENTATION_JOB_TYPES,
                    )
                    if blocking is not None:
                        raise _ApplyConflict(
                            f"{asset.display_name} already has a segmentation task queued "
                            f"or running ({blocking.id}). Wait for it or cancel it first."
                        )
                    targets.append((asset, segmentation))

                adapter.applied_at = timezone.now()
                adapter.save(update_fields=["applied_at", "updated_at"])
                adapter.applied_assets.add(*selected_ids)

                for asset, segmentation in targets:
                    job = self._queue_run(asset, segmentation, adapter, batch_id=batch_id)
                    queued.append(
                        {
                            "asset_id": str(asset.id),
                            "segmentation_id": str(segmentation.id),
                            "job_id": str(job.id),
                        }
                    )
        except _ApplyConflict as exc:
            return _detail(str(exc), status.HTTP_409_CONFLICT)
        return Response(
            {
                "batch_id": batch_id,
                "adapter_id": str(adapter.id),
                "dataset_ids": requested_dataset_ids,
                "queued": queued,
            },
            status=status.HTTP_202_ACCEPTED,
        )

    @staticmethod
    def _queue_run(asset, segmentation, adapter, *, batch_id: str):
        from quantem.jobs.constants import (  # noqa: PLC0415 -- local to this path
            JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
            QUEUE_P4_FULL,
        )

        return Job.enqueue(
            job_type=JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
            payload={
                "asset_id": str(asset.id),
                "scope": "full",
                "legs": [
                    {
                        "segmentation_id": str(segmentation.id),
                        "segmentation_type": (segmentation.segmentation_type.internal_name),
                        "source_model": adapter.base_model,
                        "adapter_id": str(adapter.id),
                    }
                ],
            },
            priority="default",
            resource_class="gpu",
            queue_name=QUEUE_P4_FULL,
            max_attempts=1,
            tags=[
                f"asset:{asset.id}",
                f"segmentation:{segmentation.id}",
                f"adapter:{adapter.id}",
            ],
            batch_id=batch_id,
        )


# ---------------------------------------------------------------------------
# 7. Existing fine-tunes
# ---------------------------------------------------------------------------


class FineTuneAdaptersView(APIView):
    """``GET /api/finetune/adapters/?segmentation_type=<id>``

    What the overwrite dropdown lists. Filtered to one organelle because a name
    is only unique within one, and a list that mixed them would offer the user a
    mitochondria model to overwrite with an ER one.
    """

    def get(self, request):
        adapters = (
            Adapter.objects.select_related("experiment")
            .prefetch_related("scope_assets")
            .exclude(name="")
        )
        segmentation_type = _segmentation_type(request)
        if segmentation_type is not None:
            adapters = adapters.filter(segmentation_type=segmentation_type)
        else:
            adapters = adapters.filter(segmentation_type__isnull=False)
        rows = []
        for adapter in adapters.order_by("-created_at"):
            scope_assets = list(adapter.scope_assets.all())
            rows.append(
                {
                    "id": str(adapter.id),
                    "name": adapter.name,
                    "base_model": adapter.base_model,
                    "status": adapter.status,
                    "created_at": adapter.created_at,
                    "experiment": (
                        {
                            "id": str(adapter.experiment_id),
                            "name": adapter.experiment.name,
                        }
                        if adapter.experiment_id
                        else None
                    ),
                    "asset_count": len(scope_assets),
                    "asset_ids": [str(asset.id) for asset in scope_assets],
                }
            )
        return Response(rows, status=status.HTTP_200_OK)
