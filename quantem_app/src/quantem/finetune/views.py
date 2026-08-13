"""Guided fine-tuning endpoints, per the API contract.

Five routes: what the user has checked, what is already in flight, start a run,
read the result, apply it. No authentication anywhere — QuantEM is single-user
and loopback-only.

Every response that carries a Dice also carries its ``split_mode``, the names of
the crops the threshold was fit on, and the oracle ceiling. That is enforced
here rather than left to the frontend, because the honesty rules are about what
the number *means*, and the backend is where the number is made.

**Refusals happen at the door.** Both ways a run can be doomed before it starts
— threshold calibration with no stored probability map, head training on a
checked area too small to cut a training window from — are decided here and
returned as a 400 with the reason. The queue used to accept both and fail
minutes later with a message the user never saw, which is the same defect twice:
the app knew, and said nothing until it was too late to act on.
"""

from __future__ import annotations

from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.assets.models import Asset
from quantem.finetune.adapt import torch_available
from quantem.finetune.job import MODE_HEAD, MODE_THRESHOLD_ONLY, MODES
from quantem.finetune.models import Adapter
from quantem.finetune.preflight import check_head_size
from quantem.inference.specs import MODEL_SPECS
from quantem.jobs.constants import JOB_DEFAULTS, JOB_TYPE_TRAIN_ORGANELLE_ADAPTER
from quantem.jobs.models import Job
from quantem.segmentation.models import ImageSegmentation
from quantem.segmentation.services.adapt import collect_crops


def _error(message: str, code: int = status.HTTP_400_BAD_REQUEST) -> Response:
    return Response({"error": message}, status=code)


def serialize_adapter(adapter: Adapter) -> dict[str, object]:
    """The ``GET /api/adapters/<id>/`` body."""
    sweep = adapter.sweep or {}
    params = adapter.params or {}
    return {
        "id": str(adapter.id),
        "base_model": adapter.base_model,
        "name": adapter.name,
        "status": adapter.status,
        "mode": adapter.mode,
        "steps": params.get("steps", 0),
        "trainable_params": adapter.trainable_params,
        "segmentation_id": (str(adapter.segmentation_id) if adapter.segmentation_id else None),
        "split_mode": adapter.split_mode,
        "train_crop_names": sweep.get("train_crop_names", []),
        "heldout_crop_names": sweep.get("heldout_crop_names", []),
        "sweep": sweep,
        "calibrated_threshold": adapter.calibrated_threshold,
        "default_threshold": _default_threshold(adapter.base_model),
        "heldout_dice": adapter.heldout_dice,
        "verified_reload": adapter.verified_reload,
        "train_seconds": adapter.train_seconds,
        "applied_at": adapter.applied_at,
        "created_at": adapter.created_at,
        "error": adapter.error,
        "caveats": adapter.caveats(),
    }


def _default_threshold(base_model: str) -> float | None:
    """The pack's published cut-off, so a change can be stated as a change.

    Without it the panel can print the new include level but not "was 0.50",
    and "0.45" on its own tells a reader nothing about which direction the
    model moved.
    """
    spec = MODEL_SPECS.get(base_model)
    return float(spec.threshold) if spec is not None else None


def _image_names(crop_dicts: list[dict]) -> dict[str, str]:
    """Display name per asset id, for the crops in this response.

    The crop's own ``name`` is derived from the asset uuid (``4f3a91c2_0``) and
    is never shown to a reader: a raw identifier in prose names nothing the user
    can see. One query, keyed by the ids actually present.
    """
    ids = {str(crop.get("image_key")) for crop in crop_dicts if crop.get("image_key")}
    if not ids:
        return {}
    return {
        str(asset_id): display_name
        for asset_id, display_name in Asset.objects.filter(id__in=ids).values_list(
            "id", "display_name"
        )
    }


class AdaptCropsView(APIView):
    """``GET /api/segmentations/<seg_id>/adapt/crops/[?base_model=<pack>]``

    ``base_model`` is optional and only affects the head-training verdict: the
    window-size rule is a property of the pack, so it cannot be answered without
    knowing which pack. Omit it and ``mode_blockers.head`` reports only the
    reasons that hold for every pack.
    """

    def get(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )
        crop_set = collect_crops(segmentation)
        body = crop_set.as_api_dict()
        # Which rungs this machine can actually offer. threshold_only needs no
        # torch and no GPU, which is exactly why it is listed unconditionally.
        body["modes"] = [MODE_THRESHOLD_ONLY] + ([MODE_HEAD] if torch_available() else [])

        base_model = str(request.query_params.get("base_model") or "").strip()
        verdict = check_head_size(crop_set.crops, base_model) if base_model else None
        body["head_size"] = verdict.as_api_dict() if verdict else None
        if verdict is not None and not verdict.ok and verdict.reason:
            blockers = dict(body.get("mode_blockers") or {})
            blockers[MODE_HEAD] = [*blockers.get(MODE_HEAD, []), verdict.reason]
            body["mode_blockers"] = blockers

        crop_dicts = body.get("crops") or []
        names = _image_names(crop_dicts)
        this_image = str(segmentation.asset_id) if segmentation.asset_id else None
        for crop in crop_dicts:
            key = str(crop.get("image_key"))
            crop["image_name"] = names.get(key, "")
            crop["is_this_image"] = key == this_image
        return Response(body, status=status.HTTP_200_OK)


class AdaptLatestView(APIView):
    """``GET /api/segmentations/<seg_id>/adapt/latest/``

    The most recent run for this segmentation, whatever its state, plus the job
    row behind it if the queue still has one.

    This is how the panel reattaches after a reload. It used to be done with a
    ``localStorage`` pointer, which meant a run was invisible on a second
    machine, invisible after a cleared browser store, and — because the pointer
    was written once and never cleared on success — the reason a second run
    could not be started at all. The server already knows; ask it.
    """

    def get(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        adapter = Adapter.objects.filter(segmentation=segmentation).order_by("-created_at").first()
        if adapter is None:
            return Response({"adapter": None, "job_id": None}, status=status.HTTP_200_OK)
        # Matched on the payload, not on ``tags``: ``JSONField.__contains`` is
        # unsupported on SQLite, which is the shipped database.
        job = (
            Job.objects.filter(
                type=JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
                payload_json__adapter_id=str(adapter.id),
            )
            .order_by("-created_at")
            .first()
        )
        return Response(
            {
                "adapter": serialize_adapter(adapter),
                "job_id": str(job.id) if job is not None else None,
            },
            status=status.HTTP_200_OK,
        )


class AdaptStartView(APIView):
    """``POST /api/segmentations/<seg_id>/adapt/``"""

    def post(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )
        data = request.data if isinstance(request.data, dict) else {}

        base_model = str(data.get("base_model") or "").strip()
        if base_model not in MODEL_SPECS:
            return _error(
                f"Unknown model {base_model!r}. Choose one of: {', '.join(sorted(MODEL_SPECS))}."
            )

        mode = str(data.get("mode") or MODE_THRESHOLD_ONLY).strip().lower()
        if mode not in MODES:
            return _error(f"mode must be one of {', '.join(MODES)}.")
        if mode == MODE_HEAD and not torch_available():
            return _error(
                "Head training needs PyTorch, which is not installed here. "
                "Threshold calibration works without it."
            )

        crop_set = collect_crops(segmentation)
        if not crop_set.ready:
            return _error(crop_set.blockers[0])

        # The two refusals that used to be discovered by a dead job. Both are
        # decided from data already in hand, so neither costs a queue slot.
        mode_blockers = crop_set.mode_blockers().get(mode) or []
        if mode_blockers:
            return _error(mode_blockers[0])
        if mode == MODE_HEAD:
            verdict = check_head_size(crop_set.crops, base_model)
            if verdict is not None and not verdict.ok and verdict.reason:
                return _error(verdict.reason)

        params = {
            "steps": int(data.get("steps") or 300),
            "lr": float(data.get("lr") or 1e-4),
            "seed": int(data.get("seed") or 0),
        }
        # One button means calibrate-and-use, not calibrate-then-hunt-for-Apply.
        # It is still opt-in per request: a run started to *look* at the numbers
        # must not silently become the model.
        apply_and_rerun = bool(data.get("apply_and_rerun"))
        adapter = Adapter.objects.create(
            segmentation=segmentation,
            base_model=base_model,
            name=str(data.get("name") or "").strip(),
            mode=mode,
            params=params,
            split_mode=crop_set.split_mode,
        )

        defaults = JOB_DEFAULTS[JOB_TYPE_TRAIN_ORGANELLE_ADAPTER]
        job = Job.enqueue(
            job_type=JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
            payload={
                "segmentation_id": str(segmentation.id),
                "adapter_id": str(adapter.id),
                "base_model": base_model,
                "mode": mode,
                **params,
                "name": adapter.name,
                "apply_and_rerun": apply_and_rerun,
            },
            priority=defaults["priority"],
            resource_class=defaults["resource_class"],
            queue_name=defaults["queue_name"],
            tags=[
                f"segmentation:{segmentation.id}",
                f"adapter:{adapter.id}",
            ],
        )
        return Response(
            {"job_id": str(job.id), "adapter_id": str(adapter.id)},
            status=status.HTTP_202_ACCEPTED,
        )


class AdapterDetailView(APIView):
    """``GET /api/adapters/<adapter_id>/``"""

    def get(self, request, adapter_id):
        adapter = get_object_or_404(Adapter, id=adapter_id)
        return Response(serialize_adapter(adapter), status=status.HTTP_200_OK)


class AdapterApplyView(APIView):
    """``POST /api/adapters/<adapter_id>/apply/``

    Stamps the adapter as the one to use for subsequent runs on its
    segmentation. Refused before the run has succeeded: applying a pending
    adapter would mean running with a threshold that has not been fitted yet.

    Applying changes nothing that already exists — no object is written, moved
    or deleted here. The objects on screen keep the include level they were
    found at until the model is run again, and that is what the response's
    ``rerun_advice`` says so the panel does not have to guess.
    """

    def post(self, request, adapter_id):
        adapter = get_object_or_404(Adapter, id=adapter_id)
        if adapter.status != "SUCCESS":
            return _error(
                f"This adapter is {adapter.status.lower()}; only a finished "
                "adapter can be applied.",
                status.HTTP_409_CONFLICT,
            )
        adapter.applied_at = timezone.now()
        adapter.save(update_fields=["applied_at", "updated_at"])
        if adapter.segmentation_type_id and adapter.segmentation_id:
            adapter.applied_assets.add(adapter.segmentation.asset_id)
        body = serialize_adapter(adapter)
        body["rerun_advice"] = rerun_advice(adapter)
        return Response(body, status=status.HTTP_200_OK)


def rerun_advice(adapter: Adapter) -> dict[str, object]:
    """What running the model again would and would not do, in the app's words.

    Kept beside the apply endpoint because the two are one decision from the
    user's side: they press a button meaning "use this", and the honest answer
    is "used — and here is what has not happened yet".

    The preservation sentence is not a reassurance, it is a description of
    :func:`quantem.seg_core.db.extraction.extract_and_save_segments`, which
    deletes only its own generated candidates and then suppresses any new guess
    landing on a kept or removed object. Asserted by
    ``test_improve_preflight.py``.
    """
    new_level = adapter.calibrated_threshold
    old_level = _default_threshold(adapter.base_model)
    changes = (
        new_level is not None
        and old_level is not None
        and abs(float(new_level) - float(old_level)) >= 5e-3
    )
    return {
        "include_level": new_level,
        "previous_include_level": old_level,
        "changes_objects": bool(changes),
        "preserves_manual_work": True,
        "summary": (
            "The objects already on screen were found at the old include level. "
            "Running the model again finds them at the new one."
            if changes
            else "This is the include level your objects were already found at, "
            "so running the model again would find the same objects."
        ),
        "preservation": (
            "Nothing you have kept, removed or drawn by hand changes when the "
            "model runs again. Only my own guesses are replaced."
        ),
    }
