"""Per-image run routes: cost a multi-organelle run, then start it as one job.

``GET  /api/assets/<id>/runs/?organelles=mito,nucleus`` costs the run and
changes nothing.
``POST /api/assets/<id>/runs/`` costs it again, creates whatever segmentations
are missing, and enqueues **one** job for all of them.

Why the plan is a separate, side-effect-free answer
---------------------------------------------------
The workspace has to price the run *before* the user commits to it -- how long
it will take, and how many megabytes have to come down first -- and the price
must not be an adjective. ``engine.estimate_tiles`` is exact and needs no
weights on disk, so an organelle whose model has not been downloaded yet can
still be costed to the tile. That is what makes "tick it anyway, it will
download in the background" an honest offer rather than a guess.

Why one job and not one per organelle
-------------------------------------
See :func:`quantem.segmentation.organelle_tasks.run_segmentation_for_image_task`.
Four rows over one image paid four cold model loads and gave the whole-image bar
four moving parts; one row decodes the image once and carries one denominator.
"""

from __future__ import annotations

from django.urls import path
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.assets.asset_openable import get_asset_openable
from quantem.assets.asset_resolver import get_active_asset
from quantem.inference import engine
from quantem.inference.specs import MODEL_SPECS
from quantem.jobs.constants import (
    JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
    QUEUE_P4_FULL,
)
from quantem.jobs.models import Job
from quantem.registry import cache as registry_cache
from quantem.registry.catalogue import (
    deduped_download_bytes,
    download_bytes,
    pack_title,
    probe_runnable,
)
from quantem.segmentation.api_views.shared import (
    active_segmentation_job,
    blocking_job_response_payload,
    completion_lock_response,
)
from quantem.segmentation.models import ImageSegmentation, SegmentationConfig
from quantem.segmentation.source_models import (
    default_source_model_for_organelle,
    normalize_source_model,
    resolve_create_segmentation_request,
)
from quantem.segmentation.type_definitions import ORGANELLE_SEGMENTATION_TYPES
from quantem.segmentation.type_service import ensure_segmentation_type

#: ``"mito"`` -> the built-in segmentation type it names.
#:
#: The short id is the one the model packs use (``quantem:mito``) and the one
#: the workspace's checklist sends; the segmentation type's own key is the
#: longer ``quantem_internal_mito``. Derived rather than retyped so the two can
#: never drift apart. The tissue mask is deliberately absent: it is painted by
#: hand and no model produces it.
ORGANELLE_TYPES = {
    definition.internal_name.rsplit("_", 1)[-1]: definition
    for definition in ORGANELLE_SEGMENTATION_TYPES
}

#: The organelles this endpoint will run, in the order the workspace lists them.
RUNNABLE_ORGANELLES: tuple[str, ...] = tuple(ORGANELLE_TYPES)


def _asset_shape(asset) -> tuple[int, int] | None:
    """``(height, width)`` inference will tile over, read as inference reads it."""
    openable = get_asset_openable(asset, require=False)
    if openable is None:
        return None
    height, width = int(openable.height), int(openable.width)
    if height <= 0 or width <= 0:
        return None
    return (height, width)


def _asset_pixel_size_nm(asset) -> float | None:
    try:
        value = float(getattr(asset, "pixel_size_nm", None) or 0.0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _requested_organelles(raw) -> list[str]:
    """The organelle ids the caller asked for, de-duplicated, in their order."""
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        items = [str(part).strip() for part in raw]
    else:
        items = []
    return list(dict.fromkeys(item for item in items if item))


def _plan_entry(
    asset,
    organelle: str,
    source_model: str | None,
    *,
    shape: tuple[int, int] | None,
    pixel_size_nm: float | None,
    segmentation: ImageSegmentation | None,
) -> dict:
    """What one ticked organelle will cost, and whether it can start yet."""
    definition = ORGANELLE_TYPES.get(organelle)
    name = definition.long_name if definition is not None else organelle
    pack_id = normalize_source_model(source_model) or (
        default_source_model_for_organelle(definition.internal_name)
        if definition is not None
        else ""
    )
    spec = MODEL_SPECS.get(pack_id) if pack_id else None

    tiles = None
    if spec is not None and shape is not None:
        # Exact, and computed with no weights on disk: this is the number the
        # tiling loop will count to, for a pack that has not been downloaded yet
        # just as much as for one that has.
        tiles = int(engine.estimate_tiles(spec, shape, pixel_size_nm=pixel_size_nm))

    installed = bool(pack_id) and registry_cache.installed(pack_id)
    runnable = probe_runnable(pack_id, installed=installed) if pack_id else None
    return {
        "organelle": organelle,
        "name": name,
        "pack_id": pack_id,
        "title": pack_title(spec) if spec is not None else (pack_id or organelle),
        "tiles": tiles,
        "model_installed": installed,
        "model_ready": bool(runnable.ok) if runnable is not None else False,
        "model_blocked_reason": (None if runnable is None or runnable.ok else runnable.reason),
        # 0 once it is on disk: a figure that keeps quoting the download size of
        # something already downloaded is how a cost screen loses its meaning.
        "download_bytes": 0 if installed or spec is None else download_bytes(spec),
        "segmentation_id": str(segmentation.id) if segmentation is not None else None,
    }


def _build_plan(asset, selected: list[str], source_models: dict) -> dict:
    """Every organelle this image can be run for, and what the ticked ones cost.

    Both halves in one answer on purpose. The checklist has to list every
    organelle -- including the ones the user has not ticked, or they could never
    tick them -- while the totals underneath have to be about the ticked ones
    only, and the deduped download figure cannot be computed client-side because
    only the server knows which packs share an encoder blob. Two endpoints for
    that would be two round trips per tick.
    """
    organelles = list(RUNNABLE_ORGANELLES)
    shape = _asset_shape(asset)
    pixel_size_nm = _asset_pixel_size_nm(asset)
    by_internal_name = {
        seg.segmentation_type.internal_name: seg
        for seg in ImageSegmentation.objects.filter(asset=asset).select_related("segmentation_type")
    }
    entries = []
    for organelle in organelles:
        definition = ORGANELLE_TYPES.get(organelle)
        entries.append(
            _plan_entry(
                asset,
                organelle,
                source_models.get(organelle),
                shape=shape,
                pixel_size_nm=pixel_size_nm,
                segmentation=(
                    by_internal_name.get(definition.internal_name)
                    if definition is not None
                    else None
                ),
            )
        )
    chosen = [entry for entry in entries if entry["organelle"] in set(selected)]
    tiles = [entry["tiles"] for entry in chosen]
    to_download = [
        entry["pack_id"] for entry in chosen if entry["pack_id"] and not entry["model_installed"]
    ]
    return {
        "asset_id": str(asset.id),
        "pixel_size_nm": pixel_size_nm,
        "organelles": entries,
        "selected": list(selected),
        # None when any organelle could not be costed, never a short total: a
        # denominator missing one organelle's tiles fills the bar with work
        # still to do.
        "tiles_total": (
            None if (not tiles or any(count is None for count in tiles)) else sum(tiles)
        ),
        # Deduped, because the three QuantEM packs share one encoder blob and
        # adding their download figures up overstates it by 2.62x.
        "download_bytes_total": deduped_download_bytes(to_download),
        "packs_to_download": to_download,
    }


class AssetRunsView(APIView):
    """Cost a run over several organelles, and start it as one job."""

    def get(self, request, asset_id):
        asset = get_active_asset(asset_id)
        # An absent parameter means "cost them all"; an empty one means the user
        # has unticked everything, which is a different answer and not an error.
        raw = request.query_params.get("organelles")
        organelles = _requested_organelles(list(RUNNABLE_ORGANELLES) if raw is None else raw)
        invalid = [item for item in organelles if item not in RUNNABLE_ORGANELLES]
        if invalid:
            return Response(
                {
                    "detail": (
                        f"This image can be segmented for {', '.join(RUNNABLE_ORGANELLES)}."
                    ),
                    "invalid": invalid,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        return Response(_build_plan(asset, organelles, {}), status=status.HTTP_200_OK)

    def post(self, request, asset_id):
        asset = get_active_asset(asset_id)
        data = request.data if isinstance(request.data, dict) else {}
        organelles = _requested_organelles(data.get("organelles"))
        if not organelles:
            return Response(
                {"detail": "Choose at least one thing to find in this image."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        invalid = [item for item in organelles if item not in RUNNABLE_ORGANELLES]
        if invalid:
            return Response(
                {
                    "detail": (
                        f"This image can be segmented for {', '.join(RUNNABLE_ORGANELLES)}."
                    ),
                    "invalid": invalid,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        raw_source_models = data.get("source_models")
        source_models = raw_source_models if isinstance(raw_source_models, dict) else {}

        legs = []
        for organelle in organelles:
            definition = ORGANELLE_TYPES.get(organelle)
            if definition is None:
                return Response(
                    {"detail": f"This build does not know how to find {organelle}."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            requested = normalize_source_model(source_models.get(organelle))
            canonical, resolved_source_model = resolve_create_segmentation_request(
                definition, requested
            )
            segmentation_type = ensure_segmentation_type(canonical)
            segmentation, _created = ImageSegmentation.objects.get_or_create(
                asset=asset, segmentation_type=segmentation_type
            )
            # The same two refusals the single-organelle run makes, made before
            # anything is queued: a locked segmentation and one already held by
            # a job are both reasons this organelle cannot start, and finding
            # out twenty minutes in is not an option.
            locked = completion_lock_response(segmentation)
            if locked is not None:
                return locked
            blocking = active_segmentation_job(segmentation)
            if blocking is not None:
                return Response(
                    blocking_job_response_payload(blocking),
                    status=status.HTTP_409_CONFLICT,
                )
            SegmentationConfig.objects.get_or_create(segmentation=segmentation)
            source_model = resolved_source_model or default_source_model_for_organelle(
                segmentation_type.internal_name
            )
            legs.append(
                {
                    "segmentation_id": str(segmentation.id),
                    "segmentation_type": segmentation_type.internal_name,
                    "source_model": source_model,
                }
            )
            source_models[organelle] = source_model

        payload = {
            "asset_id": str(asset.id),
            "scope": "full",
            "legs": legs,
        }
        job = Job.enqueue(
            job_type=JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
            payload=payload,
            priority="default",
            # The single-slot accelerated pool, as the single-organelle full run
            # already uses. Not a claim about the hardware: on a CPU-only
            # machine it is what stops a twenty-minute run from sharing a core
            # with the interactive queue.
            resource_class="gpu",
            queue_name=QUEUE_P4_FULL,
            max_attempts=1,
            tags=[f"asset:{asset.id}"] + [f"segmentation:{leg['segmentation_id']}" for leg in legs],
        )
        plan = _build_plan(asset, organelles, source_models)
        return Response(
            {"job_id": str(job.id), "plan": plan},
            status=status.HTTP_202_ACCEPTED,
        )


urlpatterns = [
    path("assets/<uuid:asset_id>/runs/", AssetRunsView.as_view(), name="asset-runs"),
]
