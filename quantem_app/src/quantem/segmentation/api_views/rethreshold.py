"""The include-level dial: read where it is, and ask for it to be moved.

The threshold is the foreground cutoff (:mod:`quantem.segmentation.run_identity`).
Moving it does **not** run the model or change candidate objects: the browser
recolors the probability map that the run already stored. Pressing Apply is the
separate commit step; it re-thresholds that stored map, replaces only the
unconfirmed candidates, and leaves confirmed/manual annotations intact. The
backend of that commit is
:func:`~quantem.seg_core.db.inference.replay_stored_probability_map`; the worker
is :mod:`quantem.jobs.handlers.rethreshold`.

Why the refusals happen *here*, before anything is queued
---------------------------------------------------------
Every reason a dial move cannot work is knowable at the moment it is asked for,
and all of them are cheap to check: no map stored, a map from an older build, a
model with more than one output, a model not installed on this machine, a run
already holding the image. Queuing anyway would put a task on screen that is
certain to go red, and hand the user the reason a minute after the moment they
could have acted on it. So ``POST`` answers 409 with the sentence, and ``GET``
answers the same question in advance so the control can be greyed out with the
reason beside it rather than failing under the user's hand.

The two unavailable-map cases say different things, and that difference is
carried all the way to the client. Both end in "run the model again"; only one
of them will keep happening until the stored result is replaced. See
:data:`~quantem.segmentation.prob_maps.persistence.NO_STORED_MAP_MESSAGE` and
:data:`~quantem.segmentation.prob_maps.persistence.LEGACY_MAP_MESSAGE`.

``urlpatterns`` at the bottom is spliced into
:mod:`quantem.segmentation.urls`, which is why the routes are defined here and
not there: four packages are adding routes this release and none of them opens
that file.
"""

from __future__ import annotations

import logging
from urllib.parse import urlencode

from django.http import FileResponse
from django.shortcuts import get_object_or_404
from django.urls import path, reverse
from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.assets.models import ImageROI
from quantem.core.error_codes import ERROR_CODE_FIELD, ErrorCode
from quantem.jobs.constants import (
    JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
    QUEUE_P1_INTERACTIVE,
)
from quantem.jobs.models import Job
from quantem.seg_core.db.prob_maps import get_prob_map_file_path
from quantem.seg_core.registry import get_segmenter_or_none
from quantem.segmentation.models import (
    ImageSegmentation,
    SegmentationResultVersion,
    SegmentObject,
)
from quantem.segmentation.prob_maps.persistence import stored_map_readiness
from quantem.segmentation.source_models import (
    normalize_source_model,
    resolve_segmenter_internal_name,
)
from quantem.segmentation.status_reconcile import reconcile_segmentation_status

from .shared import (
    _ORGANELLE_ACTION_JOB_TYPES,
    active_segmentation_job,
    blocking_job_response_payload,
    completion_lock_response,
)

logger = logging.getLogger(__name__)

#: The dial's range. It is a probability, so this is not a product choice.
INCLUDE_LEVEL_MIN = 0.0
INCLUDE_LEVEL_MAX = 1.0

#: Said when the model that produced the stored map cannot be loaded here.
#: Re-thresholding needs that model's own extraction settings -- its area floor,
#: its closing radius -- so without it this is not the same operation the run
#: performed, and doing it with another model's settings would silently produce
#: a different candidate set under the same name.
MODEL_UNAVAILABLE_MESSAGE = (
    "The model that found these objects is not available on this computer, so "
    "they cannot be redone at a different include level. Install it on the "
    "Models screen."
)

#: Said for a model whose foreground comes from several outputs combined. The
#: combination is not what gets stored, so replaying would have to redo it --
#: a second decision procedure, which is the thing the stored-map ordering
#: exists to avoid.
MULTI_OUTPUT_MESSAGE = (
    "The include level can only be moved for models that produce a single "
    "confidence map, and this one combines several."
)


class IncludeLevelSerializer(serializers.Serializer):
    """One requested dial position.

    Validated here rather than at the model layer, which does not constrain the
    field: ``ImageSegmentation.include_level`` is a plain ``FloatField``, and a
    level outside 0-1 would be accepted, stored, and handed to the segmenter's
    threshold setter, where it produces either every pixel or none of them and
    reports success.
    """

    include_level = serializers.FloatField(
        min_value=INCLUDE_LEVEL_MIN,
        max_value=INCLUDE_LEVEL_MAX,
        error_messages={
            "required": "Choose an include level between 0 and 1.",
            "invalid": "The include level has to be a number between 0 and 1.",
            "min_value": "The include level has to be between 0 and 1.",
            "max_value": "The include level has to be between 0 and 1.",
        },
    )
    source_model = serializers.CharField(required=False, allow_blank=True)
    roi_id = serializers.UUIDField(required=False, allow_null=True)


def _refusal(detail: str, *, code: ErrorCode | None = None) -> Response:
    """A 409 the client can both read and act on.

    409 rather than 400: nothing about the request is malformed. It conflicts
    with the state of the stored result, which is a state the user can change --
    by running the model once -- and the body says so.
    """
    payload: dict[str, object] = {"detail": detail}
    if code is not None:
        payload[ERROR_CODE_FIELD] = str(code)
    return Response(payload, status=status.HTTP_409_CONFLICT)


def _resolve_segmenter(segmentation: ImageSegmentation, source_model: str):
    """``(segmenter, model_name)``, or ``(None, refusal)`` when the dial cannot move."""
    segmenter_internal_name = resolve_segmenter_internal_name(
        segmentation_type_internal_name=segmentation.segmentation_type.internal_name,
        source_model=source_model,
    )
    segmenter = get_segmenter_or_none(segmenter_internal_name)
    if segmenter is None:
        return None, MODEL_UNAVAILABLE_MESSAGE

    model_names = list(segmenter.get_dl_model_names())
    if len(model_names) != 1:
        return None, MULTI_OUTPUT_MESSAGE
    return segmenter, model_names[0]


def _dial_state(segmentation: ImageSegmentation, source_model: str) -> dict:
    """Everything the control needs to render itself, including why it cannot move.

    One payload for both verbs, so the sentence a greyed-out dial shows and the
    sentence a refused move returns are the same sentence from the same check.
    Two derivations of "can this move" is how a control comes to look available
    and then fail when it is used.
    """
    segmenter, resolved = _resolve_segmenter(segmentation, source_model)
    run_version = SegmentationResultVersion.current_version_for(segmentation)
    state: dict[str, object] = {
        "include_level": segmentation.include_level,
        "default_include_level": None,
        "minimum": INCLUDE_LEVEL_MIN,
        "maximum": INCLUDE_LEVEL_MAX,
        "run_version": run_version,
        "object_count": SegmentObject.objects.filter(
            segmentation=segmentation,
            superseded_at__isnull=True,
        )
        .exclude(label_state="EXCLUDED")
        .count(),
        "can_move": False,
        "detail": "",
    }

    if segmenter is None:
        state["detail"] = resolved
        return state

    state["default_include_level"] = getattr(segmenter, "fg_threshold", None)

    readiness = stored_map_readiness(
        segmentation=segmentation,
        segmenter=segmenter,
        model_name=resolved,
    )
    if not readiness.ready:
        state["detail"] = readiness.detail
        state[ERROR_CODE_FIELD] = str(ErrorCode.PROBABILITY_MAP_MISSING)
        return state

    state["can_move"] = True
    query = urlencode({"source_model": source_model}) if source_model else ""
    preview_url = reverse("segmentation-include-level-map", args=[segmentation.id])
    state["preview_url"] = f"{preview_url}{'?' + query if query else ''}"
    return state


class SegmentationIncludeLevelMapView(APIView):
    """Serve the saved grayscale result used by the live threshold preview."""

    def get(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )
        source_model = normalize_source_model(request.query_params.get("source_model"))
        segmenter, resolved = _resolve_segmenter(segmentation, source_model)
        if segmenter is None:
            return _refusal(resolved)

        readiness = stored_map_readiness(
            segmentation=segmentation,
            segmenter=segmenter,
            model_name=resolved,
        )
        if not readiness.ready:
            return _refusal(
                readiness.detail, code=ErrorCode.PROBABILITY_MAP_MISSING
            )

        file_path = get_prob_map_file_path(
            segmentation,
            resolved,
            str(getattr(segmenter, "prob_map_prefix", "") or ""),
            None,
        )
        response = FileResponse(file_path.open("rb"), content_type="image/png")
        response["Cache-Control"] = "no-store"
        return response


class SegmentationIncludeLevelView(APIView):
    """Read the dial, or ask for it to be moved.

    ``GET`` is free: two filesystem stats and two indexed queries, no map
    decoded. It is meant to be called whenever the panel opens.

    ``POST`` queues one re-extract. It is the explicit Apply action, never a
    slider event: one request replaces the prior candidate set once.
    """

    def get(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )
        source_model = normalize_source_model(request.query_params.get("source_model"))
        return Response(
            _dial_state(segmentation, source_model), status=status.HTTP_200_OK
        )

    def post(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )

        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked

        serializer = IncludeLevelSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        include_level = float(serializer.validated_data["include_level"])
        source_model = normalize_source_model(
            serializer.validated_data.get("source_model")
        )
        roi_id = serializer.validated_data.get("roi_id")

        segmenter, resolved = _resolve_segmenter(segmentation, source_model)
        if segmenter is None:
            return _refusal(resolved)

        roi = None
        if roi_id is not None:
            roi = ImageROI.objects.filter(id=roi_id).first()
            if roi is None:
                return _refusal("That region is no longer on this image.")

        readiness = stored_map_readiness(
            segmentation=segmentation,
            segmenter=segmenter,
            model_name=resolved,
            roi=roi,
        )
        if not readiness.ready:
            return _refusal(
                readiness.detail, code=ErrorCode.PROBABILITY_MAP_MISSING
            )

        # The user is telling us nothing is running. If the stage says otherwise
        # because a worker died mid-run, correct it now rather than refusing a
        # dial move on the strength of a phantom.
        reconcile_segmentation_status(segmentation)
        blocking_job = active_segmentation_job(
            segmentation,
            job_types=_ORGANELLE_ACTION_JOB_TYPES,
        )
        if blocking_job is not None:
            return Response(
                blocking_job_response_payload(blocking_job),
                status=status.HTTP_409_CONFLICT,
            )

        payload: dict[str, object] = {
            # Required, and not by convention: this job type is in
            # ACTIVE_SEGMENTATION_JOB_TYPES, whose failure reconcilers read this
            # exact key to release an image whose worker died.
            "segmentation_id": str(segmentation.id),
            "segmentation_type": segmentation.segmentation_type.internal_name,
            "include_level": include_level,
        }
        if source_model:
            payload["source_model"] = source_model
        if roi is not None:
            payload["roi_id"] = str(roi.id)

        job = Job.enqueue(
            job_type=JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
            payload=payload,
            priority="high",
            resource_class="cpu",
            queue_name=QUEUE_P1_INTERACTIVE,
            # Deterministic failures only: a missing or unreadable stored map
            # does not improve on a second attempt, and the user is holding the
            # dial waiting for an answer.
            max_attempts=1,
            tags=[f"segmentation:{seg_id}"],
        )
        logger.info(
            "Queued a re-extract of segmentation %s at include level %s",
            segmentation.id,
            include_level,
        )
        return Response(
            {"job_id": str(job.id), "include_level": include_level},
            status=status.HTTP_202_ACCEPTED,
        )


urlpatterns = [
    path(
        "segmentations/<uuid:seg_id>/include-level/map",
        SegmentationIncludeLevelMapView.as_view(),
        name="segmentation-include-level-map",
    ),
    path(
        "segmentations/<uuid:seg_id>/include-level",
        SegmentationIncludeLevelView.as_view(),
        name="segmentation-include-level",
    ),
]
