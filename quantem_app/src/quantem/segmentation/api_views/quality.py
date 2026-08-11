"""Spot-check and count-box routes: the data behind the two-number answer.

The product rule these endpoints exist to serve is that **the quality headline
does not exist until both halves are answered**. A precision estimate on its
own is the most dangerous number this application could print: a model that
finds 511 of 1 300 real mitochondria scores beautifully on a spot check, and a
user reading "9 in 10 look right" would take counts that are 60 % low. So the
payloads here always report *both* halves and a machine-readable
``headline_ready`` with the reasons it is not, and the client is required to
render nothing headline-shaped until that is true.

Three rules the responses keep, which are the reason this is server-side:

* **The sample is stable.** Ordered by a digest of the object id and a stored
  seed, and written down at the moment it is drawn, so it survives a reload, a
  restart and a second reader. See :mod:`quantem.segmentation.quality_sampling`.
* **"Not sure" is excluded from the denominator, and is reported** so the
  sentence can say it was. Counting it either way would invent a judgement the
  user declined to make.
* **A new result version invalidates both halves.** Everything here is scoped
  to the current version, and the previous version's work is reported
  separately so the client can grey it rather than silently dropping it.

Copy lives on the client. These endpoints return counts, flags and reasons --
never phrased sentences -- so that one set of counts sits behind the screen and
behind any export, and the two cannot drift apart.
"""

from __future__ import annotations

import logging

from django.db import transaction
from django.shortcuts import get_object_or_404
from django.urls import path
from django.utils import timezone
from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.segmentation.api_views.shared import completion_lock_response
from quantem.segmentation.completion import is_locked
from quantem.segmentation.geometry_serialization import (
    GEOMETRY_DETAIL_FULL,
    geometry_coords_from_polygon,
    normalize_geometry_detail,
)
from quantem.segmentation.models import (
    CountBox,
    ImageSegmentation,
    QualityCheck,
    SegmentationResultVersion,
    SegmentObject,
)
from quantem.segmentation.overlay_ngff.dirty import merge_dirty_bboxes
from quantem.segmentation.overlay_ngff.mutations import (
    register_overlay_mutation_all_bundles,
)
from quantem.segmentation.quality_sampling import (
    COUNT_BOX_SIZE_PX,
    DEFAULT_SPOT_CHECK_SAMPLE,
    MAX_SPOT_CHECK_SAMPLE,
    MIN_SPOT_CHECK_SAMPLE,
    count_answers,
    count_box_payload,
    derive_seed,
    live_model_objects,
    order_by_sample,
    propose_count_box,
    self_confirmation,
    untouched_candidate_ids,
)
from quantem.segmentation.services.confirm_batch.feature_refresh import (
    _enqueue_segment_feature_refresh,
)
from quantem.segmentation.source_models import normalize_source_model

logger = logging.getLogger(__name__)

#: The label an answer also writes, where the answer says one unambiguously.
#:
#: "Yes, this is one whole object, outlined correctly" *is* keeping it, and
#: "this is not a <organelle>" *is* removing it, so writing the label makes the
#: check double as review rather than throwaway work.
#:
#: ``wrong_shape`` and ``unsure`` are deliberately absent. A wrong outline over
#: a real object is neither kept nor removed -- it wants fixing, and confirming
#: it would enshrine an outline the user just said was wrong, while removing it
#: would delete a real object. "Not sure" is not an instruction at all. Writing
#: a label in either case would put a decision in the data that the user did
#: not make, which is the failure this whole feature exists to avoid.
_ANSWER_LABELS = {
    QualityCheck.ANSWER_YES: "CONFIRMED",
    QualityCheck.ANSWER_NOT_THE_THING: "EXCLUDED",
}

# Machine-readable reasons the headline cannot be shown. Codes, not sentences:
# the client owns the wording, and these travel into its own copy table.
BLOCKER_NOT_ENOUGH_CHECKS = "not_enough_checks"
BLOCKER_NO_COUNT_BOX = "no_count_box"
BLOCKER_COUNT_BOX_UNFINISHED = "count_box_unfinished"


def _segmentation_or_404(seg_id) -> ImageSegmentation:
    return get_object_or_404(
        ImageSegmentation.objects.select_related("asset", "segmentation_type"),
        id=seg_id,
    )


def _requested_sample_size(raw: object) -> int:
    if raw in (None, ""):
        return DEFAULT_SPOT_CHECK_SAMPLE
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise serializers.ValidationError(
            {"n": "How many objects to check must be a whole number."}
        ) from exc
    if value < 1:
        raise serializers.ValidationError({"n": "Checking fewer than one object is not a check."})
    if value > MAX_SPOT_CHECK_SAMPLE:
        raise serializers.ValidationError(
            {
                "n": (
                    "That is more objects than one sitting can cover. "
                    f"The most that can be drawn at once is {MAX_SPOT_CHECK_SAMPLE}."
                )
            }
        )
    return value


def _segment_payload(segment: SegmentObject | None) -> dict | None:
    """Just enough of the object for the check screen to fly to it and draw it.

    Not the full segment serializer: this list is fetched for every question in
    the sample, and the check screen needs the outline and where it is, not the
    whole feature dict.
    """
    if segment is None:
        return None
    return {
        "id": str(segment.id),
        "label_state": segment.label_state,
        "source_model": segment.source_model,
        "confidence_score": segment.confidence_score,
        "centroid": [float(segment.centroid_x), float(segment.centroid_y)],
        "bbox": [
            float(segment.bbox_minx),
            float(segment.bbox_miny),
            float(segment.bbox_maxx),
            float(segment.bbox_maxy),
        ],
        "geometry_coords": geometry_coords_from_polygon(
            segment.geometry,
            geometry_detail=normalize_geometry_detail(GEOMETRY_DETAIL_FULL),
        ),
    }


def _check_payload(check: QualityCheck) -> dict:
    return {
        "id": str(check.id),
        "ordinal": int(check.ordinal),
        "answer": check.answer or None,
        "answered_at": check.answered_at.isoformat() if check.answered_at else None,
        "segment": _segment_payload(check.segment),
    }


def _current_checks(segmentation: ImageSegmentation, run_version: int):
    return list(
        QualityCheck.objects.filter(
            segmentation=segmentation,
            run_version=run_version,
            kind=QualityCheck.KIND_RANDOM_SAMPLE,
        )
        .select_related("segment")
        .order_by("ordinal")
    )


def _previous_spot_check(segmentation: ImageSegmentation, run_version: int) -> dict | None:
    """The last version's sample, so the client can grey it rather than hide it.

    A result version change invalidates the answers -- they were about objects
    that are no longer on screen -- but silently dropping them would leave a
    user who spent a minute answering twelve questions with no sign the work
    ever happened, and no idea why the headline vanished.
    """
    previous = (
        QualityCheck.objects.filter(
            segmentation=segmentation,
            kind=QualityCheck.KIND_RANDOM_SAMPLE,
        )
        .exclude(run_version=run_version)
        .order_by("-run_version")
        .first()
    )
    if previous is None:
        return None
    stale = _current_checks(segmentation, int(previous.run_version))
    return {
        "run_version": int(previous.run_version),
        "counts": count_answers(stale).as_dict(),
    }


def _current_count_box(segmentation: ImageSegmentation, run_version: int) -> CountBox | None:
    return CountBox.objects.filter(segmentation=segmentation, run_version=run_version).first()


def _previous_count_box(segmentation: ImageSegmentation, run_version: int) -> dict | None:
    previous = (
        CountBox.objects.filter(segmentation=segmentation)
        .exclude(run_version=run_version)
        .order_by("-run_version")
        .first()
    )
    return count_box_payload(previous)


def _headline_blockers(counts, box: CountBox | None) -> list[str]:
    blockers: list[str] = []
    if counts.scored < MIN_SPOT_CHECK_SAMPLE:
        blockers.append(BLOCKER_NOT_ENOUGH_CHECKS)
    if box is None:
        blockers.append(BLOCKER_NO_COUNT_BOX)
    elif not box.is_complete:
        blockers.append(BLOCKER_COUNT_BOX_UNFINISHED)
    return blockers


def _quality_payload(
    segmentation: ImageSegmentation,
    run_version: int,
    checks: list[QualityCheck],
    *,
    extra: dict | None = None,
) -> dict:
    """Everything both halves need, in one shape, from every endpoint here.

    One payload rather than three, because the headline rule is a statement
    about both halves at once: an endpoint that returned only its own half
    would leave the client to recombine them and to re-derive
    ``headline_ready``, and two derivations of one rule is how a headline comes
    to appear with only one half answered.
    """
    counts = count_answers(checks)
    box = _current_count_box(segmentation, run_version)
    blockers = _headline_blockers(counts, box)
    payload: dict = {
        "run_version": run_version,
        "minimum_checks_for_headline": MIN_SPOT_CHECK_SAMPLE,
        "sample_seed": int(checks[0].sample_seed) if checks else None,
        "object_count": live_model_objects(segmentation, run_version).count(),
        "checks": [_check_payload(check) for check in checks],
        "counts": counts.as_dict(),
        "self_confirmation": self_confirmation(checks),
        "count_box": count_box_payload(box),
        "headline_ready": not blockers,
        "headline_blockers": blockers,
        "previous_version": {
            "spot_check": _previous_spot_check(segmentation, run_version),
            "count_box": _previous_count_box(segmentation, run_version),
        },
        "locked": is_locked(segmentation),
    }
    if extra:
        payload.update(extra)
    return payload


class SpotCheckView(APIView):
    """The random spot check: draw the sample, and report the answers.

    ``GET`` writes rows, which is unusual and is the point. The sample has to
    be *the same sample* on the next request, and the only way to guarantee
    that while the pool underneath it is changing -- every answer labels an
    object and takes it out of the untouched pool -- is to write the draw down
    the first time it is made. The write is idempotent: the same request
    returns the same rows, and asking for more extends the draw rather than
    replacing it.
    """

    def get(self, request, seg_id):
        segmentation = _segmentation_or_404(seg_id)
        try:
            requested = _requested_sample_size(request.query_params.get("n"))
        except serializers.ValidationError as exc:
            return Response(exc.detail, status=status.HTTP_400_BAD_REQUEST)

        run_version = SegmentationResultVersion.current_version_for(segmentation)
        checks = _current_checks(segmentation, run_version)

        drawing_refused = None
        if len(checks) < requested:
            if is_locked(segmentation):
                # Drawing more questions on a finished image would create
                # questions that cannot be answered: answering writes a label,
                # and labels are what the lock protects. Report what exists.
                drawing_refused = "locked"
            else:
                checks = self._extend_sample(segmentation, run_version, checks, requested)

        pool_remaining = len(
            untouched_candidate_ids(
                segmentation,
                run_version,
                exclude_ids=[c.segment_id for c in checks if c.segment_id],
            )
        )
        payload = _quality_payload(
            segmentation,
            run_version,
            checks,
            extra={
                "n_requested": requested,
                "n_drawn": len(checks),
                "pool_remaining": pool_remaining,
                "drawing_refused": drawing_refused,
            },
        )
        return Response(payload, status=status.HTTP_200_OK)

    @staticmethod
    def _extend_sample(
        segmentation: ImageSegmentation,
        run_version: int,
        checks: list[QualityCheck],
        requested: int,
    ) -> list[QualityCheck]:
        """Add questions up to ``requested``, keeping the ones already drawn.

        The seed comes from the rows that already exist, so extending a sample
        never re-seeds it; only a first draw derives one. That is what makes
        "12 more" the *next* twelve of the same order rather than a reshuffle
        of the first twelve.
        """
        seed = (
            int(checks[0].sample_seed)
            if checks
            else derive_seed("spot-check", segmentation.pk, run_version)
        )
        already = [check.segment_id for check in checks if check.segment_id]
        candidates = untouched_candidate_ids(segmentation, run_version, exclude_ids=already)
        if not candidates:
            return checks

        wanted = requested - len(checks)
        chosen = order_by_sample(candidates, seed)[:wanted]
        next_ordinal = max((check.ordinal for check in checks), default=-1) + 1
        rows = [
            QualityCheck(
                segmentation=segmentation,
                run_version=run_version,
                kind=QualityCheck.KIND_RANDOM_SAMPLE,
                sample_seed=seed,
                segment_id=segment_id,
                ordinal=next_ordinal + offset,
            )
            for offset, segment_id in enumerate(chosen)
        ]
        with transaction.atomic():
            # ``ignore_conflicts`` rather than a lock: two clients opening the
            # panel at once both compute the same rows from the same seed, so
            # the loser of the race wants exactly the rows the winner wrote.
            QualityCheck.objects.bulk_create(rows, ignore_conflicts=True)
        return _current_checks(segmentation, run_version)


class SpotCheckAnswerView(APIView):
    """Record one answer, and write the label it implies.

    Refused while the image is marked finished, for the same reason every other
    label endpoint is: the answer would change an object's label, and that is
    what the lock protects. The refusal names the way out, as those endpoints do.
    """

    def post(self, request, seg_id):
        segmentation = _segmentation_or_404(seg_id)
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked

        answer = request.data.get("answer")
        valid_answers = {choice[0] for choice in QualityCheck.ANSWER_CHOICES}
        if answer not in valid_answers:
            return Response(
                {"answer": "That is not one of the answers this check offers."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        run_version = SegmentationResultVersion.current_version_for(segmentation)
        check = self._find_check(request, segmentation, run_version)
        if check is None:
            return Response(
                {
                    "detail": (
                        "That question is not part of the current check. It may "
                        "belong to an earlier version of this result."
                    )
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        check.record_answer(answer, now=timezone.now())
        check.save(update_fields=["answer", "answered_at", "updated_at"])

        overlay = self._apply_label(segmentation, check.segment, answer)

        checks = _current_checks(segmentation, run_version)
        payload = _quality_payload(
            segmentation,
            run_version,
            checks,
            extra={"check": _check_payload(check), "overlay": overlay},
        )
        return Response(payload, status=status.HTTP_200_OK)

    @staticmethod
    def _find_check(request, segmentation, run_version) -> QualityCheck | None:
        base = QualityCheck.objects.select_related("segment").filter(
            segmentation=segmentation,
            run_version=run_version,
            kind=QualityCheck.KIND_RANDOM_SAMPLE,
        )
        check_id = request.data.get("check_id")
        if check_id:
            return base.filter(id=check_id).first()
        segment_id = request.data.get("segment_id")
        if segment_id:
            return base.filter(segment_id=segment_id).first()
        return None

    @staticmethod
    def _apply_label(
        segmentation: ImageSegmentation,
        segment: SegmentObject | None,
        answer: str,
    ) -> dict | None:
        """Write the label the answer implies, if it implies one.

        Returns the overlay mutation so the client can refresh the drawing
        without a second round trip, or ``None`` when nothing changed.
        """
        label = _ANSWER_LABELS.get(answer)
        if segment is None or label is None or segment.label_state == label:
            return None

        segment.label_state = label
        segment.refined = "UNREFINED"
        segment.save()

        overlay = register_overlay_mutation_all_bundles(
            segmentation,
            dirty_bbox=merge_dirty_bboxes(segmentation, [segment.geometry]),
            source_model=normalize_source_model(segment.source_model),
        )
        try:
            _enqueue_segment_feature_refresh(
                segmentation_id=str(segmentation.id),
                segment_ids=[],
                recompute_features=True,
            )
        except Exception:
            # The label is already written, so the request succeeded; what is
            # lost is the refresh. Said out loud rather than swallowed, because
            # a queue rejecting work looks exactly like one with nothing to do.
            logger.warning(
                "Could not queue a feature refresh for segmentation %s after a spot-check answer.",
                segmentation.id,
                exc_info=True,
            )
        return overlay


class CountBoxView(APIView):
    """The marked-up box: where it goes, and what the user found in it.

    ``GET`` never creates the box. It returns the one that exists, or the
    placement the app *would* use, so that opening the panel to look costs
    nothing and the rectangle a user is shown is the rectangle that gets saved.
    """

    def get(self, request, seg_id):
        segmentation = _segmentation_or_404(seg_id)
        run_version = SegmentationResultVersion.current_version_for(segmentation)
        checks = _current_checks(segmentation, run_version)
        proposed = None
        if _current_count_box(segmentation, run_version) is None:
            proposed = propose_count_box(segmentation, run_version)
        payload = _quality_payload(
            segmentation,
            run_version,
            checks,
            extra={"proposed_count_box": proposed, "box_size_px": COUNT_BOX_SIZE_PX},
        )
        return Response(payload, status=status.HTTP_200_OK)

    def post(self, request, seg_id):
        segmentation = _segmentation_or_404(seg_id)
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked

        run_version = SegmentationResultVersion.current_version_for(segmentation)
        box = _current_count_box(segmentation, run_version)
        if box is None:
            box = self._create_box(request, segmentation, run_version)
            if isinstance(box, Response):
                return box

        error = self._apply_marks(request, box)
        if error is not None:
            return error

        checks = _current_checks(segmentation, run_version)
        payload = _quality_payload(segmentation, run_version, checks)
        return Response(payload, status=status.HTTP_200_OK)

    @staticmethod
    def _create_box(request, segmentation, run_version):
        """Place the box. The request does not get to choose where.

        There is deliberately no way to post a rectangle. A user-chosen box is
        a biased box -- people put it where the result looks interesting, and a
        recall measured there is the recall of the interesting part -- and an
        endpoint that accepted one would make the estimate quietly
        unfalsifiable, because nothing downstream could tell a drawn box from a
        drawn-where-it-looked-good box. The client renders the proposal it was
        given and posts its acceptance.
        """
        del request  # the placement is the app's, not the caller's
        proposed = propose_count_box(segmentation, run_version)
        if proposed is None:
            return Response(
                {
                    "detail": (
                        "This image cannot be read right now, so the box cannot be placed on it."
                    )
                },
                status=status.HTTP_409_CONFLICT,
            )
        if proposed["width"] <= 0 or proposed["height"] <= 0:
            return Response(
                {"detail": "A box with no area cannot be marked up."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return CountBox.objects.create(
            segmentation=segmentation,
            run_version=run_version,
            x=proposed["x"],
            y=proposed["y"],
            width=proposed["width"],
            height=proposed["height"],
            seed=int(proposed["seed"]),
            placement=str(proposed["placement"]),
        )

    @staticmethod
    def _apply_marks(request, box: CountBox):
        fields: list[str] = []
        for field in ("n_marked", "n_matched"):
            raw = request.data.get(field)
            if raw is None:
                continue
            try:
                value = int(raw)
            except (TypeError, ValueError):
                return Response(
                    {field: "A count has to be a whole number."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if value < 0:
                return Response(
                    {field: "A count cannot be negative."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            setattr(box, field, value)
            fields.append(field)

        if box.n_matched > box.n_marked:
            return Response(
                {
                    "detail": (
                        "More objects were matched than were marked, which "
                        "cannot happen: every match is one of the marks."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        complete = request.data.get("complete")
        if complete is not None:
            box.completed_at = timezone.now() if complete else None
            fields.append("completed_at")

        if fields:
            box.save(update_fields=[*dict.fromkeys(fields), "updated_at"])
        else:
            box.save()
        return None


urlpatterns = [
    path(
        "segmentations/<uuid:seg_id>/spot-check/answer",
        SpotCheckAnswerView.as_view(),
        name="segmentation-spot-check-answer",
    ),
    path(
        "segmentations/<uuid:seg_id>/spot-check/",
        SpotCheckView.as_view(),
        name="segmentation-spot-check",
    ),
    path(
        "segmentations/<uuid:seg_id>/count-box",
        CountBoxView.as_view(),
        name="segmentation-count-box",
    ),
]
