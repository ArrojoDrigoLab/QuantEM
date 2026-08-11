"""Drawing the quality sample, and counting what came back.

The two-number quality answer needs a *defensible* sample, not merely a
convenient one, so the drawing rules live here rather than inside a view: they
are the part a reviewer has to be able to read.

Three properties this module exists to guarantee
------------------------------------------------

**1. The draw is stable.** The same segmentation and the same result version
always produce the same objects, in the same order, on every call, after a
reload, after a server restart, and on a second machine reading the same
database. That rules out :func:`hash`, whose string ordering is randomised per
process by ``PYTHONHASHSEED`` -- a sample ordered by it would silently reshuffle
between the first request and the second, and "1 of 12" would mean a different
object each time. The order is a BLAKE2b digest of ``"<segment id>:<seed>"``,
which is fixed for all time.

**2. The draw is from the model's own untouched output.** A spot check answers
*"of the things it found, how many are good"*. Including an object the user has
already confirmed would count their own work as the model's, and including a
hand-drawn outline would count it as a model object that never existed. So the
pool is: live objects of the current result version, not produced by hand, and
not yet judged by the user.

**3. Answering does not disturb the draw.** Every answer also writes the normal
label, which takes that object out of the untouched pool -- so a sample
recomputed on each request would reshuffle itself as the user worked through
it. The rows are therefore written when the sample is drawn, and extending a
sample from 12 to 36 takes the *next* 24 of the same order rather than redrawing
the first 12.

What this module deliberately does not do
-----------------------------------------
It does not compute a confidence interval or phrase a sentence. The readout is
natural frequency with the sample size inside the sentence and a Wilson
interval in words, and that is rendered by the client from the counts here.
Splitting it this way keeps one set of counts behind both the screen and any
export, so the two cannot drift into disagreeing about the same sample.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from quantem.segmentation.models import (
    CountBox,
    ImageSegmentation,
    QualityCheck,
    SegmentObject,
)
from quantem.segmentation.source_models import SOURCE_MODEL_MANUAL

logger = logging.getLogger(__name__)

#: Below this many answered checks the headline sentence is not shown. Not a
#: statistical threshold so much as an honesty one: at n < 12 the interval is
#: wider than the estimate is useful, and a number with an interval that wide
#: reads as a measurement anyway.
MIN_SPOT_CHECK_SAMPLE = 12

#: Refuse to draw more than this in one request. A spot check is meant to cost
#: about a minute; a request for thousands is a client bug, and answering it
#: would write thousands of rows nobody will ever answer.
MAX_SPOT_CHECK_SAMPLE = 200

#: The default sample the screen offers.
DEFAULT_SPOT_CHECK_SAMPLE = 12

#: The count box is 512 px square in the image's own pixels. Fixed rather than
#: scaled to the image, because the point of the box is that a person can mark
#: every object inside it in about three minutes, and that budget is set by how
#: many objects fit, not by how large the image is.
COUNT_BOX_SIZE_PX = 512

#: The share of checked positives that may be the model's own guesses before
#: the readout has to say so. In practice the sample is drawn entirely from the
#: model's guesses, so this fires almost always -- which is the point: a spot
#: check *is* agreement with the model, and calling it accuracy would be the
#: single most misleading thing this feature could do.
SELF_CONFIRMATION_THRESHOLD = 0.8

#: Label states that mean "the user has not judged this object yet".
UNTOUCHED_LABEL_STATES = ("INFERRED", "CANDIDATE")


def derive_seed(*parts: object) -> int:
    """A stable non-negative 63-bit seed from the given parts.

    Derived rather than random so that the draw can be reproduced from the
    identifiers alone -- including by a reader checking that the sample really
    was arbitrary with respect to the objects, which is the whole claim a
    random spot check rests on. Persisted as well, because the derivation rule
    could change and the sample already shown to a user may not.
    """
    material = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF


def sample_order_key(segment_id: object, seed: int) -> bytes:
    """The sort key that puts one object at one place in the draw.

    BLAKE2b of ``"<segment id>:<seed>"``. Deterministic across processes,
    machines and Python versions, which :func:`hash` is not.
    """
    material = f"{segment_id}:{int(seed)}".encode()
    return hashlib.blake2b(material, digest_size=16).digest()


def order_by_sample(segment_ids, seed: int) -> list:
    """``segment_ids`` in draw order for ``seed``.

    Ties break on the id itself so the order is total even in the
    (astronomically unlikely) event of a digest collision; without that, two
    colliding objects would swap places depending on database row order.
    """
    return sorted(segment_ids, key=lambda sid: (sample_order_key(sid, seed), str(sid)))


def live_model_objects(segmentation: ImageSegmentation, run_version: int):
    """Every object of this result version the model produced and still owns.

    "Still owns" excludes superseded objects (a previous version's) and
    hand-drawn ones. This is the denominator the sentence quotes when it says
    *"I picked these 12 at random from the 511"*.
    """
    return SegmentObject.objects.filter(
        segmentation=segmentation,
        superseded_at__isnull=True,
        run_version=run_version,
    ).exclude(source_model=SOURCE_MODEL_MANUAL)


def untouched_candidate_ids(
    segmentation: ImageSegmentation,
    run_version: int,
    *,
    exclude_ids=(),
) -> list:
    """Ids of the objects a spot check may still ask about.

    ``exclude_ids`` is how a sample is extended without asking twice: the
    objects already drawn are passed in, and the next of the same order are
    taken from what is left. It is separate from the untouched filter because
    an object that has been *answered* is no longer untouched, so filtering on
    label state alone would let a re-answered object be drawn a second time.
    """
    queryset = live_model_objects(segmentation, run_version).filter(
        label_state__in=UNTOUCHED_LABEL_STATES,
        refined="UNREFINED",
    )
    if exclude_ids:
        queryset = queryset.exclude(id__in=list(exclude_ids))
    return list(queryset.values_list("id", flat=True))


@dataclass(frozen=True)
class SpotCheckCounts:
    """What the answers add up to. Every field is a count, never a rate.

    The rate is computed where the interval is, so that a number and the
    interval around it are always derived from the same denominator.
    """

    #: Rows drawn, answered or not.
    drawn: int = 0
    #: Rows with any answer, including "not sure".
    answered: int = 0
    #: Rows answered "not sure". Excluded from :attr:`scored`, and reported so
    #: the readout can say it was.
    unsure: int = 0
    #: The denominator: answered minus "not sure".
    scored: int = 0
    #: The numerator: answered "yes".
    positive: int = 0
    #: Answered "wrong shape". A real object with an outline that is wrong,
    #: which is not a good object for a measurement -- so it counts against,
    #: and is reported separately because the fix for it is different.
    wrong_shape: int = 0
    #: Answered "not a <organelle>".
    not_the_thing: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "drawn": self.drawn,
            "answered": self.answered,
            "unsure": self.unsure,
            "scored": self.scored,
            "positive": self.positive,
            "wrong_shape": self.wrong_shape,
            "not_the_thing": self.not_the_thing,
        }


def count_answers(checks) -> SpotCheckCounts:
    """Tally one sample. ``unsure`` never reaches the denominator."""
    drawn = answered = unsure = positive = wrong_shape = not_the_thing = 0
    for check in checks:
        drawn += 1
        answer = check.answer
        if not answer:
            continue
        answered += 1
        if answer == QualityCheck.ANSWER_UNSURE:
            unsure += 1
        elif answer == QualityCheck.ANSWER_YES:
            positive += 1
        elif answer == QualityCheck.ANSWER_WRONG_SHAPE:
            wrong_shape += 1
        elif answer == QualityCheck.ANSWER_NOT_THE_THING:
            not_the_thing += 1
    return SpotCheckCounts(
        drawn=drawn,
        answered=answered,
        unsure=unsure,
        scored=answered - unsure,
        positive=positive,
        wrong_shape=wrong_shape,
        not_the_thing=not_the_thing,
    )


def self_confirmation(checks) -> dict[str, object]:
    """How much of what the user agreed with started as the model's own guess.

    Computed rather than assumed, even though the pool is drawn entirely from
    model output today and the answer is therefore "all of it". Computing it
    means the number moves on its own if the pool ever widens -- and it means
    the caveat is a measurement of this sample rather than a sentence someone
    decided to always print.

    An answered row whose object has since been deleted cannot be attributed,
    so it is counted in ``unknown`` and left out of the fraction rather than
    guessed either way.
    """
    positive = 0
    from_model = 0
    unknown = 0
    for check in checks:
        if check.answer != QualityCheck.ANSWER_YES:
            continue
        positive += 1
        segment = check.segment
        if segment is None:
            unknown += 1
        elif segment.source_model != SOURCE_MODEL_MANUAL:
            from_model += 1
    attributable = positive - unknown
    fraction = (from_model / attributable) if attributable > 0 else None
    return {
        "n_positive": positive,
        "n_positive_from_model_guesses": from_model,
        "n_positive_unattributable": unknown,
        "fraction": fraction,
        "threshold": SELF_CONFIRMATION_THRESHOLD,
        "applies": fraction is not None and fraction >= SELF_CONFIRMATION_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# The count box
# ---------------------------------------------------------------------------

#: Re-exported from the model that stores them, so there is one spelling of
#: each. Reported rather than hidden: a box in the middle of an image whose
#: middle is empty resin measures nothing, and the user has to be able to see
#: that is what happened.
PLACEMENT_TISSUE_SCORED = CountBox.PLACEMENT_TISSUE_SCORED
PLACEMENT_CENTRED = CountBox.PLACEMENT_CENTRED


def propose_count_box(
    segmentation: ImageSegmentation,
    run_version: int,
    *,
    size_px: int = COUNT_BOX_SIZE_PX,
) -> dict[str, object] | None:
    """Where the app would put the count box, without putting it there.

    **The app places the box, and the user does not.** A user-chosen box is a
    biased box: people put it where the result looks interesting, and a recall
    measured there is a recall of the interesting part. So this scores windows
    for tissue content and picks among the good ones with a seeded draw --
    random within tissue, which is the property the estimate needs, rather than
    random over the whole image, most of which may be empty resin.

    Returns ``None`` only when the segmentation has no readable image at all;
    the caller reports that rather than inventing a rectangle over pixels it
    cannot see.
    """
    # Imported here rather than at module scope: this pulls in the image
    # loading stack, and the counting helpers above are used by paths that
    # never touch an image.
    from quantem.segmentation.api_views.shared import (  # noqa: PLC0415
        get_segmentation_target_image,
    )
    from quantem.segmentation.roi_selection import (  # noqa: PLC0415
        select_roi_for_image,
    )

    seed = derive_seed("count-box", segmentation.pk, run_version)
    try:
        image = get_segmentation_target_image(segmentation)
    except Exception:
        logger.info(
            "No readable image for segmentation %s; the count box cannot be placed on tissue.",
            segmentation.pk,
        )
        return None

    width = int(getattr(image, "width", 0) or 0)
    height = int(getattr(image, "height", 0) or 0)
    if width <= 0 or height <= 0:
        return None

    box_width = min(size_px, width)
    box_height = min(size_px, height)

    try:
        selection = select_roi_for_image(image, roi_size=size_px, seed=seed)
    except Exception:
        # A preview that will not load is a real state (a rendition still
        # building, a file locked by a backup) and it must not take the whole
        # quality panel down with it. Centre the box and say so.
        logger.info(
            "Could not score tissue windows for segmentation %s; centring the count box instead.",
            segmentation.pk,
            exc_info=True,
        )
        return {
            "x": float(max(0, (width - box_width) // 2)),
            "y": float(max(0, (height - box_height) // 2)),
            "width": float(box_width),
            "height": float(box_height),
            "seed": seed,
            "placement": PLACEMENT_CENTRED,
        }

    return {
        "x": float(selection.x),
        "y": float(selection.y),
        "width": float(selection.width),
        "height": float(selection.height),
        "seed": seed,
        "placement": PLACEMENT_TISSUE_SCORED,
    }


def count_box_payload(box: CountBox | None) -> dict[str, object] | None:
    """One count box as the client reads it, or ``None`` when there is none."""
    if box is None:
        return None
    return {
        "id": str(box.id),
        "run_version": int(box.run_version),
        "x": float(box.x),
        "y": float(box.y),
        "width": float(box.width),
        "height": float(box.height),
        "seed": int(box.seed),
        "placement": box.placement or None,
        "n_marked": int(box.n_marked),
        "n_matched": int(box.n_matched),
        "n_missed": int(box.n_marked) - int(box.n_matched),
        "completed_at": box.completed_at.isoformat() if box.completed_at else None,
        "is_complete": box.is_complete,
    }


__all__ = [
    "COUNT_BOX_SIZE_PX",
    "DEFAULT_SPOT_CHECK_SAMPLE",
    "MAX_SPOT_CHECK_SAMPLE",
    "MIN_SPOT_CHECK_SAMPLE",
    "PLACEMENT_CENTRED",
    "PLACEMENT_TISSUE_SCORED",
    "SELF_CONFIRMATION_THRESHOLD",
    "UNTOUCHED_LABEL_STATES",
    "SpotCheckCounts",
    "count_answers",
    "count_box_payload",
    "derive_seed",
    "live_model_objects",
    "order_by_sample",
    "propose_count_box",
    "sample_order_key",
    "self_confirmation",
    "untouched_candidate_ids",
]
