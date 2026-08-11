"""Per-object run identity: which run, and with what settings, made this object.

Every :class:`~quantem.segmentation.models.SegmentObject` produced by inference
carries a ``"run"`` entry in its ``features`` dict. A hand-drawn object carries
no such key at all, and the difference is load-bearing: **absent means "not
produced by a model", which is not the same as "produced with unknown
settings".** Nothing here ever writes a placeholder run.

The shape is a fixed contract, read by the analysis manifest
(:mod:`quantem.analysis`) and written by
:func:`quantem.seg_core.db.extraction.extract_and_save_segments`::

    "run": {
        "id": "<uuid of the inference run / job>",
        "finished_at": "<ISO-8601 UTC>",
        "pack_id": "quantem:mito",
        "threshold": 0.5,                 # the value ACTUALLY used
        "adapter_id": "<uuid>" | null,    # the adapter applied, if any
        "ran_at_nm": 8.0 | null,          # null = native scale
        "native_pixel_size_nm": 5.0 | null,
        "min_area": 60,                   # native px, the value applied
        "scope": "full",                  # "full" | "patch"
        "include_level": 0.5,             # the dial position, = threshold
        "run_version": 1,                 # which numbered result
        "prob_map_grid": "native" | null, # grid the decision was taken on
        "device": "cuda" | "cpu" | null   # where the arithmetic happened
    }

``device`` is the device the run **finished** on, which is not always the one
it was offered: a model that cannot execute on the graphics card, or a card that
runs out of memory, moves the run to the processor part-way through. Owner
ruling R5 settles what it is *for* -- a GPU run and a CPU run may be compared,
so nothing here refuses or red-flags a mixed-device analysis -- and R4 settles
that it must nevertheless be recorded: "provenance must keep recording which
device produced the numbers". ``None`` means the run predates the record or the
segmenter does not expose one, never "the processor"; guessing a default here
would put a hardware claim in the manifest that nobody measured.

The last four before it were added together, with defaults chosen so that an
object written after them carries exactly what an object written before them
carried.
``scope`` defaults to ``"full"`` because every run before patch runs existed was
a whole-image run; ``include_level`` defaults to ``threshold`` because the dial
*is* the threshold under a name a biologist can use, and the two are the same
number until a dial exists to move one of them; ``run_version`` defaults to 1
because there has always been a first result; ``prob_map_grid`` defaults to
``None`` because a run that did not record which grid it decided on did not
record it, and writing ``"native"`` there would claim provenance nobody
captured.

Why it exists: before this, an object recorded only ``source_model:
"quantem:er"``. A run made through a user-fitted adapter at threshold 0.45 was
byte-for-byte indistinguishable from released-pack output at 0.50, so an
analysis manifest could name the model but could not say which settings
produced which numbers -- and two runs of the same pack over the same image at
different thresholds were reported as one population.

``threshold`` is deliberately *the value actually used*, not the pack's
published default: applying an adapter replaces it (see
:meth:`quantem.inference.segmenter.DinoOrganelleSegmenter.apply_adapter`), and
recording the published number there would document a run that never happened.
Same for ``min_area``, which is resolved against the segmenter's own
per-organelle floor rather than the caller's generic default.

``ran_at_nm`` is the scale the model actually predicted at: a pack's
``canonical_nm`` when the asset's pixel size is known (so
:mod:`quantem.inference.resample` could resample to it), and ``None`` when the
run fell back to native pixels because either number was missing. ``None`` is
therefore a real caveat about the numbers, not a gap in the record.

Pure stdlib: no Django, no numpy. Imported by both
:mod:`quantem.seg_core` and :mod:`quantem.analysis`.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

#: The key under which the run identity lives in ``SegmentObject.features``.
RUN_FEATURE_KEY = "run"

#: Every key the contract defines, in contract order. A payload this module
#: builds always carries all of them; a value may be ``None`` where the contract
#: allows it, but a key is never omitted -- a reader must be able to tell "ran
#: at native scale" from "this writer did not know about ran_at_nm".
RUN_IDENTITY_KEYS: tuple[str, ...] = (
    "id",
    "finished_at",
    "pack_id",
    "threshold",
    "adapter_id",
    "ran_at_nm",
    "native_pixel_size_nm",
    "min_area",
    "scope",
    "include_level",
    "run_version",
    "prob_map_grid",
    "device",
)

#: The eight keys the contract had before the v2 push. Kept as a named constant
#: so the proof that adding four keys changed nothing can be written as an
#: assertion rather than as a comment: a run made before and after must agree on
#: every one of these.
LEGACY_RUN_IDENTITY_KEYS: tuple[str, ...] = RUN_IDENTITY_KEYS[:8]

#: A run over the whole image. The value every object written before patch runs
#: existed would have carried, which is why it is the default.
RUN_SCOPE_FULL = "full"
#: A run over one user-chosen rectangle.
RUN_SCOPE_PATCH = "patch"


def utc_timestamp(moment: datetime | None = None) -> str:
    """ISO-8601 UTC, e.g. ``'2026-08-07T09:15:02.481Z'``.

    Milliseconds, ``Z`` suffix, always UTC. A naive datetime is taken to be UTC
    rather than local time: these strings end up side by side in an export
    manifest, and a mixture of offsets there is worse than no offset at all.
    """
    if moment is None:
        moment = datetime.now(UTC)
    elif moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (
        moment.astimezone(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None  # drop NaN


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_run_identity(
    *,
    run_id: str,
    pack_id: str,
    threshold: float | None,
    adapter_id: str | None,
    ran_at_nm: float | None,
    native_pixel_size_nm: float | None,
    min_area: int | None,
    finished_at: datetime | str | None = None,
    scope: str = RUN_SCOPE_FULL,
    include_level: float | None = None,
    run_version: int = 1,
    prob_map_grid: str | None = None,
    device: str | None = None,
) -> dict[str, object]:
    """Build one run-identity payload, JSON-ready.

    Every field is normalised to a JSON scalar here rather than at each call
    site, so a numpy float or a UUID object cannot reach ``features`` and make
    the row unserialisable at write time.

    The four v2 fields all default to what an object written before them
    carried, so a caller that has not been updated still writes a truthful
    record rather than a gap. ``include_level`` is the one that needs saying
    out loud: passing ``None`` records the run's own ``threshold``, because
    they are the same number until something moves the dial, and recording
    ``None`` would read as "no include level" for every run ever made.
    """
    stamp = finished_at if isinstance(finished_at, str) else utc_timestamp(finished_at)
    resolved_threshold = _optional_float(threshold)
    resolved_level = _optional_float(include_level)
    if resolved_level is None:
        resolved_level = resolved_threshold
    return {
        "id": str(run_id),
        "finished_at": stamp,
        "pack_id": str(pack_id),
        "threshold": resolved_threshold,
        "adapter_id": _optional_str(adapter_id),
        "ran_at_nm": _optional_float(ran_at_nm),
        "native_pixel_size_nm": _optional_float(native_pixel_size_nm),
        "min_area": int(min_area) if min_area is not None else None,
        "scope": _optional_str(scope) or RUN_SCOPE_FULL,
        "include_level": resolved_level,
        "run_version": int(run_version) if run_version else 1,
        "prob_map_grid": _optional_str(prob_map_grid),
        "device": _optional_str(device),
    }


def resolve_ran_at_nm(
    *,
    canonical_nm: float | None,
    native_pixel_size_nm: float | None,
) -> float | None:
    """The nm/px the model actually predicted at, or None for native scale.

    Mirrors :func:`quantem.inference.resample.resample_factor`: resampling only
    happens when *both* the pack's canonical size and the asset's true pixel
    size are known. Either one missing means the run saw native pixels, and the
    honest record of that is ``None`` rather than the canonical number the model
    would have liked.
    """
    canonical = _optional_float(canonical_nm)
    native = _optional_float(native_pixel_size_nm)
    if not canonical or canonical <= 0:
        return None
    if not native or native <= 0:
        return None
    return canonical


def usable_pixel_size_nm(recorded_nm: object) -> object | None:
    """The recorded pixel size if it is a length, else ``None``.

    Zero, a negative number and an absent value all mean the same thing to
    anything that would convert with it: there is no scale here. They are not
    the same thing to a *reader* -- a recorded ``-5`` is a calibration to go and
    fix, and the value is reported unchanged wherever it is reported -- so this
    answers only the narrow question "may this be used as a scale".

    The value is returned unchanged rather than coerced, so a caller that passes
    a ``Decimal`` gets its ``Decimal`` back and nothing downstream silently
    changes type.
    """
    if not recorded_nm:
        return None
    try:
        positive = recorded_nm > 0  # type: ignore[operator]
    except TypeError:
        # A pixel size that is not a number cannot be used as one. Reached only
        # for a hand-built input; the model field is a float.
        return None
    return recorded_nm if positive else None


def produced_without_pixel_size(
    produced_pixel_size_nm: Iterable[float | None],
) -> bool:
    """True when at least one object here was produced with no pixel size.

    ``produced_pixel_size_nm`` is the set of ``native_pixel_size_nm`` values the
    objects' own stamps recorded. Empty means nothing is stamped -- hand-drawn
    outlines, or objects made before this record existed -- and says nothing
    either way, which is why this asks for a ``None`` *member* rather than for a
    missing value.
    """
    return any(value is None for value in produced_pixel_size_nm)


def calibrated_after_the_fact(
    *,
    produced_pixel_size_nm: Iterable[float | None],
    recorded_pixel_size_nm: object,
) -> bool:
    """The image has a pixel size now, and its objects were made without one.

    This is the state that reads as fine everywhere and is not: the image says
    ``5 nm/px``, the objects on top of it were produced before that number
    existed, and a pack that declares a ``canonical_nm`` therefore never
    resampled to it. The object set is the one native pixels produced, so
    converting it with the number typed in afterwards is arithmetic on the wrong
    objects -- see :func:`quantem.analysis.service.run_analysis`, which blanks
    every physical unit when this is true.

    It lives here, in the module that owns the stamp, because two screens have
    to agree about it: the labeling screen decides whether the work is finished
    before an analysis is ever run, and the finished bundle says whether its
    numbers are in physical units. Restating the predicate in either place is
    how they come to disagree, and the disagreement is invisible until someone
    compares a screen with an export.
    """
    return (
        produced_without_pixel_size(produced_pixel_size_nm)
        and usable_pixel_size_nm(recorded_pixel_size_nm) is not None
    )


def run_identity_from_segmenter(
    segmenter: object,
    *,
    run_id: str,
    pack_id_fallback: str,
    native_pixel_size_nm: float | None,
    min_area: int | None,
    finished_at: datetime | str | None = None,
    scope: str = RUN_SCOPE_FULL,
    include_level: float | None = None,
    run_version: int = 1,
    prob_map_grid: str | None = None,
) -> dict[str, object]:
    """Read a segmenter's per-run settings into a run-identity payload.

    Everything is read through the segmenter's **public** surface
    (``model_spec``, ``fg_threshold``, ``adapter_id``, ``inference_device``),
    each of which is optional: a segmenter that does not expose one records
    ``None`` for that field rather than guessing. ``pack_id_fallback`` is the
    resolved source model, used when the segmenter has no ``model_spec`` -- the
    two coincide for every released pack.

    ``inference_device`` is read here rather than passed in by the run task for
    the same reason ``fg_threshold`` is: the caller knows what it asked for, and
    only the segmenter knows what actually happened. A run that was offered the
    graphics card and finished on the processor reports ``"cpu"``.
    """
    spec = getattr(segmenter, "model_spec", None)
    pack_id = getattr(spec, "pack_id", None) or pack_id_fallback
    canonical_nm = getattr(spec, "canonical_nm", None)

    threshold = getattr(segmenter, "fg_threshold", None)
    if threshold is None:
        threshold = getattr(spec, "threshold", None)

    return build_run_identity(
        run_id=run_id,
        pack_id=str(pack_id),
        threshold=threshold,
        adapter_id=getattr(segmenter, "adapter_id", None),
        ran_at_nm=resolve_ran_at_nm(
            canonical_nm=canonical_nm,
            native_pixel_size_nm=native_pixel_size_nm,
        ),
        native_pixel_size_nm=native_pixel_size_nm,
        min_area=min_area,
        finished_at=finished_at,
        scope=scope,
        include_level=include_level,
        run_version=run_version,
        prob_map_grid=prob_map_grid,
        device=getattr(segmenter, "inference_device", None),
    )


def read_run_identity(features: object) -> dict[str, object] | None:
    """The run identity stored on a features dict, or None if there is none.

    ``None`` means "no model produced this object" -- a hand-drawn outline, or
    one saved before this record existed. Callers must not substitute defaults
    for a missing run: a fabricated ``threshold: 0.5`` on a hand-drawn object is
    exactly the kind of plausible-and-wrong number this record exists to
    prevent.
    """
    if not isinstance(features, dict):
        return None
    raw = features.get(RUN_FEATURE_KEY)
    if not isinstance(raw, dict):
        return None
    if not _optional_str(raw.get("id")):
        return None
    return dict(raw)


__all__ = [
    "LEGACY_RUN_IDENTITY_KEYS",
    "RUN_FEATURE_KEY",
    "RUN_IDENTITY_KEYS",
    "RUN_SCOPE_FULL",
    "RUN_SCOPE_PATCH",
    "build_run_identity",
    "calibrated_after_the_fact",
    "produced_without_pixel_size",
    "read_run_identity",
    "resolve_ran_at_nm",
    "run_identity_from_segmenter",
    "usable_pixel_size_nm",
    "utc_timestamp",
]
