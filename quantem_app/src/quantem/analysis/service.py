"""Run an analysis over a segmentation and write a reproducible export bundle.

This is the layer between the pure-numpy analysis functions and the app: it
loads masks from the database, calls the analysis, and writes results the user
can open in Excel or replot elsewhere.

Two house rules, both inherited from the Figure-4 pipeline's
``export_results.py`` and both non-negotiable:

* **Every chart reads from a table that is exported alongside it.** Nothing is
  computed only for display.
* **Every bundle carries a manifest** recording model ids and thresholds, pixel
  sizes, mask sources, seeds, replicate counts, band edges, the aggregation rule
  and every caveat flag. A number without its provenance is not reportable.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from quantem import __version__
from quantem.core.config import STORAGE_DIR
from quantem.segmentation.run_identity import (
    calibrated_after_the_fact,
    produced_without_pixel_size,
    usable_pixel_size_nm,
)

from . import provenance
from .compartments import CompartmentSet, area_fractions, assign_points
from .distances import DEFAULT_BAND_EDGES_NM, distance_to_boundary
from .montecarlo import (
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    NullResult,
    csr_null,
    self_check,
)
from .morphometrics import (
    CIRCULARITY_ESTIMATOR_NOTE,
    METRIC_KEYS,
    OBJECT_ROW_FIELDS,
    UNKNOWN_SOURCE,
    ObjectMetrics,
    count_by_source,
    density,
    derive,
    summarize,
)

if TYPE_CHECKING:  # pragma: no cover - the Django model is imported lazily below
    from .models import AnalysisRun

MANIFEST_NAME = "manifest.json"

#: Metrics :func:`~quantem.analysis.morphometrics.derive` produces that
#: ``objects.csv`` deliberately does not write, and the reason, which is carried
#: into the manifest so a reader of an older bundle can find out where a column
#: went.
#:
#: ``aspect_ratio`` and ``elongation`` were the same number under two names.
#: The extractor measures ``elongation`` as
#: ``major_axis_length / max(minor_axis_length, 1)``
#: (:mod:`quantem.seg_core.extraction`, and the hand-drawn path in
#: ``quantem.segmentation.features.geometry`` writes it the same way); ``derive``
#: then recomputed ``major / minor`` from those same two axes and called it
#: ``aspect_ratio``. On a real 38-row ``objects.csv`` the two columns were
#: string-identical on every row, and nothing in the manifest said so. The
#: obvious things to do with this file are a correlation matrix and a PCA, and
#: both of those silently double-weight an axis that appears twice. One ratio,
#: under the name the measurement is actually stored and served under.
OBJECT_COLUMNS_NOT_WRITTEN: dict[str, str] = {
    "aspect_ratio": (
        "Removed: it was major_axis_px / minor_axis_px, which is the elongation "
        "column recomputed — the two were identical on every row. Two columns "
        "of one number double-weight that axis in a correlation matrix or a "
        "PCA over this file. Use elongation."
    ),
}

#: ``ObjectMetrics`` key -> the ``objects.csv`` column it is written as.
#:
#: One entry, and it is not cosmetic. ``image_summary.csv`` has a
#: ``pixel_size_nm`` column meaning *what the image records*; the per-object one
#: means *what these values are in*. Both readings are defensible and they
#: disagree exactly when the bundle refuses to convert: ``image_summary.csv``
#: says ``5.0`` while every row of ``objects.csv`` is blank, because the objects
#: were produced before that 5.0 existed. The manifest tells the reader to join
#: the two files on ``image_key``, and a join across two same-named columns that
#: answer different questions is how a bundle gets quoted in units it is not in
#: — or, in pandas, how ``pixel_size_nm`` becomes ``NaN`` where 5.0 was
#: expected. Renamed so the collision cannot happen and the column says which
#: question it answers.
OBJECT_COLUMN_RENAMES: dict[str, str] = {
    "pixel_size_nm": "values_in_pixel_size_nm",
}

#: Columns of ``objects.csv``: every metric a row can carry that this file
#: writes, under the name it writes it as, plus the ones that trace it back to
#: the run, the segmentation and the image it came out of. Declared so the file
#: is written with a header even when there are no rows.
#:
#: Derived from :data:`~quantem.analysis.morphometrics.OBJECT_ROW_FIELDS` rather
#: than retyped, so a metric added to the measurement layer reaches the export
#: without a second edit -- and derived *through*
#: :data:`OBJECT_COLUMNS_NOT_WRITTEN` and :data:`OBJECT_COLUMN_RENAMES`, because
#: what a value is called in a table people join and correlate is an export
#: decision, not a measurement one. ``ObjectMetrics`` keeps every number it
#: measured whatever this file does with it.
#:
#: ``objects.csv`` is the file that goes into R or Prism, and it used to carry
#: no route at all to the sentences that qualify it: no run id, no segmentation
#: id, and nothing saying there were any caveats. ``n_caveats`` is the count and
#: ``image_key`` is the join to the ``image_summary.csv`` row whose ``caveats``
#: column holds the text -- the full text is not repeated on every object row,
#: because a two-kilobyte cell times ten thousand rows is a twenty-megabyte
#: file that no one reads either.
OBJECT_CSV_FIELDS: tuple[str, ...] = (
    *(
        OBJECT_COLUMN_RENAMES.get(name, name)
        for name in OBJECT_ROW_FIELDS
        if name not in OBJECT_COLUMNS_NOT_WRITTEN
    ),
    "image_key",
    "group",
    "segmentation_id",
    "analysis_run_id",
    "n_caveats",
)


def object_csv_row(metrics: ObjectMetrics) -> dict[str, Any]:
    """One object's ``objects.csv`` row: :meth:`ObjectMetrics.as_row` renamed.

    The measurement layer stores what it measured; this layer decides what the
    exported table is allowed to call it and what it must not offer twice. See
    :data:`OBJECT_COLUMN_RENAMES` and :data:`OBJECT_COLUMNS_NOT_WRITTEN`.
    """
    return {
        OBJECT_COLUMN_RENAMES.get(key, key): value
        for key, value in metrics.as_row().items()
        if key not in OBJECT_COLUMNS_NOT_WRITTEN
    }


#: Keys ``objects.summary`` must not describe as a distribution, and why.
#:
#: ``pixel_size_nm`` is not a measurement of the objects. It is one constant
#: written once per row, so summarising it produced ``mean 5.0, sd 0.0,
#: median 5.0, iqr 0.0, min 5.0, max 5.0`` in a table of morphometrics — a
#: measured spread of the microscope's calibration, sitting between the Feret
#: diameter and the mean probability, with an ``n`` that invites "the pixel size
#: was 5.0 ± 0.0 nm over 38 objects". The Analysis screen already refused to
#: draw the row (``ObjectsPanel.HIDDEN_METRICS``); the JSON, the manifest's
#: ``partially_measured_metrics`` and the caveat that counts how many metrics
#: there are did not. What the value actually is stays in the bundle three
#: times: ``pixel_size_nm`` and ``calibrated`` on this result,
#: ``objects.values_in_pixel_size_nm`` beside this summary, and the
#: ``values_in_pixel_size_nm`` column of ``objects.csv``.
NOT_A_DISTRIBUTION: dict[str, str] = {
    "pixel_size_nm": (
        "one constant repeated per object, not a measured spread; see "
        "objects.values_in_pixel_size_nm"
    ),
}


def summary_metric_keys(metrics: list[ObjectMetrics]) -> list[str]:
    """Which metrics ``objects.summary`` describes, in export order.

    Mirrors :func:`~quantem.analysis.morphometrics.summarize`'s own default
    ordering and then drops the keys that are not distributions of anything:
    the duplicated ratio (:data:`OBJECT_COLUMNS_NOT_WRITTEN`) and the constant
    (:data:`NOT_A_DISTRIBUTION`). Passed explicitly rather than filtered
    afterwards, so no mean of a constant is ever computed.
    """
    present = {key for m in metrics for key in m.values}
    present -= set(OBJECT_COLUMNS_NOT_WRITTEN)
    present -= set(NOT_A_DISTRIBUTION)
    ordered = [key for key in METRIC_KEYS if key in present]
    return ordered + sorted(present - set(METRIC_KEYS))


#: The columns of ``image_summary.csv`` that every run has. The rest --
#: ``area_fraction_<name>``, ``enrichment_<name>``, ``z_<key>`` -- are named
#: after the user's own compartments and can only be discovered from the rows.
IMAGE_SUMMARY_BASE_FIELDS: tuple[str, ...] = (
    "image_key",
    "segmentation_id",
    "analysis_run_id",
    "group",
    "pixel_size_nm",
    "calibrated",
    "n_objects",
    "n_hand_drawn",
    "n_model_produced",
    "n_rejected",
    "reviewed_fraction",
    "tissue_px",
    "tissue_um2",
    "objects_per_um2",
)

#: Written last, after every metric, so they never push a number off the first
#: screen -- and written *here*, in the file people open, rather than only in
#: ``manifest.json``. ``enrichment_mito = 9.456`` in a spreadsheet cell looks
#: exactly like a result whether or not it is circular by construction, and the
#: warning that says it is not was two files away.
IMAGE_SUMMARY_LAST_FIELDS: tuple[str, ...] = (
    "circular_columns",
    "n_caveats",
    "caveats",
)

#: What a column of an exported CSV means, where the name alone is not enough.
#:
#: The manifest listed every column by name and defined none of them. Several
#: needed it: one appears in both files under one name meaning two things, one
#: is a ratio whose definition decides whether a second column was a duplicate,
#: and one — ``circularity`` — is not ambiguous at all but is measured with a
#: biased estimator, which is worse, because nothing about the column looks
#: wrong. Written where the column list already is, so a reader who opens the
#: manifest to find out what they have finds the answer in the same place.
COLUMN_NOTES: dict[str, dict[str, str]] = {
    "objects.csv": {
        "values_in_pixel_size_nm": (
            "The scale this row's micron columns were converted with, blank "
            "when nothing was converted. Not the same question as the "
            "pixel_size_nm column of image_summary.csv, which is what the image "
            "records: the two disagree whenever this bundle refuses a "
            "conversion, and blank here beside 5.0 there is the expected "
            "reading of an image calibrated after its objects were produced, "
            "not a lost value. It equals image_summary.csv's pixel_size_nm on "
            "every row where that file's calibrated column is true, and is "
            "blank on every row where it is false."
        ),
        "elongation": (
            "major_axis_px / max(minor_axis_px, 1), as measured by whichever "
            "path produced the object. The only aspect-ratio column in this "
            "file; see columns_not_written."
        ),
        "calibrated": (
            "Whether this row's values are in physical units. It is not "
            "'whether the image has a pixel size' — an image calibrated after "
            "its objects were produced records one and is still false here."
        ),
        "in_reviewed_area": (
            "True for an object inside a region a person marked as reviewed. "
            "Blank means no completed area is recorded at all, which is 'nobody "
            "said', not 'outside'."
        ),
        # The one column in this file whose *estimator* is biased rather than
        # whose coverage is partial. Nothing about the column is missing, so
        # nothing else in the bundle would have flagged it here: a reader who
        # opens objects.csv, means the circularity column and compares two
        # groups has done everything right and can still get a shape result out
        # of a pure size difference. The paragraph is the same one the summary
        # carries as estimator_note; it is repeated here because this is the
        # file the number is read out of.
        "circularity": CIRCULARITY_ESTIMATOR_NOTE,
    },
    "image_summary.csv": {
        "pixel_size_nm": (
            "What the image records now, reported unchanged even when nothing "
            "was converted with it, so a refused or wrong value can be found "
            "and fixed. What the numbers are actually in is the calibrated "
            "column beside it and the values_in_pixel_size_nm column of "
            "objects.csv."
        ),
    },
}

#: ``objects.csv`` columns that are exact functions of other columns in the same
#: row, and of what.
#:
#: Not a warning about the numbers: every one of them is correct and each is
#: worth reporting. It is a warning about treating the file as a matrix of
#: independent variables, which is what a correlation heatmap or a PCA over
#: "all the morphometrics" does. The exact-duplicate case has been removed (see
#: :data:`OBJECT_COLUMNS_NOT_WRITTEN`); these are the remaining exact
#: dependencies, and the manifest names them rather than leaving each reader to
#: rediscover that ``equivalent_diameter_px`` is a monotone transform of
#: ``area_px``.
OBJECT_CSV_DERIVED_COLUMNS: dict[str, str] = {
    "elongation": "major_axis_px / max(minor_axis_px, 1)",
    "circularity": "area_px and perimeter_px",
    "equivalent_diameter_px": "area_px alone (a monotone transform of it)",
    "area_um2": "area_px and values_in_pixel_size_nm",
    "perimeter_um": "perimeter_px and values_in_pixel_size_nm",
    "major_axis_um": "major_axis_px and values_in_pixel_size_nm",
    "minor_axis_um": "minor_axis_px and values_in_pixel_size_nm",
    "feret_max_um": "feret_max_px and values_in_pixel_size_nm",
    "equivalent_diameter_um": "equivalent_diameter_px and values_in_pixel_size_nm",
}

#: Every export bundle lives under one directory, one subdirectory per run.
#: Inside the user data directory, never inside the installation.
EXPORTS_DIR = STORAGE_DIR / "exports"

#: Progress callback: ``(percent, message)``. The job layer passes one that also
#: checks for cancellation, which is why it may raise.
ProgressFn = Callable[[float, str], None]


@dataclass
class AnalysisInputs:
    """Everything one analysis run needs, already loaded into memory."""

    image_key: str
    pixel_size_nm: float | None
    compartments: CompartmentSet
    #: Per-object stored features, keyed by object id.
    object_features: dict[str, dict[str, Any]]
    #: ``SegmentObject.source_model`` per object id: ``"manual"`` for a polygon
    #: a person drew, otherwise the model that produced it. Optional, because
    #: :func:`run_analysis` is usable from a notebook with nothing but a feature
    #: dict; an object with no entry is reported as being of unrecorded origin
    #: rather than assumed to be either.
    object_sources: dict[str, str] = field(default_factory=dict)
    #: The scale each model pack behind these objects is trained and applied at,
    #: keyed by pack id: the pack's ``canonical_nm``, or ``None`` for a pack that
    #: declares none and genuinely runs native. Filled by
    #: :func:`quantem.analysis.loaders.load_inputs`. Empty from a notebook, which
    #: suppresses the scale caveat below rather than inventing one.
    canonical_nm_by_pack: dict[str, float | None] = field(default_factory=dict)
    #: Packs behind these objects whose spec this build cannot look up at all.
    #:
    #: The other half of :func:`quantem.analysis.loaders._canonical_nm`'s return
    #: value, which used to be discarded here. ``None`` in
    #: ``canonical_nm_by_pack`` means two different things -- "this pack declares
    #: no canonical scale and genuinely runs native" and "this build has never
    #: heard of this pack" -- and only the first is a reason to stay quiet. An
    #: unrecognised pack is listed here as well, so the falsy ``None`` beside it
    #: cannot be read as the harmless case.
    #: :func:`quantem.analysis.loaders._packs_that_skipped_a_resample` takes the
    #: same view for the manifest caveat; the two sites must not disagree about
    #: the same run.
    unrecognised_packs: frozenset[str] = field(default_factory=frozenset)
    #: The distinct ``native_pixel_size_nm`` values stamped on the objects by the
    #: runs that produced them -- ``None`` among them meaning "this object was
    #: produced while the image had no pixel size".
    #:
    #: Separate from the image's *current* pixel size on purpose. Setting a pixel
    #: size after an uncalibrated run is the one thing the app actively
    #: recommends ("Set the image's pixel size and re-run inference"), and a user
    #: who did only the first half got a bundle that had quietly become
    #: ``calibrated: True`` with micron columns, working distances and the
    #: wrong-scale caveat gone -- on objects a differently-scaled model had
    #: produced. Empty from a notebook or from objects made before stamping.
    produced_pixel_size_nm: frozenset[float | None] = field(default_factory=frozenset)
    #: How much of the image a person actually reviewed, if it is known:
    #: ``(reviewed_px, image_px)``. See
    #: :func:`quantem.analysis.loaders.reviewed_area`.
    reviewed_px: tuple[int, int] | None = None
    #: Whether each object is inside a region a person marked as reviewed,
    #: keyed by object id. ``None`` -- the whole mapping, not an entry -- when
    #: no completed area is recorded, because "nobody said" is not "outside".
    #: Becomes the ``in_reviewed_area`` column of ``objects.csv``, which is the
    #: distinction the reviewed-fraction caveat says the file could not make.
    object_in_reviewed_area: dict[str, bool] | None = None
    #: The completed regions themselves, in the ``provenance.section`` shape:
    #: ``regions_wkt`` plus its note, or an ``unavailable`` reason. ``None`` when
    #: nothing was marked complete. An area and a bounding box cannot be turned
    #: back into which pixels a person went through; this can.
    reviewed_regions: dict[str, Any] | None = None
    #: The segmentation these objects came out of, so ``objects.csv`` can name
    #: it. Empty from a notebook, which has no database row to point at.
    segmentation_id: str = ""
    #: Candidates a person looked at and threw away. Not zero when unknown --
    #: ``None`` means nobody recorded a rejection, which is a different fact.
    n_rejected: int | None = None
    #: Optional point set in image pixel coordinates, shape (N, 2).
    points_xy: np.ndarray | None = None
    #: Which compartment the distance analysis measures against.
    distance_target: str | None = None
    band_edges_nm: tuple[float, ...] = DEFAULT_BAND_EDGES_NM
    replicates: int = DEFAULT_REPLICATES
    seed: int = DEFAULT_SEED
    group: str = ""


#: Why "re-run inference" is not, by itself, the way out of a wrongly-scaled
#: object set -- and what is.
#:
#: The app told users to "Set the image's pixel size and re-run inference". A
#: user did exactly that. The run completed SUCCESS and returned
#: ``segment_count: 0`` with "Nothing changed: the 41 object(s) you have already
#: labelled here are exactly as they were."
#:
#: That behaviour is right:
#: :func:`quantem.seg_core.db.extraction.extract_and_save_segments` drops a
#: candidate overlapping a CONFIRMED object by >=30% or an EXCLUDED one by >=80%,
#: which is what stops a re-run destroying a day of proofreading. The
#: consequence is that the advice is a **no-op on exactly the bundles that carry
#: it**: every object measured here is CONFIRMED
#: (:func:`quantem.analysis.loaders.confirmed_objects`), so a re-run cannot
#: replace one, ``native_pixel_size_nm: null`` stays on them for good, and the
#: next bundle repeats this caveat. The user followed the instruction, got a
#: green success and was no further forward.
#:
#: So the instruction now says what has to happen first, and names the control
#: that does it. It used to name the endpoint instead --
#: ``POST /api/segmentations/<segmentation_id>/labels/clear`` -- and add "No
#: screen offers that yet". That was three of invariant I-12's classes at once
#: (an HTTP verb, an API route, a ``<placeholder>`` with nowhere to type it) in
#: a sentence that lands in an analysis bundle a biologist reads, and it stopped
#: being true when the labeling header gained its **Discard objects and
#: re-run...** button. A second segmentation of the same organelle on the same
#: image is still refused by ``unique_segmentation_per_asset``, so "make a new
#: one" is not a route either.
RERUN_NEEDS_THE_OBJECTS_GONE = (
    "Re-running inference is not by itself enough. Every object measured here "
    "is one somebody confirmed, and a new candidate that lands on an object "
    "already confirmed or excluded is dropped rather than saved — that is what "
    "stops a re-run undoing proofreading — so the run completes, reports no new "
    "objects, and leaves these ones, and their record of having been produced "
    "without a pixel size, exactly as they were. The objects have to go first. "
    "Discard objects and re-run, on the labeling screen, deletes this "
    "segmentation's confirmed and excluded objects and runs the model again; "
    "what that produces is stamped with the pixel size the image now has. "
    "Re-importing the image with its pixel size set before any organelle is "
    "run avoids the situation altogether."
)


def _scale_caveat(
    canonical_nm_by_pack: dict[str, float | None],
    *,
    calibrated_since: bool = False,
    pixel_size_nm: float | None = None,
    unrecognised_packs: Iterable[str] = (),
) -> str | None:
    """What an uncalibrated image did to the *dimensionless* numbers.

    The units guard is honest and incomplete. It blanks every ``_um``/``_um2``
    column, ``tissue_um2`` and ``objects_per_um2``, and says so -- and then
    ``n_objects``, ``area_fraction_*``, ``enrichment_*`` and ``z_*`` are printed
    in full beside them with nothing attached, because none of them has a unit
    to blank.

    They are not therefore unaffected. Six of the eight released packs declare a
    ``canonical_nm`` and resample the image to it before inference; that needs a
    pixel size, so an uncalibrated image is not resampled and the model runs at
    whatever scale the pixels happen to be. It is a different run, and the
    number it moves most is the count. On one 1400 px image the same pixels gave
    six objects imported untagged and three with 5 nm typed in -- the area
    fraction moved by 8%, the object count by half. The count is what goes in a
    bar chart, and it was the one with no warning attached.

    ``calibrated_since`` is the branch where a pixel size was typed in *after*
    the run. There the units guard is not incomplete but silent: it has a number
    to convert with, so it converts. :func:`run_analysis` now withholds that
    number under exactly the condition this function returns a sentence for --
    ``trained`` or ``unrecognised`` non-empty -- so the two halves of the bundle
    say the same thing.

    ``unrecognised_packs`` is the third state, and it used to be silently folded
    into the harmless one. ``canonical_nm_by_pack`` carries ``None`` both for a
    pack that declares no canonical scale (runs native either way, nothing was
    skipped, nothing to warn about) and for a pack this build cannot look up at
    all. Only the first is a reason to stay quiet: an unlookupable scale is not
    evidence of a scale that does not exist. ``_packs_that_skipped_a_resample``
    already says so in the manifest, in as many words, and this is the site that
    disagreed with it.
    """
    trained = {pack_id: nm for pack_id, nm in sorted(canonical_nm_by_pack.items()) if nm}
    unrecognised = sorted(set(unrecognised_packs) - set(trained))
    if not trained and not unrecognised:
        return None
    one = len(trained) == 1
    spelled = ", ".join(f"{pack_id} is applied at {nm} nm/px" for pack_id, nm in trained.items())
    if not trained:
        # Nothing to spell out: we cannot say what scale these packs run at,
        # which is the whole problem. Deliberately its own short sentence rather
        # than a variant of the two below, both of which quote a canonical scale
        # and would have to invent one here.
        many = len(unrecognised) > 1
        names = ", ".join(unrecognised)
        recalibrated = (
            f" A pixel size of {pixel_size_nm} nm/px has been set since, and "
            "setting it does not re-run inference, so it has not been used for "
            "anything in this bundle: every micron value, density and distance "
            "is blank."
            if calibrated_since
            else ""
        )
        return (
            f"{names} produced objects here and "
            f"{'are' if many else 'is'} not "
            f"{'packs' if many else 'a pack'} this build knows, so the scale "
            f"{'they run' if many else 'it runs'} at cannot be looked up and "
            "whether the image should have been resampled before inference "
            "cannot be determined. This image had no pixel size when those "
            f"objects were produced, so no resample could have happened."
            f"{recalibrated} n_objects, area_fraction_*, enrichment_* and z_* "
            "are dimensionless and are reported in full, but they were measured "
            "on whatever object set that unresampled run produced, and the "
            "count is the number a wrong scale moves most. An unlookupable "
            "scale is a reason to check, not a reason to assume nothing "
            "happened: confirm which build produced "
            f"{'these packs' if many else 'this pack'} before reporting any "
            f"count, area fraction, density or distance. "
            f"{RERUN_NEEDS_THE_OBJECTS_GONE}"
        )
    # Packs of both kinds behind one object set. The sentences below can only
    # spell out the packs whose scale is on record, so the rest are named here
    # rather than dropped -- a reader who chases the named packs and finds them
    # accounted for would otherwise conclude the run was understood.
    also_unknown = (
        (
            f" {', '.join(unrecognised)} also produced objects here and "
            f"{'are' if len(unrecognised) > 1 else 'is'} not known to this "
            "build at all, so whether "
            f"{'they' if len(unrecognised) > 1 else 'it'} should have been "
            "resampled cannot even be looked up."
        )
        if unrecognised
        else ""
    )
    if calibrated_since:
        # The dangerous state, and the one the app's own advice produces when a
        # user follows half of it: the objects were made with no pixel size and a
        # number was typed in afterwards. Every downstream guard used to key on
        # the *current* value, so the micron columns filled in, the distances
        # started working and `calibrated` read True -- over an object set a
        # model produced at the wrong scale. Nothing about the objects changed,
        # so nothing is converted now; what is left is the dimensionless half,
        # which no units guard can blank and which this sentence has to carry.
        return (
            f"These objects were produced while this image had no pixel size, and "
            f"a pixel size of {pixel_size_nm} nm/px has been set since. Setting it "
            "does not re-run inference, so the object set is still the one "
            f"produced without it. {spelled}, and "
            f"{'that resample' if one else 'those resamples'} never happened — the "
            "model saw the pixels at whatever scale they happened to be. Every "
            "micron value, density and distance in this bundle is therefore "
            "blank, as it is for an image with no pixel size at all: converting "
            "a wrongly-scaled object set into real units produces a bundle no "
            "reader can tell from a correct one. What is not blank is "
            "n_objects, area_fraction_*, enrichment_* and z_*, which are "
            "dimensionless and were measured on that same object set. The count "
            "is the most sensitive of them, and it does not merely shift: on one "
            "image the same pixels yielded 0, 19, 120 and 233 objects at "
            "5 nm/px, unset, 10 nm/px and 20 nm/px. Do not report any of this "
            "until inference has run again at the pixel size this image now "
            f"records. {RERUN_NEEDS_THE_OBJECTS_GONE}{also_unknown}"
        )
    return (
        f"Inference ran at a scale {'this model was' if one else 'these models were'} "
        f"not trained for. {spelled}, and the image would have been resampled to "
        f"{'that scale' if one else 'each of those scales'}; resampling needs a "
        "pixel size and this image has none, so the model saw the pixels at "
        "whatever scale they happen to be. This is not only a units problem. "
        "n_objects, area_fraction_*, enrichment_* and z_* are dimensionless, so "
        "the units guard above does not blank them and they are reported in "
        "full — but they were measured on the object set that scale produced. "
        "The count is the most sensitive of them, and it does not merely "
        "shift: on one image the same pixels yielded 0, 19, 120 and 233 "
        "objects at 5 nm/px, unset, 10 nm/px and 20 nm/px. Treat that as "
        "a demonstration that the count is not recoverable from a "
        "wrongly-scaled run, not as a bound. Set the image's pixel size and re-run "
        "inference before reporting any of these: setting it afterwards "
        "converts the units and leaves the object set as it is. "
        f"{RERUN_NEEDS_THE_OBJECTS_GONE}{also_unknown}"
    )


def _proofreading(inputs: AnalysisInputs, *, n_confirmed: int) -> dict[str, Any]:
    """What the human did: how much was thrown away, and how much was looked at.

    The bundle already says how many objects a person confirmed and how many
    they drew. Two things it could not say, and both change how a count reads:

    * **What was rejected.** ``n_confirmed_objects`` is the survivors. The Adapt
      wizard shows a REJECTED count off the same rows; the analysis manifest
      did not, so a set where 14 of 42 candidates were thrown away and one where
      nothing was ever rejected were written identically.
    * **What was reviewed.** Counts and fractions are over the whole image
      whatever fraction of it a person actually went through. A completed area
      covering 84--1316 px of a 1400 px image is 80% of it, and the other 20% is
      raw model output being reported at the same confidence as the rest.

    Both are ``None`` with a reason when nothing recorded them. Zero is a
    measurement ("a person rejected nothing"); ``None`` is the absence of one.
    """
    values: dict[str, Any] = {"n_confirmed_objects": n_confirmed}
    unavailable: dict[str, str] = {}

    if inputs.n_rejected is None:
        unavailable["n_rejected"] = (
            "How many candidates were reviewed and rejected is not known to this "
            "run: it is read from the segmentation's own objects, and this "
            "analysis was assembled from a feature dict without one."
        )
    else:
        values["n_rejected"] = inputs.n_rejected

    if inputs.reviewed_px is None:
        for key in ("reviewed_px", "image_px", "reviewed_fraction"):
            unavailable[key] = (
                "No completed area is recorded for this segmentation, so how "
                "much of the image a person actually reviewed is unknown. Every "
                "count and fraction in this bundle is over the whole image "
                "either way."
            )
    else:
        reviewed, image_px = inputs.reviewed_px
        values["reviewed_px"] = reviewed
        values["image_px"] = image_px
        values["reviewed_fraction"] = (reviewed / image_px) if image_px else None
        if not image_px:
            unavailable["reviewed_fraction"] = (
                "The image has no area, so the reviewed fraction is 0/0."
            )

    # The regions themselves, beside the fraction they produced. Without them a
    # reader has "80% was reviewed" and no way to say *which* 80%, so nothing
    # downstream can be recomputed over the reviewed part alone.
    if inputs.reviewed_regions is not None:
        for key, value in inputs.reviewed_regions.items():
            if key == "unavailable":
                unavailable.update(value)
            else:
                values[key] = value
    elif inputs.reviewed_px is not None:  # pragma: no cover - loader sets both
        unavailable["regions_wkt"] = (
            "A reviewed area was measured but its geometry was not passed to "
            "this run, so which pixels it covers is not recorded here."
        )

    values["note"] = (
        "Rejected candidates are in no count, area fraction or density in this "
        "bundle; they are the record of the review that produced the confirmed "
        "set. The reviewed area is the polygon a person marked complete — "
        "everything outside it is model output nobody has been through, and it "
        "is measured here on the same footing as the rest."
    )
    return provenance.section(values, unavailable)


def _proofreading_caveats(proofreading: dict[str, Any], inputs: AnalysisInputs) -> list[str]:
    """The two proofreading facts that belong in front of the reader, not in JSON."""
    out: list[str] = []
    rejected = proofreading.get("n_rejected")
    if rejected:
        confirmed = proofreading["n_confirmed_objects"]
        out.append(
            f"{rejected} candidate object{'' if rejected == 1 else 's'} in this "
            f"segmentation {'was' if rejected == 1 else 'were'} reviewed and "
            f"rejected. {'It is' if rejected == 1 else 'They are'} in no count, "
            "area fraction or density here — this bundle measures the "
            f"{confirmed} confirmed object{'' if confirmed == 1 else 's'} only — "
            "but how much a person threw away is part of what produced that set."
        )

    fraction = proofreading.get("reviewed_fraction")
    if fraction is not None and fraction < 1.0:
        reviewed = proofreading["reviewed_px"]
        total = proofreading["image_px"]
        out.append(
            f"A person marked {fraction:.0%} of this image as reviewed "
            f"({reviewed:,} of {total:,} pixels). Every count, area fraction and "
            "density in this bundle is over the whole image, including the "
            f"{1 - fraction:.0%} that was never gone through, where the objects "
            "are unreviewed model output. objects.csv separates them: its "
            "in_reviewed_area column is true for the objects inside the "
            "area a person went through. These whole-image totals do not."
        )
    elif proofreading.get("reviewed_px") is None and inputs.object_features:
        out.append(
            "No area of this image is recorded as reviewed, so the bundle "
            "cannot say how much of it a person went through. Every count and "
            "fraction covers the whole image regardless."
        )
    return out


def _flat_null_caveats(null: NullResult, n_on_tissue: int) -> list[str]:
    """What a Monte-Carlo null with no spread does and does not support.

    ``z`` has always been blank for these and the screen says a blank z means
    the null had zero spread. The *p* beside it was not blank: it read 0.048,
    which is the smallest number twenty replicates can produce and is the first
    thing anyone compares against 0.05. Both are blank now, and a blank has to
    be explained where the reader is, not only in a docstring.
    """
    undefined = [key for key, sd in null.null_sd.items() if not sd]
    if not undefined:
        return []

    names = ", ".join(sorted(undefined))
    one = len(undefined) == 1
    if null.replicates < 2:
        return [
            f"The Monte-Carlo null ran {null.replicates} replicate, so it has "
            f"no spread to measure and no z or p is reported for {names}. Ask "
            "for more replicates — the empirical p can never be smaller than "
            "1 / (replicates + 1) — or read the observed value on its own."
        ]
    floor = 1.0 / (null.replicates + 1)
    return [
        f"All {null.replicates} Monte-Carlo replicates returned the same value "
        f"for {names}, so the null has no spread and neither a z nor a p is "
        f"reported for {'it' if one else 'them'}. An empirical p against a "
        f"flat null is not a test: it would have been {floor:.4g} "
        f"(= 1 / ({null.replicates} + 1), the smallest value this many "
        "replicates can produce, and the one people read as significant) "
        "whenever the observed value differed from the null at all, and 1.0 "
        f"when it did not. {n_on_tissue} point"
        f"{'' if n_on_tissue == 1 else 's'} on the tissue is the usual cause: "
        "too few for a simulated draw ever to land in a small compartment, and "
        "a single point has no spatial structure to detect."
    ]


def run_analysis(inputs: AnalysisInputs) -> dict[str, Any]:
    """Compute every enabled analysis for one image. Pure; writes nothing."""
    caveats: list[str] = []

    # A recorded pixel size that is not a size is a different problem from an
    # absent one, and used to be reported as one ("Pixel size is not set for
    # this image"). The numbers were nearly safe -- morphometrics refused a
    # non-positive scale -- but area_fractions squared it, so -5 nm/px produced
    # the micron areas of +5 nm/px, and the sentence explaining them was false.
    recorded_nm = inputs.pixel_size_nm
    usable_nm = usable_pixel_size_nm(recorded_nm)

    # Fires when the *objects* were produced without a pixel size, whether or
    # not one has been entered since. Gating this on the asset's current value
    # meant typing a number afterwards deleted the warning about the run that
    # made the objects.
    #
    # Both predicates come from quantem.segmentation.run_identity rather than
    # being spelled out here, because the labeling screen has to answer the same
    # question before an analysis is ever run
    # (``ImageSegmentationSerializer.objects_pixel_size``). A screen that says
    # the objects are fine and a bundle that blanks every micron column is a
    # disagreement nobody sees until they compare the two.
    ran_uncalibrated = produced_without_pixel_size(inputs.produced_pixel_size_nm)
    calibrated_since = calibrated_after_the_fact(
        produced_pixel_size_nm=inputs.produced_pixel_size_nm,
        recorded_pixel_size_nm=recorded_nm,
    )
    # ...and the same condition decides the *numbers*, not only the sentence.
    # A pack that declares a canonical scale resamples the image to it before
    # inference; that needs a pixel size, so a run made without one skipped the
    # resample and produced a different object set. Converting that set into
    # microns with the number typed in afterwards is arithmetic on the wrong
    # objects, and the caveat below has always said so in as many words -- while
    # every column it describes was filled in from the asset's current value.
    #
    # This is exactly `_scale_caveat`'s own gate, so the sentence and the blanks
    # cannot disagree: a pack with no canonical_nm runs native either way
    # (`resample.resample_factor` returns 1.0 with or without a pixel size), so
    # its object set is the same one and its microns are real.
    #
    # `any(...)` on its own was not that gate. `loaders._canonical_nm` returns
    # `(None, False)` for a pack this build does not recognise, `None` is falsy,
    # and the conversion went ahead -- on an object set whose scale nobody can
    # look up. `loaders._packs_that_skipped_a_resample` takes the opposite view
    # for the manifest caveat, and it is the right one: what a pack would have
    # done cannot be looked up, and an unlookupable scale is a reason to warn,
    # not a reason to stay quiet. Unreachable from the shipped UI, which cannot
    # produce such a pack id, and still not somewhere the two may disagree.
    resample_was_skipped = calibrated_since and (
        any(inputs.canonical_nm_by_pack.values()) or bool(inputs.unrecognised_packs)
    )
    # The scale things are *converted with*, which is no longer the same
    # question as what the image records. `recorded_nm` still goes into the
    # pixel_size_nm column: a reader has to be able to see the value that was
    # refused.
    pixel_size_nm = None if resample_was_skipped else usable_nm

    if recorded_nm is not None and usable_nm is None:
        caveats.append(
            f"The pixel size recorded for this image is {recorded_nm} nm/px, "
            "which is not a length: a pixel cannot be zero or negative "
            "nanometres across. It is reported unchanged in the pixel_size_nm "
            "column so the bad value can be found and fixed, and nothing in "
            "this bundle has been converted with it — every physical unit is "
            "blank, exactly as for an image with no pixel size at all. Correct "
            "the calibration on the image and run the analysis again."
        )
    elif resample_was_skipped:
        # Not "pixel size is not set" -- it is set, and that is the problem. The
        # short sentence that pairs with the blank columns; the caveat below
        # says why the objects, not the number, are what cannot be converted.
        caveats.append(
            "Areas and distances are in pixels and have not been converted, "
            f"even though this image records {recorded_nm} nm/px: the objects "
            "were produced before it was set. Every physical unit is blank and "
            "calibrated is false, exactly as for an image with no pixel size at "
            "all — calibrated says whether this row's values are in physical "
            "units, not whether the image has been calibrated since."
        )
    elif usable_nm is None:
        caveats.append(
            "Pixel size is not set for this image: areas and distances are in "
            "pixels and cannot be converted to physical units."
        )
    if usable_nm is None or ran_uncalibrated:
        scale_caveat = _scale_caveat(
            inputs.canonical_nm_by_pack,
            calibrated_since=calibrated_since,
            pixel_size_nm=usable_nm,
            unrecognised_packs=inputs.unrecognised_packs,
        )
        if scale_caveat:
            caveats.append(scale_caveat)
    if inputs.compartments.tissue is None:
        caveats.append(
            "No tissue mask was supplied: fractions are relative to the whole "
            "image, including any empty resin or background."
        )

    areas = area_fractions(inputs.compartments, pixel_size_nm=pixel_size_nm)

    # A tissue mask that was supplied but is empty makes every fraction below a
    # ratio to zero, and makes the Monte-Carlo null unsamplable -- there is
    # nowhere to scatter a point. Named here, where it is produced, and skipped
    # below rather than left to raise out of numpy.
    tissue_is_empty = areas.tissue_px == 0
    if tissue_is_empty:
        caveats.append(
            "The tissue mask is empty: the chosen segmentation has no confirmed "
            "objects, so there is no area to measure against. Every area and "
            "point fraction is zero, and the Monte-Carlo null and its "
            "self-check were skipped."
        )

    reviewed_by_object = inputs.object_in_reviewed_area
    metrics = [
        derive(
            f,
            object_id=oid,
            pixel_size_nm=pixel_size_nm,
            source_model=inputs.object_sources.get(oid, UNKNOWN_SOURCE),
            # None for the whole run when nothing was ever marked complete, so
            # the column is blank rather than a column of False.
            in_reviewed_area=(None if reviewed_by_object is None else reviewed_by_object.get(oid)),
        )
        for oid, f in inputs.object_features.items()
    ]
    result: dict[str, Any] = {
        "image_key": inputs.image_key,
        "segmentation_id": inputs.segmentation_id,
        "group": inputs.group,
        # What the image says, not what was used: a reader who sees a blank
        # tissue_um2 next to pixel_size_nm = -5.0 can go and fix the image.
        "pixel_size_nm": recorded_nm,
        "calibrated": pixel_size_nm is not None,
        "proofreading": _proofreading(inputs, n_confirmed=len(metrics)),
        "composition": {
            "tissue_px": areas.tissue_px,
            "tissue_um2": areas.tissue_um2,
            "area_fractions": areas.fractions,
            "areas_px": areas.areas_px,
            "areas_um2": areas.areas_um2,
        },
        "objects": {
            "n": len(metrics),
            # Distributions only. The scale is a constant and is reported as one
            # on the next line rather than as a mean with a spread and an n --
            # see NOT_A_DISTRIBUTION and OBJECT_COLUMNS_NOT_WRITTEN.
            "summary": summarize(metrics, keys=summary_metric_keys(metrics)),
            # What every micron value in `summary` is in, or None when nothing
            # was converted. Not `pixel_size_nm` above: that is what the image
            # records, which is not the same question and is not the same
            # answer for an image calibrated after its objects were produced.
            "values_in_pixel_size_nm": pixel_size_nm,
            # The split every partly-populated metric refers back to. A reader
            # who sees "n=4 of 90" needs this in the same payload, not in a
            # different file.
            "by_source": count_by_source(metrics),
            "density": density(
                len(metrics),
                tissue_area_px=areas.tissue_px,
                pixel_size_nm=pixel_size_nm,
            ),
        },
        "caveats": caveats,
    }
    caveats.extend(_proofreading_caveats(result["proofreading"], inputs))

    # A metric measured on a subset of the objects is the easiest number in the
    # bundle to misquote: it sits in the same table, in the same units, next to
    # metrics that cover everything. Its own note says so, and so does the
    # caveat list, because those are the two places a reader looks.
    summary = result["objects"]["summary"]
    # The estimator note is not a coverage note. It was reaching the bundle only
    # when at least one circularity value had been blanked, so a run whose
    # objects all cleared the ceiling shipped a full circularity column with no
    # word of the bias -- and the bias is monotone in size and does not cancel
    # between groups. Scaling eight real mitochondrial outlines to 0.6x, a pure
    # size change with identical shapes, moved mean circularity 0.619 -> 0.641,
    # paired t = 3.596, p = 0.0088: a publishable "mitochondria became more
    # circular after treatment" out of a correct segmentation and a silent
    # bundle. Whenever the column is populated at all, the note goes with it.
    estimator_notes = [
        key
        for key, stats in summary.items()
        if stats.get("estimator_note") and (stats.get("n") or 0) > 0
    ]
    for key in estimator_notes:
        caveats.append(f"{key}: {summary[key]['estimator_note']}")

    partial = [key for key, stats in summary.items() if stats.get("n_missing")]
    if partial:
        one = len(partial) == 1
        caveats.append(
            f"{len(partial)} of {len(result['objects']['summary'])} metrics "
            f"{'is' if one else 'are'} measured on fewer than the {len(metrics)} "
            f"confirmed objects ({', '.join(partial)}). "
            f"{'It carries' if one else 'Each carries'} its own n and the reason "
            f"for it in the summary table; {'it is not' if one else 'none of them is'} "
            "a whole-image number."
        )

    if inputs.points_xy is not None and len(inputs.points_xy):
        assignment = assign_points(inputs.points_xy, inputs.compartments, areas=areas)
        result["points"] = {
            "n_total": assignment.n_total,
            "n_on_tissue": assignment.n_on_tissue,
            "n_off_tissue": assignment.n_off_tissue,
            "n_unreadable": assignment.n_unreadable,
            "n_out_of_bounds": assignment.n_out_of_bounds,
            "counts": assignment.counts,
            "fractions": assignment.fractions,
            "enrichment": assignment.enrichment,
        }
        if assignment.n_off_tissue:
            caveats.append(
                f"{assignment.n_off_tissue} of {assignment.n_total} points fell "
                "outside the tissue mask and were excluded from every fraction."
            )
        # The loader names the CSV line numbers when it can. This is the
        # backstop for every other route in -- centroids of a degenerate
        # polygon, a notebook caller's own array -- and it is what stops an
        # unreadable coordinate becoming a real observation at pixel (0, 0).
        if assignment.n_unreadable:
            caveats.append(
                f"{assignment.n_unreadable} of {assignment.n_total} points had "
                "a coordinate that is not a position (missing, or infinite) and "
                "were dropped before any measurement. They are in no count, "
                "fraction or enrichment here, and they are not counted as "
                "off-tissue either: a point that cannot be read is nowhere, not "
                "outside something."
            )
        if assignment.n_out_of_bounds:
            caveats.append(
                f"{assignment.n_out_of_bounds} of {assignment.n_total} points "
                "lie outside the image and were clipped onto its border, which "
                "is where they are counted — in the compartment counts here "
                "and, if a distance analysis was requested, in the distances "
                "too. Coordinates are expected in image "
                "pixels; a whole point set landing on one edge is what a CSV in "
                "nanometres, or an export from a differently cropped copy of "
                "this image, looks like. Check the units before quoting any "
                "enrichment below."
            )

        # No point on the tissue is not an enrichment of zero. Every ratio is
        # 0/0 and reported as UNDEFINED (compartments.assign_points says why);
        # the null below would be twenty identical zeros, so it is skipped
        # rather than reported as a statistic.
        no_points_on_tissue = assignment.n_on_tissue == 0
        if no_points_on_tissue and not tissue_is_empty:
            caveats.append(
                f"None of the {assignment.n_total} points is on the tissue "
                "mask, so there is no point distribution to compare against "
                "the tissue's geometry. Every enrichment is undefined rather "
                "than zero — a zero would read as maximal depletion, which is "
                "a finding, and there is no measurement here at all. The "
                "Monte-Carlo null was skipped for the same reason. If the "
                "points came from a tool that works in nanometres, or from a "
                "differently cropped export, they are in the wrong coordinates."
            )

        # A requested distance analysis that cannot run has to say so by name.
        # It used to fall through three silent conditions and leave the user
        # with a successful job, no "distances" section, and nothing but the
        # generic pixel-size caveat to explain it (honesty rule 6).
        target = inputs.distance_target
        if target and target not in inputs.compartments.masks:
            caveats.append(
                f"Distance-to-{target} was requested but skipped: {target!r} is "
                "not one of the compartments that were loaded."
            )
        elif target and pixel_size_nm:
            mask = inputs.compartments.restricted()[target]
            # The same array ``assign_points`` was given above, and now the same
            # treatment of it: both drop the rows that are not positions and
            # both clip the rest onto the image. They used to differ, and one
            # point set was simultaneously reported as pinned to the far edge
            # (here it is counted) and sitting at pixel (0, 0) (here it is
            # 3.5e+30 nm from the nearest mitochondrion).
            dist = distance_to_boundary(
                inputs.points_xy,
                mask,
                pixel_size_nm=pixel_size_nm,
                band_edges_nm=inputs.band_edges_nm,
            )
            result["distances"] = {
                "target": target,
                "band_labels": dist.band_labels,
                "band_counts": dist.band_counts,
                "band_fractions": dist.band_fractions,
                "median_nm": dist.median_nm,
                "n_inside": int(dist.inside.sum()),
                # What the median and the bands are over. Without it a reader
                # has to subtract two numbers from a different section to find
                # out that this one covers fewer points than the run does.
                "n_measured": dist.n,
                "n_unreadable": dist.n_unreadable,
                "n_out_of_image": dist.n_out_of_image,
            }
            if dist.n_unreadable:
                caveats.append(
                    f"The distance-to-{target} numbers cover {dist.n} of the "
                    f"{assignment.n_total} points. The other "
                    f"{dist.n_unreadable} have a coordinate that is not a "
                    "position, so they have no distance to anything and are in "
                    "no band, fraction or median here."
                )
            if dist.n_out_of_image:
                caveats.append(
                    f"{dist.n_out_of_image} of the {dist.n} points measured "
                    f"against {target} lie outside the image, so their distance "
                    "is measured from the border pixel they were clipped onto "
                    "— the same pixel the compartment counts use, which is why "
                    "the two sections agree — and not from the coordinates "
                    "given. Neither number is a measurement of a point at those "
                    "coordinates, and every band and median below includes them."
                )
        elif target:  # the compartment exists, so the missing piece is the scale
            if resample_was_skipped:
                # The image *does* record a pixel size, so "this image is
                # uncalibrated" would be a false sentence about a true refusal.
                # The distance is refused because the boundary it would be
                # measured to belongs to an object set produced at the wrong
                # scale, not because there is no number to convert with.
                reason = (
                    f"the objects that define {target} were produced before this "
                    f"image had a pixel size, so measuring to their boundary in "
                    f"nanometres would report the geometry of a wrongly-scaled "
                    f"object set at {recorded_nm} nm/px. Re-run inference and "
                    "the distances come back."
                )
            elif recorded_nm is not None:
                reason = (
                    f"distances need a pixel size, and this image records "
                    f"{recorded_nm} nm/px, which is not one."
                )
            else:
                reason = "distances need a pixel size, and this image is uncalibrated."
            caveats.append(f"Distance-to-{target} was requested but skipped: {reason}")

        if not tissue_is_empty and not no_points_on_tissue:
            null = csr_null(
                inputs.points_xy,
                inputs.compartments,
                image_key=inputs.image_key,
                areas=areas,
                replicates=inputs.replicates,
                seed=inputs.seed,
            )
            result["monte_carlo"] = {
                "replicates": null.replicates,
                "seed": null.seed,
                "observed": null.observed,
                "null_mean": null.null_mean,
                "null_sd": null.null_sd,
                "z": null.z,
                "p_two_sided": null.p_two_sided,
            }
            result["monte_carlo_self_check"] = self_check(
                inputs.compartments, image_key=inputs.image_key
            )
            caveats.extend(_flat_null_caveats(null, assignment.n_on_tissue))
    elif inputs.distance_target:
        caveats.append(
            f"Distance-to-{inputs.distance_target} was requested but skipped: "
            "the point set is empty, so there is nothing to measure a distance "
            "from."
        )

    result["_object_metrics"] = metrics
    return result


def circular_columns(result: dict[str, Any], row: dict[str, Any]) -> list[str]:
    """The columns of ``row`` that are circular by construction, by name.

    Enrichment measured with a compartment's own object centroids is
    ``1 / area_fraction`` and says nothing about the biology. The run names the
    compartment in its caveats and on screen; this names the *columns*, because
    the spreadsheet is where the number gets quoted from and a cell has no room
    for a sentence. ``area_fraction_<name>`` is not among them -- it is a real
    measurement, and it is the denominator that makes the others circular.
    """
    names = result.get("circular_compartments") or []
    return [
        column
        for name in names
        for column in (f"enrichment_{name}", f"z_enrichment_{name}")
        if column in row
    ]


def image_summary_row(result: dict[str, Any]) -> dict[str, Any]:
    """One image's row of ``image_summary.csv``, flat and mostly numeric.

    Shared with the group rollup in :mod:`quantem.analysis.views` so an
    aggregated group value and the exported per-image table are literally the
    same numbers -- the house rule is that every chart reads from a table that
    is exported alongside it, and a second flattener would break that quietly.

    The last three columns are text, and they are the point of this being one
    file rather than two: ``enrichment_mito = 9.456`` opened in Excel looks the
    same whether it is a result or an artefact of asking a compartment about its
    own centroids, and the sentence saying which lived only in ``manifest.json``
    and on a screen the reader has closed.
    """
    by_source = result["objects"].get("by_source") or {}
    hand_drawn = int(by_source.get("manual", 0))
    proofreading = result.get("proofreading") or {}
    row: dict[str, Any] = {
        "image_key": result["image_key"],
        # Blank from a notebook, which has neither. Present here so a row can
        # name the run that produced it: the run id used to exist only in the
        # export directory's *name*, and moving the three files anywhere lost it.
        "segmentation_id": result.get("segmentation_id") or "",
        "analysis_run_id": result.get("analysis_run_id") or "",
        "group": result["group"],
        "pixel_size_nm": result["pixel_size_nm"],
        "calibrated": result["calibrated"],
        "n_objects": result["objects"]["n"],
        # The split that explains a metric with a small n, in the table the
        # charts read from -- not only in the manifest.
        "n_hand_drawn": hand_drawn,
        "n_model_produced": result["objects"]["n"] - hand_drawn,
        # Blank, not 0, when nobody recorded a rejection or a reviewed area:
        # "a person threw nothing away" and "we do not know" are different.
        "n_rejected": proofreading.get("n_rejected"),
        "reviewed_fraction": proofreading.get("reviewed_fraction"),
        "tissue_px": result["composition"]["tissue_px"],
        "tissue_um2": result["composition"]["tissue_um2"],
        "objects_per_um2": (result["objects"].get("density") or {}).get("per_um2"),
    }
    for name, frac in (result["composition"].get("area_fractions") or {}).items():
        row[f"area_fraction_{name}"] = frac
    for name, value in ((result.get("points") or {}).get("enrichment") or {}).items():
        row[f"enrichment_{name}"] = value
    for key, z in ((result.get("monte_carlo") or {}).get("z") or {}).items():
        row[f"z_{key}"] = z
    distances = result.get("distances") or {}
    if distances:
        row["distance_median_nm"] = distances.get("median_nm")

    caveats = list(result.get("caveats") or [])
    row["circular_columns"] = " ".join(circular_columns(result, row))
    row["n_caveats"] = len(caveats)
    # One cell, because a spreadsheet reader who sorts by a column must not lose
    # the sentence that qualifies it. Newlines would end the CSV record in some
    # readers and are replaced by the separator rather than quoted.
    row["caveats"] = " | ".join(c.replace("\n", " ") for c in caveats)
    return row


def write_bundle(
    results: list[dict[str, Any]],
    out_dir: Path,
    *,
    model_provenance: dict[str, Any] | None = None,
    analysis_run_id: str | None = None,
) -> Path:
    """Write the export bundle: per-object CSV, per-image CSV, and a manifest.

    ``analysis_run_id`` is written into all three files. It used to live only in
    the name of the directory they sit in, so moving them -- into a supplementary
    zip, into a shared drive, into a paper's data deposit -- left a bundle that
    could not name its own run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        r.setdefault("analysis_run_id", analysis_run_id or "")

    # objects.csv -- one row per object, every metric, traceable to its image
    obj_rows: list[dict[str, Any]] = []
    for r in results:
        n_caveats = len(r.get("caveats") or [])
        for m in r.get("_object_metrics", []):
            row = object_csv_row(m)
            row["image_key"] = r["image_key"]
            row["group"] = r["group"]
            row["segmentation_id"] = r.get("segmentation_id") or ""
            row["analysis_run_id"] = r.get("analysis_run_id") or ""
            # The count, not the text. A reader who sorts this file and finds a
            # non-zero here knows there are sentences they have not read, and
            # image_key is the join to the image_summary.csv row that holds them.
            row["n_caveats"] = n_caveats
            obj_rows.append(row)
    obj_columns = _write_csv(out_dir / "objects.csv", obj_rows, fields=OBJECT_CSV_FIELDS)

    # image_summary.csv -- one row per image
    img_rows = [image_summary_row(r) for r in results]
    img_columns = _write_csv(
        out_dir / "image_summary.csv",
        img_rows,
        fields=IMAGE_SUMMARY_BASE_FIELDS,
        last_fields=IMAGE_SUMMARY_LAST_FIELDS,
    )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "quantem_version": __version__,
        # The run this bundle *is*. It lived only in the directory name, so a
        # manifest that had been moved could not say which run wrote it, and
        # neither CSV beside it could either.
        "analysis_run_id": analysis_run_id
        or (results[0].get("analysis_run_id") if results else None),
        "segmentation_ids": sorted(
            {r.get("segmentation_id") for r in results if r.get("segmentation_id")}
        ),
        # Not the same thing as the version string: 0.1.0 is every build
        # between two releases, and a commit is one of them.
        "release": provenance.release(),
        "environment": provenance.environment(),
        "n_images": len(results),
        "aggregation_rule": (
            "Group values are unweighted means over experimental units. Never "
            "weight by point count — doing so produced a random-data enrichment "
            "of 0.73 instead of 1.0 in the reference implementation."
        ),
        "monte_carlo": _monte_carlo_manifest(results),
        "models": model_provenance or {},
        "objects": _object_manifest(results),
        "outputs": _outputs_manifest(
            out_dir,
            {
                "objects.csv": (len(obj_rows), obj_columns, "one row per confirmed object"),
                "image_summary.csv": (
                    len(img_rows),
                    img_columns,
                    "one row per image, with that image's caveats in the last column",
                ),
            },
        ),
        "images": [
            {
                "image_key": r["image_key"],
                "segmentation_id": r.get("segmentation_id") or None,
                "group": r["group"],
                "pixel_size_nm": r["pixel_size_nm"],
                "calibrated": r["calibrated"],
                "n_objects": r["objects"]["n"],
                "n_objects_by_source": r["objects"].get("by_source") or {},
                "proofreading": r.get("proofreading") or {},
                "points": r.get("points") or None,
                "caveats": r["caveats"],
            }
            for r in results
        ],
    }
    manifest, manifest["local_paths"] = provenance.scrub_local_paths(manifest)
    # ensure_ascii=False: the notes and caveats in here are prose, and the file
    # is written as UTF-8 and read back as UTF-8. Escaping turns every em dash
    # into a literal ``—`` in a file people open in a text editor, which is
    # a worse thing to read than the character it is protecting.
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_dir


def _monte_carlo_manifest(results: list[dict[str, Any]]) -> dict[str, Any]:
    """The null's settings, or a stated reason it has none.

    ``replicates: null, seed: null`` used to sit here bare in every bundle that
    analysed no point set -- the only two nulls in the file with no entry in an
    ``unavailable`` map, which is the manifest's whole contract for a null. A
    reader could not tell "this run did not do a Monte-Carlo" from "the seed was
    lost".
    """
    first = next((r["monte_carlo"] for r in results if r.get("monte_carlo")), None)
    values: dict[str, Any] = {
        "seeding": "per (image, replicate), independent of processing order",
        "distance": ("exact KD-tree to the eroded mask boundary, observed and null alike"),
    }
    if first is not None:
        values["replicates"] = first["replicates"]
        values["seed"] = first["seed"]
        return provenance.section(values, {})

    requested = [r for r in results if (r.get("points") or {}).get("n_total")]
    if requested:
        reason = (
            "No Monte-Carlo null was run for any image in this bundle although a "
            "point set was analysed. The reason is in that image's caveats: an "
            "empty tissue mask has nowhere to scatter a point, and a point set "
            "with nothing on the tissue would give a null of identical zeros "
            "and a p-value that means nothing."
        )
    else:
        reason = (
            "No Monte-Carlo null was run: this bundle analysed no point set, so "
            "there is no observed spatial statistic to compare against complete "
            "spatial randomness. Set points_source on the run to get one."
        )
    return provenance.section(values, {"replicates": reason, "seed": reason})


def _outputs_manifest(
    out_dir: Path, described: dict[str, tuple[int, list[str], str]]
) -> dict[str, Any]:
    """Name and checksum the files this bundle just wrote.

    Every model weight in this manifest is pinned by a sha256 and the *numbers*
    were not: nothing tied a given ``objects.csv`` to a given manifest, and an
    edited cell -- a row deleted, a value tidied -- left no trace at all. A
    digest here is what makes the pairing checkable rather than assumed.

    ``manifest.json`` is deliberately absent from the list. A file cannot carry
    its own digest, and a plausible-looking entry claiming otherwise is worse
    than the gap.
    """
    files = []
    for name, (n_rows, columns, what) in described.items():
        entry = provenance.file_identity(out_dir / name, what=name)
        entry["contents"] = what
        entry["n_rows"] = n_rows
        entry["columns"] = columns
        # A list of column names is not a description of them. Two of these
        # columns are unreadable without one: see COLUMN_NOTES.
        notes = COLUMN_NOTES.get(name)
        if notes:
            entry["column_notes"] = dict(notes)
        if name == "objects.csv":
            entry["columns_not_written"] = dict(OBJECT_COLUMNS_NOT_WRITTEN)
            entry["columns_derived_from"] = dict(OBJECT_CSV_DERIVED_COLUMNS)
            entry["columns_are_not_independent"] = (
                "The columns above are exact functions of the ones they name, "
                "not separate measurements. Every one is correct and worth "
                "reporting; what they will not support is being fed together "
                "into a correlation matrix or a PCA as independent variables, "
                "which weights area, the axis lengths and the pixel size more "
                "than once."
            )
        # The caveats column is prose, so these files are not ASCII. Named here
        # because the byte-order mark that makes Excel read them correctly is
        # the same one that prefixes the first column name of anything opened
        # as plain utf-8.
        entry["encoding"] = "utf-8-sig"
        files.append(entry)
    return {
        "files": files,
        "checksum_algorithm": "sha256",
        "n_rows_excludes": "the header row, which every file here has",
        "encoding": (
            "UTF-8 with a byte-order mark, so Excel opens the caveats column "
            "as text rather than as the system codepage. Read them as "
            "'utf-8-sig' (pandas.read_csv and R's read.csv strip the mark "
            "themselves); manifest.json is UTF-8 without one."
        ),
        "verify": (
            "Recompute the digest of each file beside this manifest (sha256sum "
            "objects.csv, or Get-FileHash objects.csv). A mismatch means the "
            "file has been edited since the run wrote it, and the numbers in it "
            "are not the ones the rest of this manifest describes."
        ),
        "note": (
            "manifest.json is not listed: a file cannot contain its own digest. "
            "Checksum it externally if the bundle needs to be pinned whole. Rows "
            "in objects.csv carry the image_key of the image_summary.csv row "
            "whose caveats qualify them."
        ),
        "joining": (
            "Join objects.csv to image_summary.csv on image_key. The two files "
            "share no other column name, which is deliberate: the per-object "
            "scale is values_in_pixel_size_nm ('what these values are in') and "
            "the per-image one is pixel_size_nm ('what the image records'). "
            "They are different questions with different answers, and a join "
            "that collapsed them would silently answer one with the other. See "
            "each file's column_notes."
        ),
    }


def _object_manifest(results: list[dict[str, Any]]) -> dict[str, Any]:
    """How many objects there were, where they came from, and what is partly filled.

    The hand-drawn/model split is not trivia: it is the explanation for every
    metric in the summary whose n is below the object count, and without it a
    reader has an unexplained number and no way to chase it.

    ``partially_measured_metrics`` sums ``n`` and ``n_objects`` over every image
    in the bundle and keeps the *sentences* verbatim, one per distinct wording.
    Those are two different scopes, and with the totals and one image's sentence
    side by side the entry read ``{"n": 0, "n_objects": 16}`` beside "Measured
    on 0 of 8 confirmed objects" — which is two images of eight, not a
    contradiction, and nothing in the block said so. ``by_image`` and
    ``counts_note`` reconcile them; neither the totals nor the per-image
    sentences are worth giving up for the other.
    """
    by_source: dict[str, int] = {}
    for r in results:
        for source, n in (r["objects"].get("by_source") or {}).items():
            by_source[source] = by_source.get(source, 0) + int(n)
    total = sum(by_source.values())
    hand_drawn = by_source.get("manual", 0)

    coverage: dict[str, Any] = {}
    for r in results:
        for key, stats in (r["objects"].get("summary") or {}).items():
            # An estimator note qualifies every value the metric reports, so it
            # belongs in the manifest whether or not anything was blanked. See
            # the caveat block above for what shipping without it costs.
            if not stats.get("n_missing") and not stats.get("estimator_note"):
                continue
            entry = coverage.setdefault(key, {"n": 0, "n_objects": 0, "notes": [], "by_image": []})
            n = int(stats.get("n") or 0)
            n_objects = int(stats.get("n_objects") or 0)
            entry["n"] += n
            entry["n_objects"] += n_objects
            entry["by_image"].append(
                {
                    "image_key": r.get("image_key"),
                    "segmentation_id": r.get("segmentation_id") or None,
                    "n": n,
                    "n_objects": n_objects,
                    # The sentence that goes with *these* two numbers, so the
                    # deduplicated list above can be checked against the image
                    # it was written for.
                    "note": stats.get("note") or "",
                }
            )
            if stats.get("note") and stats["note"] not in entry["notes"]:
                entry["notes"].append(stats["note"])

    for entry in coverage.values():
        n_images = len(entry["by_image"])
        entry["n_images"] = n_images
        entry["counts_note"] = (
            f"n and n_objects are totals over the {n_images} images in "
            "by_image. Every sentence in notes was written for one of them and "
            "carries that image's numbers, not these totals: 'measured on 0 of "
            "8' beside n_objects = 16 is two images of eight apiece. Read a "
            "note against its own by_image entry."
            if n_images > 1
            else "n and n_objects are this bundle's one image, which notes describes."
        )

    return {
        "n_total": total,
        "n_by_source": dict(sorted(by_source.items())),
        "n_hand_drawn": hand_drawn,
        "n_model_produced": total - hand_drawn,
        "source_note": (
            "A hand-drawn polygon and a model-produced object do not carry the "
            "same measurements — only the model path has a probability behind "
            "it — so this split is what explains any metric below."
        ),
        "partially_measured_metrics": coverage,
    }


def export_dir_for_run(run_id: Any) -> Path:
    """Where one run's bundle lives: ``<QUANTEM_DATA_DIR>/exports/<run_id>/``."""
    return EXPORTS_DIR / str(run_id)


def non_finite_paths(value: Any, path: str = "") -> list[str]:
    """Every place in ``value`` holding a number JSON cannot represent.

    ``nan`` and ``±inf`` are floats to Python and not values to JSON.
    ``json.dumps`` writes them as the bare tokens ``NaN`` and ``Infinity``,
    which no strict parser accepts and which SQLite's ``JSON_VALID`` -- the
    check on ``AnalysisRun.results`` -- rejects outright.

    Returned as ``"path = value"`` strings rather than raised here, because the
    caller has to name them all in one sentence: a user who gets one field back
    at a time will fix one field at a time.
    """
    if isinstance(value, dict):
        found: list[str] = []
        for key, item in value.items():
            if isinstance(key, str) and key.startswith("_"):
                continue  # not stored, not written; see run_for_segmentation
            child = f"{path}.{key}" if path else str(key)
            found.extend(non_finite_paths(item, child))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for index, item in enumerate(value):
            found.extend(non_finite_paths(item, f"{path}[{index}]"))
        return found
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return [f"{path or '<root>'} = {value}"]
    return []


def run_for_segmentation(
    analysis_run: AnalysisRun, *, progress: ProgressFn | None = None
) -> dict[str, Any]:
    """Run one :class:`~quantem.analysis.models.AnalysisRun` end to end.

    Loads the masks named in ``analysis_run.params``, calls :func:`run_analysis`,
    writes the export bundle, and stores the outcome on the row. Returns the full
    result dict -- including the per-object metrics, which the *row* does not
    keep (they are in ``objects.csv``, which is the artefact a paper cites).

    Failure is recorded on the row before the exception is re-raised, so a user
    polling the run sees why it failed even though the job queue owns the retry.
    """
    # Local import: everything above this line is pure numpy and is tested
    # without Django. Only this function needs the ORM.
    from django.utils import timezone

    from .loaders import (
        centroid_self_reference_caveat,
        circular_compartments,
        load_inputs,
    )
    from .models import AnalysisRun as _AnalysisRun

    def report(percent: float, message: str) -> None:
        if progress is not None:
            progress(percent, message)

    _AnalysisRun.objects.filter(id=analysis_run.id).update(
        status=_AnalysisRun.STATUS_RUNNING,
        started_at=timezone.now(),
        error="",
    )
    analysis_run.status = _AnalysisRun.STATUS_RUNNING

    written: Path | None = None
    try:
        report(5.0, "loading masks")
        loaded = load_inputs(analysis_run)

        report(25.0, "measuring")
        result = run_analysis(loaded.inputs)

        # Named, not only described: the caveat says which compartment is
        # circular in a sentence, and image_summary.csv turns that into the
        # exact column headers so a spreadsheet reader gets it too.
        result["circular_compartments"] = circular_compartments(analysis_run, loaded.params)
        circular = centroid_self_reference_caveat(analysis_run, loaded.params)
        if circular:
            result["caveats"].append(circular)

        # What the masks' provenance could not establish. These are produced by
        # the loader, where the objects are, and lifted here because the caveat
        # list is the one the UI puts in front of the user -- a qualification
        # buried in models.compartments[i].run is not a qualification.
        for caveat in loaded.caveats:
            if caveat not in result["caveats"]:
                result["caveats"].append(caveat)

        # Checked before anything reaches the disk. A single non-finite number
        # -- ``nan`` from a distance that overflowed was the measured case --
        # cannot be stored: ``AnalysisRun.results`` carries a ``JSON_VALID``
        # check and ``nan`` is not JSON. Left until after the bundle was
        # written, the row failed on a database constraint at the last step and
        # what the user was left with was a FAILED run whose ``export_dir`` was
        # empty, three files on disk under a run id no row would ever name, and
        # "CHECK constraint failed: (JSON_VALID(...))" as the explanation.
        unstorable = non_finite_paths(result)
        if unstorable:
            raise ValueError(
                "This analysis produced "
                + (
                    "a value that is not a number: "
                    if len(unstorable) == 1
                    else f"{len(unstorable)} values that are not numbers: "
                )
                + "; ".join(unstorable)
                + ". Nothing was stored and no export was written, because a "
                "result that cannot be held in the database is not one that "
                "should be sitting in a folder either. This is a defect in "
                "QuantEM, not in your data — please report it with the point "
                "set and the compartments you used."
            )

        report(80.0, "writing export bundle")
        out_dir = export_dir_for_run(analysis_run.id)
        # Claimed before the write, not after: a write that fails half way
        # leaves the same unreachable directory a write that succeeds into a
        # run that then fails does.
        written = out_dir
        write_bundle(
            [result],
            out_dir,
            model_provenance=loaded.provenance,
            analysis_run_id=str(analysis_run.id),
        )

        stored = {k: v for k, v in result.items() if not k.startswith("_")}
        stored["n_object_rows"] = len(result.get("_object_metrics", []))
        finished = timezone.now()
        _AnalysisRun.objects.filter(id=analysis_run.id).update(
            status=_AnalysisRun.STATUS_SUCCESS,
            results=stored,
            export_dir=str(out_dir),
            group=loaded.inputs.group,
            params=loaded.params,
            finished_at=finished,
            error="",
        )
        analysis_run.status = _AnalysisRun.STATUS_SUCCESS
        analysis_run.results = stored
        analysis_run.export_dir = str(out_dir)
        analysis_run.finished_at = finished
        report(100.0, "analysis complete")
        return result
    except Exception as exc:
        message = str(exc) or exc.__class__.__name__
        # A bundle written by a run that then failed is unreachable. Every
        # download URL is built from the row's ``export_dir``, and a failed run
        # does not get one -- so the files sit under a directory named for a run
        # that disowns them, look like results to anyone who finds them, and
        # carry the numbers of an analysis the app says did not finish. Removed,
        # and said so, unless the row already points at that directory (an
        # earlier successful run of the same row, whose bundle is still its).
        if written is not None and analysis_run.export_dir != str(written):
            shutil.rmtree(written, ignore_errors=True)
            message = (
                f"{message} (the export bundle written for this run was removed: "
                "the run failed before it could be recorded, so nothing would "
                "have been able to name it.)"
            )
        _AnalysisRun.objects.filter(id=analysis_run.id).update(
            status=_AnalysisRun.STATUS_FAILED,
            finished_at=timezone.now(),
            error=message,
        )
        analysis_run.status = _AnalysisRun.STATUS_FAILED
        analysis_run.error = message
        raise


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fields: tuple[str, ...] = (),
    last_fields: tuple[str, ...] = (),
) -> list[str]:
    """Write ``rows`` as CSV, always with a header. Returns the columns written.

    ``fields`` are the columns known in advance; anything else a row carries is
    appended in first-seen order, and ``last_fields`` are held back to the end
    whatever order they were built in -- the caveat text belongs after the
    numbers, not between two of them. A run with nothing to report writes the
    header alone -- never a zero-byte file, which ``pandas.read_csv`` rejects
    with ``EmptyDataError`` rather than reading as an empty table.

    The column list goes into the manifest beside the file's digest, so a reader
    can tell a truncated file from an intact one without opening it.

    Written UTF-8 **with a BOM**, and the manifest says so beside each file.
    The caveats column is prose and carries em dashes; Excel opening a ``.csv``
    by double-click decodes it as the system codepage unless a byte-order mark
    says otherwise, and renders them as ``â€"``. The house rule for this module
    is that a result the user cannot open in Excel is not a result, and that
    covers the sentence qualifying a number as much as the number.

    The cost is that ``csv.DictReader`` over a handle opened as plain ``utf-8``
    sees ``\\ufeffimage_key`` as the first column name. Open these files as
    ``utf-8-sig`` (``pandas.read_csv`` and R's ``read.csv`` strip the mark on
    their own); ``outputs.<file>.encoding`` in the manifest is there so a reader
    does not have to find that out by hitting it.
    """
    held = set(last_fields)
    names: list[str] = [name for name in fields if name not in held]
    for row in rows:
        for key in row:
            if key not in names and key not in held:
                names.append(key)
    names.extend(last_fields)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return names
