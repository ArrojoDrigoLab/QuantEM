"""Typing a pixel size after an uncalibrated run must not launder the result.

The app tells users, in the import warning and in the uncalibrated caveat:

    "Set the image's pixel size and re-run inference before reporting any of
    these: setting it afterwards converts the units and leaves the object set
    as it is."

Nothing checked the second half of that sentence. Every downstream guard keyed
on the asset's *present* pixel size, so a user who did the first half and not
the second got a bundle that had quietly become correct-looking:

===========================  ================  ===============
                             before setting    after setting
===========================  ================  ===============
caveats                      6                 3
wrong-scale caveat           present           **gone**
``calibrated``               False             **True**
``objects_per_um2``          blank             3.6336
``distance_median_nm``       refused           313.35
===========================  ================  ===============

and the manifest asserted the opposite of the problem -- ``ran_at: "native"``,
``resampled: false``, "no pack that ran resampled it" -- beside a
``native_pixel_size_nm`` read from the asset's current value while every
object's own stamp said ``null``.

The cost, measured on one untagged image: 32 mitochondria that way against 14
with 5 nm typed in at import; 23 lipid droplets against 6. A 2.3x and 3.8x
error in the counts, with a micron distribution and a median distance to match,
carrying no warning of any kind.

The data needed to catch it was already stamped on every object.
"""

from __future__ import annotations

import csv
import json
from typing import Any

from django.test import override_settings
from rest_framework.test import APIClient

from quantem.analysis import loaders, service
from quantem.analysis.job import run_job
from quantem.analysis.models import AnalysisRun
from quantem.jobs.models import Job
from quantem.jobs.reporter import CancelToken, JobReporter
from quantem.segmentation.type_service import (
    get_or_create_er_type,
    get_or_create_lipid_droplet_type,
    get_or_create_mitochondria_type,
    get_or_create_tissue_type,
)

from .test_run_identity import RunIdentityTestCase, _square, _stamp


def _uncalibrated_stamp(**kwargs: Any) -> dict[str, Any]:
    """A run that happened while the image had no pixel size.

    ``ran_at_nm`` is None because a pack cannot resample without one -- which is
    the whole defect: the object set is the native-scale one no matter what is
    typed in later.
    """
    return _stamp(ran_at_nm=None, native_pixel_size_nm=None, **kwargs)


class CalibratedAfterTheFactTests(RunIdentityTestCase):
    """The asset is 5 nm/px; the objects were produced before that was true."""

    def _mito_produced_uncalibrated(self):
        mito = self._segmentation(get_or_create_mitochondria_type)
        for i in range(3):
            self._object(
                mito,
                _square(20 + 30 * i, 20),
                source_model="quantem:mito",
                stamp=_uncalibrated_stamp(pack_id="quantem:mito"),
            )
        return mito

    def test_the_wrong_scale_caveat_survives_a_pixel_size_typed_in_afterwards(self):
        mito = self._mito_produced_uncalibrated()
        _run, got = self._run(mito, compartments={"mito": str(mito.id)})

        caveats = " ".join(got["result"]["caveats"])
        self.assertIn("no pixel size", caveats)
        self.assertIn("does not re-run inference", caveats)
        self.assertIn(
            "0, 19, 120 and 233",
            caveats,
            "the measured demonstration belongs with the warning",
        )

    def test_the_recalibration_caveat_names_the_transition_it_used_to_miss(self):
        mito = self._mito_produced_uncalibrated()
        _run, got = self._run(mito, compartments={"mito": str(mito.id)})

        caveats = " ".join(got["result"]["caveats"])
        self.assertIn("had no pixel size", caveats)
        self.assertIn("Re-run inference", caveats)

    def test_the_manifest_does_not_call_the_missed_resample_a_design_fact(self):
        """``ran_at: "native"`` has two causes and the note stated only the
        harmless one."""
        mito = self._mito_produced_uncalibrated()
        _run, got = self._run(mito, compartments={"mito": str(mito.id)})

        scale = self._compartment(got["manifest"], "mito")["run"]["scale"]
        self.assertEqual(scale["ran_at"], "native")
        note = scale["note"]
        self.assertIn("not because no resample was called for", note)
        self.assertIn("quantem:mito", note)
        self.assertIn(
            "the image's value now",
            note,
            "native_pixel_size_nm is the asset's current value, not the run's",
        )

    def test_a_pack_with_no_canonical_scale_is_still_reported_as_benign(self):
        """quantem:er declares no canonical_nm, so its native run really is one."""
        er = self._segmentation(get_or_create_er_type)
        self._object(
            er,
            _square(30, 20),
            source_model="quantem:er",
            stamp=_uncalibrated_stamp(pack_id="quantem:er"),
        )
        _run, got = self._run(er, compartments={"er": str(er.id)})

        note = self._compartment(got["manifest"], "er")["run"]["scale"]["note"]
        self.assertIn("nothing was skipped", note)
        self.assertNotIn("not because no resample was called for", note)

    def test_a_genuinely_calibrated_run_gets_neither_caveat(self):
        """The guard must not fire on the ordinary, correct path."""
        mito = self._segmentation(get_or_create_mitochondria_type)
        for i in range(3):
            self._object(
                mito,
                _square(20 + 30 * i, 20),
                source_model="quantem:mito",
                stamp=_stamp(pack_id="quantem:mito", ran_at_nm=8.0),
            )
        _run, got = self._run(mito, compartments={"mito": str(mito.id)})

        caveats = " ".join(got["result"]["caveats"])
        self.assertNotIn("does not re-run inference", caveats)
        self.assertNotIn("had no pixel size", caveats)
        scale = self._compartment(got["manifest"], "mito")["run"]["scale"]
        self.assertEqual(scale["ran_at"], "canonical")


class ProducedPixelSizeTests(RunIdentityTestCase):
    """The field the caveat is driven from."""

    def test_it_reports_what_the_objects_recorded_not_what_the_asset_says_now(self):
        mito = self._mito_uncalibrated()
        loaded = loaders.load_inputs(self._pending_run(mito, compartments={"mito": str(mito.id)}))
        inputs = loaded.inputs if hasattr(loaded, "inputs") else loaded
        self.assertIn(None, inputs.produced_pixel_size_nm)
        self.assertEqual(float(self.asset.pixel_size_nm), 5.0)

    def _mito_uncalibrated(self):
        mito = self._segmentation(get_or_create_mitochondria_type)
        self._object(
            mito,
            _square(20, 20),
            source_model="quantem:mito",
            stamp=_uncalibrated_stamp(pack_id="quantem:mito"),
        )
        return mito

    def _pending_run(self, subject, **params):
        from quantem.analysis.models import AnalysisRun

        return AnalysisRun.objects.create(
            segmentation=subject,
            params=loaders.normalise_params(params, segmentation=subject),
        )

    def test_unstamped_objects_report_nothing_rather_than_guessing(self):
        mito = self._segmentation(get_or_create_mitochondria_type)
        self._object(mito, _square(20, 20), source_model="quantem:mito")
        _run, got = self._run(mito, compartments={"mito": str(mito.id)})

        caveats = " ".join(got["result"]["caveats"])
        self.assertNotIn(
            "does not re-run inference",
            caveats,
            "an object with no stamp says nothing about when it was produced",
        )


class NumbersNotOnlySentencesTests(RunIdentityTestCase):
    """The half of the defect the caveat described and the columns contradicted.

    The wrong-scale caveat said, in the app's own words, that every micron
    value, density and distance was "an exact conversion of a wrongly-scaled
    object set, which is more misleading than the blank columns it replaced".
    It said that beside ``calibrated: True``, a filled ``objects_per_um2`` and a
    working ``distance_median_nm``, every one of them computed from the number
    the user typed in after the fact. A sentence that contradicts the row it
    sits in loses to the row: the row is what gets sorted, plotted and pasted
    into a paper.

    So the units guard now keys on the same fact the caveat does. What it must
    *not* blank is the dimensionless half -- those are the numbers the wrong
    scale actually moved, and a bundle with nothing in it teaches nobody
    anything.
    """

    def _tissue(self):
        tissue = self._segmentation(get_or_create_tissue_type)
        self._object(tissue, _square(5, 5, 120), source_model="manual")
        return tissue

    def _uncalibrated_mito(self, n: int = 3):
        mito = self._segmentation(get_or_create_mitochondria_type)
        for i in range(n):
            self._object(
                mito,
                _square(20 + 30 * i, 20),
                source_model="quantem:mito",
                stamp=_uncalibrated_stamp(pack_id="quantem:mito"),
            )
        return mito

    def _summary_row(self, run) -> dict[str, str]:
        path = service.export_dir_for_run(run.id) / "image_summary.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        self.assertEqual(len(rows), 1)
        return rows[0]

    def _object_rows(self, run) -> list[dict[str, str]]:
        path = service.export_dir_for_run(run.id) / "objects.csv"
        return list(csv.DictReader(path.open(encoding="utf-8-sig")))

    def _full_run(self, subject, name):
        return self._run(
            subject,
            compartments={name: str(subject.id)},
            tissue_segmentation_id=str(self._tissue().id),
            points_source="centroids",
            distance_target=name,
            replicates=5,
        )

    def test_every_physical_unit_is_blank_and_calibrated_says_so(self):
        mito = self._uncalibrated_mito()
        run, got = self._full_run(mito, "mito")

        result = got["result"]
        self.assertFalse(result["calibrated"])
        self.assertIsNone(result["composition"]["tissue_um2"])
        self.assertIsNone(result["composition"]["areas_um2"])
        self.assertIsNone(result["objects"]["density"]["per_um2"])
        self.assertIsNone(result["objects"]["density"]["tissue_um2"])

        row = self._summary_row(run)
        self.assertEqual(row["calibrated"], "False")
        self.assertEqual(row["tissue_um2"], "")
        self.assertEqual(row["objects_per_um2"], "")
        # The value the user typed is still reported: they have to be able to
        # see the number that was refused in order to act on it.
        self.assertEqual(row["pixel_size_nm"], "5.0")
        self.assertNotEqual(row["tissue_px"], "")

        for obj in self._object_rows(run):
            self.assertEqual(obj["calibrated"], "False")
            self.assertEqual(obj["area_um2"], "")
            self.assertEqual(obj["perimeter_um"], "")

    def test_the_dimensionless_numbers_are_kept_and_named_as_the_ones_at_risk(self):
        """Blanking these would leave a bundle with no numbers in it at all.

        They are also the ones the wrong scale *moved* -- the count most of all
        -- and no units guard can blank a ratio, so the caveat carries them
        instead.
        """
        mito = self._uncalibrated_mito()
        run, got = self._full_run(mito, "mito")

        row = self._summary_row(run)
        self.assertEqual(row["n_objects"], "3")
        self.assertNotEqual(row["area_fraction_mito"], "")
        self.assertEqual(got["result"]["objects"]["n"], 3)
        self.assertIn("monte_carlo", got["result"])

        caveats = " ".join(got["result"]["caveats"])
        self.assertIn("n_objects, area_fraction_*, enrichment_* and z_*", caveats)
        self.assertIn("blank", caveats)

    def test_a_refused_distance_gives_the_true_reason(self):
        """The old sentence said the image was uncalibrated. It records 5 nm/px."""
        mito = self._uncalibrated_mito()
        _run, got = self._full_run(mito, "mito")

        self.assertIsNone(got["result"].get("distances"))
        skipped = [c for c in got["result"]["caveats"] if "Distance-to-mito" in c]
        self.assertEqual(len(skipped), 1, got["result"]["caveats"])
        self.assertNotIn("this image is uncalibrated", skipped[0])
        self.assertIn("produced before this image had a pixel size", skipped[0])

    def test_a_pack_that_declares_no_canonical_scale_keeps_its_microns(self):
        """The guard must fire on a *skipped resample*, not on any missing stamp.

        ``quantem:er`` declares no ``canonical_nm``, and
        ``quantem.inference.resample.resample_factor`` returns 1.0 whether or not
        a pixel size is set, so an uncalibrated run of it produced exactly the
        objects a calibrated one would have. Its microns are real, and telling
        this user to re-run inference would return the same objects.
        """
        er = self._segmentation(get_or_create_er_type)
        for i in range(3):
            self._object(
                er,
                _square(20 + 30 * i, 20),
                source_model="quantem:er",
                stamp=_uncalibrated_stamp(pack_id="quantem:er"),
            )
        run, got = self._full_run(er, "er")

        self.assertTrue(got["result"]["calibrated"])
        self.assertIsNotNone(got["result"]["composition"]["tissue_um2"])
        self.assertIsNotNone(got["result"]["objects"]["density"]["per_um2"])
        self.assertIsNotNone(got["result"].get("distances"))
        self.assertNotEqual(self._summary_row(run)["objects_per_um2"], "")

        caveats = " ".join(got["result"]["caveats"])
        self.assertNotIn("Re-run inference", caveats)
        self.assertNotIn("had no pixel size", caveats)

    def test_a_compartment_produced_uncalibrated_blanks_the_bundle_it_is_in(self):
        """The subject is not the only mask that puts a number in the row.

        ``area_fraction_mito`` and the tissue denominator are measured off other
        segmentations, and ``canonical_nm_by_pack`` has always been built over
        all of them -- so the caveat could fire for a compartment while every
        column it described stayed filled from the asset's current value.
        """
        lipids = self._segmentation(get_or_create_lipid_droplet_type)
        self._object(
            lipids,
            _square(20, 20),
            source_model="quantem:lipid_droplets",
            stamp=_stamp(pack_id="quantem:lipid_droplets", ran_at_nm=8.0),
        )
        mito = self._uncalibrated_mito(n=1)

        run, got = self._run(
            lipids,
            compartments={"lipids": str(lipids.id), "mito": str(mito.id)},
            tissue_segmentation_id=str(self._tissue().id),
            replicates=5,
        )

        self.assertFalse(got["result"]["calibrated"])
        self.assertEqual(self._summary_row(run)["objects_per_um2"], "")
        caveats = " ".join(got["result"]["caveats"])
        self.assertIn("quantem:mito", caveats)

    def test_a_genuinely_calibrated_run_keeps_every_column(self):
        """The guard must not fire on the ordinary, correct path."""
        mito = self._segmentation(get_or_create_mitochondria_type)
        for i in range(3):
            self._object(
                mito,
                _square(20 + 30 * i, 20),
                source_model="quantem:mito",
                stamp=_stamp(pack_id="quantem:mito", ran_at_nm=8.0),
            )
        run, got = self._full_run(mito, "mito")

        self.assertTrue(got["result"]["calibrated"])
        row = self._summary_row(run)
        self.assertNotEqual(row["tissue_um2"], "")
        self.assertNotEqual(row["objects_per_um2"], "")
        self.assertNotEqual(row["distance_median_nm"], "")


@override_settings(ROOT_URLCONF="quantem.analysis.tests.urls")
class ServedToTheAnalysisScreenTests(RunIdentityTestCase):
    """The user's own route: POST, run the queued job, read the run back.

    The screen's calibration badge is ``AnalysisRunSerializer.calibrated`` and
    every panel reads the flattened sections beside it. A guard that only holds
    inside ``run_analysis`` would leave the green "5.00 nm/px" badge and the
    filled density panel exactly where they were.
    """

    def setUp(self) -> None:
        super().setUp()
        self.client = APIClient()

    def test_the_screen_is_told_the_run_is_uncalibrated_and_shown_no_microns(self):
        mito = self._segmentation(get_or_create_mitochondria_type)
        for i in range(3):
            self._object(
                mito,
                _square(20 + 30 * i, 20),
                source_model="quantem:mito",
                stamp=_uncalibrated_stamp(pack_id="quantem:mito"),
            )
        tissue = self._segmentation(get_or_create_tissue_type)
        self._object(tissue, _square(5, 5, 120), source_model="manual")

        started = self.client.post(
            f"/api/segmentations/{mito.id}/analysis/",
            {
                "compartments": {"mito": str(mito.id)},
                "tissue_segmentation_id": str(tissue.id),
                "points_source": "centroids",
                "distance_target": "mito",
                "replicates": 5,
            },
            format="json",
        )
        self.assertEqual(started.status_code, 202, started.data)
        job = Job.objects.get(id=started.data["job_id"])
        run_job(
            job.payload_json,
            JobReporter(str(job.id), min_interval_seconds=0.0),
            CancelToken(str(job.id)),
        )

        run_id = started.data["analysis_run_id"]
        self.assertEqual(AnalysisRun.objects.get(id=run_id).status, AnalysisRun.STATUS_SUCCESS)
        served = self.client.get(f"/api/analysis/{run_id}/")
        self.assertEqual(served.status_code, 200, served.data)

        self.assertFalse(served.data["calibrated"])
        # The value is still served, so the screen can say what was refused.
        self.assertEqual(served.data["pixel_size_nm"], 5.0)
        self.assertIsNone(served.data["composition"]["tissue_um2"])
        self.assertIsNone(served.data["objects"]["density"]["per_um2"])
        self.assertIsNone(served.data["distances"])
        self.assertTrue(
            any("does not re-run inference" in c for c in served.data["caveats"]),
            served.data["caveats"],
        )
        self.assertFalse(
            [c for c in served.data["caveats"] if "--" in c],
            "the notice panel renders these verbatim",
        )


class MinAreaProvenanceTests(RunIdentityTestCase):
    """``min_area`` restated in microns must use the run's pixel size.

    The caveat side of "calibrated after the fact" was fixed first, and this was
    the half left behind: ``um2`` and ``model_grid_px`` were still derived from
    the asset's *current* value, so a bundle whose every other physical unit is
    deliberately blank carried ``um2: 0.0015`` -- 60 x (5/1000)^2, from a pixel
    size the run never had -- beside a ``scale`` block correctly reading
    ``resampled: false``. A reader quoting "objects below 0.0015 um2 were
    discarded" as the study's size floor would be quoting a number that never
    existed.
    """

    def _min_area(self, stamp_kwargs: dict[str, Any]) -> dict[str, Any]:
        mito = self._segmentation(get_or_create_mitochondria_type)
        self._object(
            mito,
            _square(20, 20),
            source_model="quantem:mito",
            stamp=_stamp(pack_id="quantem:mito", min_area=60, **stamp_kwargs),
        )
        _run, got = self._run(mito, compartments={"mito": str(mito.id)})
        return self._compartment(got["manifest"], "mito")["run"]["min_area"]

    def test_an_uncalibrated_run_gets_no_micron_floor(self):
        entry = self._min_area({"ran_at_nm": None, "native_pixel_size_nm": None})["by_pack"][
            "quantem:mito"
        ]

        self.assertEqual(entry["value"], 60)
        self.assertIsNone(entry["um2"], "0.0015 um2 was never a real size floor")
        self.assertIsNone(entry["model_grid_px"])
        self.assertIn("no pixel size", entry["unavailable"]["um2"])
        self.assertIn("after the run", entry["unavailable"]["um2"])

    def test_a_calibrated_run_still_restates_the_floor(self):
        entry = self._min_area({"ran_at_nm": 8.0, "native_pixel_size_nm": 5.0})["by_pack"][
            "quantem:mito"
        ]

        self.assertAlmostEqual(entry["um2"], 60 * (5.0 / 1000.0) ** 2)
        self.assertAlmostEqual(entry["model_grid_px"], round(60 * (5 / 8) ** 2, 3))


class CircularityBiasTravelsWithEveryBundleTests(RunIdentityTestCase):
    """The estimator note must not be gated on whether a value was blanked.

    Blanking impossible values (circularity > 1) was the floor. The note that
    explains the *bias* was reaching the bundle only when at least one value had
    been blanked, so a run whose objects all cleared the ceiling shipped a full
    circularity column with no word of it.

    That is the case that produces a wrong paper, because the bias is monotone
    in size and does not cancel between groups. Scaling eight real mitochondrial
    outlines to 0.6x -- a pure size change, identical shapes -- moved mean
    circularity from 0.6186 to 0.6409, paired t = 3.596, **p = 0.0088**:
    "mitochondria became more circular after treatment", from a correct
    segmentation and a silent bundle.
    """

    def _bundle_with_reportable_circularity(self):
        mito = self._segmentation(get_or_create_mitochondria_type)
        # Large enough that the estimator stays under 1 and nothing is blanked.
        for i in range(3):
            self._object(
                mito,
                _square(20 + 60 * i, 40),
                source_model="quantem:mito",
                stamp=_stamp(pack_id="quantem:mito", ran_at_nm=8.0),
            )
        return self._run(mito, compartments={"mito": str(mito.id)})

    def test_the_bias_note_ships_when_nothing_was_blanked(self):
        _run, got = self._bundle_with_reportable_circularity()
        summary = got["result"]["objects"]["summary"]["circularity"]

        self.assertGreater(summary["n"], 0, "the column must actually be populated")
        self.assertFalse(summary.get("n_missing"), "nothing should be blanked here")
        self.assertIn("estimator, not geometry", summary["estimator_note"])

    def test_it_reaches_the_caveats_a_reader_actually_sees(self):
        _run, got = self._bundle_with_reportable_circularity()
        caveats = " ".join(got["result"]["caveats"])

        self.assertIn(
            "estimator, not geometry",
            caveats,
            "the caveat list is where a reader looks; gating it on n_missing "
            "shipped a populated circularity column with no word of the bias",
        )

    def test_it_reaches_the_manifest_too(self):
        _run, got = self._bundle_with_reportable_circularity()
        blob = json.dumps(got["manifest"])

        self.assertIn("estimator, not geometry", blob)

    def test_the_column_it_qualifies_carries_it_by_name(self):
        """Somewhere in the manifest is not the same as beside the column.

        ``objects.csv`` is the file the number is read out of, and the manifest
        defines its columns one by one. A reader who opens the bundle, means the
        circularity column and compares two groups has done nothing wrong and
        can still get a shape result out of a pure size difference, so the
        warning has to be under ``circularity`` and not only in a caveat list
        attached to a different file.
        """
        _run, got = self._bundle_with_reportable_circularity()
        entry = next(
            e for e in got["manifest"]["outputs"]["files"] if e["filename"] == "objects.csv"
        )

        self.assertIn("circularity", entry["columns"])
        note = entry["column_notes"]["circularity"]
        self.assertIn("estimator, not geometry", note)
        self.assertIn("perimeter_crofton", note)
        self.assertEqual(
            note,
            got["result"]["objects"]["summary"]["circularity"]["estimator_note"],
            "the file and the summary must not describe the column differently",
        )


class UnrecognisedPackCountsAsASkippedResampleTests(RunIdentityTestCase):
    """The two sites that decide "was a resample skipped?" must agree.

    ``loaders._canonical_nm`` returns ``(canonical_nm, known)`` precisely "so
    the caller can tell them apart": ``(None, True)`` is a released pack that
    declares no canonical scale and genuinely runs native, and ``(None, False)``
    is a pack this build cannot look up at all.

    ``loaders._packs_that_skipped_a_resample`` takes the explicit view that the
    second counts as having skipped one -- "what it would have done cannot be
    looked up, and an unlookupable scale is a reason to warn, not a reason to
    stay quiet" -- and writes the manifest caveat accordingly.
    ``service.run_analysis`` discarded the flag, so ``None`` was falsy, the gate
    did not fire, and the same run got micron columns converted with a pixel
    size typed in after the objects existed. Narrow reachability: the shipped UI
    cannot produce such a pack id. Not a reason for the two to disagree.
    """

    # An organelle this build has no released model for. `parse_family` maps any
    # vendor prefix onto the released families, so a made-up *vendor* still
    # resolves; a made-up *organelle* is what `get_model_spec` cannot answer.
    UNKNOWN_PACK = "quantem:golgi"

    def _objects_from_an_unknown_pack(self):
        mito = self._segmentation(get_or_create_mitochondria_type)
        for i in range(3):
            self._object(
                mito,
                _square(20 + 30 * i, 20),
                source_model=self.UNKNOWN_PACK,
                stamp=_uncalibrated_stamp(pack_id=self.UNKNOWN_PACK),
            )
        return self._run(mito, compartments={"mito": str(mito.id)})

    def test_the_premise_this_build_really_cannot_look_the_pack_up(self):
        canonical, known = loaders._canonical_nm(self.UNKNOWN_PACK)

        self.assertIsNone(canonical)
        self.assertFalse(known, "the test is meaningless if this pack resolves")

    def test_nothing_is_converted_with_a_pixel_size_typed_in_afterwards(self):
        """The gate: ``None`` is falsy and this ran anyway."""
        _run, got = self._objects_from_an_unknown_pack()
        result = got["result"]

        self.assertEqual(
            result["pixel_size_nm"],
            5.0,
            "the image's own value is still reported so it can be checked",
        )
        self.assertFalse(
            result["calibrated"],
            "an object set whose scale cannot be looked up must not be "
            "converted with a pixel size set after it existed",
        )
        self.assertIsNone(result["objects"]["values_in_pixel_size_nm"])
        self.assertIsNone(result["objects"]["density"]["per_um2"])
        self.assertIsNone(result["composition"]["tissue_um2"])

    def test_the_blanks_come_with_the_sentence_that_explains_them(self):
        """A blank column with no caveat is the failure this guard replaces."""
        _run, got = self._objects_from_an_unknown_pack()
        caveats = " ".join(got["result"]["caveats"])

        self.assertIn(self.UNKNOWN_PACK, caveats)
        self.assertIn("not a pack this build knows", caveats)
        self.assertIn("cannot be looked up", caveats)

    def test_it_agrees_with_the_manifest_caveat_on_the_same_run(self):
        _run, got = self._objects_from_an_unknown_pack()
        blob = json.dumps(got["manifest"])

        self.assertIn("would have resampled the image to its own", blob)

    def test_a_known_pack_that_declares_no_canonical_scale_stays_quiet(self):
        """The other ``None``, which must keep behaving as it always has.

        ``quantem:er`` is released and declares no canonical scale, so its run
        was identical with or without a pixel size. Warning about it would be
        crying wolf, and blanking its microns would be withholding real numbers.
        """
        self.assertEqual(loaders._canonical_nm("quantem:er"), (None, True))

        er = self._segmentation(get_or_create_er_type)
        for i in range(3):
            self._object(
                er,
                _square(20 + 30 * i, 20),
                source_model="quantem:er",
                stamp=_uncalibrated_stamp(pack_id="quantem:er"),
            )
        _run, got = self._run(er, compartments={"er": str(er.id)})

        self.assertTrue(got["result"]["calibrated"])
        self.assertEqual(got["result"]["objects"]["values_in_pixel_size_nm"], 5.0)
