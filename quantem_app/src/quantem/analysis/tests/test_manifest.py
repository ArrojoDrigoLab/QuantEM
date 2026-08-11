"""A bundle has to say where its numbers came from, and what they are *not*.

Failures this covers, all of which shipped:

* A summary table printed ``feret_max_um, mean 1.601, sd 8.6e-5`` under a
  heading that said ninety mitochondria. The n was four, it was in the table,
  and nothing said the four were the objects a person had drawn by hand. The n
  now travels with its reason.
* A manifest recorded the model as the string ``"quantem:mito"`` and nothing
  else -- no weight digests, no threshold, no scale, no library versions, no
  checksum of the image, no commit. None of that is recoverable after the fact,
  so a bundle without it cannot be reproduced, only re-run and hoped over.
* Every model weight was pinned by a sha256 and **the numbers were not**.
  Nothing tied a given ``objects.csv`` to a given manifest, and an edited cell
  left no trace.
* The manifest named the author's machine: ``release.git_repository`` and the
  image's ``file.path``. The release-bundle path scanner, run over an analysis
  manifest, returned both.
* An uncalibrated image blanked every ``_um`` column and said so -- and printed
  ``n_objects``, ``area_fraction_*``, ``enrichment_*`` and ``z_*`` in full with
  nothing attached, though inference had run at a scale the pack is not applied
  at. On one image the same pixels gave six objects untagged and three at 5 nm.
* The bundle could say how many objects a person confirmed and not how many they
  threw away, nor how much of the image they had been through.
* ``enrichment_mito`` and ``z_enrichment_mito`` sat in ``image_summary.csv`` with
  no field saying they were circular by construction. People open that file in
  Excel.

Anything genuinely unobtainable is ``null`` *plus a reason*. A silently omitted
adapter reads exactly like a run that had none.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from django.test import TestCase
from shapely.geometry import Polygon

from quantem.analysis import loaders, provenance, service
from quantem.analysis.compartments import CompartmentSet, area_fractions
from quantem.analysis.models import AnalysisRun
from quantem.analysis.morphometrics import ObjectMetrics, summarize
from quantem.core.config import STORAGE_DIR
from quantem.registry.release import find_local_paths
from quantem.segmentation.models import (
    CompletedROI,
    ImageSegmentation,
    SegmentObject,
)
from quantem.segmentation.type_service import (
    get_or_create_mitochondria_type,
    get_or_create_tissue_type,
)
from quantem.testing import create_small_test_image

IMAGE_SIZE = 240

#: What a model-extracted object carries: the full vocabulary.
MODEL_FEATURES: dict[str, Any] = {
    "area": 400.0,
    "perimeter": 80.0,
    "eccentricity": 0.4,
    "solidity": 0.97,
    "elongation": 1.1,
    "major_axis_length": 24.0,
    "minor_axis_length": 21.0,
    "feret_diameter_max": 26.0,
    "intensity_mean": 118.0,
    "intensity_p10": 90.0,
    "intensity_p50": 119.0,
    "intensity_p90": 145.0,
    "mean_prob": 0.82,
    "mito_generated": True,
}

#: What a hand-drawn object carries: everything but the model's probability.
MANUAL_FEATURES: dict[str, Any] = {
    k: v for k, v in MODEL_FEATURES.items() if k not in {"mean_prob", "mito_generated"}
}


def _square(x: float, y: float, side: float = 20.0) -> Polygon:
    return Polygon(((x, y), (x + side, y), (x + side, y + side), (x, y + side), (x, y)))


def _walk(node: Any):
    """Every dict in a nested structure."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


class ManifestTestCase(TestCase):
    """One calibrated image; six model objects and two hand-drawn ones."""

    n_model = 6
    n_manual = 2

    def setUp(self) -> None:
        self.image = create_small_test_image("Manifest Image", width=IMAGE_SIZE, height=IMAGE_SIZE)
        self.asset = self.image.asset
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.asset, segmentation_type=get_or_create_mitochondria_type()
        )
        for i in range(self.n_model):
            self._segment(_square(20 + 30 * i, 30), MODEL_FEATURES)
        for i in range(self.n_manual):
            self._segment(_square(20 + 30 * i, 90), MANUAL_FEATURES)

        self.exports_root = STORAGE_DIR / "exports_test" / self.id().rsplit(".", 1)[-1]
        shutil.rmtree(self.exports_root, ignore_errors=True)
        patcher = mock.patch.object(service, "EXPORTS_DIR", self.exports_root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.exports_root, ignore_errors=True)

    def _segment(
        self, polygon: Polygon, features: dict, *, label_state: str = "CONFIRMED"
    ) -> SegmentObject:
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            features=dict(features),
        )

    def _rows(self, out: Path, name: str) -> list[dict[str, str]]:
        return list(csv.DictReader((out / name).open(encoding="utf-8-sig")))

    def _run(self, **params: Any) -> tuple[AnalysisRun, Path, dict]:
        params = loaders.normalise_params(params, segmentation=self.segmentation)
        run = AnalysisRun.objects.create(segmentation=self.segmentation, params=params)
        result = service.run_for_segmentation(run)
        out = service.export_dir_for_run(run.id)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        return run, out, {"result": result, "manifest": manifest}


class ObjectProvenanceTests(ManifestTestCase):
    def test_objects_csv_says_which_objects_were_drawn_by_hand(self):
        _, out, _ = self._run()
        rows = list(csv.DictReader((out / "objects.csv").open(encoding="utf-8-sig")))
        assert len(rows) == self.n_model + self.n_manual
        sources = sorted(r["source_model"] for r in rows)
        assert sources.count("manual") == self.n_manual
        assert sources.count("quantem:mito") == self.n_model

    def test_the_manifest_records_the_hand_drawn_split(self):
        _, _, got = self._run()
        objects = got["manifest"]["objects"]
        assert objects["n_total"] == self.n_model + self.n_manual
        assert objects["n_hand_drawn"] == self.n_manual
        assert objects["n_model_produced"] == self.n_model
        assert objects["n_by_source"] == {
            "manual": self.n_manual,
            "quantem:mito": self.n_model,
        }

    def test_a_partly_measured_metric_carries_its_reason(self):
        """``mean_prob`` is the honest case: a drawn polygon has no model
        probability. The number still may not be quoted as a whole-image one."""
        _, _, got = self._run()
        stats = got["result"]["objects"]["summary"]["mean_prob"]
        assert stats["n"] == self.n_model
        assert stats["n_objects"] == self.n_model + self.n_manual
        assert stats["n_missing"] == self.n_manual
        assert stats["missing_by_source"] == {"manual": self.n_manual}
        note = stats["note"]
        assert f"{self.n_model} of {self.n_model + self.n_manual}" in note
        assert "hand-drawn" in note
        assert "no model probability" in note

    def test_a_fully_measured_metric_carries_no_excuse(self):
        _, _, got = self._run()
        stats = got["result"]["objects"]["summary"]["area_px"]
        assert stats["n"] == self.n_model + self.n_manual
        assert stats["n_missing"] == 0
        assert "note" not in stats

    def test_the_caveat_list_names_the_partly_measured_metrics(self):
        _, _, got = self._run()
        caveats = " ".join(got["result"]["caveats"])
        assert "mean_prob" in caveats
        assert "whole-image number" in caveats

    def test_the_manifest_repeats_the_reason_where_the_numbers_live(self):
        _, _, got = self._run()
        partial = got["manifest"]["objects"]["partially_measured_metrics"]
        assert "mean_prob" in partial
        assert partial["mean_prob"]["n"] == self.n_model
        assert partial["mean_prob"]["notes"]


class ManifestCompletenessTests(ManifestTestCase):
    def test_the_model_is_identified_by_more_than_its_name(self):
        _, _, got = self._run()
        packs = got["manifest"]["models"]["model_packs"]
        assert [p["pack_id"] for p in packs] == ["quantem:mito"]
        pack = packs[0]
        # Architecture and scale come from the release and are always known.
        assert pack["family"] == "quantem"
        assert pack["canonical_nm"] == 8.0
        assert pack["tile_size"] == 512
        assert pack["default_threshold"] is not None
        # Digests come from the install record. On a machine where the pack is
        # not installed they are null *with the reason*, never absent.
        if "weights" in pack and pack["weights"]:
            assert pack["weights"]["head"]["sha256"]
        else:
            assert "not installed" in pack["unavailable"]["weights"]

    def test_the_threshold_that_decided_the_objects_is_recorded(self):
        _, _, got = self._run()
        run = got["manifest"]["models"]["compartments"][0]["run"]
        threshold = run["foreground_threshold"]["by_pack"]["quantem:mito"]
        assert threshold["value"] == 0.5
        assert "published default" in threshold["source"]

    def test_configured_instance_params_are_not_passed_off_as_used(self):
        """They are recorded, and the record says the segmenter ignores them.
        ``DinoOrganelleSegmenter.__init__`` absorbs ``instance_params`` into
        ``**_ignored``; claiming otherwise in a manifest would be a fabricated
        provenance."""
        _, _, got = self._run()
        params = got["manifest"]["models"]["compartments"][0]["run"]["instance_params"]
        assert params["segmentation_threshold"] is not None
        assert params["center_min_distance"] is not None
        assert params["center_confidence_threshold"] is not None
        assert "not as evidence that this run used them" in params["_note"]

    def test_the_scale_is_recorded(self):
        _, _, got = self._run()
        run = got["manifest"]["models"]["compartments"][0]["run"]
        scale = run["scale"]
        assert scale["native_pixel_size_nm"] is not None
        assert scale["canonical_nm_by_pack"] == {"quantem:mito": 8.0}
        assert scale["ran_at"] in {"native", "canonical"}
        assert scale["note"]

    def test_an_absent_adapter_is_stated_not_implied(self):
        _, _, got = self._run()
        adapter = got["manifest"]["models"]["compartments"][0]["run"]["adapter"]
        assert adapter is not None, "a missing adapter section reads as no adapter"
        assert adapter["applied"] is False

    def test_the_image_is_identified_by_content_not_by_a_local_uuid(self):
        _, _, got = self._run()
        image = got["manifest"]["models"]["image"]
        assert image["image_id"]
        file_info = image["file"]
        assert file_info.get("sha256") or file_info.get("unavailable")
        if file_info.get("sha256"):
            assert len(file_info["sha256"]) == 64
            assert file_info["filename"]

    def test_the_environment_that_produced_the_numbers_is_pinned(self):
        _, _, got = self._run()
        env = got["manifest"]["environment"]
        for name in ("numpy", "scipy", "scikit-image", "shapely"):
            assert env["packages"][name], f"{name} version missing"
        # The note used to say the min-area boundary moved with scikit-image.
        # It does not: filter_min_area counts components itself precisely so a
        # version bump cannot change an object count. A manifest that blames a
        # pin which provably cannot explain a mismatch sends its reader the
        # wrong way, so the note has to say why the pin is really there.
        note = env["skimage_note"]
        assert "binary_closing" in note and "regionprops" in note
        assert "filter_min_area" in note
        assert "cannot move with the library version" in note
        assert "inference_device" in env["unavailable"]

    def test_the_code_is_identified_by_more_than_a_dev_version_string(self):
        _, _, got = self._run()
        release = got["manifest"]["release"]
        assert release["quantem_version"]
        assert release.get("git_commit") or release["unavailable"]["git_commit"]

    def test_every_null_in_the_manifest_has_a_stated_reason(self):
        """The rule the whole file exists for: no silent omission, no guess."""
        _, _, got = self._run()
        for node in _walk(got["manifest"]):
            for field, reason in (node.get("unavailable") or {}).items():
                head = field.split(".", 1)[0]
                assert head in node, (
                    f"{field!r} is explained but not present as null in {sorted(node)}"
                )
                assert isinstance(reason, str) and len(reason) > 20, (
                    f"{field!r} is null with no usable reason: {reason!r}"
                )


class SummaryNoteUnitTests(TestCase):
    """The note is produced where the number is, so it cannot be rendered away."""

    def test_the_note_names_every_source_that_is_missing_the_metric(self):
        metrics = [
            ObjectMetrics("a", True, {"feret_max_px": 12.0}, "manual"),
            ObjectMetrics("b", True, {"feret_max_px": None}, "manual"),
            ObjectMetrics("c", True, {"feret_max_px": None}, "quantem:mito"),
            ObjectMetrics("d", True, {"feret_max_px": None}, "quantem:mito"),
        ]
        note = summarize(metrics)["feret_max_px"]["note"]
        assert "1 of 4" in note
        assert "1 hand-drawn" in note
        assert "2 from quantem:mito" in note

    def test_an_object_of_unrecorded_origin_is_named_as_such(self):
        metrics = [
            ObjectMetrics("a", True, {"area_px": 1.0}, "quantem:mito"),
            ObjectMetrics("b", True, {"area_px": None}, ""),
        ]
        assert "unrecorded origin" in summarize(metrics)["area_px"]["note"]


class ProvenanceHelperTests(TestCase):
    def test_a_missing_file_is_null_with_a_reason_not_an_exception(self):
        info = provenance.file_identity(Path(STORAGE_DIR) / "does-not-exist.tif", what="the image")
        assert info["sha256"] is None
        assert "not present" in info["unavailable"]["sha256"]

    def test_an_unreleased_pack_is_reported_rather_than_guessed(self):
        pack = provenance.model_pack("nosuch:organelle")
        assert pack["canonical_nm"] is None
        assert "not one of the released model packs" in pack["unavailable"]["canonical_nm"]

    def test_a_file_is_identified_by_name_and_digest_and_not_by_where_it_lives(self):
        """The bundle goes to a collaborator. Their copy of the file has a
        different path and the same bytes."""
        target = Path(STORAGE_DIR) / "provenance-identity-probe.bin"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"quantem")
        self.addCleanup(target.unlink, missing_ok=True)

        info = provenance.file_identity(target, what="the image")

        assert info["filename"] == "provenance-identity-probe.bin"
        assert len(info["sha256"]) == 64
        assert "path" not in info
        assert not find_local_paths(json.dumps(info))

    def test_a_dotted_reason_explains_a_nested_null_without_inventing_a_key(self):
        """``canonical_nm_by_pack.<pack>`` names one entry of a mapping the
        section already carries. Creating a literal top-level key with a dot in
        it put a second, meaningless field beside the real one."""
        built = provenance.section(
            {"canonical_nm_by_pack": {"somebody:else": None}},
            {"canonical_nm_by_pack.somebody:else": "not a pack this build knows."},
        )

        assert built["canonical_nm_by_pack"] == {"somebody:else": None}
        assert "canonical_nm_by_pack.somebody:else" not in built
        assert built["unavailable"]["canonical_nm_by_pack.somebody:else"]

    def test_a_dotted_reason_for_a_mapping_that_is_missing_still_leaves_a_null(self):
        built = provenance.section({}, {"packages.torch": "torch is not installed."})

        assert built["packages"] is None
        assert built["unavailable"]["packages.torch"]


class BundleOutputChecksumTests(ManifestTestCase):
    """The numbers get the same treatment as the weights: named, and hashed.

    ``objects.csv`` is the artefact a paper cites. Nothing tied one to a given
    manifest and nothing detected an edit, while every model weight in the same
    file carried a sha256.
    """

    def _outputs(self, manifest: dict) -> dict[str, dict]:
        return {entry["filename"]: entry for entry in manifest["outputs"]["files"]}

    def test_both_csvs_are_named_and_hashed(self):
        _, out, got = self._run()
        outputs = self._outputs(got["manifest"])

        assert set(outputs) == {"objects.csv", "image_summary.csv"}
        for name, entry in outputs.items():
            digest = provenance.sha256_file(out / name)
            assert entry["sha256"] == digest, name
            assert entry["size_bytes"] == (out / name).stat().st_size

    def test_the_row_and_column_counts_describe_the_file_that_was_written(self):
        _, out, got = self._run()
        outputs = self._outputs(got["manifest"])

        for name, entry in outputs.items():
            rows = self._rows(out, name)
            assert entry["n_rows"] == len(rows), name
            assert entry["columns"] == list(rows[0]), name
        assert outputs["objects.csv"]["n_rows"] == self.n_model + self.n_manual

    def test_editing_one_cell_of_objects_csv_breaks_the_recorded_digest(self):
        """The whole point: an edited export no longer matches its manifest."""
        _, out, got = self._run()
        recorded = self._outputs(got["manifest"])["objects.csv"]["sha256"]

        text = (out / "objects.csv").read_text(encoding="utf-8")
        (out / "objects.csv").write_text(text.replace("400.0", "4000.0", 1), "utf-8")

        assert provenance.sha256_file(out / "objects.csv") != recorded

    def test_the_manifest_does_not_claim_to_carry_its_own_digest(self):
        _, _, got = self._run()
        outputs = got["manifest"]["outputs"]

        assert "manifest.json" not in self._outputs(got["manifest"])
        assert "cannot contain its own digest" in outputs["note"]
        assert "sha256" in outputs["verify"]


class NoLocalPathsInTheBundleTests(ManifestTestCase):
    """An export bundle is emailed to a collaborator. It may not name this box.

    The user ran ``quantem.registry.release.find_local_paths`` -- the detector
    that gates a model release -- over an analysis manifest and it returned
    ``release.git_repository`` and ``models.image.file.path``.
    """

    def _manifest_text(self, out: Path) -> str:
        return (out / "manifest.json").read_text(encoding="utf-8")

    def test_the_release_scanner_finds_nothing_in_the_whole_manifest(self):
        _, out, _ = self._run()

        assert find_local_paths(self._manifest_text(out)) == []

    def test_the_checkout_location_is_gone_and_the_commit_is_what_identifies_the_code(self):
        _, _, got = self._run()
        release = got["manifest"]["release"]

        assert "git_repository" not in release
        assert release["quantem_version"]
        assert release.get("git_commit") or release["unavailable"]["git_commit"]

    def test_the_image_keeps_the_two_facts_that_travel_and_drops_the_one_that_does_not(self):
        _, _, got = self._run()
        file_info = got["manifest"]["models"]["image"]["file"]

        assert "path" not in file_info
        assert file_info["filename"]
        assert len(file_info["sha256"]) == 64

    def test_the_sweep_records_itself_so_a_clean_bundle_is_evidence_not_luck(self):
        _, _, got = self._run()
        swept = got["manifest"]["local_paths"]

        assert swept["scanner"] == "quantem.registry.release.find_local_paths"
        assert swept["clean"] is True
        assert swept["spans_removed"] == 0, (
            "every field is built without a path; the sweep is the check, not the fix"
        )

    def test_a_path_that_slips_into_a_value_is_removed_and_counted(self):
        document = {"note": r"written from D:\example\QuantEM_repo", "n": 3}

        cleaned, report = provenance.scrub_local_paths(document)

        assert report["spans_removed"] == 1
        assert report["clean"] is True
        assert find_local_paths(json.dumps(cleaned)) == []
        assert cleaned["n"] == 3
        assert "D:" not in cleaned["note"]


class UncalibratedImageTestCase(ManifestTestCase):
    """The same six model objects, on an image nobody typed a pixel size into."""

    def setUp(self) -> None:
        super().setUp()
        self.asset.pixel_size_nm = None
        self.asset.save(update_fields=["pixel_size_nm"])


class DimensionlessNumbersEscapeTheUnitsGuardTests(UncalibratedImageTestCase):
    """The units guard is honest about units and silent about the object set.

    Reported measurement, same pixels: imported untagged -> 6 objects,
    ``area_fraction_mito`` 0.0106; 5 nm typed in -> 3 objects, 0.0098. The area
    fraction barely moved. The count halved, and it is the number that goes in a
    bar chart.
    """

    def _caveats(self, got: dict) -> str:
        return " ".join(got["result"]["caveats"])

    def test_the_units_caveat_is_still_there(self):
        _, _, got = self._run()
        assert "Pixel size is not set for this image" in self._caveats(got)

    def test_the_scale_caveat_names_the_packs_own_canonical_nm(self):
        _, _, got = self._run()
        scale = [c for c in got["result"]["caveats"] if "not trained for" in c]

        assert scale, got["result"]["caveats"]
        assert "quantem:mito is applied at 8.0 nm/px" in scale[0]

    def test_the_scale_caveat_names_the_columns_the_units_guard_does_not_blank(self):
        _, _, got = self._run()
        scale = next(c for c in got["result"]["caveats"] if "not trained for" in c)

        for column in ("n_objects", "area_fraction_*", "enrichment_*", "z_*"):
            assert column in scale, column
        assert "re-run inference" in scale

    def test_those_columns_are_in_fact_populated_while_the_micron_ones_are_blank(self):
        """The caveat has to exist because the numbers do."""
        _, out, _ = self._run()
        row = self._rows(out, "image_summary.csv")[0]

        assert row["n_objects"] == str(self.n_model + self.n_manual)
        assert row["area_fraction_mito"]
        assert row["tissue_um2"] == ""
        assert row["objects_per_um2"] == ""

    def test_a_calibrated_image_gets_no_scale_caveat(self):
        """It is resampled to the pack's scale, which is what the pack wants."""
        self.asset.pixel_size_nm = 5.0
        self.asset.save(update_fields=["pixel_size_nm"])

        _, _, got = self._run()

        assert not [c for c in got["result"]["caveats"] if "not trained for" in c]

    def test_a_hand_drawn_only_compartment_gets_no_scale_caveat(self):
        """No model ran, so no model ran at the wrong scale."""
        SegmentObject.objects.filter(
            segmentation=self.segmentation, source_model="quantem:mito"
        ).delete()

        _, _, got = self._run()

        assert not [c for c in got["result"]["caveats"] if "not trained for" in c]
        assert "Pixel size is not set" in self._caveats(got)

    def test_the_scale_caveat_reaches_the_spreadsheet_too(self):
        _, out, _ = self._run()
        row = self._rows(out, "image_summary.csv")[0]

        assert "not trained for" in row["caveats"]


class ProofreadingRecordTests(ManifestTestCase):
    """What a person threw away, and how much of the image they looked at."""

    #: The reported case scaled to this image: 84--1316 px of 1400 is 88% of the
    #: width, and 77% of the area.
    ROI = _square(14.4, 14.4, side=211.2)

    def _reject(self, n: int) -> None:
        for i in range(n):
            self._segment(
                _square(20 + 12 * i, 150, side=8.0),
                MODEL_FEATURES,
                label_state="EXCLUDED",
            )

    def _review(self, polygon: Polygon) -> None:
        CompletedROI.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            bbox=polygon.envelope,
        )

    def test_the_rejected_count_is_recorded_where_the_confirmed_one_is(self):
        self._reject(14)
        _, _, got = self._run()
        block = got["manifest"]["models"]["compartments"][0]["proofreading"]

        assert block["n_confirmed"] == self.n_model + self.n_manual
        assert block["n_rejected"] == 14
        assert block["n_by_label_state"]["EXCLUDED"] == 14
        assert block["n_by_label_state"]["CONFIRMED"] == self.n_model + self.n_manual

    def test_rejections_are_a_caveat_because_they_are_in_no_number_above(self):
        self._reject(14)
        _, _, got = self._run()

        named = [c for c in got["result"]["caveats"] if "reviewed and rejected" in c]
        assert named, got["result"]["caveats"]
        assert "14 candidate objects" in named[0]

    def test_nothing_rejected_is_zero_and_reads_as_a_measurement(self):
        _, out, got = self._run()

        assert got["result"]["proofreading"]["n_rejected"] == 0
        assert self._rows(out, "image_summary.csv")[0]["n_rejected"] == "0"
        assert not [c for c in got["result"]["caveats"] if "reviewed and rejected" in c]

    def test_the_reviewed_area_is_measured_and_carried_into_the_summary(self):
        self._review(self.ROI)
        _, out, got = self._run()

        reviewed = got["manifest"]["models"]["compartments"][0]["proofreading"]["reviewed_area"]
        assert reviewed["n_regions"] == 1
        assert reviewed["image_px"] == IMAGE_SIZE * IMAGE_SIZE
        assert 0.7 < reviewed["reviewed_fraction"] < 0.8
        assert reviewed["bbox_px"] == [14.4, 14.4, 225.6, 225.6]

        row = self._rows(out, "image_summary.csv")[0]
        assert 0.7 < float(row["reviewed_fraction"]) < 0.8

    def test_a_partly_reviewed_image_says_so_beside_its_whole_image_counts(self):
        self._review(self.ROI)
        _, _, got = self._run()

        named = [c for c in got["result"]["caveats"] if "marked 77%" in c]
        assert named, got["result"]["caveats"]
        assert "over the whole image" in named[0]

    def test_no_completed_area_is_unknown_with_a_reason_and_never_zero(self):
        """ "Nobody marked a region" and "a person reviewed none of it" are
        different facts, and only one of them is knowable here."""
        _, out, got = self._run()

        reviewed = got["manifest"]["models"]["compartments"][0]["proofreading"]["reviewed_area"]
        assert reviewed["reviewed_px"] is None
        assert "unknown rather than zero" in reviewed["unavailable"]["reviewed_px"]
        assert self._rows(out, "image_summary.csv")[0]["reviewed_fraction"] == ""
        assert [c for c in got["result"]["caveats"] if "recorded as reviewed" in c]


class SpreadsheetCaveatTests(ManifestTestCase):
    """People open ``image_summary.csv`` in Excel and quote what they find."""

    def test_the_caveats_are_in_the_file_the_numbers_are_in(self):
        _, out, got = self._run()
        row = self._rows(out, "image_summary.csv")[0]

        assert int(row["n_caveats"]) == len(got["result"]["caveats"])
        for caveat in got["result"]["caveats"]:
            assert caveat in row["caveats"]

    def test_the_caveat_columns_come_after_every_number(self):
        _, out, _ = self._run()
        columns = list(self._rows(out, "image_summary.csv")[0])

        assert columns[-3:] == ["circular_columns", "n_caveats", "caveats"]

    def test_a_circular_enrichment_names_the_exact_columns_it_condemns(self):
        _, out, _ = self._run(points_source="centroids")
        row = self._rows(out, "image_summary.csv")[0]

        assert row["enrichment_mito"], "the number is there to be quoted"
        assert row["circular_columns"] == "enrichment_mito z_enrichment_mito"
        assert "circular" in row["caveats"]

    def test_the_area_fraction_is_not_condemned_with_them(self):
        """It is a real measurement, and the denominator that makes the others
        circular."""
        _, out, _ = self._run(points_source="centroids")
        row = self._rows(out, "image_summary.csv")[0]

        assert "area_fraction_mito" not in row["circular_columns"]

    def test_an_honest_enrichment_condemns_nothing(self):
        _, out, _ = self._run()

        assert self._rows(out, "image_summary.csv")[0]["circular_columns"] == ""

    def test_the_caveat_cell_survives_a_round_trip_as_one_field(self):
        _, out, _ = self._run(points_source="centroids")
        rows = self._rows(out, "image_summary.csv")

        assert len(rows) == 1, "a newline in the caveats would split the record"
        assert "\n" not in rows[0]["caveats"]


class ImpossibleCalibrationTests(TestCase):
    """A negative pixel size is not an absent one, and must not read as one."""

    def _inputs(self, pixel_size_nm: float | None) -> service.AnalysisInputs:
        mask = np.zeros((10, 10), dtype=bool)
        mask[:, :4] = True
        return service.AnalysisInputs(
            image_key="probe",
            pixel_size_nm=pixel_size_nm,
            compartments=CompartmentSet(masks={"mito": mask}, tissue=np.ones((10, 10), bool)),
            object_features={"a": {"area": 16.0, "perimeter": 16.0}},
        )

    def test_the_sentence_says_what_is_actually_wrong(self):
        caveats = service.run_analysis(self._inputs(-5.0))["caveats"]

        assert not any("Pixel size is not set" in c for c in caveats), (
            "an impossible calibration is a different problem from an absent one"
        )
        named = [c for c in caveats if "-5.0 nm/px" in c]
        assert named, caveats
        assert "not a length" in named[0]

    def test_an_absent_pixel_size_still_gets_its_own_sentence(self):
        caveats = service.run_analysis(self._inputs(None))["caveats"]

        assert any("Pixel size is not set" in c for c in caveats)
        assert not any("not a length" in c for c in caveats)

    def test_the_bad_value_is_reported_rather_than_hidden(self):
        result = service.run_analysis(self._inputs(-5.0))

        assert result["pixel_size_nm"] == -5.0, "so it can be found and fixed"
        assert result["calibrated"] is False

    def test_no_micron_value_is_computed_from_it(self):
        """``(-5/1000)**2 == (5/1000)**2``: a truthiness test alone turned an
        impossible calibration into the micron areas of a plausible one."""
        result = service.run_analysis(self._inputs(-5.0))

        assert result["composition"]["tissue_um2"] is None
        assert result["composition"]["areas_um2"] is None
        assert (result["objects"]["density"] or {}).get("per_um2") is None

    def test_area_fractions_refuses_a_non_positive_scale_directly(self):
        comp = CompartmentSet(masks={"mito": np.ones((4, 4), bool)}, tissue=np.ones((4, 4), bool))

        assert area_fractions(comp, pixel_size_nm=-5.0).tissue_um2 is None
        assert area_fractions(comp, pixel_size_nm=0.0).tissue_um2 is None
        assert area_fractions(comp, pixel_size_nm=5.0).tissue_um2 == 16 * 5e-3**2


class EmptyCompartmentSetTests(TestCase):
    """Unreachable through ``normalise_params``; this package is documented as
    usable from a notebook, where nothing normalises anything."""

    def test_a_set_with_no_masks_says_what_is_missing(self):
        """It used to raise ``StopIteration`` out of a property, which is both
        unreadable and, inside a generator, silently the end of a loop. Anything
        other than the ValueError fails this, including that."""
        empty = CompartmentSet(masks={})

        with self.assertRaises(ValueError) as caught:
            _shape = empty.shape

        assert "neither a tissue mask nor any compartment" in str(caught.exception)

    def test_a_tissue_mask_alone_is_enough(self):
        assert CompartmentSet(masks={}, tissue=np.ones((3, 5), bool)).shape == (3, 5)


class EncoderRunDirTests(TestCase):
    """A raw install records the encoder run by its path on the training box.

    The run id -- the directory's *name* -- is what identifies the encoder and
    has to survive; its parents name a disk. Reduced with the release module's
    own :func:`quantem.registry.release.encoder_run_dir`, so the analysis
    manifest and a shipped pack agree on what a run dir is.
    """

    RECORD = {
        "head": {"filename": "head.pt", "sha256": "0" * 64, "size_bytes": 12},
        "checkpoint_step": 674999,
        "source": "raw",
    }

    def _pack(self, encoder_run_dir: str) -> dict:
        record = dict(self.RECORD, encoder_run_dir=encoder_run_dir)
        with mock.patch("quantem.registry.cache.read_record", return_value=record):
            return provenance.model_pack("quantem:mito")

    def test_an_absolute_run_dir_is_reduced_to_the_run_id_and_says_so(self):
        pack = self._pack(r"D:\example\legacy\runs\m1_dinov3_vitb")

        assert pack["encoder_run_dir"] == "m1_dinov3_vitb"
        assert pack["encoder_run_id"] == "m1_dinov3_vitb"
        assert "the parents identify somebody's disk" in pack["encoder_run_dir_note"]
        assert find_local_paths(json.dumps(pack)) == []

    def test_a_relative_run_dir_is_how_a_run_identifies_itself_and_is_kept(self):
        pack = self._pack("foundation_weights/m1_dinov3_vitb")

        assert pack["encoder_run_dir"] == "foundation_weights/m1_dinov3_vitb"
        assert pack["encoder_run_id"] == "m1_dinov3_vitb"
        assert "encoder_run_dir_note" not in pack
        assert find_local_paths(json.dumps(pack)) == []


class FabricatedPointObservationTests(ManifestTestCase):
    """Three unreadable coordinates used to produce a 31-fold enrichment.

    ``float("nan")``, ``float("inf")`` and ``1e400`` all parse, and
    ``np.round(nan).astype(int)`` is ``INT_MIN``, which the clip to the image
    turns into 0. Every unusable row therefore became a genuine observation at
    pixel (0, 0), and the only trace was a ``RuntimeWarning`` on stderr.
    """

    def setUp(self) -> None:
        super().setUp()
        # A tissue mask over the whole image, so nothing is excluded for being
        # off-tissue and the enrichment below is about the coordinates alone.
        self.tissue = ImageSegmentation.objects.create(
            asset=self.asset, segmentation_type=get_or_create_tissue_type()
        )
        whole = _square(0, 0, side=IMAGE_SIZE)
        SegmentObject.objects.create(
            segmentation=self.tissue,
            geometry=whole,
            centroid=whole.centroid,
            bbox=whole.envelope,
            label_state="CONFIRMED",
            features=dict(MANUAL_FEATURES),
        )

    def _junk_run(self):
        return self._run(
            tissue_segmentation_id=str(self.tissue.id),
            points_source="csv",
            # One real point, then three rows that are not positions at all.
            points_csv="x,y\n30,40\nnan,nan\ninf,0\n-inf,0\n",
        )

    def test_unreadable_rows_never_reach_the_measurement(self):
        _, _, got = self._junk_run()
        points = got["result"]["points"]

        assert points["n_total"] == 1, "the three junk rows are gone before this"
        assert points["n_unreadable"] == 0
        assert points["n_on_tissue"] == 1

    def test_the_dropped_rows_are_named_by_line_in_the_caveats(self):
        _, _, got = self._junk_run()

        named = [c for c in got["result"]["caveats"] if "not a position" in c]
        assert named, got["result"]["caveats"]
        assert "3 of 4 rows" in named[0]
        assert "line 3 (nan,nan)" in named[0]
        assert "line 4 (inf,0)" in named[0]
        assert "not (0, 0)" in named[0]

    def test_the_manifest_records_the_rows_it_could_not_read(self):
        _, _, got = self._junk_run()
        block = got["manifest"]["models"]["points"]

        assert block["source"] == "csv"
        assert block["n_rows_read"] == 4
        assert block["n_unreadable"] == 3
        assert block["n_points"] == 1
        assert [entry["line"] for entry in block["unreadable_lines"]] == [3, 4, 5]

    def test_a_csv_of_nothing_but_unreadable_rows_is_refused_at_request_time(self):
        with self.assertRaises(loaders.AnalysisInputError) as caught:
            loaders.normalise_params(
                {"points_source": "csv", "points_csv": "x,y\nnan,nan\ninf,inf\n"},
                segmentation=self.segmentation,
            )
        assert "nothing to measure" in str(caught.exception)


class NoPointOnTissueBundleTests(ManifestTestCase):
    """``enrichment_mito = 0.0`` is maximal depletion. 0/0 is not a number."""

    def setUp(self) -> None:
        super().setUp()
        # Tissue is a small square in the corner; the points below all miss it.
        self.tissue = ImageSegmentation.objects.create(
            asset=self.asset, segmentation_type=get_or_create_tissue_type()
        )
        patch = _square(0, 0, side=60.0)
        SegmentObject.objects.create(
            segmentation=self.tissue,
            geometry=patch,
            centroid=patch.centroid,
            bbox=patch.envelope,
            label_state="CONFIRMED",
            features=dict(MANUAL_FEATURES),
        )

    def _off_tissue_run(self):
        return self._run(
            tissue_segmentation_id=str(self.tissue.id),
            points_source="csv",
            points_csv="x,y\n200,200\n210,215\n230,190\n",
        )

    def test_every_enrichment_is_undefined_rather_than_zero(self):
        _, _, got = self._off_tissue_run()
        points = got["result"]["points"]

        assert points["n_on_tissue"] == 0
        assert points["n_off_tissue"] == 3
        # The compartment does have area, so this is not the empty-compartment
        # case: the 0/0 is on the numerator side.
        assert got["result"]["composition"]["area_fractions"]["mito"] > 0
        assert points["enrichment"]["mito"] is None

    def test_the_spreadsheet_cell_is_blank_and_not_a_hard_zero(self):
        _, out, _ = self._off_tissue_run()
        row = self._rows(out, "image_summary.csv")[0]

        assert row["enrichment_mito"] == "", "someone sorts on this column"

    def test_no_p_value_is_reported_from_a_null_of_identical_zeros(self):
        _, _, got = self._off_tissue_run()

        assert got["result"].get("monte_carlo") is None
        mc = got["manifest"]["monte_carlo"]
        assert mc["replicates"] is None
        assert "null of identical zeros" in mc["unavailable"]["replicates"]

    def test_the_reader_is_told_why_in_words(self):
        _, _, got = self._off_tissue_run()

        named = [c for c in got["result"]["caveats"] if "undefined rather than zero" in c]
        assert named, got["result"]["caveats"]
        assert "None of the 3 points is on the tissue mask" in named[0]


class ObjectsCsvTraceabilityTests(ManifestTestCase):
    """``objects.csv`` is the file that goes into R or Prism."""

    ROI = _square(0, 0, side=120.0)

    def test_every_row_names_its_run_its_segmentation_and_its_caveat_count(self):
        run, out, got = self._run()
        rows = self._rows(out, "objects.csv")

        assert rows
        for row in rows:
            assert row["analysis_run_id"] == str(run.id)
            assert row["segmentation_id"] == str(self.segmentation.id)
            assert row["image_key"] == str(self.asset.id)
            assert int(row["n_caveats"]) == len(got["result"]["caveats"])

    def test_the_reviewed_and_the_unreviewed_are_distinguishable(self):
        """The caveat said in so many words that they were not."""
        CompletedROI.objects.create(
            segmentation=self.segmentation,
            geometry=self.ROI,
            bbox=self.ROI.envelope,
        )
        _, out, got = self._run()
        rows = self._rows(out, "objects.csv")

        flags = {row["object_id"]: row["in_reviewed_area"] for row in rows}
        assert set(flags.values()) == {"True", "False"}, flags
        # Six model objects at x = 20 + 30i, y = 30 (centroids x = 30 + 30i,
        # y = 40) and two hand-drawn at y = 90 (centroids y = 100), all side 20.
        # Inside the 120 px square: the four model centroids at x <= 120, plus
        # both hand-drawn ones.
        assert sum(value == "True" for value in flags.values()) == 6
        assert sum(value == "False" for value in flags.values()) == 2

        partly = [c for c in got["result"]["caveats"] if "never gone through" in c]
        assert partly, "this is the caveat the column answers"

    def test_no_completed_area_leaves_the_column_blank_not_false(self):
        _, out, _ = self._run()

        for row in self._rows(out, "objects.csv"):
            assert row["in_reviewed_area"] == "", "nobody said is not 'outside'"

    def test_the_reviewed_regions_themselves_are_in_the_manifest(self):
        CompletedROI.objects.create(
            segmentation=self.segmentation,
            geometry=self.ROI,
            bbox=self.ROI.envelope,
        )
        _, _, got = self._run()
        block = got["manifest"]["images"][0]["proofreading"]

        assert block["n_regions"] == 1
        # An area and a bbox cannot be turned back into which pixels were swept.
        from shapely import wkt as shapely_wkt

        recovered = shapely_wkt.loads(block["regions_wkt"])
        assert abs(recovered.area - 120.0 * 120.0) < 1e-6
        assert recovered.bounds == (0.0, 0.0, 120.0, 120.0)


class BundleIdentityTests(ManifestTestCase):
    """A bundle that has been moved must still be able to name its own run."""

    def test_the_manifest_carries_the_run_id_not_only_the_directory_name(self):
        run, out, got = self._run()

        assert got["manifest"]["analysis_run_id"] == str(run.id)
        assert got["manifest"]["segmentation_ids"] == [str(self.segmentation.id)]
        # The directory is only where it happens to sit, and moving it loses it.
        assert out.name == str(run.id)

    def test_both_csvs_carry_it_too(self):
        run, out, _ = self._run()

        assert self._rows(out, "image_summary.csv")[0]["analysis_run_id"] == str(run.id)
        assert self._rows(out, "objects.csv")[0]["analysis_run_id"] == str(run.id)

    def test_a_run_with_no_point_set_says_why_it_has_no_monte_carlo(self):
        _, _, got = self._run()
        mc = got["manifest"]["monte_carlo"]

        assert mc["replicates"] is None and mc["seed"] is None
        assert "analysed no point set" in mc["unavailable"]["replicates"]
        assert "analysed no point set" in mc["unavailable"]["seed"]

    def test_who_supplied_the_pixel_size_is_recorded(self):
        """0 -> 25 -> 134 objects on identical pixels turns on this number."""
        _, _, got = self._run()
        block = got["manifest"]["models"]["image"]["pixel_size_provenance"]

        assert block["effective_nm"] == self.asset.pixel_size_nm
        assert block["source"] == "entered_by_hand"
        assert block["file_declared_nm"] is None
        assert "typed by a person" in block["note"]

    def test_a_digest_of_a_converted_file_does_not_pass_for_the_source(self):
        self.asset.original_filename = "twin_untagged_c.tif"
        self.asset.save()
        _, _, got = self._run()
        image = got["manifest"]["models"]["image"]

        assert image["original_filename"] == "twin_untagged_c.tif"
        assert image["file"]["filename"].endswith(".png")
        assert image["source_file_sha256"] is None
        reason = image["unavailable"]["source_file_sha256"]
        assert "does not record a checksum of the uploaded bytes" in reason


class MultiImageCoverageTotalsTests(ManifestTestCase):
    """``partially_measured_metrics`` mixed two scopes and reconciled neither.

    ``n`` and ``n_objects`` are summed over every image in the bundle; the
    sentences in ``notes`` are kept verbatim, one per distinct wording, and each
    was written for a single image. Together that produced

        "mean_prob": {"n": 0, "n_objects": 16, "notes": [
            "... Measured on 0 of 8 confirmed objects ..."]}

    which is two images of eight, not a contradiction -- and nothing in the
    block said which reading was right.
    """

    def _two_image_bundle(self) -> dict:
        import copy

        _, _, got = self._run()
        first = got["result"]
        second = copy.deepcopy(first)
        second["image_key"] = "second-image"
        out = self.exports_root / "two-images"
        service.write_bundle([first, second], out)
        return json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    def test_the_totals_and_the_sentence_no_longer_read_as_one_number(self):
        manifest = self._two_image_bundle()
        entry = manifest["objects"]["partially_measured_metrics"]["mean_prob"]
        per_image = self.n_model + self.n_manual

        # The totals are still totals -- they are the bundle-level answer.
        assert entry["n"] == 2 * self.n_model
        assert entry["n_objects"] == 2 * per_image
        assert entry["n_images"] == 2

        # The sentence is still one image's, verbatim.
        assert f"of {per_image}" in " ".join(entry["notes"])

        # And now something says which is which.
        assert "totals over the 2 images" in entry["counts_note"]
        assert "one of them" in entry["counts_note"]

    def test_by_image_reconciles_them_object_by_object(self):
        manifest = self._two_image_bundle()
        entry = manifest["objects"]["partially_measured_metrics"]["mean_prob"]
        per_image = self.n_model + self.n_manual

        assert len(entry["by_image"]) == 2
        assert {e["image_key"] for e in entry["by_image"]} == {
            str(self.asset.id),
            "second-image",
        }
        assert sum(e["n"] for e in entry["by_image"]) == entry["n"]
        assert sum(e["n_objects"] for e in entry["by_image"]) == entry["n_objects"]
        for e in entry["by_image"]:
            assert e["n_objects"] == per_image
            assert f"of {per_image}" in e["note"]

    def test_a_single_image_bundle_says_so_rather_than_going_silent(self):
        _, _, got = self._run()
        entry = got["manifest"]["objects"]["partially_measured_metrics"]["mean_prob"]

        assert entry["n_images"] == 1
        assert entry["n_objects"] == self.n_model + self.n_manual
        assert "one image" in entry["counts_note"]


class SizeFloorAppliesToOneProvenanceTests(ManifestTestCase):
    """The minimum-area floor is a model filter, and the note read as a global.

    ``filter_min_area`` runs inside inference. A polygon a person drew is stored
    directly and never passes through it, so a 36 px hand-drawn object ships in
    ``objects.csv`` out of a compartment whose model floor is 60 px -- while a
    model-produced component of the same 36 px was discarded before it could
    become an object at all.

    The manifest scoped the value correctly, under ``by_pack``, and then said
    "Connected components below this area ... are not in any count, area
    fraction or density in this bundle", which is a claim about the bundle and
    is false. The consequence is not wording: a size distribution pooled over
    both provenances is left-truncated for one of them and not the other.
    """

    #: quantem:mito's default_min_area. Asserted below rather than assumed.
    FLOOR_PX = 60
    SMALL_PX = 36.0

    def _draw_one_below_the_floor(self):
        side = self.SMALL_PX**0.5
        self._segment(
            _square(150, 150, side=side),
            {**MANUAL_FEATURES, "area": self.SMALL_PX},
        )

    @staticmethod
    def _min_area(manifest: dict) -> dict:
        for block in manifest["models"]["compartments"]:
            if block["compartment"] == "mito":
                return block["run"]["min_area"]
        raise AssertionError("no mito compartment in the manifest")

    def test_the_premise_the_floor_is_the_one_this_pack_uses(self):
        _, _, got = self._run(compartments={"mito": str(self.segmentation.id)})
        entry = self._min_area(got["manifest"])["by_pack"]["quantem:mito"]
        assert entry["value"] == self.FLOOR_PX

    def test_the_note_no_longer_claims_the_whole_bundle(self):
        _, _, got = self._run(compartments={"mito": str(self.segmentation.id)})
        note = self._min_area(got["manifest"])["note"]

        assert "by the pack that produced them" in note
        assert "in this bundle" not in note

    def test_the_manifest_names_what_the_floor_did_not_filter(self):
        self._draw_one_below_the_floor()
        _, _, got = self._run(compartments={"mito": str(self.segmentation.id)})
        bypassed = self._min_area(got["manifest"])["not_applied_to"]

        assert bypassed["hand_drawn_objects"] == self.n_manual + 1
        assert bypassed["smallest_floor_px"] == self.FLOOR_PX
        assert bypassed["n_hand_drawn_below_the_floor"] == 1
        assert bypassed["smallest_hand_drawn_px"] == self.SMALL_PX
        assert "filter_min_area runs inside" in bypassed["why"]

    def test_the_object_really_does_ship_below_the_model_floor(self):
        """The hazard, in the file people read, not only in the manifest."""
        self._draw_one_below_the_floor()
        _, out, _ = self._run(compartments={"mito": str(self.segmentation.id)})
        rows = self._rows(out, "objects.csv")

        small = [r for r in rows if float(r["area_px"]) < self.FLOOR_PX]
        assert len(small) == 1
        assert small[0]["source_model"] == "manual"

    def test_the_hazard_reaches_the_top_level_caveats(self):
        """Four levels down in the JSON is not a qualification."""
        self._draw_one_below_the_floor()
        _, _, got = self._run(compartments={"mito": str(self.segmentation.id)})
        caveats = " ".join(got["result"]["caveats"])

        assert "applies to model-produced objects only" in caveats
        assert "left-truncated" in caveats
        assert "source_model column" in caveats

    def test_it_stays_quiet_when_no_drawn_object_is_below_the_floor(self):
        """MANUAL_FEATURES is 400 px, well above 60. Crying wolf is a defect."""
        _, _, got = self._run(compartments={"mito": str(self.segmentation.id)})
        bypassed = self._min_area(got["manifest"])["not_applied_to"]
        caveats = " ".join(got["result"]["caveats"])

        assert bypassed["hand_drawn_objects"] == self.n_manual
        assert bypassed["n_hand_drawn_below_the_floor"] == 0
        assert "smallest_hand_drawn_px" not in bypassed
        assert "left-truncated" not in caveats
