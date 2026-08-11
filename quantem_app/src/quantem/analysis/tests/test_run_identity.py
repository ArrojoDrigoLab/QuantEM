"""The manifest must describe the run that happened, not the settings that exist.

Four things a user found by reading a bundle field by field, all of which
produced a confident, wrong sentence:

* An empty tissue mask made ``cytoplasm`` **1.0** of a zero-pixel tissue, and
  1.0 is truthy enough to give it a *defined* enrichment of 0.0 -- maximal
  depletion of a compartment with no area -- printed beside the caveat saying
  every fraction is zero.
* A compartment with no confirmed objects declared *"inference ran at native
  resolution"* about a pack that resamples to 8 nm. Nothing had run at all.
* An ER threshold was recorded as ``0.45, "calibrated by adapter a33e4160"``
  beside 85 objects produced at 0.50, nineteen minutes before that adapter
  existed. The manifest admitted this three levels down and said nothing in the
  caveat list the UI shows.
* ``min_area``, the sliding-window overlap and the encoder checkpoint step were
  absent or bare ``null`` -- next to a paragraph about ``min_area`` moving every
  object count, and a ``pack.json`` that carries the checkpoint step.

Two more found on the second reading:

* A run recorded its ``adapter_id`` and nothing else. What that adapter *was* --
  base model, mode, steps, split mode, held-out Dice -- appeared only under
  ``adapter_applied_now``, which describes the adapter applied to the
  segmentation today; and its ``head.pt`` was named by path and never hashed.
* ``inference_device`` was ``null`` with an honest reason and no route to a
  value, though it is a reproducibility variable like the library versions.

The fix is that per-object run stamps are the evidence and the segmentation's
current configuration is the fallback of last resort, which announces itself.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from django.test import TestCase
from django.utils import timezone
from shapely.geometry import Polygon

from quantem.analysis import loaders, provenance, service
from quantem.analysis.compartments import CompartmentSet, area_fractions, assign_points
from quantem.analysis.models import AnalysisRun
from quantem.core.config import STORAGE_DIR
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.run_identity import RUN_IDENTITY_KEYS, build_run_identity
from quantem.segmentation.type_service import (
    get_or_create_er_type,
    get_or_create_lipid_droplet_type,
    get_or_create_mitochondria_type,
    get_or_create_nucleus_type,
    get_or_create_tissue_type,
)
from quantem.testing import TEST_PIXEL_SIZE_NM, create_small_test_image

IMAGE_SIZE = 240

RUN_A = "11111111-1111-4111-8111-111111111111"
RUN_B = "22222222-2222-4222-8222-222222222222"
ADAPTER_ID = "a33e4160-0000-4000-8000-000000000001"


def _square(x: float, y: float, side: float = 20.0) -> Polygon:
    return Polygon(((x, y), (x + side, y), (x + side, y + side), (x, y + side), (x, y)))


def _stamp(
    *,
    run_id: str = RUN_A,
    pack_id: str = "quantem:er",
    threshold: float = 0.5,
    adapter_id: str | None = None,
    # quantem:er declares no canonical_nm, so a real ER run is a native one.
    ran_at_nm: float | None = None,
    native_pixel_size_nm: float | None = TEST_PIXEL_SIZE_NM,
    min_area: int = 100,
) -> dict[str, Any]:
    """One object's ``features["run"]``.

    Built by the *writer* -- :func:`quantem.segmentation.run_identity.
    build_run_identity`, the function inference itself calls -- rather than
    hand-assembled here. A hand-assembled dict would let this file keep passing
    while the two halves of the contract drifted apart, which is the one failure
    a shared contract cannot afford.
    """
    return build_run_identity(
        run_id=run_id,
        pack_id=pack_id,
        threshold=threshold,
        adapter_id=adapter_id,
        ran_at_nm=ran_at_nm,
        native_pixel_size_nm=native_pixel_size_nm,
        min_area=min_area,
    )


class RunIdentityTestCase(TestCase):
    """One calibrated 5 nm/px image; segmentations are added per test."""

    def setUp(self) -> None:
        self.image = create_small_test_image(
            "Run Identity", width=IMAGE_SIZE, height=IMAGE_SIZE
        )
        self.asset = self.image.asset
        self.exports_root = STORAGE_DIR / "exports_test" / self.id().rsplit(".", 1)[-1]
        shutil.rmtree(self.exports_root, ignore_errors=True)
        patcher = mock.patch.object(service, "EXPORTS_DIR", self.exports_root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.exports_root, ignore_errors=True)

    # --- fixture helpers ---------------------------------------------------

    def _segmentation(self, type_factory) -> ImageSegmentation:
        return ImageSegmentation.objects.create(
            asset=self.asset, segmentation_type=type_factory()
        )

    def _object(
        self,
        segmentation: ImageSegmentation,
        polygon: Polygon,
        *,
        source_model: str = "",
        stamp: dict[str, Any] | None = None,
        label_state: str = "CONFIRMED",
    ) -> SegmentObject:
        features: dict[str, Any] = {
            "area": polygon.area,
            "perimeter": polygon.length,
            "eccentricity": 0.25,
            "solidity": 0.98,
            "intensity_mean": 100.0,
        }
        if stamp is not None:
            features[loaders.RUN_STAMP_KEY] = stamp
        return SegmentObject.objects.create(
            segmentation=segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            source_model=source_model,
            features=features,
        )

    def _run(self, subject: ImageSegmentation, **params) -> tuple[AnalysisRun, dict]:
        normalised = loaders.normalise_params(params, segmentation=subject)
        run = AnalysisRun.objects.create(segmentation=subject, params=normalised)
        result = service.run_for_segmentation(run)
        manifest = json.loads(
            (service.export_dir_for_run(run.id) / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        return run, {"result": result, "manifest": manifest}

    @staticmethod
    def _compartment(manifest: dict, name: str) -> dict:
        for block in manifest["models"]["compartments"]:
            if block["compartment"] == name:
                return block
        raise AssertionError(
            f"no compartment {name!r} in "
            f"{[b['compartment'] for b in manifest['models']['compartments']]}"
        )


# ---------------------------------------------------------------------------
# 1. The empty tissue mask
# ---------------------------------------------------------------------------


class EmptyTissueCytoplasmTests(TestCase):
    """``cytoplasm = 1 - nucleus`` is only true of a tissue that has area."""

    def test_a_zero_area_tissue_gives_cytoplasm_no_fraction_of_it(self):
        shape = (16, 16)
        comp = CompartmentSet(
            masks={"nucleus": np.zeros(shape, dtype=bool)},
            tissue=np.zeros(shape, dtype=bool),
        )

        areas = area_fractions(comp, pixel_size_nm=5.0)

        assert areas.tissue_px == 0
        assert areas.areas_px == {"nucleus": 0, "cytoplasm": 0}
        assert areas.fractions == {"nucleus": 0.0, "cytoplasm": 0.0}, (
            "1.0 - 0.0 asserts that all of a zero-pixel tissue is cytoplasm"
        )

    def test_enrichment_in_a_zero_area_compartment_is_undefined_not_zero(self):
        """0.0 is a number a reader can quote: 'maximally depleted'. There is no
        area to be depleted *of*, so the ratio is undefined and must render as
        such."""
        shape = (16, 16)
        comp = CompartmentSet(
            masks={"nucleus": np.zeros(shape, dtype=bool)},
            tissue=np.zeros(shape, dtype=bool),
        )
        points = np.array([[4.0, 4.0], [9.0, 9.0]])

        assignment = assign_points(points, comp)

        assert assignment.enrichment == {"nucleus": None, "cytoplasm": None}

    def test_a_tissue_with_area_still_derives_cytoplasm_as_the_complement(self):
        nucleus = np.zeros((10, 10), dtype=bool)
        nucleus[:, :4] = True
        comp = CompartmentSet(masks={"nucleus": nucleus}, tissue=np.ones((10, 10), bool))

        areas = area_fractions(comp)

        assert areas.areas_px == {"nucleus": 40, "cytoplasm": 60}
        assert areas.fractions == {"nucleus": 0.4, "cytoplasm": 0.6}

    def test_membership_masks_cover_every_point_not_only_the_on_tissue_ones(self):
        """The docstring used to say "the on-tissue points". Indexing these
        against ``pts[on_tissue]`` silently misaligns them."""
        tissue = np.zeros((10, 10), dtype=bool)
        tissue[:, :5] = True
        nucleus = np.zeros((10, 10), dtype=bool)
        nucleus[:, :2] = True
        comp = CompartmentSet(masks={"nucleus": nucleus}, tissue=tissue)
        points = np.array([[1.0, 1.0], [8.0, 8.0], [3.0, 3.0]])

        assignment = assign_points(points, comp)

        for name, mask in assignment.membership.items():
            assert mask.shape == (3,), f"{name} is not one entry per input point"
        assert assignment.membership["nucleus"].tolist() == [True, False, False]
        assert assignment.membership["cytoplasm"].tolist() == [False, False, True]


class EmptyTissueEndToEndTests(RunIdentityTestCase):
    def test_an_empty_tissue_mask_with_a_nucleus_reports_no_defined_enrichment(self):
        """The ordinary path: a nucleus compartment and a tissue segmentation
        with nothing confirmed in it. Every number must agree with the caveat
        printed beside it."""
        mito = self._segmentation(get_or_create_mitochondria_type)
        nucleus = self._segmentation(get_or_create_nucleus_type)
        tissue = self._segmentation(get_or_create_tissue_type)
        for i in range(3):
            self._object(mito, _square(30 + 40 * i, 40))
        self._object(nucleus, _square(120, 120, side=40))

        _run, got = self._run(
            mito,
            compartments={"mito": str(mito.id), "nucleus": str(nucleus.id)},
            tissue_segmentation_id=str(tissue.id),
            points_source="centroids",
        )
        composition = got["result"]["composition"]
        enrichment = got["result"]["points"]["enrichment"]

        assert composition["tissue_px"] == 0
        assert composition["area_fractions"]["cytoplasm"] == 0.0
        assert composition["areas_px"]["cytoplasm"] == 0
        assert enrichment == {"mito": None, "nucleus": None, "cytoplasm": None}, (
            "a compartment with no area cannot have a defined enrichment"
        )
        assert any("tissue mask is empty" in c for c in got["result"]["caveats"])

    def test_the_exported_summary_row_carries_the_same_undefined(self):
        """``image_summary.csv`` is what a chart reads. A 0.0 that became 100%
        in the UI got there through this row."""
        mito = self._segmentation(get_or_create_mitochondria_type)
        nucleus = self._segmentation(get_or_create_nucleus_type)
        tissue = self._segmentation(get_or_create_tissue_type)
        self._object(mito, _square(30, 40))
        self._object(nucleus, _square(120, 120, side=40))

        _run, got = self._run(
            mito,
            compartments={"mito": str(mito.id), "nucleus": str(nucleus.id)},
            tissue_segmentation_id=str(tissue.id),
            points_source="centroids",
        )
        row = service.image_summary_row(got["result"])

        assert row["area_fraction_cytoplasm"] == 0.0
        assert row["enrichment_cytoplasm"] is None


# ---------------------------------------------------------------------------
# 2. The scale of a compartment nothing ran on
# ---------------------------------------------------------------------------


class UnknownScaleTests(RunIdentityTestCase):
    def test_a_compartment_with_no_confirmed_objects_does_not_claim_native(self):
        mito = self._segmentation(get_or_create_mitochondria_type)
        empty = self._segmentation(get_or_create_er_type)
        self._object(mito, _square(30, 40), source_model="quantem:mito")

        _run, got = self._run(
            mito, compartments={"mito": str(mito.id), "er": str(empty.id)}
        )
        scale = self._compartment(got["manifest"], "er")["run"]["scale"]

        assert scale["ran_at"] == loaders.SCALE_UNKNOWN
        assert scale["ran_at_nm"] is None
        assert scale["resampled"] is None
        reason = scale["unavailable"]["ran_at_nm"]
        assert "no confirmed objects at all" in reason
        assert "not 'native'" in reason

    def test_the_unknown_scale_is_a_top_level_caveat_not_a_buried_field(self):
        mito = self._segmentation(get_or_create_mitochondria_type)
        empty = self._segmentation(get_or_create_er_type)
        self._object(mito, _square(30, 40), source_model="quantem:mito")

        _run, got = self._run(
            mito, compartments={"mito": str(mito.id), "er": str(empty.id)}
        )

        named = [c for c in got["result"]["caveats"] if "'er'" in c and "unknown" in c]
        assert named, got["result"]["caveats"]
        assert "not\nnative" not in named[0]
        assert "resampled" in named[0]

    def test_a_hand_drawn_compartment_says_no_inference_ran_rather_than_native(self):
        mito = self._segmentation(get_or_create_mitochondria_type)
        drawn = self._segmentation(get_or_create_nucleus_type)
        self._object(mito, _square(30, 40), source_model="quantem:mito")
        for i in range(2):
            self._object(drawn, _square(120 + 30 * i, 120), source_model="manual")

        _run, got = self._run(
            mito, compartments={"mito": str(mito.id), "nucleus": str(drawn.id)}
        )
        scale = self._compartment(got["manifest"], "nucleus")["run"]["scale"]

        assert scale["ran_at"] == loaders.SCALE_UNKNOWN
        assert "drawn by hand" in scale["unavailable"]["resampled"]

    def test_a_pack_that_produced_objects_still_reports_its_canonical_scale(self):
        """The unknown case must not swallow the case that does have evidence."""
        mito = self._segmentation(get_or_create_mitochondria_type)
        self._object(mito, _square(30, 40), source_model="quantem:mito")

        _run, got = self._run(mito, compartments={"mito": str(mito.id)})
        scale = self._compartment(got["manifest"], "mito")["run"]["scale"]

        assert scale["canonical_nm_by_pack"] == {"quantem:mito": 8.0}
        assert scale["ran_at"] == "canonical"
        assert scale["resampled"] is True

    def test_a_stamped_run_reports_the_scale_it_actually_ran_at(self):
        """``quantem:ld`` declares 8.0 nm: on a 5 nm image it resamples. This is
        the compartment the reader watched take nine tiles while the manifest
        said native."""
        ld = self._segmentation(get_or_create_lipid_droplet_type)
        for i in range(4):
            self._object(
                ld,
                _square(30 + 30 * i, 40),
                source_model="quantem:ld",
                stamp=_stamp(pack_id="quantem:ld", ran_at_nm=8.0, min_area=40),
            )

        _run, got = self._run(ld, compartments={"ld": str(ld.id)})
        scale = self._compartment(got["manifest"], "ld")["run"]["scale"]

        assert scale["recorded_from"] == "the objects"
        assert scale["ran_at"] == "canonical"
        assert scale["ran_at_nm"] == 8.0
        assert scale["resampled"] is True

    def test_a_pack_that_declares_no_canonical_scale_is_a_native_run_not_unknown(self):
        """``quantem:er``'s ``canonical_nm`` is null. A native run it recorded is
        evidence, and must not be downgraded to unknown."""
        er = self._segmentation(get_or_create_er_type)
        for i in range(3):
            self._object(
                er,
                _square(30 + 30 * i, 40),
                source_model="quantem:er",
                stamp=_stamp(ran_at_nm=None),
            )

        _run, got = self._run(er, compartments={"er": str(er.id)})
        scale = self._compartment(got["manifest"], "er")["run"]["scale"]

        assert scale["ran_at"] == "native"
        assert scale["resampled"] is False
        assert scale["canonical_nm_by_pack"] == {"quantem:er": None}

    def test_a_recalibrated_image_says_the_run_used_a_different_pixel_size(self):
        er = self._segmentation(get_or_create_er_type)
        for i in range(3):
            self._object(
                er,
                _square(30 + 30 * i, 40),
                source_model="quantem:er",
                stamp=_stamp(native_pixel_size_nm=2.0),
            )

        _run, got = self._run(er, compartments={"er": str(er.id)})

        named = [c for c in got["result"]["caveats"] if "pixel size has changed" in c]
        assert named, got["result"]["caveats"]
        assert "2.0" in named[0] and str(TEST_PIXEL_SIZE_NM) in named[0]


# ---------------------------------------------------------------------------
# 3. Current settings are not the run's settings
# ---------------------------------------------------------------------------


class ThresholdProvenanceTests(RunIdentityTestCase):
    def _er_with_stamps(self, n: int = 5, **stamp_kwargs) -> ImageSegmentation:
        er = self._segmentation(get_or_create_er_type)
        for i in range(n):
            self._object(
                er,
                _square(20 + 20 * (i % 8), 20 + 30 * (i // 8)),
                source_model="quantem:er",
                stamp=_stamp(**stamp_kwargs),
            )
        return er

    def test_the_threshold_comes_from_the_objects_not_the_applied_adapter(self):
        er = self._er_with_stamps(threshold=0.5)

        _run, got = self._run(er, compartments={"er": str(er.id)})
        threshold = self._compartment(got["manifest"], "er")["run"][
            "foreground_threshold"
        ]["by_pack"]["quantem:er"]

        assert threshold["value"] == 0.5
        assert threshold["recorded_from"] == "the objects"
        assert threshold["n_objects"] == 5

    def test_an_adapter_applied_after_inference_is_a_top_level_caveat(self):
        """The reported case: proofread, fine-tune, apply, analyse. The adapter
        calibrated nothing that is in the bundle."""
        from quantem.finetune.models import Adapter

        er = self._er_with_stamps(threshold=0.5, adapter_id=None)
        # An unsaved row rather than a Mock: the manifest reads the adapter's
        # derived values too (heldout_dice off the sweep, head_file off
        # head_path, caveats()), and a Mock answers those with more Mocks.
        adapter = Adapter(
            id=ADAPTER_ID,
            segmentation=er,
            name="ER on my crops",
            base_model="quantem:er",
            mode="head_only",
            calibrated_threshold=0.45,
            split_mode="random",
            head_path="",
            verified_reload=True,
            applied_at=timezone.now(),
        )

        with mock.patch(
            "quantem.finetune.models.active_adapter_for", return_value=adapter
        ):
            _run, got = self._run(er, compartments={"er": str(er.id)})

        block = self._compartment(got["manifest"], "er")["run"]
        threshold = block["foreground_threshold"]["by_pack"]["quantem:er"]
        assert threshold["value"] == 0.5, "the adapter did not produce these objects"
        assert threshold["superseded_for_future_runs"]["calibrated_threshold"] == 0.45
        assert block["adapter"]["applied"] is False
        assert block["adapter_applied_now"]["applied"] is True

        named = [c for c in got["result"]["caveats"] if ADAPTER_ID in c]
        assert named, got["result"]["caveats"]
        assert "does not re-infer" in named[0]

    def test_objects_from_two_runs_at_two_thresholds_report_neither_as_the_value(self):
        er = self._segmentation(get_or_create_er_type)
        for i in range(3):
            self._object(
                er,
                _square(20 + 25 * i, 20),
                source_model="quantem:er",
                stamp=_stamp(run_id=RUN_A, threshold=0.5),
            )
        for i in range(2):
            self._object(
                er,
                _square(20 + 25 * i, 80),
                source_model="quantem:er",
                stamp=_stamp(run_id=RUN_B, threshold=0.45, adapter_id=ADAPTER_ID),
            )

        _run, got = self._run(er, compartments={"er": str(er.id)})
        block = self._compartment(got["manifest"], "er")["run"]
        threshold = block["foreground_threshold"]["by_pack"]["quantem:er"]

        assert threshold["value"] is None
        assert sorted(threshold["values"]) == [0.45, 0.5]
        assert "more than one threshold" in threshold["unavailable"]["value"]
        assert [r["n_objects"] for r in block["runs"]] == [3, 2]
        assert any("not all produced at the same foreground threshold" in c
                   for c in got["result"]["caveats"])
        assert any("not all produced under the same adapter" in c
                   for c in got["result"]["caveats"])
        assert any("2 different inference runs" in c
                   for c in got["result"]["caveats"])

    def test_objects_made_before_stamping_fall_back_and_say_so_at_the_top(self):
        er = self._segmentation(get_or_create_er_type)
        for i in range(4):
            self._object(er, _square(20 + 25 * i, 20), source_model="quantem:er")

        _run, got = self._run(er, compartments={"er": str(er.id)})
        block = self._compartment(got["manifest"], "er")["run"]
        threshold = block["foreground_threshold"]["by_pack"]["quantem:er"]

        assert threshold["value"] == 0.5
        assert threshold["recorded_from"] == "the segmentation's current configuration"
        assert "predate per-run stamping" in block["recorded_from"]
        named = [
            c
            for c in got["result"]["caveats"]
            if "current ones, not the run's" in c and "'er'" in c
        ]
        assert named, got["result"]["caveats"]

    def test_a_partly_stamped_compartment_names_how_many_are_unaccounted_for(self):
        er = self._segmentation(get_or_create_er_type)
        for i in range(3):
            self._object(
                er, _square(20 + 25 * i, 20), source_model="quantem:er", stamp=_stamp()
            )
        for i in range(2):
            self._object(er, _square(20 + 25 * i, 80), source_model="quantem:er")

        _run, got = self._run(er, compartments={"er": str(er.id)})

        named = [c for c in got["result"]["caveats"] if "carry no record" in c]
        assert named, got["result"]["caveats"]
        assert "2 of the 5" in named[0]


# ---------------------------------------------------------------------------
# 4. The gaps a reader found field by field
# ---------------------------------------------------------------------------


class ManifestGapTests(RunIdentityTestCase):
    def test_min_area_is_recorded_with_its_units_and_its_model_grid_equivalent(self):
        ld = self._segmentation(get_or_create_lipid_droplet_type)
        for i in range(3):
            self._object(
                ld,
                _square(20 + 25 * i, 20),
                source_model="quantem:ld",
                stamp=_stamp(pack_id="quantem:ld", ran_at_nm=8.0, min_area=40),
            )

        _run, got = self._run(ld, compartments={"ld": str(ld.id)})
        min_area = self._compartment(got["manifest"], "ld")["run"]["min_area"]
        entry = min_area["by_pack"]["quantem:ld"]

        assert entry["value"] == 40
        assert entry["recorded_from"] == "the objects"
        assert min_area["units"] == "native image pixels"
        # 40 native px at 5 nm, seen by a pack whose canonical scale is 8 nm.
        assert entry["model_grid_px"] == round(40 * (5.0 / 8.0) ** 2, 3)
        assert entry["um2"] == 40 * (5.0 / 1000.0) ** 2
        # The note used to tell readers the boundary moved with scikit-image
        # 0.26 ("smaller than" -> "smaller than or equal to"). That is false
        # here, and specifically so: `remove_small_objects` is called nowhere in
        # the run path, because `inference.postprocess.filter_min_area` counts
        # the components itself precisely so a library upgrade cannot change an
        # object count. A manifest that blames a pin which provably cannot
        # explain a mismatch sends whoever is reconciling one the wrong way.
        assert "QuantEM's own" in min_area["note"]
        assert "does not change with the scikit-image version" in min_area["note"]
        assert "exactly this many pixels is kept" in min_area["note"]

    def test_min_area_falls_back_to_the_organelle_default_when_unstamped(self):
        mito = self._segmentation(get_or_create_mitochondria_type)
        self._object(mito, _square(30, 40), source_model="quantem:mito")

        _run, got = self._run(mito, compartments={"mito": str(mito.id)})
        entry = self._compartment(got["manifest"], "mito")["run"]["min_area"][
            "by_pack"
        ]["quantem:mito"]

        assert entry["value"] == 60, "quantem:mito's default_min_area"
        assert entry["recorded_from"] == "the segmentation's current configuration"
        assert entry["caveat"]

    def test_the_sliding_window_overlap_is_recorded_beside_the_tile_size(self):
        from quantem.inference import tiling

        mito = self._segmentation(get_or_create_mitochondria_type)
        self._object(mito, _square(30, 40), source_model="quantem:mito")

        _run, got = self._run(mito, compartments={"mito": str(mito.id)})
        pack = got["manifest"]["models"]["model_packs"][0]

        assert pack["tile_size"] == 512
        assert pack["tiling"]["window_overlap"] == tiling.DEFAULT_OVERLAP == 0.25
        assert pack["tiling"]["stride_px"] == 384
        assert "Hann" in pack["tiling"]["blend"]

    def test_the_encoder_checkpoint_is_read_or_explained_never_a_bare_null(self):
        mito = self._segmentation(get_or_create_mitochondria_type)
        self._object(mito, _square(30, 40), source_model="quantem:mito")

        _run, got = self._run(mito, compartments={"mito": str(mito.id)})
        pack = got["manifest"]["models"]["model_packs"][0]
        unavailable = pack.get("unavailable") or {}

        if "weights" in unavailable:
            # The pack is not installed on this machine; the record cannot be
            # read at all and the reason for that already covers it.
            assert "not installed" in unavailable["weights"]
            return
        for key in ("checkpoint_step", "encoder_run_id", "encoder_run_dir"):
            assert key in pack, f"{key} absent entirely"
            assert pack[key] is not None or unavailable.get(key), (
                f"{key} is a bare null with no stated reason"
            )
        assert pack["checkpoint_step"] == 674999
        assert pack["encoder_run_id"] == "m1_dinov3_vitb"

    def test_every_null_in_the_new_sections_has_a_stated_reason(self):
        """The rule the manifest exists for, re-checked over the run block."""
        er = self._segmentation(get_or_create_er_type)
        empty = self._segmentation(get_or_create_nucleus_type)
        for i in range(2):
            self._object(
                er,
                _square(20 + 25 * i, 20),
                source_model="quantem:er",
                stamp=_stamp(run_id=RUN_A if i else RUN_B, threshold=0.5 - 0.05 * i),
            )

        _run, got = self._run(
            er, compartments={"er": str(er.id), "nucleus": str(empty.id)}
        )

        def walk(node: Any):
            if isinstance(node, dict):
                yield node
                for value in node.values():
                    yield from walk(value)
            elif isinstance(node, list):
                for value in node:
                    yield from walk(value)

        for node in walk(got["manifest"]):
            for field_name, reason in (node.get("unavailable") or {}).items():
                head = field_name.split(".", 1)[0]
                assert head in node, (
                    f"{field_name!r} is explained but not present as null in "
                    f"{sorted(node)}"
                )
                assert isinstance(reason, str) and len(reason) > 20, (
                    f"{field_name!r} is null with no usable reason: {reason!r}"
                )


class RunStampReaderTests(RunIdentityTestCase):
    def test_this_module_reads_every_field_the_writer_promises(self):
        """The contract is shared with :mod:`quantem.segmentation.run_identity`.
        If it grows a field, this side has to be told, not left reading a shape
        that no longer exists."""
        assert loaders.RUN_STAMP_FIELDS == RUN_IDENTITY_KEYS
        assert loaders.RUN_STAMP_KEY == "run"
        stamp = _stamp()
        assert set(stamp) == set(RUN_IDENTITY_KEYS)

    def test_a_stamp_with_no_run_id_is_not_treated_as_a_recorded_run(self):
        """The writer's reader rejects it, and so must the counting here: a
        half-written stamp is an unrecorded run, not a run with blank settings."""
        seg = self._segmentation(get_or_create_er_type)
        broken = _stamp()
        broken["id"] = ""
        self._object(seg, _square(20, 20), source_model="quantem:er", stamp=broken)

        stamps = loaders.run_stamps(seg)

        assert stamps.stamps == []
        assert stamps.n_unstamped == 1

    def test_a_deleted_adapter_leaves_its_id_and_a_reason_not_a_crash(self):
        """The id lives on the objects and outlives the record. Provenance must
        never fail a run (nor invent a name for a row that is gone)."""
        seg = self._segmentation(get_or_create_er_type)
        for i in range(2):
            self._object(
                seg,
                _square(20 + 30 * i, 20),
                source_model="quantem:er",
                stamp=_stamp(adapter_id=ADAPTER_ID),
            )

        _run, got = self._run(seg, compartments={"er": str(seg.id)})
        adapter = self._compartment(got["manifest"], "er")["run"]["adapter"]

        assert adapter["applied"] is True
        assert adapter["adapter_id"] == ADAPTER_ID
        assert adapter["name"] is None
        assert "no longer exists" in adapter["unavailable"]["name"]

    def test_a_malformed_adapter_id_is_reported_rather_than_raised(self):
        seg = self._segmentation(get_or_create_er_type)
        self._object(
            seg,
            _square(20, 20),
            source_model="quantem:er",
            stamp=_stamp(adapter_id="not-a-uuid"),
        )

        _run, got = self._run(seg, compartments={"er": str(seg.id)})
        adapter = self._compartment(got["manifest"], "er")["run"]["adapter"]

        assert adapter["adapter_id"] == "not-a-uuid"
        assert adapter["name"] is None
        assert adapter["unavailable"]["name"]

    def test_a_hand_drawn_object_without_a_stamp_is_not_an_unrecorded_run(self):
        """Absence of a stamp means two different things. Confusing them is what
        turns "nobody ran a model here" into "a model ran, settings unknown"."""
        seg = self._segmentation(get_or_create_er_type)
        self._object(seg, _square(20, 20), source_model="manual")
        self._object(seg, _square(60, 20), source_model="quantem:er", stamp=_stamp())

        stamps = loaders.run_stamps(seg)

        assert stamps.n_objects == 2
        assert stamps.n_hand_drawn == 1
        assert stamps.n_model_produced == 1
        assert stamps.n_unstamped == 0
        assert stamps.packs() == ["quantem:er"]

    def test_only_confirmed_objects_are_read(self):
        seg = self._segmentation(get_or_create_er_type)
        self._object(seg, _square(20, 20), source_model="quantem:er", stamp=_stamp())
        self._object(
            seg,
            _square(60, 20),
            source_model="quantem:er",
            stamp=_stamp(threshold=0.9),
            label_state="CANDIDATE",
        )

        assert [s["threshold"] for s in loaders.run_stamps(seg).stamps] == [0.5]


class RealInferenceToManifestTests(TestCase):
    """The contract end to end: real inference writes, the manifest reads.

    No weights -- the model's forward is the stand-in seam
    :func:`quantem.inference.engine.predict_region` documents -- but everything
    between the segmenter and the exported ``manifest.json`` is the real code
    path, including the stamp itself. A unit test on either half can pass while
    the halves disagree; this one cannot.
    """

    CALIBRATED = 0.31

    def setUp(self) -> None:
        from quantem.inference import engine
        from quantem.inference.specs import MODEL_SPECS
        from quantem.testing import create_mitochondria_segmentation

        image = create_small_test_image("inference to manifest", width=256, height=256)
        self.asset = image.asset
        self.segmentation = create_mitochondria_segmentation(image)

        def forward(tile: np.ndarray) -> np.ndarray:
            # 0.6 clears both the pack's published 0.5 and the adapter's 0.31,
            # so the two runs below differ only in the threshold they record.
            prob = np.full(tile.shape[:2], 0.05, dtype=np.float32)
            prob[50:150, 50:150] = 0.6
            return prob

        engine.clear_model_cache()
        engine.cache_model(
            engine.LoadedModel(
                spec=MODEL_SPECS["quantem:mito"],
                device="cpu",
                module=None,
                forward=forward,
                encoder_tier="stand-in",
            )
        )
        self.addCleanup(engine.clear_model_cache)

        self.exports_root = STORAGE_DIR / "exports_test" / self.id().rsplit(".", 1)[-1]
        shutil.rmtree(self.exports_root, ignore_errors=True)
        patcher = mock.patch.object(service, "EXPORTS_DIR", self.exports_root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.exports_root, ignore_errors=True)

    def _adapter(self, **overrides):
        from quantem.finetune.models import STATUS_SUCCESS, Adapter

        fields = {
            "segmentation": self.segmentation,
            "base_model": "quantem:mito",
            "name": "mito @ liver",
            "status": STATUS_SUCCESS,
            "mode": "threshold_only",
            "calibrated_threshold": self.CALIBRATED,
            "split_mode": "image-disjoint",
            "applied_at": timezone.now(),
        }
        fields.update(overrides)
        return Adapter.objects.create(**fields)

    def _infer_and_confirm(self) -> int:
        from quantem.segmentation.organelle_tasks import run_segmentation_full_task

        run_segmentation_full_task(
            segmentation_id=str(self.segmentation.id),
            segmentation_type=self.segmentation.segmentation_type.internal_name,
            source_model="quantem:mito",
        )
        return SegmentObject.objects.filter(segmentation=self.segmentation).update(
            label_state="CONFIRMED"
        )

    def _manifest(self) -> dict:
        params = loaders.normalise_params({}, segmentation=self.segmentation)
        run = AnalysisRun.objects.create(segmentation=self.segmentation, params=params)
        result = service.run_for_segmentation(run)
        manifest = json.loads(
            (service.export_dir_for_run(run.id) / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        return {"result": result, "manifest": manifest}

    def test_the_manifest_reports_what_the_run_stamped(self):
        adapter = self._adapter()
        assert self._infer_and_confirm() >= 1

        got = self._manifest()
        block = got["manifest"]["models"]["compartments"][0]["run"]

        assert block["foreground_threshold"]["by_pack"]["quantem:mito"] == {
            "value": self.CALIBRATED,
            "source": (
                "recorded on the 1 object quantem:mito produced here, by the "
                "run that produced it"
            ),
            "recorded_from": "the objects",
            "pack_default": 0.5,
            "n_objects": 1,
        }
        assert block["adapter"]["adapter_id"] == str(adapter.id)
        assert block["adapter"]["calibrated_threshold"] == self.CALIBRATED
        assert block["min_area"]["by_pack"]["quantem:mito"]["value"] == 60
        assert block["scale"]["ran_at_nm"] == 8.0
        assert block["scale"]["resampled"] is True
        assert block["caveats"] == []
        assert not [
            c for c in got["result"]["caveats"] if "not the run's" in c
        ], got["result"]["caveats"]

    def test_the_manifest_names_the_device_a_real_run_finished_on(self):
        """The other end of the wire the reader was written against.

        ``_device_provenance`` was landed forward-compatibly and had nothing to
        read: no writer put a device on a run, so this field was null in every
        bundle QuantEM had ever produced. Owner ruling R4 requires the record
        (R5 requires that it not be used to police a comparison), and this is
        the whole path -- segmenter, run identity, object features, manifest --
        rather than a fabricated stamp.
        """
        assert self._infer_and_confirm() >= 1

        got = self._manifest()
        device = got["manifest"]["models"]["compartments"][0]["run"][
            "inference_device"
        ]

        assert device["value"] == "cpu"
        assert device["recorded_from"] == "the objects"
        assert device["n_objects"] >= 1
        assert "unavailable" not in device

    def test_an_adapter_applied_after_the_run_does_not_rewrite_its_threshold(self):
        """Proofread, fine-tune, apply, analyse. The manifest used to report the
        adapter's 0.45 beside objects the released model made at 0.50."""
        assert self._infer_and_confirm() >= 1
        later = self._adapter(calibrated_threshold=0.45, name="fitted afterwards")

        got = self._manifest()
        block = got["manifest"]["models"]["compartments"][0]["run"]
        threshold = block["foreground_threshold"]["by_pack"]["quantem:mito"]

        assert threshold["value"] == 0.5, "the released model's own threshold ran"
        assert threshold["superseded_for_future_runs"]["adapter_id"] == str(later.id)
        named = [c for c in got["result"]["caveats"] if str(later.id) in c]
        assert named, got["result"]["caveats"]
        assert "does not re-infer" in named[0]


class BundleFileTests(RunIdentityTestCase):
    def test_the_caveats_reach_the_bundle_on_disk(self):
        er = self._segmentation(get_or_create_er_type)
        for i in range(3):
            self._object(er, _square(20 + 25 * i, 20), source_model="quantem:er")

        run, got = self._run(er, compartments={"er": str(er.id)})
        manifest = json.loads(
            (Path(run.export_dir) / "manifest.json").read_text(encoding="utf-8")
        )

        image_caveats = manifest["images"][0]["caveats"]
        assert any("current ones, not the run's" in c for c in image_caveats)
        run.refresh_from_db()
        assert any("current ones, not the run's" in c for c in run.results["caveats"])


class AdapterBelongsToTheRunThatUsedItTests(RunIdentityTestCase):
    """``runs[1].adapter_id`` was a foreign key into a table the reader has not got.

    The adapter's base model, mode, steps, split mode and held-out Dice were in
    the manifest only under ``adapter_applied_now`` -- the adapter applied to the
    segmentation *today*, which need not be the one that produced the objects,
    and in the reported case was not. The adapter's ``head.pt`` was named by
    path and never hashed, though it is as much a part of what made these
    objects as the released pack's weights are.
    """

    def _adapter(self, **overrides):
        from quantem.finetune.models import STATUS_SUCCESS, Adapter

        fields = {
            "id": ADAPTER_ID,
            "base_model": "quantem:er",
            "name": "ER on my crops",
            "status": STATUS_SUCCESS,
            "mode": "head_only",
            "params": {"steps": 250, "lr": 0.001},
            "sweep": {"heldout_dice_at_calibrated": 0.87},
            "calibrated_threshold": 0.45,
            "split_mode": "image-disjoint",
            "verified_reload": True,
        }
        fields.update(overrides)
        return Adapter.objects.create(**fields)

    def _er_under(self, adapter_id: str, *, n: int = 3):
        seg = self._segmentation(get_or_create_er_type)
        for i in range(n):
            self._object(
                seg,
                _square(20 + 30 * i, 20),
                source_model="quantem:er",
                stamp=_stamp(threshold=0.45, adapter_id=adapter_id),
            )
        return seg

    def _the_run(self, got: dict) -> dict:
        runs = self._compartment(got["manifest"], "er")["run"]["runs"]
        assert len(runs) == 1, runs
        return runs[0]

    def test_the_run_carries_the_adapter_it_used_expanded_in_place(self):
        adapter = self._adapter()
        seg = self._er_under(str(adapter.id))

        _run, got = self._run(seg, compartments={"er": str(seg.id)})
        run = self._the_run(got)

        assert run["adapter_id"] == str(adapter.id)
        assert run["adapter"]["base_model"] == "quantem:er"
        assert run["adapter"]["mode"] == "head_only"
        assert run["adapter"]["steps"] == 250
        assert run["adapter"]["split_mode"] == "image-disjoint"
        assert run["adapter"]["heldout_dice"] == 0.87
        assert run["adapter"]["calibrated_threshold"] == 0.45

    def test_the_adapter_head_is_pinned_by_digest_like_every_other_weight(self):
        head = Path(STORAGE_DIR) / "adapter-head-probe.pt"
        head.parent.mkdir(parents=True, exist_ok=True)
        head.write_bytes(b"a trained head")
        self.addCleanup(head.unlink, missing_ok=True)
        adapter = self._adapter(head_path=str(head))
        seg = self._er_under(str(adapter.id))

        _run, got = self._run(seg, compartments={"er": str(seg.id)})
        recorded = self._the_run(got)["adapter"]["head"]

        assert recorded["filename"] == "adapter-head-probe.pt"
        assert recorded["sha256"] == provenance.sha256_file(head)
        assert "path" not in recorded

    def test_a_threshold_only_adapter_says_why_there_is_no_head_to_hash(self):
        adapter = self._adapter(mode="threshold_only", head_path="")
        seg = self._er_under(str(adapter.id))

        _run, got = self._run(seg, compartments={"er": str(seg.id)})
        recorded = self._the_run(got)["adapter"]

        assert recorded["head"] is None
        assert "fits a threshold and no weights" in recorded["unavailable"]["head"]

    def test_no_held_out_split_is_null_with_the_split_mode_as_the_reason(self):
        from quantem.finetune.models import SPLIT_NO_HELDOUT

        adapter = self._adapter(sweep={}, split_mode=SPLIT_NO_HELDOUT)
        seg = self._er_under(str(adapter.id))

        _run, got = self._run(seg, compartments={"er": str(seg.id)})
        recorded = self._the_run(got)["adapter"]

        assert recorded["heldout_dice"] is None
        assert SPLIT_NO_HELDOUT in recorded["unavailable"]["heldout_dice"]

    def test_the_run_reports_the_adapter_that_made_it_not_the_applied_one(self):
        """The reported shape: fine-tune again, apply the new one, then analyse.
        ``adapter_applied_now`` moves; the objects do not."""
        made_them = self._adapter()
        seg = self._er_under(str(made_them.id))
        applied_now = self._adapter(
            id="a33e4160-0000-4000-8000-000000000002",
            name="fitted afterwards",
            segmentation=seg,
            calibrated_threshold=0.6,
            sweep={"heldout_dice_at_calibrated": 0.5},
            applied_at=timezone.now(),
        )

        _run, got = self._run(seg, compartments={"er": str(seg.id)})
        block = self._compartment(got["manifest"], "er")["run"]

        assert self._the_run(got)["adapter"]["heldout_dice"] == 0.87
        assert block["adapter_applied_now"]["adapter_id"] == str(applied_now.id)
        assert block["adapter_applied_now"]["heldout_dice"] == 0.5
        assert block["adapter"]["adapter_id"] == str(made_them.id)

    def test_a_run_with_no_adapter_says_so_rather_than_borrowing_one(self):
        self._adapter(applied_at=timezone.now())
        seg = self._segmentation(get_or_create_er_type)
        self._object(
            seg, _square(20, 20), source_model="quantem:er", stamp=_stamp()
        )

        _run, got = self._run(seg, compartments={"er": str(seg.id)})
        run = self._the_run(got)

        assert run["adapter_id"] is None
        assert run["adapter"] is None


class InferenceDeviceTests(RunIdentityTestCase):
    """``cuda`` and ``cpu`` do not agree to the last bit, so it is provenance.

    The bundle's ``environment`` block can only describe the machine that wrote
    the bundle; the analysis job runs in another process. The honest place is
    the run stamp, beside the threshold that run used.
    """

    def test_an_object_set_that_records_no_device_says_so_about_itself(self):
        """The reason is about these objects, not about the format.

        It used to be about the format: the contract had no device field, so
        nothing could be read even from a fully stamped object. It has one now
        (this asserts it), which makes a null here a statement about objects
        made before the writer landed -- and the sentence has to say that
        rather than blaming a gap that has been filled.
        """
        seg = self._segmentation(get_or_create_er_type)
        self._object(seg, _square(20, 20), source_model="quantem:er", stamp=_stamp())

        _run, got = self._run(seg, compartments={"er": str(seg.id)})
        device = self._compartment(got["manifest"], "er")["run"]["inference_device"]

        assert device["value"] is None
        reason = device["unavailable"]["value"]
        assert "quantem.segmentation.run_identity" in reason
        assert loaders.DEVICE_STAMP_FIELD in reason
        assert loaders.DEVICE_STAMP_FIELD in RUN_IDENTITY_KEYS
        assert "before QuantEM recorded the device" in reason

    def test_the_environment_block_points_at_the_run_rather_than_answering(self):
        seg = self._segmentation(get_or_create_er_type)
        self._object(seg, _square(20, 20), source_model="quantem:er", stamp=_stamp())

        _run, got = self._run(seg, compartments={"er": str(seg.id)})
        reason = got["manifest"]["environment"]["unavailable"]["inference_device"]

        assert "models.compartments[].run" in reason
        assert got["manifest"]["environment"]["torch_devices_available"]

    def test_a_stamped_device_is_read_the_moment_one_is_written(self):
        """Forward-compatible on purpose: the reader is not the blocker."""
        seg = self._segmentation(get_or_create_er_type)
        for i in range(2):
            stamp = _stamp()
            stamp[loaders.DEVICE_STAMP_FIELD] = "cuda"
            self._object(
                seg, _square(20 + 30 * i, 20), source_model="quantem:er", stamp=stamp
            )

        _run, got = self._run(seg, compartments={"er": str(seg.id)})
        device = self._compartment(got["manifest"], "er")["run"]["inference_device"]

        assert device["value"] == "cuda"
        assert device["recorded_from"] == "the objects"
        assert device["n_objects"] == 2

    def test_two_devices_report_neither_as_the_value(self):
        seg = self._segmentation(get_or_create_er_type)
        for i, name in enumerate(("cuda", "cpu")):
            stamp = _stamp()
            stamp[loaders.DEVICE_STAMP_FIELD] = name
            self._object(
                seg, _square(20 + 30 * i, 20), source_model="quantem:er", stamp=stamp
            )

        _run, got = self._run(seg, compartments={"er": str(seg.id)})
        device = self._compartment(got["manifest"], "er")["run"]["inference_device"]

        assert device["value"] is None
        assert device["values"] == ["cpu", "cuda"]
        assert "more than one device" in device["unavailable"]["value"]
