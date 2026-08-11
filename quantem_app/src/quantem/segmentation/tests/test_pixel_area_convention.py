"""One pixel convention, from the polygon a person drew to ``area_fraction_*``.

Reported: ``features.area`` was the pixel count of a **boundary-inclusive** fill.
``cv2.fillPoly`` rounds each vertex to a pixel centre and paints both boundaries
of every span, so a polygon spanning *s* pixels covered *s + 1* and a hand-drawn
square of side *s* measured ``(s + 1) ** 2``:

===========  ===============  ==================  =========
drawn side   polygon area     stored ``area``     bias
===========  ===============  ==================  =========
5            25               36                  **+44.0%**
10           100              121                 **+21.0%**
20           400              441                 +10.25%
50           2500             2601                +4.04%
100          10000            10201               +2.01%
===========  ===============  ==================  =========

A model-found object of the same size was measured correctly, because that path
never rasterises a polygon at all -- ``seg_core.extraction`` reads
``region.area`` straight off the label mask. So **one objects.csv could hold a
model object and the hand-drawn correction of the same organelle whose areas
differed by 21% purely by provenance**, with ``source_model`` the only hint.
Drawing the objects the model missed is this app's central workflow.

The convention now, stated at :mod:`quantem.seg_core.rasterize` and used by
every mask that gets counted: **a pixel belongs to a shape when its centre
does**, ties broken half-open. It is the convention ``regionprops`` already
measures a model's label mask in, so the two provenances agree by construction
rather than by coincidence, and ``perimeter`` -- measured off the same mask --
describes the same outline that ``area`` does.

These tests drive the endpoints the drawing tools actually post to
(``segments/`` and ``segments/confirm-batch/``) and the real analysis service,
because a fix that only holds in a unit test is not a fix. Every number below
fails on the boundary-inclusive fill; the parity tests fail on either error
alone.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from unittest import mock

import numpy as np
from django.test import TestCase
from rest_framework.test import APIClient
from shapely.geometry import Polygon
from skimage.measure import label as sk_label
from skimage.measure import regionprops

from quantem.analysis import loaders, service
from quantem.analysis.models import AnalysisRun
from quantem.core.config import STORAGE_DIR
from quantem.seg_core.extraction import build_segment_from_region
from quantem.seg_core.rasterize import fill_ring, fill_rings
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.overlay_ngff.render import geometry_to_rings, rasterize_region
from quantem.segmentation.type_service import (
    get_or_create_mitochondria_type,
    get_or_create_tissue_type,
)
from quantem.testing import create_small_test_image

SIZE = 320

#: The sizes from the report, and the areas the boundary-inclusive fill gave.
REPORTED = (
    # side, true area, what used to be stored
    (5, 25, 36),
    (10, 100, 121),
    (20, 400, 441),
    (50, 2500, 2601),
    (100, 10000, 10201),
)


def _square_coords(x0: int, y0: int, side: int) -> list[list[int]]:
    return [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]]


def _square(x: float, y: float, side: float) -> Polygon:
    return Polygon(((x, y), (x + side, y), (x + side, y + side), (x, y + side), (x, y)))


def _circularity(features: dict) -> float:
    return 4.0 * math.pi * features["area"] / (features["perimeter"] ** 2)


class RasterConventionTests(TestCase):
    """The rule itself: pixel count == polygon area, ties resolved once."""

    def test_an_axis_aligned_rectangle_covers_exactly_its_area(self):
        for side, area, inclusive in REPORTED:
            ring = np.array(_square_coords(20, 20, side), dtype=float)
            covered = int(fill_ring(ring, x0=0, y0=0, x1=SIZE, y1=SIZE).sum())
            self.assertEqual(
                covered,
                area,
                f"a {side} px square covered {covered} px "
                f"(the boundary-inclusive fill gave {inclusive})",
            )

    def test_a_shared_edge_is_counted_once(self):
        """Two squares meeting at x=40 must cover 2 * 400, not overlap in a column.

        The half-open rule is what makes this true, and it is the same rule that
        makes one square's area come out right.
        """
        left = fill_ring(
            np.array(_square_coords(20, 20, 20), dtype=float),
            x0=0,
            y0=0,
            x1=SIZE,
            y1=SIZE,
        )
        right = fill_ring(
            np.array(_square_coords(40, 20, 20), dtype=float),
            x0=0,
            y0=0,
            x1=SIZE,
            y1=SIZE,
        )
        self.assertFalse((left & right).any(), "the shared edge was painted twice")
        self.assertEqual(int((left | right).sum()), 800)

    def test_a_model_label_mask_survives_the_round_trip_exactly(self):
        """mask -> contour -> raster gives back the same pixels.

        This is the parity that makes the two provenances comparable: the
        polygon ``seg_core.extraction`` stores for a model object rasterises to
        the very pixels ``regionprops`` counted as its ``area``.
        """
        rng = np.random.default_rng(11)
        for _ in range(12):
            source = np.zeros((SIZE, SIZE), dtype=bool)
            rows, cols = np.mgrid[0:SIZE, 0:SIZE]
            a = float(rng.uniform(4, 60))
            b = float(rng.uniform(4, 45))
            theta = float(rng.uniform(0.0, math.pi))
            u = (cols - 160) * math.cos(theta) + (rows - 160) * math.sin(theta)
            v = -(cols - 160) * math.sin(theta) + (rows - 160) * math.cos(theta)
            source = (u / a) ** 2 + (v / b) ** 2 <= 1.0
            if not source.any():
                continue

            region = regionprops(sk_label(source))[0]
            segment = build_segment_from_region(
                region,
                sk_label(source),
                {},
                np.ones((SIZE, SIZE), dtype=np.float32),
                "mito_generated",
                0.0,
                0.0,
            )
            ring = np.asarray(segment.polygon_coords, dtype=float)
            painted = fill_ring(ring, x0=0, y0=0, x1=SIZE, y1=SIZE)

            self.assertEqual(int(painted.sum()), int(region.area))
            self.assertTrue(
                np.array_equal(painted, source),
                "the stored outline did not rasterise back to the model's own mask",
            )

    def test_a_hole_is_taken_out_of_the_area(self):
        outer = np.array(_square_coords(20, 20, 60), dtype=float)
        inner = np.array(_square_coords(40, 40, 20), dtype=float)
        covered = int(fill_rings([outer, inner], x0=0, y0=0, x1=SIZE, y1=SIZE).sum())
        self.assertEqual(covered, 60 * 60 - 20 * 20)

    def test_an_object_hanging_off_the_edge_keeps_its_shape(self):
        """Only the pixels outside the window are dropped, never the geometry.

        The OpenCV path clipped the *coordinates* to stay in bounds, which
        folded the outline flat along the border and then measured the folded
        shape: 26 x 26 instead of the 25 x 25 that is actually on the image.
        """
        ring = np.array(_square_coords(-15, -15, 40), dtype=float)
        self.assertEqual(int(fill_ring(ring, x0=0, y0=0, x1=64, y1=64).sum()), 25 * 25)


class DrawnObjectMeasurementTests(TestCase):
    """The reported table, through the endpoints the drawing tools post to."""

    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image(
            "Area convention", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def _confirm_batch(self, coords_list) -> list[SegmentObject]:
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/segments/confirm-batch/",
            {
                "segments": [{"geometry_coords": c} for c in coords_list],
                "manual_creation": True,
            },
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)
        return list(
            SegmentObject.objects.filter(
                segmentation=self.segmentation, label_state="CONFIRMED"
            ).order_by("created_at")
        )

    def test_a_drawn_square_measures_its_drawn_area(self):
        for side, area, inclusive in REPORTED:
            with self.subTest(side=side):
                SegmentObject.objects.filter(segmentation=self.segmentation).delete()
                (segment,) = self._confirm_batch([_square_coords(20, 20, side)])
                self.assertEqual(
                    segment.features["area"],
                    float(area),
                    f"a {side} px square drawn through confirm-batch stored "
                    f"{segment.features['area']} (was {inclusive}, "
                    f"{100 * (inclusive / area - 1):+.2f}%)",
                )

    def test_the_drawn_area_is_the_shapely_area_of_what_was_sent(self):
        """Not just for squares: whatever geometry arrives, ``area`` is its area."""
        coords = [
            [40, 40],
            [140, 60],
            [180, 150],
            [90, 190],
            [30, 120],
        ]
        (segment,) = self._confirm_batch([coords])
        drawn = Polygon([(float(x), float(y)) for x, y in coords]).area
        self.assertAlmostEqual(
            segment.features["area"] / drawn,
            1.0,
            delta=0.01,
            msg=f"stored {segment.features['area']} for a polygon of area {drawn}",
        )

    def test_a_drawn_object_and_a_model_object_of_one_shape_measure_the_same(self):
        """The headline: same organelle, two provenances, one set of numbers.

        A 10x10 object came out ``area 100, perimeter 36.0, circularity 0.970``
        from the model and ``area 121, perimeter 40.0, circularity 0.950`` by
        hand. ``source_model`` was the only hint that the two rows were not
        comparable.
        """
        source = np.zeros((SIZE, SIZE), dtype=bool)
        source[40:50, 40:50] = True
        region = regionprops(sk_label(source))[0]
        model = build_segment_from_region(
            region,
            sk_label(source),
            {},
            np.ones((SIZE, SIZE), dtype=np.float32),
            "mito_generated",
            0.0,
            0.0,
        )

        (drawn,) = self._confirm_batch([_square_coords(40, 40, 10)])

        self.assertEqual(model.features["area"], 100)
        self.assertEqual(drawn.features["area"], 100.0)
        self.assertAlmostEqual(drawn.features["perimeter"], model.features["perimeter"], places=6)
        self.assertAlmostEqual(_circularity(drawn.features), _circularity(model.features), places=6)
        for key in ("eccentricity", "solidity", "elongation", "feret_diameter_max"):
            self.assertAlmostEqual(drawn.features[key], model.features[key], places=6, msg=key)

    def test_the_single_segment_endpoint_agrees_with_confirm_batch(self):
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/segments/",
            {"geometry_coords": _square_coords(60, 60, 20)},
            format="json",
        )
        self.assertEqual(response.status_code, 201, response.data)
        segment = SegmentObject.objects.get(id=response.data["id"])
        self.assertEqual(segment.features["area"], 400.0)

    def test_re_measuring_a_model_object_does_not_move_its_area(self):
        """The feature-refresh job runs a stored polygon back through this path.

        It used to come back different twice over. The rasteriser added a
        boundary; and before that, a 1 px Douglas-Peucker pass -- put in to
        speed the rasteriser up -- moved the outline itself, by **-16%** of the
        area of a 4 px-radius object, -13.8% at 3 px, and a couple of percent
        either way up to 20 px. So a model object that had been refreshed no
        longer matched the ``region.area`` it was extracted with, and the two
        provenances disagreed again for objects that had merely been touched.

        The simplification is gone. It also cost more than it saved: ~540 us of
        Douglas-Peucker to spare ~20 us of fill on a 30 px blob.
        """
        from quantem.segmentation.tasks import compute_segment_features_task

        rows, cols = np.mgrid[0:SIZE, 0:SIZE]
        for radius in (3, 4, 6, 8, 10, 20):
            with self.subTest(radius=radius):
                SegmentObject.objects.filter(segmentation=self.segmentation).delete()
                source = (rows - 100) ** 2 + (cols - 100) ** 2 <= radius**2
                labelled = sk_label(source)
                region = regionprops(labelled)[0]
                extracted = build_segment_from_region(
                    region,
                    labelled,
                    {},
                    np.ones((SIZE, SIZE), dtype=np.float32),
                    "mito_generated",
                    0.0,
                    0.0,
                )
                polygon = Polygon(extracted.polygon_coords)
                segment = SegmentObject.objects.create(
                    segmentation=self.segmentation,
                    geometry=polygon,
                    centroid=polygon.centroid,
                    bbox=polygon.envelope,
                    label_state="CONFIRMED",
                    features=dict(extracted.features),
                )

                compute_segment_features_task(str(segment.id))
                segment.refresh_from_db()

                self.assertEqual(
                    segment.features["area"],
                    float(region.area),
                    "a refresh moved a model object's area away from the "
                    "region.area it was extracted with",
                )


class OverlayAndCompositionTests(TestCase):
    """The label map the viewer draws is the one the analysis counts."""

    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image("Composition", width=SIZE, height=SIZE, textured=True)
        self.asset = self.image.asset
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.asset, segmentation_type=get_or_create_mitochondria_type()
        )
        self.tissue = ImageSegmentation.objects.create(
            asset=self.asset, segmentation_type=get_or_create_tissue_type()
        )
        for x, y in ((40, 40), (120, 60), (80, 160)):
            self._object(self.segmentation, _square(x, y, 20))
        self._object(self.tissue, _square(20, 20, 160))

        self.exports_root = STORAGE_DIR / "exports_test" / self.id().rsplit(".", 1)[-1]
        shutil.rmtree(self.exports_root, ignore_errors=True)
        patcher = mock.patch.object(service, "EXPORTS_DIR", self.exports_root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.exports_root, ignore_errors=True)

    @staticmethod
    def _object(segmentation, polygon: Polygon) -> SegmentObject:
        return SegmentObject.objects.create(
            segmentation=segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CONFIRMED",
            features={
                "area": polygon.area,
                "perimeter": polygon.length,
                "eccentricity": 0.25,
                "solidity": 0.98,
                "major_axis_length": 22.0,
                "minor_axis_length": 20.0,
                "intensity_mean": 100.0,
            },
        )

    def test_the_overlay_label_map_covers_the_polygon_area(self):
        rings = geometry_to_rings(_square(40, 40, 20))
        labels, _border = rasterize_region(
            [{"label": 7, "priority": 0, "area": 0.0, "rings": rings}],
            x0=0,
            y0=0,
            x1=SIZE,
            y1=SIZE,
        )
        self.assertEqual(int((labels == 7).sum()), 400)

    def test_the_compartment_masks_count_what_was_drawn(self):
        mask = loaders.segmentation_mask(self.segmentation, (SIZE, SIZE))
        self.assertEqual(int(mask.sum()), 3 * 400)
        tissue = loaders.segmentation_mask(self.tissue, (SIZE, SIZE))
        self.assertEqual(int(tissue.sum()), 160 * 160)

    def test_area_fraction_is_the_drawn_geometry(self):
        """The reported headline number, end to end through the real service.

        Three 20 px squares in a 160 px tissue square reported
        ``areas_px.mito = 1323`` over ``tissue_px = 25921``, giving
        ``area_fraction_mito = 0.05104`` where the drawn geometry gives
        0.046875 -- **+8.9%**. It does not cancel because the small objects in
        the numerator inflate proportionally more than the large denominator.
        """
        params = loaders.normalise_params(
            {"tissue_segmentation_id": str(self.tissue.id), "replicates": 5},
            segmentation=self.segmentation,
        )
        run = AnalysisRun.objects.create(
            segmentation=self.segmentation, params=params, group=params["group"]
        )
        service.run_for_segmentation(run)
        run.refresh_from_db()
        self.assertEqual(run.status, AnalysisRun.STATUS_SUCCESS, run.error)

        composition = run.results["composition"]
        self.assertEqual(composition["tissue_px"], 160 * 160)
        self.assertEqual(composition["areas_px"]["mito"], 3 * 400)
        self.assertAlmostEqual(composition["area_fractions"]["mito"], 1200 / 25600.0, places=9)

        manifest = json.loads((Path(run.export_dir) / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["models"]["compartments"])
