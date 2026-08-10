"""What happens to objects measured before the pixel convention was fixed.

The fix changes what is written from now on. A number already in the database
stays as it was, so a pre-fix database holds hand-drawn objects whose ``area``
is ``(s + 1) ** 2`` sitting beside model objects and freshly drawn ones that are
right, with nothing on the row to tell them apart. The queued feature refresh
will not find them either: it sweeps for objects with *no* ``area``, and a stale
measurement is not a missing one.

``manage.py remeasure_segment_features`` is the deliberate pass. These tests
plant an object carrying the old convention's number and drive the command.
"""

from __future__ import annotations

from io import StringIO

from django.core.management import call_command
from django.test import TestCase
from shapely.geometry import Polygon

from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

SIZE = 256

#: A 20 px square as the boundary-inclusive fill measured it, and as it is.
STALE_AREA = 21.0 * 21.0
TRUE_AREA = 400.0


def _square(x: float, y: float, side: float) -> Polygon:
    return Polygon(
        ((x, y), (x + side, y), (x + side, y + side), (x, y + side), (x, y))
    )


class RemeasureCommandTests(TestCase):
    def setUp(self):
        self.image = create_small_test_image(
            "Remeasure", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        polygon = _square(40, 40, 20)
        self.segment = SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CONFIRMED",
            confidence_score=0.82,
            features={
                # As a pre-fix hand-drawn object was stored.
                "area": STALE_AREA,
                "perimeter": 80.0,
                "intensity_mean": 128.0,
                "mean_prob": 0.82,
                "mito_generated": True,
            },
        )

    def _run(self, *args) -> str:
        out = StringIO()
        call_command("remeasure_segment_features", *args, stdout=out)
        return out.getvalue()

    def test_a_dry_run_reports_the_correction_and_writes_nothing(self):
        output = self._run("--segmentation", str(self.segmentation.id))

        self.assertIn("dry run", output)
        self.assertIn("area_changed=1", output)
        self.assertIn("--apply", output)

        self.segment.refresh_from_db()
        self.assertEqual(self.segment.features["area"], STALE_AREA)
        self.assertEqual(self.segment.features["intensity_mean"], 128.0)

    def test_apply_puts_the_object_in_the_convention_it_is_read_in(self):
        self._run("--segmentation", str(self.segmentation.id), "--apply")

        self.segment.refresh_from_db()
        self.assertEqual(self.segment.features["area"], TRUE_AREA)

    def test_it_re_measures_rather_than_rescaling(self):
        """Not ``area * (s / (s + 1)) ** 2``: the polygon is measured again.

        Every other measurement moves with it -- perimeter came off the same
        inflated mask -- and a rescale would leave those describing the mask
        rather than the object.
        """
        self._run("--segmentation", str(self.segmentation.id), "--apply")

        self.segment.refresh_from_db()
        # 20 px square under perimeter_crofton (the shipped estimator); the old
        # walk gave 4*(20-1) = 76.0.
        self.assertAlmostEqual(self.segment.features["perimeter"], 74.734, places=3)
        self.assertNotEqual(self.segment.features["intensity_mean"], 128.0)

    def test_the_object_keeps_everything_that_is_not_a_measurement(self):
        """Nothing here reshapes an outline, so the model's opinion of it stands.

        ``mean_prob`` and ``confidence_score`` describe the model's confidence
        in a polygon that has not moved. Dropping them -- which is right when a
        *geometry edit* invalidates them -- would destroy a column of
        objects.csv for no reason.
        """
        self._run("--segmentation", str(self.segmentation.id), "--apply")

        self.segment.refresh_from_db()
        self.assertEqual(self.segment.features["mean_prob"], 0.82)
        self.assertEqual(self.segment.confidence_score, 0.82)
        self.assertTrue(self.segment.features["mito_generated"])

    def test_confirmed_only_leaves_the_candidates_alone(self):
        polygon = _square(120, 120, 20)
        candidate = SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CANDIDATE",
            features={"area": STALE_AREA},
        )

        self._run("--segmentation", str(self.segmentation.id), "--confirmed-only", "--apply")

        self.segment.refresh_from_db()
        candidate.refresh_from_db()
        self.assertEqual(self.segment.features["area"], TRUE_AREA)
        self.assertEqual(candidate.features["area"], STALE_AREA)
