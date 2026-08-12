"""The one v2 migration applies forward and backward on a database with rows.

**Why this is worth a slow test.** ``0003_v2_push`` is the single segmentation
migration for a release built by nine packages in parallel, so it carries
columns for five features at once and it lands before most of the code that
uses them. If it cannot be unapplied, a half-landed push has no way back except
a restored backup; and on SQLite, dropping a column is a table rebuild, which is
exactly the operation that loses rows when it goes wrong. Testing it on an empty
database would prove nothing about either.

So this creates real rows -- an image, a segmentation, objects with geometry, a
stored probability map, and one of each new model -- then walks the migration
down and back up, and checks what survived.

**Two things it asserts that are easy to get wrong:**

* Reversing *does* destroy the quality answers and the version headers, because
  their tables go with them. That is stated here rather than discovered later:
  the way back from a half-landed push costs the checks a user has answered
  since it landed, and nothing else.
* The forward data step is not a no-op. A probability map that recorded which
  interpolator carried it to native pixels comes out of the migration with that
  interpolator in its own column, rather than on the column default.
"""

from __future__ import annotations

from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase
from shapely.geometry import Polygon

from quantem.segmentation.models import (
    CountBox,
    ImageSegmentation,
    ProbabilityMap,
    QualityCheck,
    SegmentationResultVersion,
    SegmentObject,
)
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

BEFORE = ("segmentation", "0002_segmentationcompletionarchive")
AFTER = ("segmentation", "0003_v2_push")
LATEST = ("segmentation", "0006_segmentationtype_measurement_mode")


def _migrate_to(target) -> None:
    executor = MigrationExecutor(connection)
    executor.loader.build_graph()
    executor.migrate([target])


class MigrationRoundTripTests(TransactionTestCase):
    #: Real migrations, not the fast in-memory table creation, because the
    #: point is the migration operations themselves.
    available_apps = None

    def setUp(self):
        self.image = create_small_test_image("Migration round trip", width=128, height=128)
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        polygon = Polygon(((4, 4), (40, 4), (40, 40), (4, 40), (4, 4)))
        self.segment = SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CONFIRMED",
            source_model="quantem:mito",
            confidence_score=0.77,
            features={"area": 1296.0},
        )
        self.prob_map = ProbabilityMap.objects.create(
            segmentation=self.segmentation,
            name="DINO",
            file_path="prob_maps/fixture.png",
            metadata={
                "native_coordinates": True,
                "resample_interpolation": "INTER_AREA",
                "quantization": "uint8_255_round",
                "quantization_levels": 255,
            },
        )
        self.version = SegmentationResultVersion.objects.create(
            segmentation=self.segmentation, version=1, object_count=1
        )
        self.check = QualityCheck.objects.create(
            segmentation=self.segmentation,
            segment=self.segment,
            sample_seed=99,
            ordinal=0,
        )
        self.box = CountBox.objects.create(
            segmentation=self.segmentation,
            x=1.0,
            y=2.0,
            width=64.0,
            height=64.0,
            n_marked=9,
            n_matched=5,
        )

    def tearDown(self):
        # Leave the database where the rest of the suite expects it, whatever
        # happened above. Idempotent: migrating to a target already applied is
        # an empty plan.
        _migrate_to(LATEST)

    @staticmethod
    def _segmentation_columns() -> set[str]:
        with connection.cursor() as cursor:
            description = connection.introspection.get_table_description(
                cursor, "segmentation_segmentobject"
            )
        return {column.name for column in description}

    @staticmethod
    def _table_names() -> set[str]:
        with connection.cursor() as cursor:
            return set(connection.introspection.table_names(cursor))

    def test_the_columns_and_tables_arrive_and_leave_together(self):
        self.assertIn("run_version", self._segmentation_columns())
        self.assertIn("segmentation_qualitycheck", self._table_names())

        _migrate_to(BEFORE)

        columns = self._segmentation_columns()
        self.assertNotIn("run_version", columns)
        self.assertNotIn("superseded_at", columns)
        tables = self._table_names()
        for table in (
            "segmentation_qualitycheck",
            "segmentation_countbox",
            "segmentation_segmentationresultversion",
        ):
            self.assertNotIn(table, tables)

        _migrate_to(AFTER)

        self.assertIn("run_version", self._segmentation_columns())
        self.assertIn("segmentation_qualitycheck", self._table_names())

    def test_the_objects_and_their_measurements_survive_the_round_trip(self):
        _migrate_to(BEFORE)
        _migrate_to(AFTER)

        segment = SegmentObject.objects.get(id=self.segment.id)
        self.assertEqual(segment.label_state, "CONFIRMED")
        self.assertEqual(segment.source_model, "quantem:mito")
        self.assertAlmostEqual(segment.confidence_score, 0.77)
        self.assertEqual(segment.features, {"area": 1296.0})
        self.assertIsNotNone(segment.geometry)
        self.assertAlmostEqual(segment.geometry.area, 1296.0)

    def test_the_re_added_columns_come_back_on_their_defaults(self):
        """Nullable-or-defaulted, so a half-landed push never fails a write."""
        SegmentObject.objects.filter(id=self.segment.id).update(run_version=4)
        _migrate_to(BEFORE)
        _migrate_to(AFTER)

        segment = SegmentObject.objects.get(id=self.segment.id)
        self.assertEqual(segment.run_version, 1)
        self.assertIsNone(segment.superseded_at)

        segmentation = ImageSegmentation.objects.values("preview_rows_ready", "include_level").get(
            id=self.segmentation.id
        )
        self.assertEqual(segmentation["preview_rows_ready"], 0)
        self.assertIsNone(segmentation["include_level"])

    def test_reversing_costs_the_quality_answers_and_nothing_else(self):
        _migrate_to(BEFORE)
        _migrate_to(AFTER)

        self.assertEqual(QualityCheck.objects.count(), 0)
        self.assertEqual(CountBox.objects.count(), 0)
        self.assertEqual(SegmentationResultVersion.objects.count(), 0)
        self.assertEqual(SegmentObject.objects.count(), 1)
        self.assertEqual(ProbabilityMap.objects.count(), 1)

    def test_a_probability_map_is_described_from_what_it_recorded(self):
        """The forward data step, proved by a value the defaults cannot give."""
        _migrate_to(BEFORE)
        _migrate_to(AFTER)

        prob_map = ProbabilityMap.objects.get(id=self.prob_map.id)
        self.assertEqual(prob_map.grid, ProbabilityMap.GRID_NATIVE)
        self.assertEqual(prob_map.resample_kernel, "INTER_AREA")
        self.assertEqual(prob_map.quantisation, "uint8_255_round")
        self.assertEqual(prob_map.value_range, [0.0, 1.0])

    def test_a_map_that_recorded_nothing_says_so_rather_than_guessing(self):
        bare = ProbabilityMap.objects.create(
            segmentation=self.segmentation,
            name="Bare",
            file_path="prob_maps/bare.png",
            metadata={},
        )
        _migrate_to(BEFORE)
        _migrate_to(AFTER)

        refreshed = ProbabilityMap.objects.get(id=bare.id)
        self.assertEqual(refreshed.resample_kernel, "")
        self.assertIsNone(refreshed.value_range)
        # The default is right rather than a guess: every map this application
        # has ever written was stored at native scale.
        self.assertEqual(refreshed.grid, ProbabilityMap.GRID_NATIVE)

    def test_a_map_written_on_the_model_grid_is_not_relabelled_native(self):
        model_grid = ProbabilityMap.objects.create(
            segmentation=self.segmentation,
            name="Model grid",
            file_path="prob_maps/model_grid.png",
            metadata={"native_coordinates": False},
        )
        _migrate_to(BEFORE)
        _migrate_to(AFTER)

        refreshed = ProbabilityMap.objects.get(id=model_grid.id)
        self.assertEqual(refreshed.grid, ProbabilityMap.GRID_MODEL)
