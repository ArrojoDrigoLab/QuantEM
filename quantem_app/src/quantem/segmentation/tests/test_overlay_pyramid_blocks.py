"""The pyramid pass costs what is annotated, not what the canvas measures.

**The defect.** ``_build_pyramid`` enumerated every ``PYRAMID_BLOCK_SIZE`` block
of every level over the full extent. On a 165 231 x 153 701 asset that is 2 114
blocks per array, 4 228 across the two, and each one opened the staged zarr
group and materialised a 4096-square region only to find it empty and return.
Cost scaled with canvas area; the annotated area made no difference at all.

The parallelism switch made the worst case worse rather than better:
``use_pool`` came from ``len(objects) >= RASTER_POOL_MIN_OBJECTS``, so a huge
canvas with *few* objects -- exactly a segmentation someone has just created,
which reads zero objects -- took the fully sequential path over all 4 228
blocks. That is what made every annotation-triggered rebuild slow enough to hold
the bundle for the length of an annotation session.

**What is pinned here.**

1. :func:`_pyramid_level_blocks` returns exactly the blocks that can hold
   content: the level-1 set straight off the bboxes, each level above it the
   previous one halved, and the pixel of padding that keeps a block on the far
   side of a halving boundary.
2. ``content_bboxes=None`` still visits everything and ``[]`` visits nothing --
   two different answers that are easy to collapse into one.
3. Pooling is decided by the block count, not the object count.
4. The restriction is *invisible in the output*: a rebuild that visits 22 blocks
   writes byte-for-byte the bundle a rebuild that visits every block writes.
   That is the test that actually checks the correctness argument (no level-0
   pixel ever lands outside its object's bbox), because it checks it against
   real geometry -- filled polygons, punched holes, baked borders, objects
   overlapping and objects hard against the image edge -- rather than against a
   restatement of the claim.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from django.test import TestCase
from shapely.geometry import Polygon, box

from quantem.assets.models import Asset, Rendition
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.overlay_ngff import mutations
from quantem.segmentation.overlay_ngff.constants import (
    OVERLAY_ARRAY_KEYS,
    PYRAMID_BLOCK_SIZE,
    RASTER_POOL_MIN_PYRAMID_BLOCKS,
)
from quantem.segmentation.overlay_ngff.paths import get_overlay_active_bundle_path
from quantem.segmentation.overlay_ngff.store import _level_shapes
from quantem.segmentation.type_service import get_or_create_mitochondria_type

#: The owner's canvas, and the one the before/after numbers were measured on.
REFERENCE_WIDTH = 165_231
REFERENCE_HEIGHT = 153_701


def _blocks(level_shapes, content_bboxes):
    return mutations._pyramid_level_blocks(level_shapes, content_bboxes)


def _total(blocks_by_level) -> int:
    """Blocks visited across both arrays -- the unit the pool gate counts."""
    return sum(len(blocks) for blocks in blocks_by_level.values()) * len(
        OVERLAY_ARRAY_KEYS
    )


def _indices(blocks_by_level, level: int) -> set[tuple[int, int]]:
    return {
        (block[0] // PYRAMID_BLOCK_SIZE, block[2] // PYRAMID_BLOCK_SIZE)
        for block in blocks_by_level[level]
    }


# ---------------------------------------------------------------------------
# 1. The block set itself
# ---------------------------------------------------------------------------
class PyramidBlockSetTests(TestCase):
    """Exact block sets, stated as sets and not as counts wherever it matters."""

    # 40 000 square: nine levels, and a level-1 block grid (10 x 10) big enough
    # for the halving walk to visit different indices on the way up.
    SHAPES = _level_shapes(40_000, 40_000)

    def test_the_shape_ladder_this_file_reasons_about(self):
        """State the fixture, so a change to _level_shapes fails here first."""
        self.assertEqual(len(self.SHAPES), 9)
        self.assertEqual(self.SHAPES[1], (20_000, 20_000))
        self.assertEqual(self.SHAPES[8], (157, 157))

    def test_none_visits_every_block_of_every_level(self):
        blocks_by_level = _blocks(self.SHAPES, None)

        self.assertEqual(sorted(blocks_by_level), [1, 2, 3, 4, 5, 6, 7, 8])
        # ceil(20000/2048) = 10, ceil(10000/2048) = 5, ceil(5000/2048) = 3, ...
        self.assertEqual(
            [len(blocks_by_level[level]) for level in sorted(blocks_by_level)],
            [100, 25, 9, 4, 1, 1, 1, 1],
        )
        self.assertEqual(_total(blocks_by_level), 284)

    def test_an_empty_list_visits_nothing_and_is_not_the_same_as_none(self):
        """``[]`` is "no content", ``None`` is "unknown". Never the same answer."""
        empty = _blocks(self.SHAPES, [])
        unknown = _blocks(self.SHAPES, None)

        self.assertEqual(_total(empty), 0)
        self.assertTrue(all(blocks == [] for blocks in empty.values()))
        self.assertNotEqual(empty, unknown)
        self.assertEqual(_total(unknown), 284)

    def test_one_small_object_visits_one_block_per_level(self):
        blocks_by_level = _blocks(self.SHAPES, [(100, 100, 200, 200)])

        self.assertEqual(_total(blocks_by_level), 16)  # 8 levels x 2 arrays
        for level in sorted(blocks_by_level):
            self.assertEqual(_indices(blocks_by_level, level), {(0, 0)})
        # Clamped to the level extent, not run off the end of it.
        self.assertEqual(blocks_by_level[1], [(0, 2048, 0, 2048)])
        self.assertEqual(blocks_by_level[8], [(0, 157, 0, 157)])

    def test_two_distant_objects_merge_as_the_levels_halve(self):
        """The child ``i`` -> parent ``i // 2`` walk, asserted index by index.

        The far object sits in level-1 block (7, 7). Halving takes it to (3, 3),
        (1, 1), then (0, 0), where it merges with the near object -- so the block
        count *falls* up the pyramid instead of doubling.
        """
        blocks_by_level = _blocks(
            self.SHAPES,
            [(100, 100, 200, 200), (30_000, 30_000, 30_100, 30_100)],
        )

        self.assertEqual(_indices(blocks_by_level, 1), {(0, 0), (7, 7)})
        self.assertEqual(_indices(blocks_by_level, 2), {(0, 0), (3, 3)})
        self.assertEqual(_indices(blocks_by_level, 3), {(0, 0), (1, 1)})
        self.assertEqual(_indices(blocks_by_level, 4), {(0, 0)})
        self.assertEqual(_indices(blocks_by_level, 8), {(0, 0)})
        self.assertEqual(_total(blocks_by_level), 22)
        # The level-3 block of the far object, clamped to a 5 000 px level.
        self.assertIn((2048, 4096, 2048, 4096), blocks_by_level[3])

    def test_the_padding_keeps_the_block_past_a_halving_boundary(self):
        """``y_max = 4095`` lands on level-1 pixel 2047: the last of block 0.

        Without the pixel of padding the set would stop there. One rounding
        convention away -- in the downsample, in a future bbox that is inclusive
        rather than exclusive at the top -- and the first row of block 1 holds
        content nobody visits, which is a silently missing tile at every zoom
        level above it. The padding costs one block and removes the class.
        """
        padded = _indices(_blocks(self.SHAPES, [(0, 0, 100, 4095)]), 1)
        unpadded_rows = {0}  # 4095 // 2 // 2048 == 0

        self.assertEqual(padded, {(0, 0), (1, 0)})
        self.assertNotEqual({row for row, _ in padded}, unpadded_rows)

    def test_the_padding_keeps_the_block_before_a_halving_boundary(self):
        """The same at the low edge: ``y_min = 4096`` is the first of block 1."""
        padded = _indices(_blocks(self.SHAPES, [(4096, 4096, 4200, 4200)]), 1)

        self.assertEqual(padded, {(0, 0), (0, 1), (1, 0), (1, 1)})

    def test_padding_never_names_a_block_the_level_does_not_have(self):
        """An object hard against the bottom-right corner stays inside the grid."""
        blocks_by_level = _blocks(
            self.SHAPES, [(39_900, 39_900, 40_000, 40_000)]
        )

        self.assertEqual(_indices(blocks_by_level, 1), {(9, 9)})
        for level, blocks in blocks_by_level.items():
            level_height, level_width = self.SHAPES[level]
            for block_y0, block_y1, block_x0, block_x1 in blocks:
                self.assertLess(block_y0, level_height)
                self.assertLess(block_x0, level_width)
                self.assertLessEqual(block_y1, level_height)
                self.assertLessEqual(block_x1, level_width)

    def test_a_bbox_running_off_the_top_left_is_clamped_not_negative(self):
        """Ring coordinates may be negative; a block index may not be."""
        blocks_by_level = _blocks(self.SHAPES, [(-500, -500, 50, 50)])

        self.assertEqual(_indices(blocks_by_level, 1), {(0, 0)})

    def test_a_single_level_store_has_no_blocks_at_all(self):
        """Nothing above level 0 means nothing to downsample, either way."""
        shapes = _level_shapes(200, 200)

        self.assertEqual(len(shapes), 1)
        self.assertEqual(_blocks(shapes, None), {})
        self.assertEqual(_blocks(shapes, [(0, 0, 100, 100)]), {})


class ReferenceCanvasBlockCountTests(TestCase):
    """The measured before/after, kept as an assertion rather than a memory."""

    SHAPES = _level_shapes(REFERENCE_WIDTH, REFERENCE_HEIGHT)

    @staticmethod
    def _twenty_one_objects(*, origin: int, span: int):
        """21 ER-sized objects on a 5-column grid inside a ``span`` square."""
        step = span // 5
        return [
            (
                origin + (index % 5) * step,
                origin + (index // 5) * step,
                origin + (index % 5) * step + 400,
                origin + (index // 5) * step + 400,
            )
            for index in range(21)
        ]

    def test_the_full_enumeration_is_the_4228_blocks_that_were_slow(self):
        self.assertEqual(_total(_blocks(self.SHAPES, None)), 4228)

    def test_twenty_one_objects_in_one_working_area_visit_twenty_six_blocks(self):
        """The shape of a real annotation session: 21 objects, one region.

        26 blocks against 4 228. The owner measured 28 on the same canvas with
        their own 21 objects; the exact number is a function of how far apart
        the objects are, which is the whole point -- it is annotated *extent*
        that sets the cost now, and no longer canvas extent.
        """
        blocks_by_level = _blocks(
            self.SHAPES, self._twenty_one_objects(origin=60_000, span=4_000)
        )

        self.assertEqual(_total(blocks_by_level), 26)
        self.assertLess(_total(blocks_by_level), RASTER_POOL_MIN_PYRAMID_BLOCKS)

    def test_the_cost_tracks_how_spread_out_the_annotation_is(self):
        """Same 21 objects over a fifth of the canvas: more blocks, still few.

        Worth pinning next to the clustered case so the property being claimed
        is unambiguous. It is not "few objects are cheap" -- it is "the pyramid
        visits the annotated region", and a diagonal smear across 140 000 px
        costs more than a cluster and still nothing like the whole canvas.
        """
        spread = [
            (x, x, x + 400, x + 400)
            for x in range(2_000, 2_000 + 21 * 7_000, 7_000)
        ]
        self.assertEqual(len(spread), 21)

        blocks_by_level = _blocks(self.SHAPES, spread)

        self.assertEqual(_total(blocks_by_level), 150)
        self.assertLess(_total(blocks_by_level), _total(_blocks(self.SHAPES, None)) / 20)


# ---------------------------------------------------------------------------
# 2. The pool gate counts blocks, not objects
# ---------------------------------------------------------------------------
class _RecordingGroup:
    """Stands in for the staged zarr group; counts opens, blocks and closes."""

    def __init__(self) -> None:
        self.blocks: list[tuple[str, int, tuple[int, int, int, int]]] = []
        self.closed = 0


class PyramidPoolGateTests(TestCase):
    """The switch is the pyramid's own work unit, not somebody else's."""

    def _drive(self, *, width: int, height: int, content_bboxes):
        group = _RecordingGroup()
        pools: list[dict] = []

        class _FakePool:
            def __init__(self, **kwargs):
                pools.append(kwargs)

            def map(self, function, tasks):
                for task in tasks:
                    group.blocks.append((task[1], task[2], task[3]))
                return []

            def shutdown(self, wait: bool = True, **kwargs) -> None:
                return None

        def _open(_stage_root):
            return group

        def _close(_group):
            group.closed += 1

        def _downsample(_group, array_key, level, block):
            group.blocks.append((array_key, level, block))

        def _worker(task):
            # A level with a single block skips ``map`` and calls the by-path
            # worker directly; it still counts as a visited block.
            group.blocks.append((task[1], task[2], task[3]))

        with (
            patch.object(mutations, "ProcessPoolExecutor", _FakePool),
            patch.object(mutations.render_module, "open_staged_group", _open),
            patch.object(mutations.render_module, "close_staged_group", _close),
            patch.object(mutations.render_module, "downsample_block", _downsample),
            patch.object(
                mutations.render_module, "downsample_block_worker", _worker
            ),
        ):
            mutations._build_pyramid(
                "unused",
                width=width,
                height=height,
                content_bboxes=content_bboxes,
            )
        return group, pools

    def test_a_huge_canvas_with_a_few_objects_no_longer_spawns_a_pool(self):
        """The case that was slowest: enormous extent, almost nothing on it."""
        group, pools = self._drive(
            width=REFERENCE_WIDTH,
            height=REFERENCE_HEIGHT,
            content_bboxes=[(1_000, 1_000, 1_400, 1_400)],
        )

        self.assertEqual(pools, [], "a pool was spawned for a handful of blocks")
        self.assertEqual(len(group.blocks), 20)  # 10 levels x 2 arrays
        self.assertEqual(group.closed, 1, "the store handle was not released")

    def test_the_store_is_opened_once_for_the_whole_in_process_pass(self):
        """One handle across every level and array, not one per block.

        Re-opening the group per block is what the by-path pool worker has to do
        for process isolation. In-process there is no such constraint, and 4 228
        group opens against a directory store is not free.
        """
        opens: list[str] = []
        group = _RecordingGroup()

        def _open(stage_root):
            opens.append(stage_root)
            return group

        def _downsample(_group, array_key, level, block):
            group.blocks.append((array_key, level, block))

        with (
            patch.object(mutations.render_module, "open_staged_group", _open),
            patch.object(
                mutations.render_module,
                "close_staged_group",
                lambda _group: group.__setattr__("closed", group.closed + 1),
            ),
            patch.object(mutations.render_module, "downsample_block", _downsample),
        ):
            mutations._build_pyramid(
                "stage-root",
                width=REFERENCE_WIDTH,
                height=REFERENCE_HEIGHT,
                content_bboxes=[(1_000, 1_000, 1_400, 1_400)],
            )

        self.assertEqual(opens, ["stage-root"])
        self.assertEqual(len(group.blocks), 20)
        self.assertEqual(group.closed, 1)

    def test_the_handle_is_released_even_when_a_block_blows_up(self):
        """The release is in a ``finally``: the caller has a directory to move."""
        group = _RecordingGroup()

        def _boom(*_args):
            raise RuntimeError("chunk write failed")

        with (
            patch.object(
                mutations.render_module, "open_staged_group", lambda _root: group
            ),
            patch.object(
                mutations.render_module,
                "close_staged_group",
                lambda _group: group.__setattr__("closed", group.closed + 1),
            ),
            patch.object(mutations.render_module, "downsample_block", _boom),
            self.assertRaises(RuntimeError),
        ):
            mutations._build_pyramid(
                "stage-root",
                width=REFERENCE_WIDTH,
                height=REFERENCE_HEIGHT,
                content_bboxes=[(1_000, 1_000, 1_400, 1_400)],
            )

        self.assertEqual(group.closed, 1)

    def test_enough_blocks_still_fans_out_to_a_pool(self):
        group, pools = self._drive(
            width=REFERENCE_WIDTH,
            height=REFERENCE_HEIGHT,
            content_bboxes=None,
        )

        self.assertEqual(len(pools), 1)
        self.assertEqual(pools[0]["initializer"], mutations.django_pool_initializer)
        self.assertEqual(len(group.blocks), 4228)
        self.assertEqual(
            group.closed, 0, "the pool path must not touch the in-process handle"
        )

    def test_no_content_does_no_work_and_opens_nothing(self):
        group, pools = self._drive(
            width=REFERENCE_WIDTH, height=REFERENCE_HEIGHT, content_bboxes=[]
        )

        self.assertEqual(pools, [])
        self.assertEqual(group.blocks, [])
        self.assertEqual(group.closed, 0)


# ---------------------------------------------------------------------------
# 3. The restriction does not change the published bundle
# ---------------------------------------------------------------------------
def _sha256_of_tree(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = 0
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
        digest.update(path.read_bytes())
        files += 1
    return digest.hexdigest(), files


class RestrictedPyramidWritesTheSameBundleTests(TestCase):
    """The correctness argument, checked against pixels instead of restated.

    The claim is that no level-0 pixel ever lands outside the bbox of the draw op
    that produced it, so blocks the bboxes cannot reach are background. The
    geometry below is chosen to attack that: a polygon with a hole (background
    punched *inside* a bbox), two overlapping objects (a label boundary, so a
    baked border, in the middle of nothing), an object flush against the right
    edge of the image, and one far away in the opposite corner so the block sets
    are genuinely sparse and genuinely have to merge on the way up.
    """

    EXTENT = 12_288  # 6 macro tiles a side; five pyramid levels above level 0

    def setUp(self):
        self.asset = Asset.objects.create(
            display_name=f"Pyramid block identity {uuid4().hex[:8]}",
            original_filename="pyramid_identity.tif",
            logical_width=self.EXTENT,
            logical_height=self.EXTENT,
            channels=1,
            bit_depth=8,
            pixel_size_nm=5.0,
            preprocess_stage="DONE",
            preprocess_progress=100.0,
        )
        Rendition.objects.create(
            asset=self.asset,
            type=Rendition.TYPE_FULL,
            storage_root="DATA_DIR",
            stored_path=f"images/pyramid_identity_{self.asset.id}.png",
            path_exists=False,
            is_directory=False,
            stored_width=self.EXTENT,
            stored_height=self.EXTENT,
            stored_channels=1,
            stored_bit_depth=8,
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self._seed_objects()

    def _seed_objects(self) -> None:
        ring_with_hole = Polygon(
            [(600, 600), (1400, 600), (1400, 1400), (600, 1400)],
            [[(800, 800), (1200, 800), (1200, 1200), (800, 1200)]],
        )
        geometries = [
            ring_with_hole,
            box(2000, 2000, 2300, 2300),
            box(2250, 2250, 2600, 2600),  # overlaps the one above
            box(self.EXTENT - 120, 5000, self.EXTENT, 5300),  # flush right edge
            box(self.EXTENT - 900, self.EXTENT - 900, self.EXTENT - 600,
                self.EXTENT - 600),  # opposite corner
            box(6100, 6100, 6300, 6300),  # astride a macro-tile seam
        ]
        SegmentObject.objects.bulk_create(
            [
                SegmentObject(
                    segmentation=self.segmentation,
                    geometry=geometry,
                    centroid=geometry.centroid,
                    bbox=geometry.envelope,
                    label_state="INFERRED",
                    confidence_score=0.8,
                    features={},
                )
                for geometry in geometries
            ]
        )

    def test_visiting_only_the_content_blocks_writes_the_same_bytes(self):
        state = mutations.rebuild_overlay_full(self.segmentation)
        restricted_sha, restricted_files = _sha256_of_tree(
            get_overlay_active_bundle_path(state)
        )
        self.assertGreater(restricted_files, 20, "the store came out empty")

        exhaustive = mutations._build_pyramid

        def _visit_everything(stage_root, *, width, height, content_bboxes=None):
            # The pre-fix behaviour: content_bboxes=None is the escape hatch that
            # still enumerates every block of every level.
            return exhaustive(stage_root, width=width, height=height)

        with patch.object(mutations, "_build_pyramid", _visit_everything):
            state = mutations.rebuild_overlay_full(self.segmentation)
        full_sha, full_files = _sha256_of_tree(get_overlay_active_bundle_path(state))

        self.assertEqual(full_files, restricted_files)
        self.assertEqual(
            full_sha,
            restricted_sha,
            "restricting the pyramid to the content blocks changed the "
            "published overlay bytes -- some level-0 content lies outside the "
            "draw-op bboxes",
        )

    def test_the_restricted_build_really_did_visit_fewer_blocks(self):
        """Otherwise the identity test above is comparing a thing to itself."""
        shapes = _level_shapes(self.EXTENT, self.EXTENT)
        objects = list(SegmentObject.objects.filter(segmentation=self.segmentation))
        label_map = {obj.id: index + 1 for index, obj in enumerate(objects)}
        draw_ops = mutations._build_draw_ops(objects, label_map=label_map)

        restricted = _total(_blocks(shapes, [op["bbox"] for op in draw_ops]))
        exhaustive = _total(_blocks(shapes, None))

        self.assertLess(restricted, exhaustive)
        self.assertGreater(exhaustive, 0)
