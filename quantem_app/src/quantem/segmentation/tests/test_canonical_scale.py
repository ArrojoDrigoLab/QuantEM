"""The segmenter is told the asset's pixel size.

Six of the eight released packs declare a ``canonical_nm`` -- 8 nm for mito and
lipid droplets, 25 nm for nuclei. Without the asset's own pixel size the
segmenter has nothing to resample from, so it runs at native scale: a 5 nm/px
image is fed to a model trained and benchmarked at 8 nm, the objects are the
wrong apparent size, and every number derived from the result -- counts, areas,
densities, enrichments, calibrated thresholds, group means -- is off in a way
nothing in the UI or the manifest would have mentioned.

The only trace it ever left was a ``logger.warning``.
"""

from __future__ import annotations

from types import SimpleNamespace

from django.test import SimpleTestCase

from quantem.segmentation.organelle_tasks import (
    _asset_pixel_size_nm,
    _build_segmenter_kwargs,
)


def _segmentation(pixel_size_nm, internal_name="dino_mito"):
    return SimpleNamespace(
        asset=SimpleNamespace(pixel_size_nm=pixel_size_nm),
        segmentation_type=SimpleNamespace(internal_name=internal_name),
    )


class _Config:
    def get_instance_params(self):
        return {}


class CanonicalScaleTests(SimpleTestCase):
    def test_pixel_size_reaches_the_segmenter(self):
        kwargs = _build_segmenter_kwargs(
            _segmentation(5.0),
            _Config(),
            segmenter_internal_name="dino_mito",
            source_model="quantem:mito",
        )
        self.assertEqual(kwargs["pixel_size_nm"], 5.0)

    def test_uncalibrated_image_passes_none_not_a_guess(self):
        """None means "run at native scale and say so", not "assume 5 nm"."""
        kwargs = _build_segmenter_kwargs(
            _segmentation(None),
            _Config(),
            segmenter_internal_name="dino_mito",
            source_model="quantem:mito",
        )
        self.assertIn("pixel_size_nm", kwargs)
        self.assertIsNone(kwargs["pixel_size_nm"])

    def test_nonsense_pixel_sizes_are_treated_as_uncalibrated(self):
        for value in (0, -1.0, "", "abc", float("nan")):
            with self.subTest(value=value):
                self.assertIsNone(_asset_pixel_size_nm(_segmentation(value)))

    def test_decimal_pixel_size_survives(self):
        """Asset.pixel_size_nm is a DecimalField; float() must happen here."""
        from decimal import Decimal

        self.assertEqual(_asset_pixel_size_nm(_segmentation(Decimal("4.25"))), 4.25)

    def test_missing_asset_does_not_raise(self):
        self.assertIsNone(_asset_pixel_size_nm(SimpleNamespace()))


class ResampleConsequenceTests(SimpleTestCase):
    """Show the difference the fix makes, in tiles rather than in principle."""

    def test_tile_count_changes_with_the_pixel_size(self):
        from quantem.inference import engine
        from quantem.inference.specs import get_model_spec

        spec = get_model_spec("quantem", "mito")
        self.assertEqual(spec.canonical_nm, 8.0)
        shape = (2921, 3228)

        native = engine.estimate_tiles(spec, shape, pixel_size_nm=None)
        canonical = engine.estimate_tiles(spec, shape, pixel_size_nm=5.0)

        # 5 nm -> 8 nm is a 0.625x resample, so the canonical run covers the
        # same field in far fewer tiles. The user who reported this counted
        # 72 tiles in the log and knew from that alone it had run at 5 nm.
        self.assertEqual(native, 72)
        self.assertLess(canonical, native)
        self.assertEqual(canonical, 25)

    def test_er_declares_no_canonical_scale_and_is_unaffected(self):
        from quantem.inference import engine
        from quantem.inference.specs import get_model_spec

        spec = get_model_spec("quantem", "er")
        self.assertIsNone(spec.canonical_nm)
        shape = (1024, 1024)
        self.assertEqual(
            engine.estimate_tiles(spec, shape, pixel_size_nm=None),
            engine.estimate_tiles(spec, shape, pixel_size_nm=5.0),
        )
