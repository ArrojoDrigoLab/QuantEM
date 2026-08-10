import sys
from unittest.mock import patch

from django.test import SimpleTestCase

from quantem.seg_core.registry import (
    DEFAULT_SEGMENTERS,
    get_segmenter,
    register_default_segmenters,
    register_segmenter,
)


class SegmenterRegistryTests(SimpleTestCase):
    def test_get_segmenter_imports_lazy_registration_only_on_demand(self):
        fixture_module = "quantem.seg_core.tests.lazy_segmenter_fixture"
        sys.modules.pop(fixture_module, None)

        with patch.dict("quantem.seg_core.registry._registry", {}, clear=True):
            register_segmenter(
                "lazy_type",
                f"{fixture_module}.LazyTestSegmenter",
            )

            self.assertNotIn(fixture_module, sys.modules)

            segmenter = get_segmenter("lazy_type")

            self.assertEqual(segmenter.name, "lazy")
            self.assertIn(fixture_module, sys.modules)

    def test_default_registrations_cover_the_four_released_organelles(self):
        with patch.dict("quantem.seg_core.registry._registry", {}, clear=True):
            register_default_segmenters()

            from quantem.seg_core.registry import _registry

            self.assertEqual(
                sorted(_registry),
                ["dino_er", "dino_ld", "dino_mito", "dino_nucleus"],
            )

    def test_default_registrations_are_import_paths_not_imported_eagerly(self):
        # Registration must not drag torch into the process at app startup.
        for import_path in DEFAULT_SEGMENTERS.values():
            self.assertTrue(import_path.startswith("quantem.inference.segmenter."))
