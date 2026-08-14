"""Pyramid encoding must leave compute capacity for the interactive app."""

from types import SimpleNamespace
from unittest.mock import patch

import zarr
from django.test import SimpleTestCase
from numcodecs import blosc as blosc_runtime

from quantem.assets.ngff import bounded_ngff_build_resources


class NgffBuildResourceTests(SimpleTestCase):
    def test_build_uses_one_codec_thread_per_profile_sized_chunk_worker(self):
        original_concurrency = zarr.config.get("async.concurrency")
        original_blosc_threads = blosc_runtime.get_nthreads()

        with (
            patch(
                "quantem.assets.ngff.get_machine_profile",
                return_value=SimpleNamespace(raster_workers=4),
            ),
            bounded_ngff_build_resources(),
        ):
            self.assertEqual(zarr.config.get("async.concurrency"), 4)
            self.assertEqual(blosc_runtime.get_nthreads(), 1)

        self.assertEqual(zarr.config.get("async.concurrency"), original_concurrency)
        self.assertEqual(blosc_runtime.get_nthreads(), original_blosc_threads)
