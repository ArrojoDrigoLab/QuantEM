"""The two caches: the loaded model, and the per-crop embedding.

They are different things and both are required. The model cache stops the
weights being reloaded per request; the embedding cache stops the encoder being
re-run over a crop the user is still working in. A box in a *fresh* region
misses the second and must still hit the first.
"""

from __future__ import annotations

import numpy as np
from django.test import SimpleTestCase

from quantem.sam.backends import get_backend, loaded_backend_keys, reset_backend
from quantem.sam.backends.base import Embedding
from quantem.sam.embedding_cache import EmbeddingCache, cache_key
from quantem.sam.geometry import Box, Crop, plan_crop
from quantem.sam.tests.support import stub_environment


def _embedding(value: float = 1.0) -> Embedding:
    return Embedding(
        features=np.full((1, 4, 4, 4), value, dtype=np.float32),
        original_size=(64, 64),
        input_size=(64, 64),
    )


class EmbeddingCacheTests(SimpleTestCase):
    def test_a_miss_then_a_hit(self):
        cache = EmbeddingCache(max_entries=4)
        key = cache_key("seg", "backend", Crop(0, 0, 64, 64))

        self.assertIsNone(cache.get(key))
        cache.put(key, _embedding())
        self.assertIsNotNone(cache.get(key))
        self.assertEqual((cache.hits, cache.misses), (1, 1))

    def test_the_key_is_the_crop_window_not_the_box(self):
        """Two boxes in one grid cell must produce the same key."""
        first = plan_crop(Box(1200, 1200, 1260, 1260), 4096, 3072)
        second = plan_crop(Box(1700, 1500, 1760, 1560), 4096, 3072)
        self.assertEqual(
            cache_key("seg", "backend", first),
            cache_key("seg", "backend", second),
        )

    def test_a_different_segmentation_is_a_different_entry(self):
        crop = Crop(0, 0, 64, 64)
        self.assertNotEqual(
            cache_key("seg-a", "backend", crop),
            cache_key("seg-b", "backend", crop),
        )

    def test_new_weights_cannot_be_served_an_old_embedding(self):
        crop = Crop(0, 0, 64, 64)
        self.assertNotEqual(
            cache_key("seg", "microsam:vit_b_em_organelles", crop),
            cache_key("seg", "meta:vit_b", crop),
        )

    def test_it_is_bounded_and_evicts_least_recently_used(self):
        cache = EmbeddingCache(max_entries=2)
        keys = [cache_key("seg", "b", Crop(index, 0, 64, 64)) for index in range(3)]

        cache.put(keys[0], _embedding(0))
        cache.put(keys[1], _embedding(1))
        cache.get(keys[0])  # keys[0] is now the most recent
        cache.put(keys[2], _embedding(2))

        self.assertEqual(len(cache), 2)
        self.assertIsNone(cache.get(keys[1]), "the least recently used survived")
        self.assertIsNotNone(cache.get(keys[0]))
        self.assertIsNotNone(cache.get(keys[2]))

    def test_it_reports_what_it_is_holding(self):
        cache = EmbeddingCache(max_entries=4)
        cache.put(cache_key("seg", "b", Crop(0, 0, 64, 64)), _embedding())
        self.assertEqual(cache.bytes_held, 4 * 4 * 4 * 4)


class ModelCacheTests(SimpleTestCase):
    """The weights load once per process, not once per request.

    This is the cache the owner asked for by name: in the labeling view a user
    draws many boxes in quick succession, and reloading hundreds of megabytes
    between them would make the feature unusable.
    """

    def setUp(self):
        self.enterContext(stub_environment())
        reset_backend()
        self.addCleanup(reset_backend)

    def test_the_model_is_built_once_and_reused(self):
        first = get_backend()
        second = get_backend()
        self.assertIs(
            first, second, "a second prompt rebuilt the model instead of reusing it"
        )
        self.assertEqual(len(loaded_backend_keys()), 1)

    def test_a_box_in_a_fresh_region_reuses_the_model(self):
        """A cold embedding cache must not mean a cold model."""
        backend = get_backend()
        far_away = plan_crop(Box(3000, 2500, 3100, 2600), 4096, 3072)
        near_by = plan_crop(Box(100, 100, 200, 200), 4096, 3072)
        self.assertNotEqual(far_away.key(), near_by.key())
        self.assertIs(get_backend(), backend)

    def test_resetting_drops_it(self):
        first = get_backend()
        reset_backend()
        self.assertEqual(loaded_backend_keys(), [])
        self.assertIsNot(first, get_backend())
