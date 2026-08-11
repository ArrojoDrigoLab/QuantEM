"""Draw a box, get an object -- through the real endpoint, with a stub backend.

Everything except the weights is the shipping code path: the view, crop
planning, the embedding cache, the coordinate round trip, and the segmentation
service that stores and measures the object.
"""

from __future__ import annotations

from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from quantem.sam.backends import get_backend
from quantem.sam.embedding_cache import EMBEDDINGS
from quantem.sam.tests.support import TEST_URLCONF, stub_environment
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

SIZE = 512


@override_settings(ROOT_URLCONF=TEST_URLCONF)
class BoxPromptApiTests(TestCase):
    def setUp(self):
        self.enterContext(stub_environment())
        self.client = APIClient()
        self.image = create_small_test_image("Box prompt", width=SIZE, height=SIZE, textured=True)
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def _prompt(self, x0, y0, x1, y1):
        return self.client.post(
            f"/api/sam/segmentations/{self.segmentation.id}/box/",
            {"box": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}},
            format="json",
        )

    # --- the happy path ---------------------------------------------------

    def test_a_box_creates_one_confirmed_object(self):
        response = self._prompt(100, 100, 200, 200)
        self.assertEqual(response.status_code, 201, response.data)

        stored = SegmentObject.objects.filter(segmentation=self.segmentation)
        self.assertEqual(stored.count(), 1)
        self.assertEqual(stored.first().label_state, "CONFIRMED")

    def test_the_object_lands_where_the_box_was_drawn(self):
        """The coordinate round trip, asserted end to end through HTTP."""
        response = self._prompt(150, 160, 260, 250)
        self.assertEqual(response.status_code, 201, response.data)

        segment = SegmentObject.objects.get(segmentation=self.segmentation)
        minx, miny, maxx, maxy = segment.geometry.bounds
        self.assertGreaterEqual(minx, 149.0)
        self.assertGreaterEqual(miny, 159.0)
        self.assertLessEqual(maxx, 261.0)
        self.assertLessEqual(maxy, 251.0)

    def test_the_response_carries_the_object_and_the_runners_up(self):
        response = self._prompt(100, 100, 200, 200)
        body = response.data

        self.assertIn("geometry_coords", body["object"])
        self.assertGreater(len(body["object"]["geometry_coords"]), 3)
        self.assertIsInstance(body["object"]["score"], float)
        self.assertGreaterEqual(len(body["other_candidates"]), 1)
        for candidate in body["other_candidates"]:
            self.assertIn("geometry_coords", candidate)
            self.assertIn("score", candidate)
        self.assertLessEqual(body["other_candidates"][0]["score"], body["object"]["score"])

    def test_the_stored_object_keeps_its_score(self):
        self._prompt(100, 100, 200, 200)
        segment = SegmentObject.objects.get(segmentation=self.segmentation)
        self.assertIn("sam_score", segment.features)

    def test_the_object_survives_a_reload(self):
        """Persistence: a fresh read of the database still has it."""
        self._prompt(120, 120, 220, 220)
        reloaded = ImageSegmentation.objects.get(id=self.segmentation.id)
        self.assertEqual(
            SegmentObject.objects.filter(segmentation=reloaded, label_state="CONFIRMED").count(),
            1,
        )

    # --- the caches -------------------------------------------------------

    def test_a_second_box_in_the_same_region_reuses_the_embedding(self):
        first = self._prompt(100, 100, 200, 200)
        self.assertEqual(first.status_code, 201, first.data)
        self.assertFalse(first.data["timing"]["cache_hit"])

        second = self._prompt(300, 300, 380, 380)
        self.assertEqual(second.status_code, 201, second.data)
        self.assertTrue(
            second.data["timing"]["cache_hit"],
            "the second box in the same crop window re-ran the encoder",
        )
        self.assertEqual(second.data["timing"]["encode_ms"], 0.0)

    def test_the_encoder_really_only_ran_once(self):
        """Asserted on the backend's own counter, not only on the reported flag."""
        backend = get_backend()
        self._prompt(100, 100, 200, 200)
        self._prompt(300, 300, 380, 380)
        self._prompt(150, 400, 250, 480)
        self.assertEqual(backend.encode_calls, 1)

    def test_a_box_in_a_fresh_region_re_encodes_but_keeps_the_model(self):
        backend = get_backend()
        self._prompt(100, 100, 200, 200)
        self.assertEqual(backend.encode_calls, 1)

        # 4096 is well outside this 512 px image, so use a second, larger one.
        big = create_small_test_image("Far away", width=3000, height=3000, textured=True)
        other = ImageSegmentation.objects.create(
            asset=big.asset, segmentation_type=get_or_create_mitochondria_type()
        )
        response = self.client.post(
            f"/api/sam/segmentations/{other.id}/box/",
            {"box": {"x0": 2000, "y0": 2000, "x1": 2100, "y1": 2100}},
            format="json",
        )
        self.assertEqual(response.status_code, 201, response.data)
        self.assertFalse(response.data["timing"]["cache_hit"])
        self.assertEqual(backend.encode_calls, 2)
        self.assertIs(get_backend(), backend, "the model was rebuilt for a new region")

    def test_the_embedding_cache_stays_bounded(self):
        for index in range(12):
            EMBEDDINGS.put(
                ("seg", "stub:threshold", (index, 0, 64, 64)),
                get_backend().encode(__import__("numpy").zeros((64, 64, 3), dtype="uint8")),
            )
        self.assertLessEqual(len(EMBEDDINGS), 8)

    # --- refusals ---------------------------------------------------------

    def test_a_box_with_no_area_is_refused_in_plain_language(self):
        response = self._prompt(100, 100, 100, 100)
        self.assertEqual(response.status_code, 400)
        self.assertIn("box", response.data["detail"].lower())

    def test_a_malformed_body_is_refused(self):
        response = self.client.post(
            f"/api/sam/segmentations/{self.segmentation.id}/box/",
            {"bbox": [1, 2, 3, 4]},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.data)

    def test_a_box_off_the_image_is_refused_rather_than_crashing(self):
        response = self._prompt(-500, -500, -400, -400)
        self.assertEqual(response.status_code, 400)

    def test_an_unknown_segmentation_is_a_404(self):
        response = self.client.post(
            "/api/sam/segmentations/00000000-0000-0000-0000-000000000000/box/",
            {"box": {"x0": 1, "y0": 1, "x1": 9, "y1": 9}},
            format="json",
        )
        self.assertEqual(response.status_code, 404)

    def test_no_error_copy_names_a_drive_or_an_identifier(self):
        """The app's copy rules: no absolute paths, no UUIDs in a sentence."""
        for response in (
            self._prompt(100, 100, 100, 100),
            self.client.post(
                f"/api/sam/segmentations/{self.segmentation.id}/box/",
                {},
                format="json",
            ),
        ):
            detail = response.data["detail"]
            self.assertNotIn(":\\", detail)
            self.assertNotIn("/api/", detail)
            self.assertNotIn(str(self.segmentation.id), detail)


@override_settings(ROOT_URLCONF=TEST_URLCONF)
class ModelStatusApiTests(TestCase):
    def setUp(self):
        self.enterContext(stub_environment())
        self.client = APIClient()

    def test_status_reports_what_the_client_needs(self):
        response = self.client.get("/api/sam/model/")
        self.assertEqual(response.status_code, 200)
        for key in ("model", "installed", "download", "size_bytes", "stub_mode"):
            self.assertIn(key, response.data)
        self.assertIn("percent", response.data["download"])
        self.assertTrue(response.data["stub_mode"])
