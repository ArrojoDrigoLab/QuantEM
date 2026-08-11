"""Four keys were added to the run identity, and nothing else changed.

Three packages needed a new field on the per-object run stamp -- the run's
scope, the include level it was extracted at, and which numbered result it
belongs to -- and the native-coordinate probability work needed a fourth. All
four were added in one edit, because the contract is asserted key-for-key in
two places (``segmentation/tests/test_run_identity.py`` and
``analysis/tests/test_run_identity.py``) and three separate additions would
have been three chances to leave the tuple and the builder disagreeing.

**The property that matters is that nothing moved.** Every object written today
must record exactly what it recorded yesterday for the eight fields that
already existed, or an analysis comparing objects from before and after this
release is comparing two different records and does not know it. So the
pre-existing eight are rebuilt here from the *old* code, literally, and
compared.

The defaults were chosen for the same reason, and each is a claim about
history:

* ``scope="full"`` -- every run before patch runs existed was a whole image.
* ``include_level=threshold`` -- the dial is the threshold under a name a
  biologist can use, and until something moves it they are one number.
  Defaulting to ``None`` would have read as "no include level" for every run
  ever made.
* ``run_version=1`` -- there has always been a first result.
* ``prob_map_grid=None`` -- a run that did not record which grid it decided on
  did not record it. Writing ``"native"`` there would claim provenance nobody
  captured.
"""

from __future__ import annotations

from django.test import SimpleTestCase

from quantem.analysis import loaders
from quantem.segmentation.run_identity import (
    LEGACY_RUN_IDENTITY_KEYS,
    RUN_IDENTITY_KEYS,
    RUN_SCOPE_FULL,
    RUN_SCOPE_PATCH,
    build_run_identity,
)

#: The arguments a caller written before this release passes.
LEGACY_CALL = {
    "run_id": "6f1b0c2e-0000-4000-8000-000000000001",
    "pack_id": "quantem:mito",
    "threshold": 0.45,
    "adapter_id": "6f1b0c2e-0000-4000-8000-000000000002",
    "ran_at_nm": 8.0,
    "native_pixel_size_nm": 5.0,
    "min_area": 60,
    "finished_at": "2026-08-01T09:15:02.481Z",
}


def _payload_as_the_old_builder_made_it() -> dict[str, object]:
    """The eight-key dict the pre-release builder returned, written out.

    Copied rather than imported, because the point is to compare against code
    that no longer exists. If someone changes how a field is normalised, this
    is what notices.
    """
    return {
        "id": str(LEGACY_CALL["run_id"]),
        "finished_at": LEGACY_CALL["finished_at"],
        "pack_id": str(LEGACY_CALL["pack_id"]),
        "threshold": float(LEGACY_CALL["threshold"]),
        "adapter_id": str(LEGACY_CALL["adapter_id"]),
        "ran_at_nm": float(LEGACY_CALL["ran_at_nm"]),
        "native_pixel_size_nm": float(LEGACY_CALL["native_pixel_size_nm"]),
        "min_area": int(LEGACY_CALL["min_area"]),
    }


class LegacyFieldsAreUnchangedTests(SimpleTestCase):
    def test_the_eight_original_keys_still_come_first_and_in_order(self):
        self.assertEqual(
            LEGACY_RUN_IDENTITY_KEYS,
            (
                "id",
                "finished_at",
                "pack_id",
                "threshold",
                "adapter_id",
                "ran_at_nm",
                "native_pixel_size_nm",
                "min_area",
            ),
        )
        self.assertEqual(RUN_IDENTITY_KEYS[:8], LEGACY_RUN_IDENTITY_KEYS)

    def test_a_pre_release_call_records_exactly_what_it_used_to(self):
        payload = build_run_identity(**LEGACY_CALL)
        legacy = {key: payload[key] for key in LEGACY_RUN_IDENTITY_KEYS}
        self.assertEqual(legacy, _payload_as_the_old_builder_made_it())

    def test_the_payload_is_still_in_contract_order(self):
        """``test_run_identity.py`` asserts ``tuple(payload) == KEYS``."""
        self.assertEqual(tuple(build_run_identity(**LEGACY_CALL)), RUN_IDENTITY_KEYS)

    def test_the_analysis_side_follows_without_being_told(self):
        self.assertEqual(loaders.RUN_STAMP_FIELDS, RUN_IDENTITY_KEYS)


class NewFieldDefaultsTests(SimpleTestCase):
    def test_an_unupdated_caller_still_writes_a_truthful_record(self):
        payload = build_run_identity(**LEGACY_CALL)
        self.assertEqual(payload["scope"], RUN_SCOPE_FULL)
        self.assertEqual(payload["include_level"], payload["threshold"])
        self.assertEqual(payload["run_version"], 1)
        self.assertIsNone(payload["prob_map_grid"])

    def test_the_include_level_follows_the_threshold_rather_than_a_constant(self):
        for threshold in (0.1, 0.45, 0.9):
            with self.subTest(threshold=threshold):
                payload = build_run_identity(**{**LEGACY_CALL, "threshold": threshold})
                self.assertEqual(payload["include_level"], threshold)

    def test_a_run_with_no_threshold_has_no_include_level_either(self):
        payload = build_run_identity(**{**LEGACY_CALL, "threshold": None})
        self.assertIsNone(payload["threshold"])
        self.assertIsNone(payload["include_level"])

    def test_an_explicit_include_level_overrides_the_threshold(self):
        payload = build_run_identity(**LEGACY_CALL, include_level=0.62)
        self.assertEqual(payload["threshold"], 0.45)
        self.assertEqual(payload["include_level"], 0.62)

    def test_the_four_new_fields_can_all_be_set(self):
        payload = build_run_identity(
            **LEGACY_CALL,
            scope=RUN_SCOPE_PATCH,
            include_level=0.62,
            run_version=3,
            prob_map_grid="native",
        )
        self.assertEqual(payload["scope"], RUN_SCOPE_PATCH)
        self.assertEqual(payload["include_level"], 0.62)
        self.assertEqual(payload["run_version"], 3)
        self.assertEqual(payload["prob_map_grid"], "native")

    def test_a_blank_scope_falls_back_to_the_historical_value(self):
        payload = build_run_identity(**LEGACY_CALL, scope="  ")
        self.assertEqual(payload["scope"], RUN_SCOPE_FULL)

    def test_every_value_stays_a_json_scalar(self):
        payload = build_run_identity(
            **LEGACY_CALL, scope=RUN_SCOPE_PATCH, run_version=2, prob_map_grid="native"
        )
        for key, value in payload.items():
            with self.subTest(key=key):
                self.assertIsInstance(value, (str, int, float, type(None)))
