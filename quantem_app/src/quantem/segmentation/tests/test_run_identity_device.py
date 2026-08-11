"""A run records which device produced its numbers.

Owner ruling R4: *"provenance must keep recording which device produced the
numbers"*. Owner ruling R5 settles what that record is **not** for -- a GPU run
and a CPU run may be compared, so no banner, no refusal, no extra caveat -- but
it keeps the record, "because it is cheap and useful".

It was neither recorded nor cheap-and-useful, because it was not recorded at
all. ``RUN_IDENTITY_KEYS`` had no ``device`` entry, so
``models.compartments[].run.inference_device`` in every export manifest was
null and the manifest said in as many words that there was nothing to read. The
reader (``quantem.analysis.loaders._device_provenance``) had been written
forward-compatibly and was waiting.

Without the writer these tests fail at the first assertion in each case:
``"device"`` is not in the contract and ``build_run_identity`` raises
``TypeError`` for the keyword.
"""

from __future__ import annotations

from django.test import SimpleTestCase

from quantem.segmentation.run_identity import (
    LEGACY_RUN_IDENTITY_KEYS,
    RUN_IDENTITY_KEYS,
    build_run_identity,
    run_identity_from_segmenter,
)

_CALL = {
    "run_id": "11111111-1111-4111-8111-111111111111",
    "pack_id": "quantem:mito",
    "threshold": 0.45,
    "adapter_id": None,
    "ran_at_nm": 8.0,
    "native_pixel_size_nm": 5.0,
    "min_area": 60,
}


class _Spec:
    pack_id = "quantem:mito"
    canonical_nm = 8.0
    threshold = 0.5


class _Segmenter:
    """Only the public surface the builder is allowed to read."""

    model_spec = _Spec()
    fg_threshold = 0.45
    adapter_id = None

    def __init__(self, device):
        self.inference_device = device


class _OldSegmenter:
    model_spec = _Spec()
    fg_threshold = 0.45
    adapter_id = None


class DeviceIsPartOfTheContractTests(SimpleTestCase):
    def test_the_contract_carries_a_device_field(self):
        self.assertIn("device", RUN_IDENTITY_KEYS)

    def test_the_eight_pre_v2_keys_are_untouched(self):
        """Adding a field must not renumber the ones a released build wrote."""
        self.assertEqual(RUN_IDENTITY_KEYS[:8], LEGACY_RUN_IDENTITY_KEYS)

    def test_a_caller_that_does_not_know_the_device_records_none(self):
        """Never ``"cpu"`` by default.

        A default would put a hardware claim in a manifest that nobody
        measured, and "we did not record it" and "it ran on the processor" are
        different statements about a number's reproducibility.
        """
        self.assertIsNone(build_run_identity(**_CALL)["device"])

    def test_the_value_is_a_json_scalar_like_every_other(self):
        payload = build_run_identity(**_CALL, device="cuda")
        self.assertEqual(payload["device"], "cuda")
        for key, value in payload.items():
            with self.subTest(key=key):
                self.assertIsInstance(value, (str, int, float, type(None)))


class DeviceComesFromTheSegmenterTests(SimpleTestCase):
    def _identity(self, segmenter):
        return run_identity_from_segmenter(
            segmenter,
            run_id=_CALL["run_id"],
            pack_id_fallback="quantem:mito",
            native_pixel_size_nm=5.0,
            min_area=60,
        )

    def test_the_run_records_where_it_finished(self):
        self.assertEqual(self._identity(_Segmenter("cuda"))["device"], "cuda")

    def test_a_run_that_fell_back_records_the_processor_not_the_card(self):
        """``inference_device`` is where the run *finished*.

        A model that cannot execute on the graphics card, or a card that ran
        out of memory, moves the run to the processor. Recording the device it
        was offered would document a run that never happened -- the same rule
        that makes ``threshold`` the value actually used rather than the pack's
        published default.
        """
        self.assertEqual(self._identity(_Segmenter("cpu"))["device"], "cpu")

    def test_a_segmenter_that_reports_no_device_records_none(self):
        self.assertIsNone(self._identity(_OldSegmenter())["device"])

    def test_a_blank_device_is_not_a_device(self):
        self.assertIsNone(self._identity(_Segmenter("   "))["device"])
