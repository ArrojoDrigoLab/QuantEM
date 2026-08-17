"""A job type may be declared before it can be run -- but never enqueued.

``registry.job_handler`` refuses to register a handler for a type that is not
in ``ALLOWED_JOB_TYPES``, so a release built by several packages at once has to
declare every new type up front, in one edit, before any of the packages that
implement them land. That leaves a window in which the queue knows a name it
cannot dispatch.

The hazard is specific and it is in ``JobCreateSerializer``: its ``type`` field
is a ``ChoiceField`` over ``ALLOWED_JOB_TYPES``, so during that window a client
could post one of the undeclared-but-unimplemented types, get a ``201``, and
watch the row fail at dispatch with a message about a missing handler -- a
failure with no visible cause and nothing the user can do. These tests pin both
halves: the declaration is allowed to run ahead, and the door stays shut until
a handler exists.
"""

from __future__ import annotations

from django.test import TestCase
from rest_framework.test import APIClient

from quantem.jobs.constants import (
    ACTIVE_SEGMENTATION_JOB_TYPES,
    ALLOWED_JOB_TYPES,
    HEAVY_BACKGROUND_JOB_TYPES,
    HEAVY_CPU_JOB_TYPES,
    HEAVY_INTERACTIVE_JOB_TYPES,
    JOB_DEFAULTS,
    JOB_TYPE_LABELS,
    JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
    JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
    JOB_TYPE_RUN_ANALYSIS,
    JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
    NO_RETRY_JOB_TYPES,
)
from quantem.jobs.models import Job
from quantem.jobs.registry import _HANDLERS

#: The types this release declares ahead of their handlers.
DECLARED_AHEAD = (
    JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
    JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
)


class DeclarationCompletenessTests(TestCase):
    """A half-declared type is worse than an undeclared one.

    ``JOB_DEFAULTS[job_type]`` is read unguarded by the create serializer, so a
    type in ``ALLOWED_JOB_TYPES`` with no defaults entry is a ``KeyError`` on a
    request rather than a validation error. ``JOB_TYPE_LABELS`` is what the
    Tasks & Queues panel shows; a missing entry there is a task with no name.
    """

    def test_every_allowed_type_has_defaults_and_a_label(self):
        for job_type in sorted(ALLOWED_JOB_TYPES):
            with self.subTest(job_type=job_type):
                self.assertIn(job_type, JOB_DEFAULTS)
                self.assertIn(job_type, JOB_TYPE_LABELS)

    def test_the_labels_are_english_and_never_the_type_string(self):
        for job_type, label in JOB_TYPE_LABELS.items():
            with self.subTest(job_type=job_type):
                self.assertNotIn(job_type, label)
                self.assertNotIn("_", label)

    def test_the_new_types_are_declared_with_a_queue_that_matches_their_cost(self):
        one_image = JOB_DEFAULTS[JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE]
        self.assertEqual(one_image["resource_class"], "gpu")
        self.assertEqual(one_image["queue_name"], "p4_full")

        # Seconds of CPU with the user holding the dial: interactive queue, and
        # no GPU slot it has no use for.
        reextract = JOB_DEFAULTS[JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL]
        self.assertEqual(reextract["resource_class"], "cpu")
        self.assertEqual(reextract["queue_name"], "p1_interactive")

        # Analysis is user-requested scientific work; overlay rebuilds maintain
        # an already-saved display cache and must yield to it.
        self.assertEqual(JOB_DEFAULTS[JOB_TYPE_RUN_ANALYSIS]["queue_name"], "p1_interactive")
        self.assertEqual(
            JOB_DEFAULTS[JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY]["queue_name"],
            "p4_full",
        )

    def test_both_new_types_hold_a_segmentation_and_do_not_retry(self):
        for job_type in DECLARED_AHEAD:
            with self.subTest(job_type=job_type):
                self.assertIn(job_type, ACTIVE_SEGMENTATION_JOB_TYPES)
                self.assertIn(job_type, NO_RETRY_JOB_TYPES)


class HeavyJobClassificationTests(TestCase):
    """The heavy gate is only as good as which half a type was put in.

    ``runner._available_slots`` treats the two halves differently -- background
    maintenance waits for every heavy job, interactive work waits only for
    other interactive work -- so a type filed on the wrong side is not a
    cosmetic mistake. Filed as background it can be parked behind a display
    rebuild the user is not waiting on; filed as interactive it gets to run
    next to one.
    """

    def test_the_two_halves_partition_the_heavy_set(self):
        self.assertEqual(
            HEAVY_BACKGROUND_JOB_TYPES & HEAVY_INTERACTIVE_JOB_TYPES,
            frozenset(),
        )
        self.assertEqual(
            HEAVY_BACKGROUND_JOB_TYPES | HEAVY_INTERACTIVE_JOB_TYPES,
            HEAVY_CPU_JOB_TYPES,
        )

    def test_every_heavy_type_is_a_declared_cpu_job(self):
        """A gpu type here would be gated on a slot pool it never draws from."""
        for job_type in sorted(HEAVY_CPU_JOB_TYPES):
            with self.subTest(job_type=job_type):
                self.assertIn(job_type, ALLOWED_JOB_TYPES)
                self.assertEqual(JOB_DEFAULTS[job_type]["resource_class"], "cpu")

    def test_the_halves_agree_with_the_queue_each_type_was_declared_on(self):
        """The queue already records who is waiting; the sets must not disagree.

        A heavy job on the interactive queue but filed as background would be
        told to wait by the gate and told to go first by the scheduler, which
        is how the release ended up claiming a priority policy it did not have.
        """
        for job_type in sorted(HEAVY_INTERACTIVE_JOB_TYPES):
            with self.subTest(job_type=job_type):
                self.assertEqual(JOB_DEFAULTS[job_type]["queue_name"], "p1_interactive")
        for job_type in sorted(HEAVY_BACKGROUND_JOB_TYPES):
            with self.subTest(job_type=job_type):
                self.assertNotEqual(JOB_DEFAULTS[job_type]["queue_name"], "p1_interactive")


class UnimplementedTypesCannotBeEnqueuedTests(TestCase):
    """The door. This is the test that makes declaring ahead safe."""

    def setUp(self):
        self.client = APIClient()

    def test_a_declared_type_with_no_handler_is_refused_at_the_door(self):
        for job_type in DECLARED_AHEAD:
            if job_type in _HANDLERS:
                # Its package has landed; the gap this guards is closed for it.
                continue
            with self.subTest(job_type=job_type):
                response = self.client.post(
                    "/api/jobs/", {"type": job_type, "payload": {}}, format="json"
                )
                self.assertEqual(response.status_code, 400, response.data)
                self.assertEqual(Job.objects.filter(type=job_type).count(), 0)

    def test_the_refusal_says_what_happened_without_naming_the_task(self):
        unimplemented = [t for t in DECLARED_AHEAD if t not in _HANDLERS]
        if not unimplemented:
            self.skipTest("every declared type now has a handler")
        response = self.client.post(
            "/api/jobs/", {"type": unimplemented[0], "payload": {}}, format="json"
        )
        detail = " ".join(str(message) for message in response.data["type"])
        self.assertNotIn(unimplemented[0], detail)
        self.assertNotIn("handler", detail.lower())
        self.assertIn("cannot", detail.lower())

    def test_an_implemented_type_still_goes_through(self):
        """The gate must refuse the gap and nothing else."""
        response = self.client.post(
            "/api/jobs/",
            {
                "type": "rebuild_segmentation_overlay",
                "payload": {"segmentation_id": "not-used-by-validation"},
            },
            format="json",
        )
        self.assertEqual(response.status_code, 201, response.data)

    def test_every_registered_handler_is_a_type_a_client_may_post(self):
        """No handler is reachable by the worker but unreachable by a request.

        The inverse of the gate above: if the two lists drift the other way,
        work exists that nothing can start.
        """
        self.assertTrue(set(_HANDLERS).issubset(ALLOWED_JOB_TYPES))
