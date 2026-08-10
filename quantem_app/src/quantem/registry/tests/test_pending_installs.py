"""The installer's model checkboxes must become real install jobs (Ruling C).

The NSIS installer offers the eight packs at install time but downloads
nothing itself -- the app's install machinery is the tested path (digest
verification, progress, cancel, AV-retry). It writes a one-shot request::

    <data dir>/pending-model-installs.json
    {"packs": ["omniem:mito", ...]}

and on the first ``quantem serve`` (the frozen build runs the same command)
the server queues an ordinary ``install_model_pack`` job per not-yet-installed
pack and deletes the file. The file is a request, not state: deleted even on
partial queueing, and a malformed one is logged and deleted -- the server must
never refuse to start over it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

from django.test import TestCase

from quantem.jobs.constants import JOB_TYPE_INSTALL_MODEL_PACK, QUEUE_P2_UPLOAD
from quantem.jobs.models import Job
from quantem.registry.pending_installs import (
    PENDING_INSTALLS_FILENAME,
    pending_installs_path,
    process_pending_model_installs,
)


def _write_request(payload) -> Path:
    path = pending_installs_path()
    if isinstance(payload, (dict, list)):
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_text(str(payload), encoding="utf-8")
    return path


class PendingModelInstallsTests(TestCase):
    def tearDown(self):
        # The function under test deletes the file itself; this is the guard
        # against a failing test leaking the request into another one.
        pending_installs_path().unlink(missing_ok=True)

    def test_the_request_becomes_install_jobs_and_the_file_is_deleted(self):
        path = _write_request({"packs": ["omniem:mito", "quantem:er"]})

        with patch("quantem.registry.cache.installed", return_value=False):
            queued = process_pending_model_installs()

        self.assertEqual(queued, ["omniem:mito", "quantem:er"])
        self.assertFalse(path.exists(), "the request file is one-shot")

        jobs = Job.objects.filter(type=JOB_TYPE_INSTALL_MODEL_PACK).order_by(
            "created_at"
        )
        self.assertEqual(
            sorted(job.payload_json["pack_id"] for job in jobs),
            ["omniem:mito", "quantem:er"],
        )
        for job in jobs:
            # The exact shape the Models screen's install endpoint enqueues, so
            # the handler, the progress polling and the failed-install
            # surfacing all see a job they already know.
            self.assertEqual(job.status, "PENDING")
            self.assertEqual(job.payload_json["source"], "huggingface")
            self.assertFalse(job.payload_json["force"])
            self.assertIn("repo_id", job.payload_json)
            self.assertIn("revision", job.payload_json)
            self.assertEqual(job.queue_name, QUEUE_P2_UPLOAD)
            self.assertEqual(job.max_attempts, 1)
            self.assertIn(f"model:{job.payload_json['pack_id']}", job.tags)

    def test_already_installed_packs_are_not_queued_again(self):
        _write_request({"packs": ["omniem:mito", "quantem:er"]})

        with patch(
            "quantem.registry.cache.installed",
            side_effect=lambda pack_id: pack_id == "omniem:mito",
        ):
            queued = process_pending_model_installs()

        self.assertEqual(queued, ["quantem:er"])
        self.assertEqual(Job.objects.filter(type=JOB_TYPE_INSTALL_MODEL_PACK).count(), 1)

    def test_an_unknown_pack_id_is_skipped_and_the_rest_still_queue(self):
        _write_request({"packs": ["definitely:not-a-pack", "omniem:mito"]})

        with patch("quantem.registry.cache.installed", return_value=False):
            queued = process_pending_model_installs()

        self.assertEqual(queued, ["omniem:mito"])
        self.assertFalse(pending_installs_path().exists())

    def test_a_pack_with_an_active_install_job_is_not_queued_twice(self):
        """serve can restart between queueing and the download finishing."""
        Job.enqueue(
            job_type=JOB_TYPE_INSTALL_MODEL_PACK,
            payload={"pack_id": "omniem:mito", "source": "huggingface"},
            queue_name=QUEUE_P2_UPLOAD,
            max_attempts=1,
        )
        _write_request({"packs": ["omniem:mito"]})

        with patch("quantem.registry.cache.installed", return_value=False):
            queued = process_pending_model_installs()

        self.assertEqual(queued, [])
        self.assertEqual(Job.objects.filter(type=JOB_TYPE_INSTALL_MODEL_PACK).count(), 1)

    def test_a_utf8_bom_on_the_request_file_is_accepted(self):
        """UAT round 13, paper-cut 7: a BOM'd request died for an invisible reason.

        The real NSIS writer emits BOM-less bytes, but a hand-edited file
        (Notepad's default was UTF-8-with-BOM for years) must not be rejected
        over three invisible bytes -- the reader decodes ``utf-8-sig``.
        """
        path = pending_installs_path()
        path.write_bytes(
            b"\xef\xbb\xbf" + json.dumps({"packs": ["omniem:mito"]}).encode("utf-8")
        )

        with patch("quantem.registry.cache.installed", return_value=False):
            queued = process_pending_model_installs()

        self.assertEqual(queued, ["omniem:mito"])
        self.assertFalse(path.exists())
        job = Job.objects.get(type=JOB_TYPE_INSTALL_MODEL_PACK)
        self.assertEqual(job.payload_json["pack_id"], "omniem:mito")

    def test_a_malformed_file_is_deleted_and_never_crashes(self):
        for payload in ("this is not JSON {{{", {"packs": "omniem:mito"}, ["nope"], {}):
            path = _write_request(payload)

            queued = process_pending_model_installs()  # must not raise

            self.assertEqual(queued, [])
            self.assertFalse(path.exists(), f"malformed request {payload!r} kept")
        self.assertEqual(Job.objects.filter(type=JOB_TYPE_INSTALL_MODEL_PACK).count(), 0)

    def test_no_file_is_a_quiet_noop(self):
        self.assertFalse(pending_installs_path().exists())
        self.assertEqual(process_pending_model_installs(), [])
        self.assertEqual(Job.objects.count(), 0)


class ServeEntryPointTests(TestCase):
    """The hook must run from the real ``quantem serve`` startup sequence."""

    def tearDown(self):
        pending_installs_path().unlink(missing_ok=True)

    def test_cmd_serve_consumes_the_request_before_serving(self):
        from quantem.cli import cmd_serve
        from quantem.core.config import STORAGE_DIR

        path = _write_request({"packs": ["omniem:mito"]})
        self.assertEqual(path.parent, Path(STORAGE_DIR))
        self.assertEqual(path.name, PENDING_INSTALLS_FILENAME)

        env = {
            # The suite must not gain a scheduler thread from this call.
            "QUANTEM_AUTOSTART_JOBS": "0",
            "QUANTEM_DATA_DIR": str(STORAGE_DIR),
        }
        with (
            patch.dict(os.environ, env),
            patch("waitress.serve") as serve,
            patch("django.core.management.call_command") as migrate,
            patch("quantem.registry.cache.installed", return_value=False),
        ):
            rc = cmd_serve(argparse.Namespace(port=0, data_dir=str(STORAGE_DIR)))

        self.assertEqual(rc, 0)
        self.assertTrue(serve.called)
        self.assertTrue(migrate.called)
        self.assertFalse(path.exists(), "cmd_serve must consume the request file")
        job = Job.objects.get(type=JOB_TYPE_INSTALL_MODEL_PACK)
        self.assertEqual(job.payload_json["pack_id"], "omniem:mito")
        self.assertEqual(job.status, "PENDING")
