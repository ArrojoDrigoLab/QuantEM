"""``/api/system/status/`` publishes the upload size limit.

The server already refuses an oversized upload correctly: ``quantem serve``
hands ``settings.QUANTEM_MAX_UPLOAD_BYTES`` to waitress as
``max_request_body_size`` and a request declaring a larger body is answered
``413`` with a sentence a person can read.

The user never saw that sentence. waitress rejects from the request headers and
closes the socket while the browser is still streaming the body, which the
browser surfaces as a failed network request -- indistinguishable from the
machine going to sleep -- after however many minutes the doomed upload took. The
only way the refusal can reach a person *before* the wait is for the client to
know the number and check the file itself, and there was nowhere to read it
from: ``/api/system/status/`` answered ``cuda_available`` and
``supported_upload_formats`` and nothing about size.

So the endpoint that already publishes *which formats* the upload accepts now
also publishes *how large* an upload it accepts. Same contract, same endpoint,
same reason: the file picker must not have to guess, and a hard-coded number in
the client would drift from the one waitress is enforcing.
"""

from __future__ import annotations

from django.conf import settings
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from quantem import __version__


class SystemStatusUploadLimitTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def _status(self):
        response = self.client.get("/api/system/status/")
        self.assertEqual(response.status_code, 200)
        return response.data

    def test_the_limit_is_reported(self):
        body = self._status()

        self.assertIn("max_upload_bytes", body)
        self.assertEqual(body["max_upload_bytes"], settings.QUANTEM_MAX_UPLOAD_BYTES)

    def test_the_limit_is_a_positive_integer_of_bytes(self):
        """Not a float, not a string, not a "64 GB".

        The client compares it against ``File.size``, which is a number of
        bytes. A float would arrive in JavaScript as a value that cannot be
        compared exactly, and a formatted string would have to be parsed by the
        very code that is trying to avoid guessing.
        """
        value = self._status()["max_upload_bytes"]

        self.assertIsInstance(value, int)
        self.assertNotIsInstance(value, bool)
        self.assertGreater(value, 0)

    @override_settings(QUANTEM_MAX_UPLOAD_BYTES=123_456_789)
    def test_it_is_read_from_the_setting_waitress_is_given(self):
        """The number reported is the number enforced.

        ``quantem.cli._waitress_options`` reads the same setting. If this view
        answered a constant of its own, the two could disagree and the client
        would refuse files the server would have taken -- or, worse, promise to
        accept files it then rejects after a ten-minute upload.
        """
        self.assertEqual(self._status()["max_upload_bytes"], 123_456_789)

    def test_the_formats_and_the_gpu_flag_still_come_back(self):
        """The file picker reads all three from this one response."""
        body = self._status()

        self.assertIn("cuda_available", body)
        self.assertIsInstance(body["cuda_available"], bool)
        self.assertIn(".tif", body["supported_upload_formats"])

    def test_the_installed_application_version_is_published(self):
        self.assertEqual(self._status()["app_version"], __version__)
