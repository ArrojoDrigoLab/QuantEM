"""Free text typed on the import form survives the import, and is findable.

The import form had a box labelled *"Tags (comma-separated, optional)"*. The
client collected it, ``uploadAsset`` posted it as ``tag_names``, and
``AssetUploadView.post`` never looked at it -- there is no tag field on
:class:`~quantem.assets.models.Asset` and no tag model anywhere in this tree.
A user typed ``PV`` into it and nothing happened: the library showed no tags and
its search still matched only names and filenames. No number was wrong, which is
what made it durable -- a field that accepts input and discards it is how people
come to believe their images are grouped when they are not.

``notes`` is the field that already existed. It is a real column, it is patchable
through :func:`~quantem.assets.asset_mutations.update_asset`, and
``AssetListView``'s ``search`` has always matched it. Upload was simply the one
door that could not set it, so the form's free-text box now writes there.

These tests are about the round trip, because that is exactly what was missing:
posting the value, reading it back, and finding the image by it.
"""

from __future__ import annotations

from django.test import TestCase
from rest_framework.test import APIClient

from quantem.assets.models import Asset
from quantem.testing import build_test_upload_file


class UploadNotesTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def _upload(self, **extra):
        payload = {"file": build_test_upload_file(), "display_name": "Scan 1"}
        payload.update(extra)
        response = self.client.post("/api/assets/upload/", payload, format="multipart")
        self.assertEqual(response.status_code, 201, response.data)
        return response.data

    def test_notes_are_stored_and_returned(self):
        body = self._upload(notes="PV, day 14")

        self.assertEqual(body["notes"], "PV, day 14")
        asset = Asset.objects.get(id=body["id"])
        self.assertEqual(asset.notes, "PV, day 14")

    def test_the_library_search_finds_the_image_by_its_notes(self):
        """The whole point of typing a word there.

        ``_filtered_asset_queryset`` matches display name, original filename and
        notes; the display name here deliberately shares nothing with the search
        term, so a hit can only have come from the notes.
        """
        self._upload(notes="PV")

        response = self.client.get("/api/assets/", {"search": "PV"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual([entry["display_name"] for entry in response.data], ["Scan 1"])

    def test_an_omitted_note_is_empty_rather_than_null(self):
        """``Asset.notes`` is ``TextField(blank=True, default="")``.

        Writing ``None`` into it would make every consumer -- the search, the
        serializer, the client's optional ``notes?: string`` -- handle a second
        empty value for no gain.
        """
        body = self._upload()

        self.assertEqual(body["notes"], "")
        self.assertEqual(Asset.objects.get(id=body["id"]).notes, "")

    def test_surrounding_whitespace_is_not_stored(self):
        # " PV " and "PV" are the same note, and only one of them is found by a
        # search for "PV " typed with a trailing space.
        body = self._upload(notes="  PV  ")

        self.assertEqual(body["notes"], "PV")

    def test_tag_names_is_not_quietly_accepted(self):
        """The field the form used to post, and the reason this file exists.

        Nothing reads ``tag_names``, so the guarantee worth pinning is the
        honest one: posting it changes nothing at all. If a tag feature is ever
        built, this test failing is the correct way to find out that a client
        was relying on the old name in the meantime.
        """
        body = self._upload(tag_names="PV,control")

        self.assertEqual(body["notes"], "")
        response = self.client.get("/api/assets/", {"search": "PV"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(list(response.data), [])
