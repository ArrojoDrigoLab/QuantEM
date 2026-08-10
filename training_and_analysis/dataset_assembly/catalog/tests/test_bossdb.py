from __future__ import annotations

import json
import unittest
from pathlib import Path

from catalog.models import Candidate
from catalog.sources.bossdb import (
    _candidates_from_project_records,
)


class BossDBScannerTests(unittest.TestCase):
    def test_lee_bock_project_records_to_candidates(self):
        fixture = Path(__file__).parent / "fixtures" / "bossdb_projects_lee_bock.json"
        records = json.loads(fixture.read_text(encoding="utf-8"))["data"]

        result = _candidates_from_project_records(records, limit=100)

        self.assertEqual(result.errors, [])
        candidates = {candidate.source_record_id: candidate for candidate in result.candidates}
        self.assertEqual(set(candidates), {"lee2016", "bock2011"})

        lee = candidates["lee2016"]
        self.assertEqual(lee.source_name, "bossdb")
        self.assertEqual(lee.landing_url, "https://bossdb.org/project/lee2016")
        self.assertEqual(lee.dataset_doi, "10.60533/BOSS-2016-8MUX")
        self.assertEqual(lee.publication_doi, "10.1038/nature17192")
        self.assertIn("TEM", lee.modality)
        self.assertIn("bossdb://lee/lee16/image", lee.download_or_manifest_urls)
        self.assertIn("BossDB project metadata ID: lee2016", lee.evidence_text)

        bock = candidates["bock2011"]
        self.assertEqual(bock.landing_url, "https://bossdb.org/project/bock2011")
        self.assertIn("bossdb://bock/bock11/image", bock.raw_metadata["bossdb_uris"])
        self.assertEqual(bock.license, "CC-BY 4.0")

    def test_malformed_project_records_emit_parser_errors(self):
        fixture = Path(__file__).parent / "fixtures" / "bossdb_malformed_projects.json"
        records = json.loads(fixture.read_text(encoding="utf-8"))["data"]

        result = _candidates_from_project_records(records, limit=100)

        self.assertEqual(result.candidates, [])
        self.assertEqual(len(result.errors), 3)
        self.assertTrue(all(error.source_name == "bossdb" for error in result.errors))
        self.assertIn("missing ID", result.errors[0].message)

if __name__ == "__main__":
    unittest.main()
