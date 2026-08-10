from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from catalog.http import HttpFetchError
from literature_screen import literature


FIXTURES = Path(__file__).parent / "fixtures"


class LiteratureScannerTests(unittest.TestCase):
    def test_literature_config_has_journal_groups_terms_and_source_defaults(self):
        config = literature.load_literature_config()

        self.assertIn("source_journal_groups", config)
        self.assertGreaterEqual(len(config["source_journal_groups"]), 2)
        starter_issns = [
            issn
            for group in config["source_journal_groups"]
            for journal in group["journals"]
            for issn in journal["issn"]
        ]
        self.assertIn("0021-9525", starter_issns)
        for key in ("tem", "sbf_sem", "fib_sem", "volume_em", "ultrastructure", "organelle", "source_data", "repository", "dataset"):
            self.assertIn(key, config["query_terms"])
        for source in ("openalex", "crossref", "europepmc", "pubmed"):
            self.assertIn("limit", config["per_source"][source])
            self.assertIn("date_window_days", config["per_source"][source])

    def test_crossref_scanner_emits_metadata_only_article_candidate(self):
        fixture = _fixture("crossref_literature_search.json")

        def fake_get_json(url):
            self.assertTrue(url.startswith(literature.CROSSREF_WORKS_URL))
            self.assertIn("query.bibliographic=FIB-SEM+source+data", url)
            self.assertIn("from-pub-date%3A2025-01-01", url)
            return fixture

        with patch.object(literature, "get_json", side_effect=fake_get_json):
            result = literature.scan_literature(since="2025-01-01", query="FIB-SEM source data", limit=5, sources=["crossref"])

        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.source_name, "literature_crossref")
        self.assertEqual(candidate.source_record_id, "10.1234/jcb.2026.001")
        self.assertEqual(candidate.publication_doi, "10.1234/jcb.2026.001")
        self.assertEqual(candidate.landing_url, "https://doi.org/10.1234/jcb.2026.001")
        self.assertIn("FIB-SEM source data", candidate.title)
        self.assertIn("https://figshare.com/articles/dataset/fibsem_source_data/12345", candidate.download_or_manifest_urls)
        self.assertIn("https://doi.org/10.6084/m9.figshare.12345", candidate.download_or_manifest_urls)
        self.assertFalse(any("pdf" in url.lower() for url in candidate.download_or_manifest_urls))
        self.assertEqual(candidate.raw_metadata["journal"], "Journal of Cell Biology")
        self.assertEqual(candidate.raw_metadata["year"], "2026")
        self.assertIn("FIB-SEM", candidate.raw_metadata["matched_evidence_terms"])
        self.assertNotIn("<jats", candidate.raw_metadata["abstract"])

    def test_full_history_overrides_default_literature_window_and_limit_cap(self):
        fixture = _fixture("crossref_literature_search.json")

        def fake_get_json(url):
            self.assertTrue(url.startswith(literature.CROSSREF_WORKS_URL))
            self.assertIn("rows=250", url)
            self.assertNotIn("from-pub-date", url)
            return fixture

        with patch.object(literature, "get_json", side_effect=fake_get_json):
            result = literature.scan_literature(
                query="FIB-SEM source data",
                limit=250,
                sources=["crossref"],
                full_history=True,
            )

        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.candidates), 1)

    def test_europepmc_scanner_emits_metadata_only_article_candidate(self):
        fixture = _fixture("europepmc_literature_search.json")

        def fake_get_json(url):
            self.assertTrue(url.startswith(literature.EUROPEPMC_SEARCH_URL))
            self.assertIn("resultType=core", url)
            self.assertIn("FIRST_PDATE", url)
            return fixture

        with patch.object(literature, "get_json", side_effect=fake_get_json):
            result = literature.scan_literature(since="2024-01-01", query="TEM organelle source data", limit=5, sources=["europepmc"])

        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.source_name, "literature_europepmc")
        self.assertEqual(candidate.source_record_id, "10.7554/elife.99999")
        self.assertEqual(candidate.publication_doi, "10.7554/elife.99999")
        self.assertEqual(candidate.raw_metadata["pmid"], "40600001")
        self.assertEqual(candidate.raw_metadata["pmcid"], "PMC1234567")
        self.assertEqual(candidate.raw_metadata["journal"], "eLife")
        self.assertEqual(candidate.raw_metadata["year"], "2025")
        self.assertEqual(candidate.download_or_manifest_urls, ["https://www.ebi.ac.uk/biostudies/studies/S-BSST999"])
        self.assertIn("TEM", candidate.raw_metadata["matched_evidence_terms"])
        self.assertFalse(any("pdf" in url.lower() for url in candidate.download_or_manifest_urls))

    def test_pubmed_scanner_uses_esearch_esummary_and_elink_metadata(self):
        esearch = _fixture("pubmed_esearch_literature.json")
        esummary = _fixture("pubmed_esummary_literature.json")
        elink = _fixture("pubmed_elink_literature.json")

        def fake_get_json(url):
            if url.startswith(literature.PUBMED_ESEARCH_URL):
                self.assertIn("db=pubmed", url)
                self.assertIn("SBF-SEM", url)
                return esearch
            if url.startswith(literature.PUBMED_ESUMMARY_URL):
                self.assertIn("id=40600001", url)
                return esummary
            if url.startswith(literature.PUBMED_ELINK_URL):
                self.assertIn("cmd=llinks", url)
                return elink
            raise AssertionError(f"unexpected URL {url}")

        with patch.object(literature, "get_json", side_effect=fake_get_json):
            result = literature.scan_literature(since="2025-01-01", query="SBF-SEM source data", limit=5, sources=["pubmed"])

        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.source_name, "literature_pubmed")
        self.assertEqual(candidate.source_record_id, "40600001")
        self.assertEqual(candidate.publication_doi, "10.5555/jcb.2026.2")
        self.assertEqual(candidate.raw_metadata["pmid"], "40600001")
        self.assertEqual(candidate.raw_metadata["pmcid"], "PMC7654321")
        self.assertEqual(candidate.raw_metadata["journal"], "Journal of Cell Biology")
        self.assertEqual(candidate.raw_metadata["year"], "2026")
        self.assertEqual(candidate.download_or_manifest_urls, ["https://zenodo.org/records/12345"])
        self.assertIn("SBF-SEM", candidate.raw_metadata["matched_evidence_terms"])
        self.assertFalse(any("pdf" in url.lower() for url in candidate.download_or_manifest_urls))

    def test_literature_malformed_payloads_become_scanner_errors(self):
        cases = [
            ("crossref", "crossref_literature_malformed.json", "Crossref literature payload missing message.items list"),
            ("europepmc", "europepmc_literature_malformed.json", "Europe PMC literature payload missing resultList.result list"),
            ("pubmed", "pubmed_esearch_malformed.json", "PubMed ESearch payload missing esearchresult.idlist"),
        ]
        for source, fixture_name, expected_message in cases:
            with self.subTest(source=source):
                with patch.object(literature, "get_json", return_value=_fixture(fixture_name)):
                    result = literature.scan_literature(query="source data", limit=5, sources=[source])

                self.assertEqual(result.candidates, [])
                self.assertEqual(len(result.errors), 1)
                self.assertEqual(result.errors[0].source_name, "literature")
                self.assertEqual(result.errors[0].adapter, f"{source}_literature")
                self.assertEqual(result.errors[0].error_type, "ValueError")
                self.assertIn(expected_message, result.errors[0].message)

    def test_pubmed_malformed_esummary_and_elink_payloads_become_scanner_errors(self):
        esearch = _fixture("pubmed_esearch_literature.json")
        esummary = _fixture("pubmed_esummary_literature.json")
        cases = [
            ("esummary", "pubmed_esummary_malformed.json", "PubMed ESummary payload missing object for uid 40600001"),
            ("elink", "pubmed_elink_malformed.json", "PubMed ELink payload missing linksets list"),
        ]
        for stage, fixture_name, expected_message in cases:
            with self.subTest(stage=stage):
                def fake_get_json(url):
                    if url.startswith(literature.PUBMED_ESEARCH_URL):
                        return esearch
                    if url.startswith(literature.PUBMED_ESUMMARY_URL):
                        return _fixture(fixture_name) if stage == "esummary" else esummary
                    if url.startswith(literature.PUBMED_ELINK_URL):
                        return _fixture(fixture_name)
                    raise AssertionError(f"unexpected URL {url}")

                with patch.object(literature, "get_json", side_effect=fake_get_json):
                    result = literature.scan_literature(query="source data", limit=5, sources=["pubmed"])

                self.assertEqual(result.candidates, [])
                self.assertEqual(len(result.errors), 1)
                self.assertEqual(result.errors[0].adapter, "pubmed_literature")
                self.assertEqual(result.errors[0].error_type, "ValueError")
                self.assertIn(expected_message, result.errors[0].message)

    def test_literature_api_failures_become_scanner_errors(self):
        for source in ("crossref", "europepmc", "pubmed"):
            with self.subTest(source=source):
                error = HttpFetchError("https://example.org/api", "HTTP 503", status=503, excerpt="maintenance")
                with patch.object(literature, "get_json", side_effect=error):
                    result = literature.scan_literature(query="source data", limit=5, sources=[source])

                self.assertEqual(result.candidates, [])
                self.assertEqual(len(result.errors), 1)
                self.assertEqual(result.errors[0].adapter, f"{source}_literature")
                self.assertEqual(result.errors[0].response_status, 503)
                self.assertEqual(result.errors[0].response_excerpt, "maintenance")


def _fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
