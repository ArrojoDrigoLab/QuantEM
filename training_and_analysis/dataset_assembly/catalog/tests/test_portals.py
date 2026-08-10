from __future__ import annotations

import json
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from catalog.http import HttpFetchError
from catalog.sources import SOURCE_NAMES
from catalog.sources import portals as generic


FIXTURES = Path(__file__).parent / "fixtures"


class GenericPortalScannerTests(unittest.TestCase):
    def test_default_portal_query_groups_cover_high_recall_terms(self):
        joined = " ".join(generic.PORTAL_QUERY_GROUPS)
        for term in generic.HIGH_RECALL_TERMS:
            self.assertIn(term, joined)

    def test_figshare_scanner_emits_enriched_candidate(self):
        search = _fixture("figshare_search.json")
        article = _fixture("figshare_article.json")

        def fake_post_json(url, payload):
            self.assertEqual(url, generic.FIGSHARE_SEARCH_URL)
            self.assertEqual(payload["item_type"], 3)
            self.assertEqual(payload["search_for"], "FIB-SEM ultrastructure mitochondria")
            return search

        def fake_get_json(url):
            self.assertEqual(url, "https://api.figshare.com/v2/articles/19898404")
            return article

        with patch.object(generic, "post_json", side_effect=fake_post_json), patch.object(generic, "get_json", side_effect=fake_get_json):
            result = generic.scan_generic_source("figshare", query="FIB-SEM ultrastructure mitochondria", limit=1)

        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.source_name, "figshare")
        self.assertEqual(candidate.source_record_id, "19898404")
        self.assertEqual(candidate.dataset_doi, "10.6084/m9.figshare.19898404.v1")
        self.assertEqual(candidate.publication_doi, "10.1101/2022.01.01.000001")
        self.assertEqual(candidate.landing_url, "https://figshare.com/articles/dataset/DeepContact_Training_Data/19898404")
        self.assertIn("CC BY 4.0", candidate.license)
        self.assertEqual(candidate.dimensions_or_image_count, "2 files; 3072 bytes total")
        self.assertIn("TIFF", candidate.file_formats)
        self.assertIn("ZIP", candidate.file_formats)
        self.assertIn("image/tiff", candidate.file_formats)
        self.assertIn("mitochondria", candidate.evidence_text)
        self.assertNotIn("<p>", candidate.evidence_text)
        self.assertEqual(candidate.download_or_manifest_urls, ["https://api.figshare.com/v2/articles/19898404"])
        self.assertNotIn("references", candidate.raw_metadata["article"])
        self.assertNotIn("download_url", candidate.raw_metadata["article"]["files"][0])

    def test_figshare_scanner_paginates_for_larger_limits(self):
        search = _fixture("figshare_search.json")
        article = _fixture("figshare_article.json")
        second_search = [dict(search[0], id=19898405, url_public_api="https://api.figshare.com/v2/articles/19898405")]
        second_article = dict(article, id=19898405, title="Second FIB-SEM dataset")
        pages: list[int] = []

        def fake_post_json(url, payload):
            self.assertEqual(url, generic.FIGSHARE_SEARCH_URL)
            pages.append(payload["page"])
            if payload["page"] == 1:
                return [search[0], dict(search[0])]
            if payload["page"] == 2:
                return second_search
            return []

        def fake_get_json(url):
            if url == "https://api.figshare.com/v2/articles/19898404":
                return article
            if url == "https://api.figshare.com/v2/articles/19898405":
                return second_article
            raise AssertionError(f"unexpected URL {url}")

        with patch.object(generic, "post_json", side_effect=fake_post_json), patch.object(generic, "get_json", side_effect=fake_get_json):
            result = generic.scan_generic_source("figshare", query="FIB-SEM ultrastructure mitochondria", limit=2)

        self.assertEqual(result.errors, [])
        self.assertEqual([candidate.source_record_id for candidate in result.candidates], ["19898404", "19898405"])
        self.assertEqual(pages, [1, 2])

    def test_figshare_malformed_search_record_becomes_scanner_error(self):
        malformed = _fixture("figshare_malformed_search.json")
        with patch.object(generic, "post_json", return_value=malformed):
            result = generic.scan_generic_source("figshare", limit=5)

        self.assertEqual(result.candidates, [])
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].source_name, "figshare")
        self.assertEqual(result.errors[0].error_type, "ValueError")
        self.assertIn("missing id", result.errors[0].message)

    def test_figshare_api_failure_becomes_scanner_error(self):
        error = HttpFetchError(generic.FIGSHARE_SEARCH_URL, "HTTP 500", status=500, excerpt="upstream failure")
        with patch.object(generic, "post_json", side_effect=error):
            result = generic.scan_generic_source("figshare", limit=5)

        self.assertEqual(result.candidates, [])
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].response_status, 500)
        self.assertEqual(result.errors[0].response_excerpt, "upstream failure")

    def test_figshare_oai_source_is_registered(self):
        self.assertIn("figshare_oai", SOURCE_NAMES)

    def test_figshare_oai_scanner_emits_relevant_datacite_candidate(self):
        xml = _text_fixture("figshare_oai_datacite_records.xml")
        seen_urls: list[str] = []

        def fake_get_text(url):
            seen_urls.append(url)
            qs = parse_qs(urlparse(url).query)
            self.assertEqual(qs["verb"], ["ListRecords"])
            self.assertEqual(qs["metadataPrefix"], ["oai_datacite"])
            return xml

        with patch.object(generic, "get_text", side_effect=fake_get_text):
            result = generic.scan_generic_source("figshare_oai", limit=1, full_history=True)

        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.source_name, "figshare_oai")
        self.assertEqual(candidate.source_record_id, "24502954")
        self.assertEqual(candidate.dataset_doi, "10.6084/m9.figshare.24502954.v1")
        self.assertEqual(candidate.publication_doi, "10.1038/s41586-example")
        self.assertEqual(candidate.landing_url, "https://janelia.figshare.com/articles/dataset/Raw_FIB-SEM_mitochondria_volume/24502954")
        self.assertEqual(candidate.download_or_manifest_urls, ["https://api.figshare.com/v2/articles/24502954"])
        self.assertEqual(candidate.dimensions_or_image_count, "1 related file URLs")
        self.assertIn("TIFF", candidate.file_formats)
        self.assertIn("application/zip", candidate.file_formats)
        self.assertIn("Creative Commons Attribution 4.0", candidate.license)
        self.assertIn("intracellular ultrastructure", candidate.evidence_text)
        self.assertNotIn("download_url", candidate.raw_metadata["oai"])
        self.assertFalse(result.cursor_complete)
        self.assertEqual(result.cursor["resumption_token"], "abc-token")
        self.assertEqual(result.cursor["records_examined_in_batch"], 2)
        self.assertEqual(len(seen_urls), 1)

    def test_figshare_oai_scanner_resumes_from_resumption_token(self):
        xml = _text_fixture("figshare_oai_datacite_second_page.xml")
        seen_url = None

        def fake_get_text(url):
            nonlocal seen_url
            seen_url = url
            return xml

        cursor = {
            "adapter": "figshare_oai",
            "metadata_prefix": "oai_datacite",
            "resumption_token": "abc-token",
            "from": "2026-05-01",
            "complete": False,
        }
        with patch.object(generic, "get_text", side_effect=fake_get_text):
            result = generic.scan_generic_source("figshare_oai", limit=10, full_history=True, cursor=cursor)

        self.assertEqual(result.errors, [])
        self.assertTrue(result.cursor_complete)
        self.assertEqual(result.cursor, {"complete": True})
        self.assertEqual(result.candidates[0].source_record_id, "24512767")
        qs = parse_qs(urlparse(seen_url).query)
        self.assertEqual(qs["verb"], ["ListRecords"])
        self.assertEqual(qs["resumptionToken"], ["abc-token"])
        self.assertNotIn("metadataPrefix", qs)
        self.assertNotIn("from", qs)

    def test_figshare_oai_empty_final_page_completes_cursor(self):
        xml = _text_fixture("figshare_oai_empty_final_page.xml")
        seen_url = None

        def fake_get_text(url):
            nonlocal seen_url
            seen_url = url
            return xml

        cursor = {
            "adapter": "figshare_oai",
            "metadata_prefix": "oai_datacite",
            "resumption_token": "stale-token",
            "from": None,
            "complete": False,
        }
        with patch.object(generic, "get_text", side_effect=fake_get_text):
            result = generic.scan_generic_source("figshare_oai", limit=10, full_history=True, cursor=cursor)

        self.assertEqual(result.errors, [])
        self.assertEqual(result.candidates, [])
        self.assertTrue(result.cursor_complete)
        self.assertEqual(result.cursor, {"complete": True})
        qs = parse_qs(urlparse(seen_url).query)
        self.assertEqual(qs["resumptionToken"], ["stale-token"])

    def test_figshare_oai_incremental_since_uses_from_date(self):
        xml = _text_fixture("figshare_oai_no_records.xml")
        seen: dict[str, str] = {}

        def fake_get_text(url):
            qs = parse_qs(urlparse(url).query)
            seen["from"] = qs["from"][0]
            seen["metadataPrefix"] = qs["metadataPrefix"][0]
            return xml

        with patch.object(generic, "get_text", side_effect=fake_get_text):
            result = generic.scan_generic_source("figshare_oai", limit=5, since="2026-05-01")

        self.assertEqual(result.errors, [])
        self.assertEqual(result.candidates, [])
        self.assertTrue(result.cursor_complete)
        self.assertEqual(seen["from"], "2026-05-01")
        self.assertEqual(seen["metadataPrefix"], "oai_datacite")

    def test_mendeley_scanner_emits_datacite_and_file_metadata(self):
        datacite = _fixture("mendeley_datacite_search.json")
        public_api = _fixture("mendeley_public_api.json")

        def fake_get_json(url):
            if url.startswith(generic.DATACITE_DOIS_URL):
                self.assertIn("client-id=bl.mendeley", url)
                self.assertIn("resource-type-id=dataset", url)
                self.assertIn("SBF-SEM", url)
                return datacite
            if url == "https://data.mendeley.com/public-api/datasets/emdata123":
                return public_api
            raise AssertionError(f"unexpected URL {url}")

        with patch.object(generic, "get_json", side_effect=fake_get_json):
            result = generic.scan_generic_source("mendeley", query="SBF-SEM serial block face ultrastructure", limit=1)

        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.source_name, "mendeley")
        self.assertEqual(candidate.source_record_id, "emdata123:v2")
        self.assertEqual(candidate.dataset_doi, "10.17632/emdata123.2")
        self.assertEqual(candidate.publication_doi, "10.1016/j.example.2025.1")
        self.assertEqual(candidate.landing_url, "https://data.mendeley.com/datasets/emdata123")
        self.assertIn("CC BY 4.0", candidate.license)
        self.assertEqual(candidate.dimensions_or_image_count, "3 files; 4096 bytes total")
        self.assertIn("MRC", candidate.file_formats)
        self.assertIn("TIFF", candidate.file_formats)
        self.assertIn("image/tiff", candidate.file_formats)
        self.assertIn("intracellular ultrastructure", candidate.evidence_text)
        self.assertIn("https://data.mendeley.com/public-api/datasets/emdata123", candidate.download_or_manifest_urls)
        self.assertNotIn("xml", candidate.raw_metadata["datacite"]["attributes"])
        self.assertNotIn("download_url", candidate.raw_metadata["mendeley_public_api"]["files"][0])

    def test_datacite_scanner_emits_cross_repository_dataset_candidate(self):
        datacite = _fixture("mendeley_datacite_search.json")

        def fake_get_json(url):
            self.assertTrue(url.startswith(generic.DATACITE_DOIS_URL))
            self.assertIn("resource-type-id=dataset", url)
            self.assertIn("page%5Bnumber%5D=1", url)
            self.assertIn("FIB-SEM", url)
            return datacite

        with patch.object(generic, "get_json", side_effect=fake_get_json):
            result = generic.scan_generic_source("datacite", query="FIB-SEM ultrastructure", limit=1)

        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.source_name, "datacite")
        self.assertEqual(candidate.dataset_doi, "10.17632/emdata123.2")
        self.assertIn("intracellular ultrastructure", candidate.evidence_text)

    def test_incremental_since_is_sent_to_supported_portals(self):
        seen: dict[str, str] = {}

        def fake_get_json(url):
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            if url.startswith("https://zenodo.org/api/records"):
                seen["zenodo_q"] = qs["q"][0]
                return {"hits": {"hits": []}}
            if url.startswith(generic.DATACITE_DOIS_URL):
                seen["datacite_since"] = qs["from-updated-date"][0]
                return {"data": []}
            raise AssertionError(f"unexpected URL {url}")

        with patch.object(generic, "get_json", side_effect=fake_get_json):
            zenodo = generic.scan_generic_source("zenodo", query="electron microscopy", limit=1, since="2026-05-01")
            datacite = generic.scan_generic_source("datacite", query="FIB-SEM", limit=1, since="2026-05-01")

        self.assertEqual(zenodo.errors, [])
        self.assertEqual(datacite.errors, [])
        self.assertEqual(seen["zenodo_q"], "(electron microscopy) AND updated:[2026-05-01 TO *]")
        self.assertEqual(seen["datacite_since"], "2026-05-01")

    def test_zenodo_cursor_resumes_from_next_link(self):
        calls: list[str] = []
        first = {
            "hits": {
                "hits": [
                    {
                        "id": 1,
                        "metadata": {"title": "First TEM organelle dataset"},
                        "links": {"html": "https://zenodo.org/records/1"},
                        "doi": "10.5281/zenodo.1",
                    }
                ]
            },
            "links": {"next": "https://zenodo.org/api/records?page=2"},
        }
        second = {
            "hits": {
                "hits": [
                    {
                        "id": 2,
                        "metadata": {"title": "Second TEM organelle dataset"},
                        "links": {"html": "https://zenodo.org/records/2"},
                        "doi": "10.5281/zenodo.2",
                    }
                ]
            }
        }

        def fake_get_json(url):
            calls.append(url)
            if url == "https://zenodo.org/api/records?page=2":
                return second
            return first

        with patch.object(generic, "get_json", side_effect=fake_get_json):
            first_result = generic.scan_generic_source("zenodo", query="TEM", limit=1, full_history=True)
            second_result = generic.scan_generic_source(
                "zenodo",
                query="TEM",
                limit=1,
                full_history=True,
                cursor=first_result.cursor,
            )

        self.assertEqual(first_result.candidates[0].source_record_id, "1")
        self.assertFalse(first_result.cursor_complete)
        self.assertEqual(first_result.cursor["next_url"], "https://zenodo.org/api/records?page=2")
        self.assertEqual(second_result.candidates[0].source_record_id, "2")
        self.assertTrue(second_result.cursor_complete)
        self.assertEqual(calls[1], "https://zenodo.org/api/records?page=2")

    def test_dataverse_cursor_resumes_from_start_offset(self):
        starts: list[str] = []

        def fake_get_json(url):
            qs = parse_qs(urlparse(url).query)
            starts.append(qs["start"][0])
            start = int(qs["start"][0])
            return {
                "data": {
                    "items": [
                        {
                            "global_id": f"doi:10.7910/DVN/EMDATA{start}",
                            "name": f"FIB-SEM dataset {start}",
                            "persistentUrl": f"https://doi.org/10.7910/DVN/EMDATA{start}",
                            "description": "Public FIB-SEM mitochondria image volumes.",
                            "fileCount": 1,
                        }
                    ]
                }
            }

        with patch.object(generic, "get_json", side_effect=fake_get_json):
            first = generic.scan_generic_source("dataverse", query="FIB-SEM", limit=1, full_history=True)
            second = generic.scan_generic_source("dataverse", query="FIB-SEM", limit=1, full_history=True, cursor=first.cursor)

        self.assertEqual(starts, ["0", "1"])
        self.assertEqual(first.cursor["start"], 1)
        self.assertEqual(second.candidates[0].source_record_id, "doi:10.7910/DVN/EMDATA1")

    def test_dataverse_scanner_emits_metadata_only_dataset_candidate(self):
        response = {
            "data": {
                "items": [
                    {
                        "global_id": "doi:10.7910/DVN/EMDATA",
                        "name": "FIB-SEM intracellular ultrastructure dataset",
                        "persistentUrl": "https://doi.org/10.7910/DVN/EMDATA",
                        "description": "Public FIB-SEM mitochondria image volumes.",
                        "fileCount": 3,
                        "size_in_bytes": 4096,
                        "subjects": ["Medicine, Health and Life Sciences"],
                    }
                ]
            }
        }

        def fake_get_json(url):
            self.assertTrue(url.startswith(generic.DATAVERSE_SEARCH_URL))
            self.assertIn("type=dataset", url)
            self.assertIn("per_page=1", url)
            return response

        with patch.object(generic, "get_json", side_effect=fake_get_json):
            result = generic.scan_generic_source("dataverse", query="FIB-SEM ultrastructure", limit=1)

        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.source_name, "dataverse")
        self.assertEqual(candidate.dataset_doi, "10.7910/DVN/EMDATA")
        self.assertEqual(candidate.dimensions_or_image_count, "3 files; 4096 bytes total")
        self.assertEqual(candidate.download_or_manifest_urls, ["https://doi.org/10.7910/DVN/EMDATA"])

    def test_mendeley_malformed_datacite_record_becomes_scanner_error(self):
        malformed = _fixture("mendeley_malformed_datacite.json")
        with patch.object(generic, "get_json", return_value=malformed):
            result = generic.scan_generic_source("mendeley", limit=5)

        self.assertEqual(result.candidates, [])
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].source_name, "mendeley")
        self.assertEqual(result.errors[0].error_type, "ValueError")
        self.assertIn("missing attributes", result.errors[0].message)

    def test_mendeley_api_failure_becomes_scanner_error(self):
        error = HttpFetchError(generic.DATACITE_DOIS_URL, "HTTP 503", status=503, excerpt="maintenance")
        with patch.object(generic, "get_json", side_effect=error):
            result = generic.scan_generic_source("mendeley", limit=5)

        self.assertEqual(result.candidates, [])
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].response_status, 503)
        self.assertEqual(result.errors[0].response_excerpt, "maintenance")


def _fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _text_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
