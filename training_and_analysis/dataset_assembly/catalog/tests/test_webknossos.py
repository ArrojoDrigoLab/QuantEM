from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from catalog.sources.webknossos import scan_webknossos


FIXTURES = Path(__file__).parent / "fixtures"
SEED_URL = "https://webknossos.org/datasets/b2275d664e4c2a96/HuaLab-CBA_Ca-mouse-unexposed-M2"
DISAMBIGUATE_URL = "https://webknossos.org/api/datasets/disambiguate/b2275d664e4c2a96/HuaLab-CBA_Ca-mouse-unexposed-M2/toId"
DATASET_API_URL = "https://webknossos.org/api/datasets/652d563301000068049c066e"
SHARE_URL = "https://webknossos.tnw.tudelft.nl/links/r0uP-eaG4YNbBnSN"
SHARE_DATASET_API_URL = "https://webknossos.tnw.tudelft.nl/api/datasets/6672e292010000bc00a1be4e"


class WebknossosScannerTests(unittest.TestCase):
    def test_seed_url_metadata_to_candidate(self):
        fixture = _load_fixture("webknossos_hua_dataset.json")
        result = scan_webknossos(
            seed_urls=[SEED_URL],
            fetch_json=_fake_fetch(
                {
                    DISAMBIGUATE_URL: {"id": "652d563301000068049c066e"},
                    DATASET_API_URL: fixture,
                }
            ),
        )

        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.source_name, "webknossos")
        self.assertEqual(candidate.source_record_id, "b2275d664e4c2a96/HuaLab-CBA_Ca-mouse-unexposed-M2")
        self.assertEqual(candidate.title, "HuaLab-CBA_Ca-mouse-unexposed-M2")
        self.assertEqual(candidate.landing_url, SEED_URL)
        self.assertEqual(candidate.modality, "SBEM")
        self.assertEqual(candidate.organism, "Mouse")
        self.assertEqual(candidate.tissue_or_sample, "cochlea")
        self.assertEqual(candidate.file_formats, ["zarr3"])
        self.assertIn("10329x25183x2558", candidate.dimensions_or_image_count or "")
        self.assertIn("acquisition=SBEM", candidate.evidence_text or "")
        self.assertEqual(candidate.raw_metadata["id"], "652d563301000068049c066e")

    def test_dataset_api_url_metadata_to_candidate(self):
        fixture = _load_fixture("webknossos_hua_dataset.json")
        result = scan_webknossos(
            seed_urls=[DATASET_API_URL],
            fetch_json=_fake_fetch({DATASET_API_URL: fixture}),
        )

        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].source_record_id, "b2275d664e4c2a96/HuaLab-CBA_Ca-mouse-unexposed-M2")

    def test_share_link_resolves_dataset_id_from_html(self):
        fixture = _load_fixture("webknossos_hua_dataset.json")
        fixture["id"] = "6672e292010000bc00a1be4e"
        fixture["name"] = "20231212_MCF7_NdAc"
        fixture["owningOrganization"] = "hoogenboom-group"

        result = scan_webknossos(
            seed_urls=[SHARE_URL],
            fetch_text=_fake_fetch_text(
                {
                    SHARE_URL: (
                        '<meta property="og:title" content="20231212_MCF7_NdAc | WEBKNOSSOS" />'
                        '<meta property="og:image" content="https://webknossos.tnw.tudelft.nl/api/datasets/'
                        '6672e292010000bc00a1be4e/layers/EM/thumbnail?w=1000&amp;h=300" />'
                    )
                }
            ),
            fetch_json=_fake_fetch({SHARE_DATASET_API_URL: fixture}),
        )

        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].source_record_id, "hoogenboom-group/20231212_MCF7_NdAc")
        self.assertEqual(
            result.candidates[0].landing_url,
            "https://webknossos.tnw.tudelft.nl/datasets/hoogenboom-group/20231212_MCF7_NdAc",
        )

    def test_malformed_metadata_shape_emits_scanner_error(self):
        fixture = _load_fixture("webknossos_malformed_dataset.json")
        result = scan_webknossos(
            seed_urls=[DATASET_API_URL],
            fetch_json=_fake_fetch({DATASET_API_URL: fixture}),
        )

        self.assertEqual(result.candidates, [])
        self.assertEqual(len(result.errors), 1)
        error = result.errors[0]
        self.assertEqual(error.source_name, "webknossos")
        self.assertEqual(error.adapter, "webknossos_public_dataset_api")
        self.assertIn("dataSource", error.message)
        self.assertEqual(error.reproduction_command, "python scripts/scan_sources.py --source webknossos")


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _fake_fetch(mapping: dict[str, Any]):
    def fetch(url: str) -> Any:
        if url not in mapping:
            raise AssertionError(f"unexpected URL: {url}")
        return mapping[url]

    return fetch


def _fake_fetch_text(mapping: dict[str, str]):
    def fetch(url: str) -> str:
        if url not in mapping:
            raise AssertionError(f"unexpected URL: {url}")
        return mapping[url]

    return fetch


if __name__ == "__main__":
    unittest.main()
