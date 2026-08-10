from __future__ import annotations

import json
import unittest
from pathlib import Path

from catalog.sources.empiar import _candidate_from_entry
from catalog.sources.openorganelle import _candidate_from_manifest


class EmpiarOpenOrganelleScannerTests(unittest.TestCase):
    def test_empiar_entry_to_candidate(self):
        payload = _load_empiar_fixture("empiar_entry_10094.json", "EMPIAR-10094")
        candidate = _candidate_from_entry("EMPIAR-10094", payload)
        self.assertEqual(candidate.source_name, "empiar")
        self.assertEqual(candidate.dataset_doi, "10.6019/EMPIAR-10094")
        self.assertEqual(candidate.modality, "SBF-SEM")
        self.assertEqual(candidate.organism, "Human")
        self.assertEqual(candidate.tissue_or_sample, "HeLa cell pellet")
        self.assertIn("DM4", candidate.file_formats)
        self.assertIn("scale: cell", candidate.evidence_text or "")
        self.assertIn("HeLa cell benchmark data", candidate.evidence_text or "")
        self.assertEqual(candidate.raw_metadata, payload)

    def test_empiar_human_liver_segmentations_to_candidate(self):
        payload = _load_empiar_fixture("empiar_entry_13356.json", "EMPIAR-13356")
        candidate = _candidate_from_entry("EMPIAR-13356", payload)
        self.assertEqual(candidate.organism, "Human")
        self.assertEqual(candidate.tissue_or_sample, "liver tissue sample")
        self.assertEqual(candidate.modality, "SBF-SEM")
        self.assertEqual(candidate.dimensions_or_image_count, "597 images/series")
        self.assertIn("Segmentation: ER mask", candidate.evidence_text or "")
        self.assertIn("Segmentation: Mitochondrial mask", candidate.evidence_text or "")
        self.assertIn("scale: tissue", candidate.evidence_text or "")
        self.assertEqual(candidate.raw_metadata, payload)

    def test_empiar_sparse_entry_keeps_missing_biology_null(self):
        payload = _load_empiar_fixture("empiar_entry_sparse.json", "EMPIAR-19999")
        candidate = _candidate_from_entry("EMPIAR-19999", payload)
        self.assertEqual(candidate.modality, "SBF-SEM")
        self.assertIsNone(candidate.organism)
        self.assertIsNone(candidate.tissue_or_sample)
        self.assertIn("Automated volume electron microscopy benchmark", candidate.evidence_text or "")
        self.assertIn("Aligned image stack", candidate.evidence_text or "")

    def test_openorganelle_manifest_to_candidate(self):
        fixture = Path(__file__).parent / "fixtures" / "openorganelle_jrc_hela_1_manifest.json"
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        manifest_url = "https://raw.githubusercontent.com/janelia-cellmap/fibsem-metadata/stable/api/jrc_hela-1/manifest.json"
        candidate = _candidate_from_manifest("jrc_hela-1", payload, manifest_url=manifest_url)
        self.assertEqual(candidate.source_name, "openorganelle")
        self.assertEqual(candidate.source_record_id, "jrc_hela-1")
        self.assertEqual(candidate.title, "Interphase HeLa cell")
        self.assertEqual(candidate.dataset_doi, "10.25378/janelia.13123415")
        self.assertEqual(candidate.publication_doi, "10.1038/s41586-021-03977-3")
        self.assertEqual(candidate.modality, "FIB-SEM")
        self.assertEqual(candidate.organism, "Human")
        self.assertIn("HeLa", candidate.tissue_or_sample or "")
        self.assertIn("n5", candidate.file_formats)
        self.assertIn("precomputed", candidate.file_formats)
        self.assertIn(manifest_url, candidate.download_or_manifest_urls)
        self.assertIn("s3://janelia-cosem-datasets/jrc_hela-1/jrc_hela-1.n5/labels/er_seg", candidate.download_or_manifest_urls)
        self.assertIn("FIB-SEM Data", candidate.evidence_text or "")

    def test_openorganelle_missing_fields_to_candidate(self):
        fixture = Path(__file__).parent / "fixtures" / "openorganelle_missing_fields_manifest.json"
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        manifest_url = "https://raw.githubusercontent.com/janelia-cellmap/fibsem-metadata/stable/api/jrc_missing-fields/manifest.json"
        candidate = _candidate_from_manifest("jrc_missing-fields", payload, manifest_url=manifest_url)
        self.assertEqual(candidate.source_name, "openorganelle")
        self.assertEqual(candidate.source_record_id, "jrc_missing-fields")
        self.assertEqual(candidate.title, "jrc_missing-fields")
        self.assertEqual(candidate.organism, "Mouse")
        self.assertEqual(candidate.modality, "volume EM")
        self.assertEqual(candidate.download_or_manifest_urls, [manifest_url])


def _load_empiar_fixture(name: str, accession: str) -> dict:
    fixture = Path(__file__).parent / "fixtures" / name
    return json.loads(fixture.read_text(encoding="utf-8"))[accession]


if __name__ == "__main__":
    unittest.main()
