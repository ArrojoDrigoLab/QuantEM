from __future__ import annotations

import unittest

from catalog.eligibility import classify_candidate
from catalog.models import Candidate


class EligibilityTests(unittest.TestCase):
    def classify(self, **kwargs):
        base = {"source_name": "test", "source_record_id": "row1", "title": "x"}
        base.update(kwargs)
        return classify_candidate(Candidate(**base))

    def test_sbf_sem_cellular_is_eligible(self):
        result = self.classify(modality="SBF-SEM", evidence_text="HeLa cellular ultrastructure with mitochondria")
        self.assertEqual(result["eligibility_status"], "eligible")
        self.assertFalse(result["needs_codex_review"])
        self.assertFalse(result["needs_manual_review"])

    def test_sbem_cellular_is_eligible(self):
        result = self.classify(modality="SBEM", evidence_text="mouse cochlea tissue with mitochondria segmentation")
        self.assertEqual(result["eligibility_status"], "eligible")
        self.assertEqual(result["qualifying_modality"], "SBEM")
        self.assertFalse(result["needs_codex_review"])

    def test_vem_cellular_is_eligible(self):
        result = self.classify(modality="vEM", evidence_text="cellular volume with nucleus and organelles")
        self.assertEqual(result["eligibility_status"], "eligible")
        self.assertEqual(result["qualifying_modality"], "vEM")

    def test_tem_organelle_is_eligible(self):
        result = self.classify(modality="TEM", evidence_text="intracellular organelle ultrastructure")
        self.assertEqual(result["eligibility_status"], "eligible")

    def test_immunoelectron_tomography_nuclear_pore_is_eligible(self):
        result = self.classify(
            modality="large-scale EM ; immunoelectron tomography",
            evidence_text="cultured human cells with nuclear envelope herniations and nuclear pore complexes",
        )
        self.assertEqual(result["eligibility_status"], "eligible")

    def test_cryo_et_is_excluded(self):
        result = self.classify(modality="cryo-ET", evidence_text="cellular cryo electron tomography")
        self.assertEqual(result["eligibility_status"], "ineligible")
        self.assertIn("cryo-ET", result["exclusion_reason"])
        self.assertFalse(result["needs_codex_review"])

    def test_hyphenated_cryo_electron_tomography_is_excluded(self):
        result = self.classify(
            title="Cryo-electron tomography of NIH 3T3 fibroblasts",
            modality="electron tomography",
            evidence_text="Reconstructed cryo-electron tomograms acquired on cryo-FIB lamellae.",
        )
        self.assertEqual(result["eligibility_status"], "ineligible")
        self.assertIn("cryo-ET", result["exclusion_reason"])

    def test_cryo_et_with_volume_alias_is_still_excluded(self):
        result = self.classify(modality="vEM cryo-ET", evidence_text="cellular organelles")
        self.assertEqual(result["eligibility_status"], "ineligible")
        self.assertIn("cryo-ET", result["exclusion_reason"])

    def test_surface_sem_is_excluded(self):
        result = self.classify(modality="SEM", evidence_text="surface SEM of epithelial topography")
        self.assertEqual(result["eligibility_status"], "ineligible")

    def test_sem_edx_is_excluded(self):
        result = self.classify(modality="SEM-EDX", evidence_text="elemental particle analysis")
        self.assertEqual(result["eligibility_status"], "ineligible")

    def test_materials_are_excluded(self):
        result = self.classify(modality="TEM", title="nanoparticle scaffold TEM characterization")
        self.assertEqual(result["eligibility_status"], "ineligible")

    def test_openorganelle_polymerize_sample_prep_is_not_materials_exclusion(self):
        result = self.classify(
            source_name="openorganelle",
            modality="FIB-SEM",
            evidence_text="Mouse kidney tissue with FIB-SEM data and intracellular organelles.",
            raw_metadata={"sample": {"protocol": "Durcupan resin; polymerize the sample in a 60 C oven."}},
        )
        self.assertEqual(result["eligibility_status"], "eligible")
        self.assertIsNone(result["exclusion_reason"])

    def test_iodp_sem_dataset_is_excluded(self):
        result = self.classify(
            source_name="zenodo",
            source_record_id="10668453",
            title="IODP Expedition 390 Scanning electron microscope images",
            evidence_text=(
                "Microscopic images of discrete samples were acquired using a scanning electron microscope "
                "and captured as image files. International Ocean Discovery Program IODP JOIDES Resolution "
                "Expedition 390 South Atlantic Transect Mid-Atlantic Ridge."
            ),
            raw_metadata={"files": [{"key": "U1559C.zip"}, {"key": "SEM-README.txt"}]},
        )
        self.assertEqual(result["eligibility_status"], "ineligible")
        self.assertIn("ocean-drilling", result["exclusion_reason"])
        self.assertFalse(result["needs_codex_review"])

    def test_patch_dataset_is_eligible(self):
        result = self.classify(title="CEM1.5M cellular EM patches", evidence_text="cellular EM organelle mitochondria")
        self.assertEqual(result["eligibility_status"], "eligible")

    def test_literature_leads_are_routed_to_codex_review(self):
        result = self.classify(
            source_name="literature_europepmc",
            modality="volume electron microscopy",
            evidence_text="Article abstract mentions cellular organelles and source data.",
        )
        self.assertEqual(result["eligibility_status"], "uncertain")
        self.assertTrue(result["needs_codex_review"])

    def test_incomplete_metadata_is_uncertain(self):
        result = self.classify(title="Electron microscopy dataset")
        self.assertEqual(result["eligibility_status"], "uncertain")
        self.assertTrue(result["needs_codex_review"])
        self.assertTrue(result["needs_manual_review"])

    def test_zenodo_csv_without_em_reference_is_ineligible(self):
        result = self.classify(
            source_name="zenodo",
            source_record_id="17979678",
            title="Air Quality Turano-Gerace (ATMOTUBE)",
            evidence_text="Air Quality Turano-Gerace (ATMOTUBE)",
            raw_metadata={
                "files": [
                    {
                        "key": "results-01a1ad07-c4a1-442d-b0eb-a9b46a29dc99.csv",
                        "size": 300230457,
                    }
                ]
            },
        )
        self.assertEqual(result["eligibility_status"], "ineligible")
        self.assertIn("no electron microscopy reference", result["exclusion_reason"])
        self.assertFalse(result["needs_codex_review"])

    def test_zenodo_zip_without_em_reference_is_ineligible(self):
        result = self.classify(
            source_name="zenodo",
            title="Air quality sensor data archive",
            evidence_text="Citizen science environmental monitoring data.",
            raw_metadata={"files": [{"key": "sensor-results.zip"}]},
        )
        self.assertEqual(result["eligibility_status"], "ineligible")
        self.assertIn("no electron microscopy reference", result["exclusion_reason"])
        self.assertFalse(result["needs_codex_review"])

    def test_zenodo_tiff_tem_organelle_is_eligible(self):
        result = self.classify(
            source_name="zenodo",
            title="TEM source images",
            modality="TEM",
            evidence_text="intracellular organelle ultrastructure",
            raw_metadata={"files": [{"key": "cell_001.tif"}]},
        )
        self.assertEqual(result["eligibility_status"], "eligible")
        self.assertFalse(result["needs_codex_review"])

    def test_zenodo_jpeg_em_images_can_be_eligible(self):
        result = self.classify(
            source_name="zenodo",
            title="Thin section electron microscopy lung image dataset",
            evidence_text="Transmission electron microscopy images show alveolar epithelium cells and lung tissue ultrastructure.",
            raw_metadata={"files": [{"key": "C08_A_montage.jpg"}]},
            file_formats=["JPG"],
        )
        self.assertEqual(result["eligibility_status"], "eligible")
        self.assertFalse(result["needs_codex_review"])

    def test_zenodo_5682693_is_eligible_from_metadata(self):
        result = self.classify(
            source_name="zenodo",
            source_record_id="5682693",
            title="A detailed ultrastructural examination of lung cryobiopsy samples from a COVID-19 patient case series - Data set 08",
            evidence_text=(
                "Thin section electron microscopy. Stitched image montages of thin sections through the lung "
                "acquired by scanning electron microscopy or transmission electron microscopy. Images show "
                "pathological changes of the alveolar epithelium including type-1-cells and type-2-cell hyperplasia."
            ),
            raw_metadata={
                "files": [{"key": "Data set 08 Alveolar epithelium.zip"}],
                "metadata": {
                    "keywords": [
                        "COVID-19",
                        "electron microscopy",
                        "SARS-CoV-2",
                        "lung",
                        "patient",
                        "cryobiopsy",
                        "thin section",
                        "alveolar damage",
                    ]
                },
            },
        )
        self.assertEqual(result["eligibility_status"], "eligible")
        self.assertFalse(result["needs_codex_review"])


if __name__ == "__main__":
    unittest.main()
