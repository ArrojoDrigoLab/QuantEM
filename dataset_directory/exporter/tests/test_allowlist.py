"""The privacy gate must fail closed, including on vocabulary it has never seen."""
from __future__ import annotations

import pytest

from quantem_directory import allowlist


class TestTagGroupGate:
    def test_known_groups_pass(self):
        allowlist.check_tag_groups(["kingdom", "species", "lab", "patient id"])

    def test_an_unclassified_group_is_a_hard_error(self):
        # Tag.group is free text in the source database, so a new group can
        # appear at any time. Defaulting it into the output would be a leak;
        # this forces a human to classify it. The example has to be a name no
        # withheld pattern already covers, or it would be classified and pass.
        with pytest.raises(allowlist.DisallowedTagGroup) as raised:
            allowlist.check_tag_groups(["kingdom", "collection_site"])
        assert "collection_site" in str(raised.value)

    def test_the_error_names_every_unclassified_group(self):
        with pytest.raises(allowlist.DisallowedTagGroup) as raised:
            allowlist.check_tag_groups(["alpha", "beta"])
        message = str(raised.value)
        assert "alpha" in message and "beta" in message

    def test_published_and_withheld_never_overlap(self):
        assert not (allowlist.PUBLISHED_TAG_GROUPS & allowlist.WITHHELD_TAG_GROUPS)

    def test_the_published_set_is_the_five_facet_groups(self):
        # Widening this set is a privacy decision, not a refactor. If this test
        # is failing, that decision is what needs review.
        assert allowlist.PUBLISHED_TAG_GROUPS == {
            "kingdom",
            "species",
            "organ",
            "Tissue Region",
            "modality",
        }

    def test_licence_is_withheld_deliberately(self):
        # Reuse terms come from the depositor, not from this directory.
        assert "license" in allowlist.WITHHELD_TAG_GROUPS


class TestByteScan:
    def test_catches_an_email_address(self):
        assert allowlist.scan_bytes("contact a.person@example.ac.uk", source="x.json")

    def test_catches_an_orcid(self):
        assert allowlist.scan_bytes("0000-0002-1825-0097", source="x.json")

    def test_catches_a_leaked_contributor_record(self):
        assert allowlist.scan_bytes("{'name': 'A Person', 'orcid':", source="x.json")

    def test_reports_each_distinct_hit_once(self):
        findings = allowlist.scan_bytes("a@b.com a@b.com a@b.com", source="x.json")
        assert len(findings) == 1

    def test_ordinary_dataset_prose_is_clean(self):
        assert not allowlist.scan_bytes(
            "Mouse Liver FIB-SEM Volume, 4 nm/px, deposited at EMPIAR-12585", source="x.json"
        )


class TestVocabularyScan:
    def test_a_specimen_code_is_not_a_valid_facet_value(self):
        assert allowlist.scan_vocabulary(["Patient A01"], source="facets.json/Tissue Region")
        assert allowlist.scan_vocabulary(["Donor 7"], source="facets.json/Tissue Region")

    def test_real_tissue_contexts_are_clean(self):
        assert not allowlist.scan_vocabulary(
            ["Lung cryobiopsy", "Virus-particle regions", "Pancreatic islet"],
            source="facets.json/Tissue Region",
        )

    def test_depositor_titles_are_not_held_to_the_vocabulary_bar(self):
        # Dataset names are the depositors' own published titles, reproduced
        # verbatim so they match the source repository. They are checked by
        # scan_bytes, never by scan_vocabulary.
        title = "Example Tissue Series — Data Set 18: patient A01"
        assert not allowlist.scan_bytes(title, source="datasets.json")
        assert allowlist.scan_vocabulary([title], source="facets.json/organ")


class TestVocabularyDrop:
    def test_a_specimen_code_is_dropped_rather_than_listed(self):
        # The transform drops these by shape, so no individual value has to be
        # written down anywhere in order to suppress it.
        assert not allowlist.is_publishable_vocabulary_value("Patient A01")
        assert allowlist.is_publishable_vocabulary_value("Lung cryobiopsy")

    def test_a_group_is_withheld_by_shape_as_well_as_by_name(self):
        # A corpus that grows a new study-variable group stays covered without
        # this repository having to name the variable.
        assert allowlist.is_withheld_tag_group("some_compound dosage")
        assert allowlist.is_withheld_tag_group("infusion time (minutes)")
        assert allowlist.is_withheld_tag_group("pooled_mouse_count")
        assert not allowlist.is_withheld_tag_group("kingdom")
