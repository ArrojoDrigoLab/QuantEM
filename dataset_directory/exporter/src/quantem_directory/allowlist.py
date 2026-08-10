"""The privacy gate: what may be published, expressed as an allow-list.

The corpus database tags assets with a free-text ``group`` field — there is no
enum constraining it, so new groups can appear at any time. A deny-list would
therefore fail open. This module is an allow-list: a tag group that is not named
here is never published, and the transform raises if it encounters one it does
not recognise.

Only a small number of groups are eligible for publication. Everything else is
withheld, and the published files are re-scanned byte by byte afterwards so a
regression fails the build rather than the review.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

#: Tag groups whose values become public facets or public display strings.
#: Anything absent from this set is dropped at transform time.
PUBLISHED_TAG_GROUPS = frozenset(
    {
        "kingdom",
        "species",
        "organ",
        "Tissue Region",
        "modality",
    }
)

#: Facet vocabularies the export computes itself rather than reading from a tag
#: group. They share a name with a withheld tag group but not its contents —
#: ``repository`` is derived from each dataset's public link rather than from
#: the corpus's 43-value free-text ``repository`` tag, ``resolution`` is banded
#: from parsed nm/px rather than the free-text ``resolution`` tag, and
#: ``dimensionality`` comes from the reconciled 2D/3D rule rather than the tag
#: of that name, which does not reproduce the published split.
DERIVED_DICTIONARIES = frozenset({"repository", "resolution", "dimensionality"})

#: Directly or indirectly identifying. Never published under any circumstance.
_IDENTIFYING = {
    "lab",
    "author",
    "institution",
    "service request",
    "patient id",
    "mouse id",
    "animal",
    "case",
    "sample",
    "sample id",
    "sample_id",
    "sample name",
    "sample_site",
    "specimen_group",
    "original_filename",
    "volume_name",
    "experiment date",
    "acquisition_date",
    "acquisition_month",
    "source_file_date",
    "source_file_time",
}

#: Experimental conditions. Individually innocuous, but a corpus-wide facet over
#: study variables would assemble a study design the site has no reason to
#: publish. Groups of this kind are also matched by shape — see
#: :data:`_WITHHELD_PATTERNS` — so a corpus that grows a new one stays covered
#: without this list having to name it.
_EXPERIMENTAL = {
    "age",
    "age_group",
    "developmental_age",
    "developmental_stage",
    "sex",
    "gender",
    "genotype",
    "strain",
    "organism_strain",
    "gene_perturbation",
    "genetic_perturbation",
    "differentiation_state",
    "treatment",
    "condition",
    "sample_condition",
    "sample_state",
    "exposure_condition",
    "disease_context",
    "infection",
    "pathogen",
    "pathogen_species",
    "virus",
    "symbiont",
    "host_species",
    "immunolabel",
    "fasting",
    "dpi",
    "sample prep",
    "sample_preparation",
}

#: Shapes of tag group that are withheld whatever they are called. Naming a
#: group family by pattern rather than by instance keeps the classification
#: complete as the corpus grows, and keeps this file free of the study
#: variables the groups happen to record.
#: Group names are written in several styles — ``some dosage``, ``mouse_id``,
#: ``PooledMouseCount`` — so the boundary has to be "not another letter or
#: digit" rather than ``\b``, which treats an underscore as part of the word.
_EDGE = r"(?<![a-z0-9])(?:{})(?![a-z0-9])"
_WITHHELD_PATTERNS = (
    re.compile(r"(?i)" + _EDGE.format("dose|dosage|concentration|dilution")),
    re.compile(r"(?i)^(?:diet|infusion|osmotic|perfusion|injection)(?![a-z0-9])"),
    re.compile(r"(?i)" + _EDGE.format("mouse|mice|animal|patient|donor|specimen|subject|cohort")),
    re.compile(r"(?i)" + _EDGE.format("labelling|labeling|label|tracer|isotope|pulse|chase")),
    re.compile(r"(?i)" + _EDGE.format("timepoint|duration|elapsed")
               + r"|\((?:seconds|minutes|hours|days|weeks|months)\)$"),
)

#: Finer-grained anatomy and cell identity. Withheld to keep one vocabulary per
#: rank: the site facets on organ and tissue context, and adding four partially
#: overlapping alternatives would make the counts unexplainable.
_REDUNDANT_VOCABULARY = {
    "tissue",
    "organism",
    "taxon",
    "cell",
    "cell type",
    "cell_type",
    "cell line",
    "cell_line",
    "Cell Line",
    "brain_region",
    "cortical_layer",
    "liver region",
    "sample_region",
    "volume_layer",
    "cellular_component",
    "target_organelle",
    "area",
    "context",
    "sample_type",
    "correlative_modality",
    "dataset",
    "experiment",
    "repository",
    "repository_collection",
    "source",
    "dimensionality",
    "resolution",
    "OutreachProject",
    "published_data",
}

#: Acquisition parameters. Not withheld for privacy — simply not useful as
#: facets over a corpus this heterogeneous, and mostly unpopulated.
_ACQUISITION = {
    "Voltage",
    "camera",
    "magnification",
    "model",
    "microscope",
    "imaging_platform",
    "pixel_calibration",
    "scale_bar",
    "section_orientation",
    "voxel_type",
    "format",
    "file_format",
}

#: Bibliographic pointers. The dataset's own link is a first-class field; these
#: per-asset duplicates are inconsistent and would compete with it.
_BIBLIOGRAPHIC = {
    "doi",
    "dataset_doi",
    "preprint_doi",
    "publication",
    "publication_doi",
    "publication_journal",
    "publication_pmid",
    "publication_title",
    "publication_url",
    "publication_year",
    "related_data_doi",
    "source_record_doi",
    "source_record_id",
    "version_of_record_doi",
    "accession",
}

#: Corpus-assembly machinery with no meaning outside the pipeline that made it.
_INTERNAL = {
    "",
    "asset_role",
    "dataset_split",
    "image_inventory",
    "materialization",
    "materialization_allowed",
    "processing_state",
    "triage_decision",
}

#: Licence terms are set by each depositor and are deliberately not restated
#: here — a reader should get them from the source repository, which is
#: authoritative and current. See ``DATA_LICENSE.md``.
_LICENCE = {"license"}

#: Groups known to exist and deliberately withheld. This is not what enforces
#: the gate — :data:`PUBLISHED_TAG_GROUPS` is — but classifying every group
#: makes the decision auditable, and :func:`check_tag_groups` fails on any group
#: that appears in the corpus without a ruling here.
WITHHELD_TAG_GROUPS = frozenset(
    _IDENTIFYING
    | _EXPERIMENTAL
    | _REDUNDANT_VOCABULARY
    | _ACQUISITION
    | _BIBLIOGRAPHIC
    | _INTERNAL
    | _LICENCE
)

#: Asset/dataset columns from the source extract that must never be published,
#: whether because they are internal machinery or because they identify people.
WITHHELD_FIELDS = frozenset(
    {
        "notes",
        "raw_metadata",
        "normalized_metadata",
        "original_filename",
        "legacy_key",
        "file_path",
        "stored_path",
        "storage_root",
        "provenance",
        "lifecycle_status",
        "is_eval_set",
        "cells_count",
        "refined_cells_count",
        "canonical_manifest_ref",
        "tiles_summary",
        "DatasetGrouping",
        "new_subtype",
        "new_or_public",
        "n_outreach_assets",
    }
)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_ORCID = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")
# Contributor records can reach a text field as a serialised dict; catch the
# shape rather than any single instance.
_DICT_LITERAL = re.compile(r"\{'(?:name|affiliation|orcid)'")
_SPECIMEN_CODE = re.compile(r"(?i)\b(?:patient|donor|subject)\s+[A-Z]?\d")

#: Never acceptable anywhere in the published output.
ALWAYS_FORBIDDEN = (
    ("email address", _EMAIL),
    ("ORCID", _ORCID),
    ("contributor record", _DICT_LITERAL),
)

#: Forbidden in vocabularies we curate — facet values are our editorial choice,
#: and a specimen code is never a valid tissue or organ. Deliberately *not*
#: applied to dataset and asset names: those are the depositor's own published
#: titles, and rewriting them would break correspondence with the source
#: repository, which is the whole point of listing them.
VOCABULARY_FORBIDDEN = (("specimen identifier", _SPECIMEN_CODE),)


class DisallowedTagGroup(Exception):
    """Raised when the transform meets a tag group it has no ruling for."""


def is_withheld_tag_group(group: str) -> bool:
    """True if the group is withheld by name or by shape."""
    return group in WITHHELD_TAG_GROUPS or any(p.search(group) for p in _WITHHELD_PATTERNS)


def is_publishable_vocabulary_value(value: str) -> bool:
    """False for a value unfit to be a public facet label, whatever its group.

    Applied at transform time so such a value is dropped rather than merely
    reported, and so no individual value has to be listed anywhere to suppress
    it. Deliberately *not* applied to dataset and asset names: those are the
    depositor's own published titles, and rewriting them would break
    correspondence with the source repository.
    """
    return not any(pattern.search(value) for _, pattern in VOCABULARY_FORBIDDEN)


def check_tag_groups(groups: Iterable[str]) -> None:
    """Raise unless every group is explicitly published or explicitly withheld.

    A group that is neither is a *new* group the corpus has grown since this
    module was written. Failing here forces a human to classify it rather than
    letting it default into the published output.
    """
    unknown = sorted(
        g for g in set(groups) if g not in PUBLISHED_TAG_GROUPS and not is_withheld_tag_group(g)
    )
    if unknown:
        raise DisallowedTagGroup(
            "tag group(s) with no publish/withhold ruling: "
            + ", ".join(repr(g) for g in unknown)
            + " — add each to PUBLISHED_TAG_GROUPS or WITHHELD_TAG_GROUPS in allowlist.py"
        )


def scan_bytes(text: str, *, source: str) -> list[str]:
    """Flag anything that must never appear in any published file."""
    findings = []
    for label, pattern in ALWAYS_FORBIDDEN:
        for match in sorted({m.group(0) for m in pattern.finditer(text)}):
            findings.append(f"{source}: {label} — {match!r}")
    return findings


def scan_vocabulary(values: Iterable[str], *, source: str) -> list[str]:
    """Flag values unfit to be a public facet label."""
    findings = []
    for value in values:
        for label, pattern in VOCABULARY_FORBIDDEN:
            if pattern.search(value):
                findings.append(f"{source}: {label} used as a facet value — {value!r}")
    return findings
