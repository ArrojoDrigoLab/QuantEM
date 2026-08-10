"""Rule-based eligibility classification for public intracellular EM datasets."""

from __future__ import annotations

import re
from typing import Any

from .models import Candidate


QUALIFYING_MODALITY_TERMS = [
    "sbf-sem",
    "sbf sem",
    "sbfsem",
    "sbem",
    "serial block face",
    "serial block-face",
    "serial block face sem",
    "serial block-face sem",
    "fib-sem",
    "fib sem",
    "fibsem",
    "array tomography sem",
    "atum-sem",
    "atum sem",
    "atum",
    "at-sem",
    "at sem",
    "multibeam sem",
    "multi-beam sem",
    "msem",
    "sstem",
    "ss-tem",
    "serial section tem",
    "serial-section tem",
    "tem",
    "transmission electron microscopy",
    "electron tomography",
    "immunoelectron tomography",
    "resin-section et",
    "volume em",
    "volume electron microscopy",
    "vem",
    "cellular em",
    "cem1.5m",
    "cem500k",
]

DISPLAY_MODALITY_ALIASES = {
    "array tomography sem": "array tomography SEM",
    "atum": "ATUM",
    "atum sem": "ATUM-SEM",
    "atum-sem": "ATUM-SEM",
    "multi-beam sem": "multibeam SEM",
    "multibeam sem": "multibeam SEM",
    "sbem": "SBEM",
    "sbf sem": "SBF-SEM",
    "sbf-sem": "SBF-SEM",
    "sbfsem": "SBF-SEM",
    "serial block face": "serial block-face SEM",
    "serial block-face": "serial block-face SEM",
    "serial block face sem": "serial block-face SEM",
    "serial block-face sem": "serial block-face SEM",
    "vem": "vEM",
    "volume electron microscopy": "volume electron microscopy",
}

INTRACELLULAR_TERMS = [
    "intracellular",
    "ultrastructure",
    "ultrastructural",
    "organelle",
    "organelles",
    "mitochond",
    "nucleus",
    "nuclei",
    "nuclear envelope",
    "nuclear pore",
    "endoplasmic reticulum",
    " er ",
    "golgi",
    "lysosome",
    "vesicle",
    "annulate lamellae",
    "cell membrane",
    "plasma membrane",
    "cytoplasm",
    "whole cell",
    "epithelium",
    "epithelial",
    "cellular",
    "tissue",
    "pancreas",
    "pancreatic",
    "heLa".lower(),
    "mcf7",
    "mcf-7",
    "label map",
    "segmentation",
]

EXCLUSION_PATTERNS = [
    (r"\bcryo[- ]?et\b", "cryo-ET is excluded by project scope"),
    (r"\bcryo[- ]?electron tomo(?:graphy|grams?|graphic)\b", "cryo-ET is excluded by project scope"),
    (r"\bcryo[- ]?em\b", "structural cryo-EM is excluded"),
    (r"single particle", "single-particle structural EM is excluded"),
    (r"protein structure", "structural EM is excluded"),
    (r"\bsem[- /]?(edx|eds)\b", "SEM-EDX/EDS is excluded"),
    (r"\b(edx|eds)\b", "elemental/particle analysis is excluded"),
    (r"surface[- ]only sem", "surface-only SEM is excluded"),
    (r"surface sem", "surface SEM is excluded without internal ultrastructure evidence"),
    (r"topographic sem", "topographic SEM is excluded"),
    (
        r"\b(nanoparticle|nanoparticles|scaffold|scaffolds|hydrogel|hydrogels|implant|implants|"
        r"alloy|alloys|polymer|polymers|catalyst|catalysts)\b",
        "materials/device EM is excluded",
    ),
    (
        r"\b(iodp expedition|international ocean discovery program|integrated ocean drilling program|joides resolution)\b",
        "geological/ocean-drilling EM is excluded",
    ),
]

def normalize_text(*parts: Any) -> str:
    joined = " ".join(str(p) for p in parts if p is not None)
    return re.sub(r"\s+", " ", joined).strip().lower()


def classify_candidate(candidate: Candidate) -> dict[str, Any]:
    """Classify one candidate using conservative deterministic rules."""
    text = normalize_text(
        candidate.title,
        candidate.modality,
        candidate.tissue_or_sample,
        candidate.evidence_text,
        " ".join(candidate.file_formats),
        candidate.raw_metadata,
    )
    modality_text = normalize_text(candidate.modality, candidate.evidence_text, candidate.title)

    for pattern, reason in EXCLUSION_PATTERNS:
        if re.search(pattern, text):
            return _result(candidate, "ineligible", 0.95, None, None, reason, False)

    if candidate.source_name.startswith("literature_"):
        return _result(
            candidate,
            "uncertain",
            0.5,
            None,
            None,
            "Literature lead requires Codex extraction of public dataset links and eligibility evidence.",
            True,
        )

    modality_hits = _modality_hits(modality_text)
    intracellular_hits = [term.strip() for term in INTRACELLULAR_TERMS if term in text]

    if candidate.source_name == "openorganelle" and modality_hits:
        modality = _display_modality(candidate.modality, modality_hits[0])
        evidence = (
            f"OpenOrganelle record includes a qualifying public EM layer ({modality_hits[0]}) "
            "from a cellular/organelle volume EM source."
        )
        return _result(candidate, "eligible", 0.95, modality, evidence, None, False)

    if candidate.source_name == "webknossos" and _webknossos_has_em_layer(candidate.raw_metadata):
        if intracellular_hits:
            evidence = (
                "webKnossos record exposes public EM image layers and has cellular/tissue evidence "
                f"({intracellular_hits[0]})."
            )
            return _result(candidate, "eligible", 0.9, "volume EM", evidence, None, False)
        return _result(
            candidate,
            "uncertain",
            0.65,
            "volume EM",
            "webKnossos record exposes public EM image layers, but intracellular/cellular context is incomplete.",
            None,
            True,
        )

    if candidate.source_name == "zenodo" and not _zenodo_has_em_reference(text, modality_hits):
        return _result(
            candidate,
            "ineligible",
            0.9,
            None,
            None,
            "Zenodo record has no electron microscopy reference in title, evidence, or metadata.",
            False,
        )

    if modality_hits and intracellular_hits:
        modality = _display_modality(candidate.modality, modality_hits[0])
        evidence = f"Found qualifying modality ({modality_hits[0]}) and intracellular/cellular evidence ({intracellular_hits[0]})."
        return _result(candidate, "eligible", 0.9, modality, evidence, None, False)

    if modality_hits:
        modality = _display_modality(candidate.modality, modality_hits[0])
        evidence = f"Found qualifying modality ({modality_hits[0]}), but intracellular evidence is incomplete."
        return _result(candidate, "uncertain", 0.62, modality, evidence, None, True)

    if "sem" in text and not any(
        term in text for term in ["fib", "sbf", "sbem", "serial block", "atum", "volume", "multibeam"]
    ):
        return _result(
            candidate,
            "uncertain",
            0.75,
            None,
            None,
            "SEM mention lacks volume/cross-sectional intracellular evidence; route to Codex review.",
            True,
        )

    return _result(
        candidate,
        "uncertain",
        0.35,
        None,
        None,
        "No qualifying intracellular EM modality could be confirmed from supplied metadata; route to Codex review.",
        True,
    )


def _zenodo_has_em_reference(text: str, modality_hits: list[str]) -> bool:
    if modality_hits:
        return True
    terms = [
        "electron microscopy",
        "electron micrograph",
        "electron microscope",
        "em",
        "tem",
        "sem",
        "fibsem",
        "fib-sem",
        "sbf-sem",
        "sbfsem",
        "sbem",
        "serial block face",
        "serial section",
        "ultrastructure",
    ]
    variants = {text, re.sub(r"[-_/]+", " ", text)}
    return any(_contains_term(variant, term) for variant in variants for term in terms)


def _modality_hits(text: str) -> list[str]:
    variants = {text, re.sub(r"[-_/]+", " ", text)}
    hits: list[str] = []
    for term in QUALIFYING_MODALITY_TERMS:
        if any(_contains_term(variant, term) for variant in variants):
            hits.append(term)
    return hits


def _contains_term(text: str, term: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _webknossos_has_em_layer(raw_metadata: Any) -> bool:
    if not isinstance(raw_metadata, dict):
        return False
    data_source = raw_metadata.get("dataSource")
    if not isinstance(data_source, dict):
        return False
    layers = data_source.get("dataLayers")
    if not isinstance(layers, list):
        return False
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        name = str(layer.get("name") or "").lower()
        category = str(layer.get("category") or "").lower()
        if category == "color" and re.search(r"(?<![a-z0-9])em(?![a-z0-9])|electron", name):
            return True
    return False


def _display_modality(modality: str | None, hit: str) -> str:
    if modality:
        return modality
    if hit in DISPLAY_MODALITY_ALIASES:
        return DISPLAY_MODALITY_ALIASES[hit]
    if hit in {"cem1.5m", "cem500k", "cellular em"}:
        return "cellular EM patch dataset"
    return hit.upper() if "tem" in hit or "sem" in hit else hit


def _dedupe_keys(candidate: Candidate) -> list[str]:
    keys: list[str] = []
    for prefix, value in [
        ("dataset_doi", candidate.dataset_doi),
        ("landing_url", candidate.landing_url),
        ("source_record", f"{candidate.source_name}:{candidate.source_record_id}"),
    ]:
        if value:
            keys.append(f"{prefix}:{str(value).strip().lower()}")
    return keys


def _result(
    candidate: Candidate,
    status: str,
    confidence: float,
    qualifying_modality: str | None,
    evidence: str | None,
    exclusion_reason: str | None,
    needs_codex_review: bool,
) -> dict[str, Any]:
    fields = {
        "title": candidate.title,
        "landing_url": candidate.landing_url,
        "download_or_manifest_urls": candidate.download_or_manifest_urls,
        "publication_doi": candidate.publication_doi,
        "publication_dois": candidate.publication_dois,
        "dataset_doi": candidate.dataset_doi,
        "modality": candidate.modality,
        "organism": candidate.organism,
        "tissue_or_sample": candidate.tissue_or_sample,
        "dimensions_or_image_count": candidate.dimensions_or_image_count,
        "file_formats": candidate.file_formats,
        "license": candidate.license,
        "source_name": candidate.source_name,
        "source_record_id": candidate.source_record_id,
        "evidence_text": candidate.evidence_text,
        "raw_metadata": candidate.raw_metadata,
        "discovered_at": candidate.discovered_at,
    }
    classification = {
        "candidate_id": f"{candidate.source_name}:{candidate.source_record_id}",
        "eligibility_status": status,
        "confidence": confidence,
        "qualifying_modality": qualifying_modality,
        "intracellular_ultrastructure_evidence": evidence,
        "exclusion_reason": exclusion_reason,
        "needs_codex_review": needs_codex_review,
        "needs_manual_review": needs_codex_review,
        "dedupe_keys": _dedupe_keys(candidate),
        "recommended_catalog_fields": fields,
    }
    return classification
