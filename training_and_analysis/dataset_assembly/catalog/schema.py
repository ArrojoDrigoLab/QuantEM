"""Validation helpers for classifier output records."""

from __future__ import annotations

import math
from typing import Any

from .models import ELIGIBILITY_STATUSES


CLASSIFIER_BASE_REQUIRED_KEYS = [
    "candidate_id",
    "eligibility_status",
    "confidence",
    "qualifying_modality",
    "intracellular_ultrastructure_evidence",
    "exclusion_reason",
    "dedupe_keys",
    "recommended_catalog_fields",
]
CLASSIFIER_REVIEW_KEYS = ["needs_codex_review", "needs_manual_review"]
CLASSIFIER_ALLOWED_KEYS = [*CLASSIFIER_BASE_REQUIRED_KEYS, *CLASSIFIER_REVIEW_KEYS]


def validate_classification(row: dict[str, Any], *, context: str = "classification") -> None:
    """Validate one classifier result against the loader-facing contract."""
    if not isinstance(row, dict):
        raise ValueError(f"{context}: expected a JSON object")

    missing = [key for key in CLASSIFIER_BASE_REQUIRED_KEYS if key not in row]
    if missing:
        raise ValueError(f"{context}: classification missing required keys: {missing}")
    _normalize_review_flags(row, context=context)

    unexpected = sorted(set(row) - set(CLASSIFIER_ALLOWED_KEYS))
    if unexpected:
        raise ValueError(f"{context}: classification has unexpected keys: {unexpected}")

    candidate_id = row["candidate_id"]
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError(f"{context}: candidate_id must be a non-empty string")

    status = row["eligibility_status"]
    if status not in ELIGIBILITY_STATUSES:
        raise ValueError(f"{context}: bad eligibility_status: {status}")

    confidence = row["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
        raise ValueError(f"{context}: confidence must be a finite number")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(f"{context}: confidence must be between 0.0 and 1.0")

    for key in ("qualifying_modality", "intracellular_ultrastructure_evidence", "exclusion_reason"):
        value = row[key]
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{context}: {key} must be a string or null")

    dedupe_keys = row["dedupe_keys"]
    if not isinstance(dedupe_keys, list) or any(not isinstance(value, str) for value in dedupe_keys):
        raise ValueError(f"{context}: dedupe_keys must be a list of strings")

    fields = row["recommended_catalog_fields"]
    if not isinstance(fields, dict):
        raise ValueError(f"{context}: recommended_catalog_fields must be an object")
    _validate_recommended_catalog_fields(fields, context=context)


def _normalize_review_flags(row: dict[str, Any], *, context: str) -> None:
    has_codex = "needs_codex_review" in row
    has_manual = "needs_manual_review" in row
    if not has_codex and not has_manual:
        raise ValueError(
            f"{context}: classification missing required review flag: needs_codex_review "
            "(the alias needs_manual_review is also accepted)"
        )

    if has_codex and not isinstance(row["needs_codex_review"], bool):
        raise ValueError(f"{context}: needs_codex_review must be a boolean")
    if has_manual and not isinstance(row["needs_manual_review"], bool):
        raise ValueError(f"{context}: needs_manual_review must be a boolean")

    review_value = row["needs_codex_review"] if has_codex else row["needs_manual_review"]
    if has_manual and row["needs_manual_review"] != review_value:
        raise ValueError(f"{context}: needs_codex_review and needs_manual_review disagree")

    row.setdefault("needs_codex_review", review_value)
    row.setdefault("needs_manual_review", review_value)


def _validate_recommended_catalog_fields(fields: dict[str, Any], *, context: str) -> None:
    nullable_string_fields = [
        "title",
        "landing_url",
        "publication_doi",
        "dataset_doi",
        "modality",
        "organism",
        "tissue_or_sample",
        "dimensions_or_image_count",
        "license",
        "source_name",
        "source_record_id",
        "duplicate_of_source_name",
        "duplicate_of_source_record_id",
        "duplicate_scope",
        "duplicate_reason",
        "evidence_text",
        "discovered_at",
    ]
    for key in nullable_string_fields:
        if key in fields and fields[key] is not None and not isinstance(fields[key], str):
            raise ValueError(f"{context}: recommended_catalog_fields.{key} must be a string or null")

    for key in ("download_or_manifest_urls", "file_formats"):
        if key in fields and (
            not isinstance(fields[key], list) or any(not isinstance(value, str) for value in fields[key])
        ):
            raise ValueError(f"{context}: recommended_catalog_fields.{key} must be a list of strings")
    if "publication_dois" in fields and (
        not isinstance(fields["publication_dois"], list) or any(not isinstance(value, str) for value in fields["publication_dois"])
    ):
        raise ValueError(f"{context}: recommended_catalog_fields.publication_dois must be a list of strings")

    if "raw_metadata" in fields and not isinstance(fields["raw_metadata"], dict):
        raise ValueError(f"{context}: recommended_catalog_fields.raw_metadata must be an object")


# The strict JSON Schema `catalog.classify` hands to the model as --output-schema when it reviews the
# candidates that deterministic triage left undecided. The prompt of record,
# ../prompts/01_dataset_classification.txt, ends by embedding this schema through its
# {output_schema_json} placeholder.
REVIEW_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "candidate_id",
        "eligibility_status",
        "confidence",
        "qualifying_modality",
        "intracellular_ultrastructure_evidence",
        "exclusion_reason",
        "needs_codex_review",
        "needs_manual_review",
        "dedupe_keys",
        "recommended_catalog_fields",
    ],
    "properties": {
        "candidate_id": {"type": "string"},
        "eligibility_status": {"type": "string", "enum": ["eligible", "ineligible", "uncertain"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "qualifying_modality": {"type": ["string", "null"]},
        "intracellular_ultrastructure_evidence": {"type": ["string", "null"]},
        "exclusion_reason": {"type": ["string", "null"]},
        "needs_codex_review": {"type": "boolean"},
        "needs_manual_review": {"type": "boolean"},
        "dedupe_keys": {"type": "array", "items": {"type": "string"}},
        "recommended_catalog_fields": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "title",
                "landing_url",
                "download_or_manifest_urls",
                "publication_doi",
                "publication_dois",
                "dataset_doi",
                "modality",
                "organism",
                "tissue_or_sample",
                "dimensions_or_image_count",
                "file_formats",
                "license",
                "source_name",
                "source_record_id",
                "duplicate_of_source_name",
                "duplicate_of_source_record_id",
                "duplicate_scope",
                "duplicate_reason",
                "evidence_text",
                "raw_metadata",
                "discovered_at",
            ],
            "properties": {
                "title": {"type": ["string", "null"]},
                "landing_url": {"type": ["string", "null"]},
                "download_or_manifest_urls": {"type": "array", "items": {"type": "string"}},
                "publication_doi": {"type": ["string", "null"]},
                "publication_dois": {"type": "array", "items": {"type": "string"}},
                "dataset_doi": {"type": ["string", "null"]},
                "modality": {"type": ["string", "null"]},
                "organism": {"type": ["string", "null"]},
                "tissue_or_sample": {"type": ["string", "null"]},
                "dimensions_or_image_count": {"type": ["string", "null"]},
                "file_formats": {"type": "array", "items": {"type": "string"}},
                "license": {"type": ["string", "null"]},
                "source_name": {"type": ["string", "null"]},
                "source_record_id": {"type": ["string", "null"]},
                "duplicate_of_source_name": {"type": ["string", "null"]},
                "duplicate_of_source_record_id": {"type": ["string", "null"]},
                "duplicate_scope": {"type": ["string", "null"]},
                "duplicate_reason": {"type": ["string", "null"]},
                "evidence_text": {"type": ["string", "null"]},
                "raw_metadata": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                    "required": [],
                },
                "discovered_at": {"type": ["string", "null"]},
            },
        },
    },
}
