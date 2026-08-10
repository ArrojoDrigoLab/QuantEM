"""Shared record helpers for scanner and loader code."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


ELIGIBILITY_STATUSES = {"eligible", "ineligible", "uncertain"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [text]


@dataclass
class Candidate:
    source_name: str
    source_record_id: str
    title: str | None = None
    landing_url: str | None = None
    download_or_manifest_urls: list[str] = field(default_factory=list)
    publication_doi: str | None = None
    publication_dois: list[str] = field(default_factory=list)
    dataset_doi: str | None = None
    modality: str | None = None
    organism: str | None = None
    tissue_or_sample: str | None = None
    dimensions_or_image_count: str | None = None
    file_formats: list[str] = field(default_factory=list)
    license: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    evidence_text: str | None = None
    discovered_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_record_id": self.source_record_id,
            "title": self.title,
            "landing_url": self.landing_url,
            "download_or_manifest_urls": self.download_or_manifest_urls,
            "publication_doi": self.publication_doi,
            "publication_dois": self.publication_dois,
            "dataset_doi": self.dataset_doi,
            "modality": self.modality,
            "organism": self.organism,
            "tissue_or_sample": self.tissue_or_sample,
            "dimensions_or_image_count": self.dimensions_or_image_count,
            "file_formats": self.file_formats,
            "license": self.license,
            "raw_metadata": self.raw_metadata,
            "evidence_text": self.evidence_text,
            "discovered_at": self.discovered_at,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Candidate":
        return cls(
            source_name=str(row["source_name"]),
            source_record_id=str(row["source_record_id"]),
            title=clean_text(row.get("title")),
            landing_url=clean_text(row.get("landing_url")),
            download_or_manifest_urls=as_list(row.get("download_or_manifest_urls")),
            publication_doi=clean_text(row.get("publication_doi")),
            publication_dois=as_list(row.get("publication_dois")),
            dataset_doi=clean_text(row.get("dataset_doi")),
            modality=clean_text(row.get("modality")),
            organism=clean_text(row.get("organism")),
            tissue_or_sample=clean_text(row.get("tissue_or_sample")),
            dimensions_or_image_count=clean_text(row.get("dimensions_or_image_count")),
            file_formats=as_list(row.get("file_formats")),
            license=clean_text(row.get("license")),
            raw_metadata=dict(row.get("raw_metadata") or {}),
            evidence_text=clean_text(row.get("evidence_text")),
            discovered_at=clean_text(row.get("discovered_at")) or utc_now_iso(),
        )


@dataclass
class ScannerError:
    source_name: str
    adapter: str
    url: str | None
    error_type: str
    message: str
    response_status: int | None = None
    response_excerpt: str | None = None
    raw_payload_path: str | None = None
    reproduction_command: str | None = None
    stack_trace: str | None = None
    adapter_version: str = "v1"
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "adapter": self.adapter,
            "url": self.url,
            "response_status": self.response_status,
            "response_excerpt": self.response_excerpt,
            "raw_payload_path": self.raw_payload_path,
            "reproduction_command": self.reproduction_command,
            "error_type": self.error_type,
            "message": self.message,
            "stack_trace": self.stack_trace,
            "adapter_version": self.adapter_version,
            "created_at": self.created_at,
        }
