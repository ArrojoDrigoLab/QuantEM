"""Zenodo metadata-dump scanner.

The full Zenodo search API is too broad and rate-limited for an all-history
pass. This scanner uses Zenodo's metadata dump as the complete record list,
then fetches individual record JSON only for dump records with EM-like metadata
so downstream deterministic rules still have file-level metadata.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import tarfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..http import USER_AGENT, get_json
from ..jsonl import read_jsonl
from ..models import Candidate
from .base import ScannerResult, safe_collect


EXPORTER_URL = "https://zenodo.org/api/exporter"
DUMP_KEY = "records-xml.tar.gz"
ZENODO_RECORD_API_URL = "https://zenodo.org/api/records/{record_id}"
DEFAULT_DUMP_DIR_NAME = "zenodo_dump"
DEFAULT_CACHE_FILENAME = "candidates.jsonl"
MAX_FILE_METADATA = 25
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
PROGRESS_INTERVAL_RECORDS = 100_000
CANDIDATE_CACHE_FILTER_VERSION = 2

DATACITE_NS = {"d": "http://datacite.org/schema/kernel-4"}
OAI_DATACITE_NS = {"o": "http://schema.datacite.org/oai/oai-1.1/", **DATACITE_NS}

ALLOWED_RESOURCE_TYPES = {"dataset", "image", "other"}
EXCLUDED_RESOURCE_TYPES = {"text", "software", "poster", "presentation", "video", "lesson"}

STRONG_EM_PATTERNS = [
    r"\belectron microscopy\b",
    r"\belectron micrographs?\b",
    r"\belectron microscope\b",
    r"\btransmission electron\b",
    r"\bscanning electron\b",
    r"\bvolume electron microscopy\b",
    r"\bvolume em\b",
    r"\bserial block[- ]face\b",
    r"\bserial section(?:ing)?\b",
    r"\bfib[- ]?sem\b",
    r"\bsbf[- ]?sem\b",
    r"\bsbem\b",
    r"\barray tomography\b",
    r"\bultrastructure\b",
]
TEM_CONTEXT_TERMS = [
    "ultrastructure",
    "mitochond",
    "organelle",
    "intracellular",
    "cellular",
    "cell",
    "tissue",
    "membrane",
    "neuron",
    "axon",
    "synapse",
]
SEM_CONTEXT_TERMS = [
    "fib",
    "sbf",
    "sbem",
    "serial block",
    "volume",
    "ultrastructure",
    "intracellular",
    "cellular",
    "tissue",
]
@dataclass(frozen=True)
class DumpInfo:
    version_id: str
    url: str
    size: int | None = None
    checksum: str | None = None
    created: str | None = None


@dataclass(frozen=True)
class CacheInfo:
    path: Path
    meta_path: Path
    candidate_count: int
    dump_version_id: str


@dataclass(frozen=True)
class DumpRecord:
    record_id: str
    title: str | None
    description: str | None
    subjects: list[str]
    resource_type_general: str | None
    resource_type: str | None
    landing_url: str | None
    dataset_doi: str | None
    rights: list[dict[str, str | None]]
    related_identifiers: list[dict[str, str | None]]
    creators: list[dict[str, str | None]]
    publication_year: str | None
    dates: list[dict[str, str | None]]
    member_name: str

    def evidence_text(self) -> str:
        return _evidence_text(
            [
                self.title,
                self.description,
                " ".join(self.subjects),
                self.resource_type_general,
                self.resource_type,
                " ".join(str(item.get("identifier") or "") for item in self.related_identifiers),
            ]
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "title": self.title,
            "description": self.description,
            "subjects": self.subjects,
            "resource_type_general": self.resource_type_general,
            "resource_type": self.resource_type,
            "landing_url": self.landing_url,
            "dataset_doi": self.dataset_doi,
            "rights": self.rights,
            "related_identifiers": self.related_identifiers,
            "creators": self.creators,
            "publication_year": self.publication_year,
            "dates": self.dates,
            "member_name": self.member_name,
        }


def scan_zenodo_dump(
    *,
    root: str | Path,
    limit: int = 1500,
    cursor: dict[str, Any] | None = None,
    dump_dir: str | Path | None = None,
    dump_path: str | Path | None = None,
    candidate_cache_path: str | Path | None = None,
    force_rebuild_cache: bool = False,
    **_: Any,
) -> ScannerResult:
    return safe_collect(
        "zenodo",
        "zenodo_dump_adapter",
        _scan_zenodo_dump,
        root=root,
        limit=limit,
        cursor=cursor,
        dump_dir=dump_dir,
        dump_path=dump_path,
        candidate_cache_path=candidate_cache_path,
        force_rebuild_cache=force_rebuild_cache,
    )


def _scan_zenodo_dump(
    *,
    root: str | Path,
    limit: int,
    cursor: dict[str, Any] | None,
    dump_dir: str | Path | None,
    dump_path: str | Path | None,
    candidate_cache_path: str | Path | None,
    force_rebuild_cache: bool,
) -> ScannerResult:
    if cursor and cursor.get("complete"):
        return ScannerResult(cursor=cursor, cursor_complete=True)
    if limit < 1:
        raise ValueError("limit must be at least 1")

    root_path = Path(root)
    resolved_dump_dir = Path(dump_dir) if dump_dir else root_path / "runs" / DEFAULT_DUMP_DIR_NAME
    resolved_cache_path = (
        Path(candidate_cache_path) if candidate_cache_path else resolved_dump_dir / DEFAULT_CACHE_FILENAME
    )
    cache_info = _ensure_candidate_cache(
        dump_dir=resolved_dump_dir,
        dump_path=Path(dump_path) if dump_path else None,
        candidate_cache_path=resolved_cache_path,
        force_rebuild_cache=force_rebuild_cache,
    )
    offset = _cursor_int(cursor, "candidate_offset", 0)
    if offset >= cache_info.candidate_count:
        return ScannerResult(
            cursor=_complete_cursor(cache_info),
            cursor_complete=True,
        )

    candidates: list[Candidate] = []
    for index, row in enumerate(read_jsonl(cache_info.path)):
        if index < offset:
            continue
        candidates.append(Candidate.from_dict(row))
        if len(candidates) >= limit:
            break

    next_offset = offset + len(candidates)
    complete = next_offset >= cache_info.candidate_count
    cursor_payload = (
        _complete_cursor(cache_info)
        if complete
        else {
            "adapter": "zenodo_dump",
            "candidate_offset": next_offset,
            "candidate_count": cache_info.candidate_count,
            "cache_path": str(cache_info.path),
            "dump_version_id": cache_info.dump_version_id,
            "complete": False,
        }
    )
    return ScannerResult(candidates=candidates, cursor=cursor_payload, cursor_complete=complete)


def _ensure_candidate_cache(
    *,
    dump_dir: Path,
    dump_path: Path | None,
    candidate_cache_path: Path,
    force_rebuild_cache: bool,
) -> CacheInfo:
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_info = _local_dump_info(dump_path) if dump_path else _latest_dump_info()
    resolved_dump_path = dump_path or _download_dump(dump_info, dump_dir)
    meta_path = _cache_meta_path(candidate_cache_path)
    if not force_rebuild_cache:
        cache = _read_cache_meta(candidate_cache_path, meta_path)
        if cache and cache.dump_version_id == dump_info.version_id:
            return cache

    return _build_candidate_cache(
        dump_path=resolved_dump_path,
        dump_info=dump_info,
        candidate_cache_path=candidate_cache_path,
        meta_path=meta_path,
    )


def _latest_dump_info() -> DumpInfo:
    payload = get_json(EXPORTER_URL)
    entries = payload.get(DUMP_KEY) if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Zenodo exporter response does not list {DUMP_KEY}")
    entry = next((item for item in entries if isinstance(item, dict) and item.get("is_head")), entries[0])
    if not isinstance(entry, dict):
        raise ValueError(f"Zenodo exporter {DUMP_KEY} entry is not an object")
    links = entry.get("links") if isinstance(entry.get("links"), dict) else {}
    url = links.get("self_head") or links.get("self")
    version_id = entry.get("version_id") or "head"
    if not isinstance(url, str) or not url:
        raise ValueError(f"Zenodo exporter {DUMP_KEY} entry has no download URL")
    return DumpInfo(
        version_id=str(version_id),
        url=url,
        size=_int_or_none(entry.get("size")),
        checksum=str(entry["checksum"]) if entry.get("checksum") else None,
        created=str(entry["created"]) if entry.get("created") else None,
    )


def _local_dump_info(path: Path | None) -> DumpInfo:
    if path is None:
        raise ValueError("dump path is required")
    if not path.exists():
        raise FileNotFoundError(f"Zenodo dump file not found: {path}")
    return DumpInfo(version_id=f"local:{path.resolve()}:{path.stat().st_size}", url=str(path), size=path.stat().st_size)


def _download_dump(info: DumpInfo, dump_dir: Path) -> Path:
    safe_version = re.sub(r"[^A-Za-z0-9_.-]+", "_", info.version_id or "head")
    target = dump_dir / f"records-xml-{safe_version}.tar.gz"
    if _download_complete(target, info):
        return target

    partial = target.with_suffix(target.suffix + ".part")
    _download_binary(info.url, target, partial=partial, expected_size=info.size)
    if info.checksum:
        _verify_checksum(target, info.checksum)
    return target


def _download_complete(path: Path, info: DumpInfo) -> bool:
    if not path.exists():
        return False
    if info.size is not None and path.stat().st_size != info.size:
        return False
    return True


def _download_binary(url: str, target: Path, *, partial: Path, expected_size: int | None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    resume_at = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if resume_at:
        headers["Range"] = f"bytes={resume_at}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - Zenodo metadata dump URL.
            append = resume_at > 0 and getattr(response, "status", None) == 206
            mode = "ab" if append else "wb"
            downloaded = resume_at if append else 0
            next_report = downloaded + 512 * 1024 * 1024
            with partial.open(mode) as handle:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        _print_download_progress(downloaded, expected_size)
                        next_report = downloaded + 512 * 1024 * 1024
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to download Zenodo metadata dump: {exc}") from exc
    partial.replace(target)


def _print_download_progress(downloaded: int, expected_size: int | None) -> None:
    if expected_size:
        percent = downloaded / expected_size * 100
        print(f"zenodo dump download: {downloaded}/{expected_size} bytes ({percent:.1f}%)")
    else:
        print(f"zenodo dump download: {downloaded} bytes")


def _verify_checksum(path: Path, checksum: str) -> None:
    if not checksum.startswith("md5:"):
        return
    expected = checksum.split(":", 1)[1].strip().lower()
    digest = hashlib.md5()  # noqa: S324 - verifying an upstream md5 checksum, not security use.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    actual = digest.hexdigest().lower()
    if actual != expected:
        raise ValueError(f"Zenodo dump checksum mismatch for {path}: expected {expected}, got {actual}")


def _build_candidate_cache(
    *,
    dump_path: Path,
    dump_info: DumpInfo,
    candidate_cache_path: Path,
    meta_path: Path,
) -> CacheInfo:
    candidate_cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = candidate_cache_path.with_suffix(candidate_cache_path.suffix + ".tmp")
    tmp_meta_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    if tmp_meta_path.exists():
        tmp_meta_path.unlink()

    stats = {
        "records_scanned": 0,
        "metadata_prefilter_matches": 0,
        "record_details_fetched": 0,
        "candidates_written": 0,
        "parse_errors": 0,
    }
    seen: set[str] = set()
    with tmp_path.open("w", encoding="utf-8") as handle, tarfile.open(dump_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".xml"):
                continue
            stats["records_scanned"] += 1
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            raw_xml = extracted.read()
            try:
                dump_record = _record_from_datacite_xml(raw_xml, member.name)
            except (ET.ParseError, ValueError):
                stats["parse_errors"] += 1
                continue
            if dump_record.record_id in seen:
                continue
            seen.add(dump_record.record_id)
            if not _passes_dump_metadata_prefilter(dump_record):
                _maybe_print_scan_progress(stats)
                continue
            stats["metadata_prefilter_matches"] += 1
            detail = get_json(ZENODO_RECORD_API_URL.format(record_id=dump_record.record_id))
            stats["record_details_fetched"] += 1
            candidate = _candidate_from_zenodo_detail(detail, dump_record)
            handle.write(json.dumps(candidate.to_dict(), ensure_ascii=True, sort_keys=True) + "\n")
            stats["candidates_written"] += 1
            _maybe_print_scan_progress(stats)

    meta = {
        "complete": True,
        "candidate_cache_filter_version": CANDIDATE_CACHE_FILTER_VERSION,
        "dump_key": DUMP_KEY,
        "dump_version_id": dump_info.version_id,
        "dump_created": dump_info.created,
        "dump_size": dump_info.size,
        "dump_checksum": dump_info.checksum,
        "candidate_count": stats["candidates_written"],
        "stats": stats,
    }
    tmp_meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(candidate_cache_path)
    tmp_meta_path.replace(meta_path)
    return CacheInfo(
        path=candidate_cache_path,
        meta_path=meta_path,
        candidate_count=stats["candidates_written"],
        dump_version_id=dump_info.version_id,
    )


def _maybe_print_scan_progress(stats: dict[str, int]) -> None:
    scanned = stats["records_scanned"]
    if scanned and scanned % PROGRESS_INTERVAL_RECORDS == 0:
        print(
            "zenodo dump scan: "
            f"scanned={stats['records_scanned']} "
            f"metadata_matches={stats['metadata_prefilter_matches']} "
            f"details={stats['record_details_fetched']} "
            f"candidates={stats['candidates_written']}"
        )


def _read_cache_meta(candidate_cache_path: Path, meta_path: Path) -> CacheInfo | None:
    if not candidate_cache_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict) or not meta.get("complete"):
        return None
    if meta.get("candidate_cache_filter_version") != CANDIDATE_CACHE_FILTER_VERSION:
        return None
    candidate_count = _int_or_none(meta.get("candidate_count"))
    dump_version_id = meta.get("dump_version_id")
    if candidate_count is None or not isinstance(dump_version_id, str):
        return None
    return CacheInfo(
        path=candidate_cache_path,
        meta_path=meta_path,
        candidate_count=candidate_count,
        dump_version_id=dump_version_id,
    )


def _cache_meta_path(candidate_cache_path: Path) -> Path:
    return candidate_cache_path.with_name(candidate_cache_path.stem + ".meta.json")


def _record_from_datacite_xml(raw_xml: bytes, member_name: str) -> DumpRecord:
    root = ET.fromstring(raw_xml)
    resource = root.find(".//d:resource", OAI_DATACITE_NS)
    if resource is None:
        raise ValueError("DataCite XML missing resource payload")

    title = _text(resource.find("d:titles/d:title", DATACITE_NS))
    descriptions = [_text(node) for node in resource.findall("d:descriptions/d:description", DATACITE_NS)]
    subjects = [_text(node) for node in resource.findall("d:subjects/d:subject", DATACITE_NS)]
    resource_type_node = resource.find("d:resourceType", DATACITE_NS)
    resource_type_general = resource_type_node.get("resourceTypeGeneral") if resource_type_node is not None else None
    resource_type = _text(resource_type_node)
    dataset_doi = _text(resource.find("d:identifier[@identifierType='DOI']", DATACITE_NS))
    alternate_identifiers = [
        {
            "type": node.get("alternateIdentifierType"),
            "identifier": _text(node),
        }
        for node in resource.findall("d:alternateIdentifiers/d:alternateIdentifier", DATACITE_NS)
    ]
    landing_url = _landing_url_from_alternates(alternate_identifiers)
    record_id = _record_id_from_metadata(member_name, landing_url, alternate_identifiers)
    return DumpRecord(
        record_id=record_id,
        title=title,
        description=_evidence_text(descriptions),
        subjects=[item for item in subjects if item],
        resource_type_general=resource_type_general,
        resource_type=resource_type,
        landing_url=landing_url,
        dataset_doi=dataset_doi,
        rights=_rights(resource),
        related_identifiers=_related_identifiers(resource),
        creators=_creators(resource),
        publication_year=_text(resource.find("d:publicationYear", DATACITE_NS)),
        dates=_dates(resource),
        member_name=member_name,
    )


def _landing_url_from_alternates(alternate_identifiers: list[dict[str, str | None]]) -> str | None:
    for item in alternate_identifiers:
        value = item.get("identifier")
        if value and "zenodo.org/records/" in value:
            return value
    for item in alternate_identifiers:
        value = item.get("identifier")
        if value and value.startswith("http"):
            return value
    return None


def _record_id_from_metadata(
    member_name: str,
    landing_url: str | None,
    alternate_identifiers: list[dict[str, str | None]],
) -> str:
    for value in [landing_url, *(item.get("identifier") for item in alternate_identifiers)]:
        if not value:
            continue
        match = re.search(r"(?:zenodo\.org/records/|oai:zenodo\.org:)(\d+)", value)
        if match:
            return match.group(1)
    stem = Path(member_name).stem
    if re.fullmatch(r"\d+", stem):
        return stem
    raise ValueError(f"Could not infer Zenodo record id from dump member {member_name}")


def _rights(resource: ET.Element) -> list[dict[str, str | None]]:
    rows: list[dict[str, str | None]] = []
    for node in resource.findall("d:rightsList/d:rights", DATACITE_NS):
        rows.append(
            {
                "rights": _text(node),
                "rights_uri": node.get("rightsURI"),
                "rights_identifier": node.get("rightsIdentifier"),
            }
        )
    return rows


def _related_identifiers(resource: ET.Element) -> list[dict[str, str | None]]:
    rows: list[dict[str, str | None]] = []
    for node in resource.findall("d:relatedIdentifiers/d:relatedIdentifier", DATACITE_NS):
        rows.append(
            {
                "identifier": _text(node),
                "identifier_type": node.get("relatedIdentifierType"),
                "relation_type": node.get("relationType"),
            }
        )
    return rows


def _creators(resource: ET.Element) -> list[dict[str, str | None]]:
    rows: list[dict[str, str | None]] = []
    for node in resource.findall("d:creators/d:creator", DATACITE_NS)[:20]:
        rows.append(
            {
                "name": _text(node.find("d:creatorName", DATACITE_NS)),
                "affiliation": _text(node.find("d:affiliation", DATACITE_NS)),
            }
        )
    return rows


def _dates(resource: ET.Element) -> list[dict[str, str | None]]:
    rows: list[dict[str, str | None]] = []
    for node in resource.findall("d:dates/d:date", DATACITE_NS):
        rows.append({"date": _text(node), "date_type": node.get("dateType")})
    return rows


def _passes_dump_metadata_prefilter(record: DumpRecord) -> bool:
    resource_type = (record.resource_type_general or record.resource_type or "").strip().lower()
    if resource_type in EXCLUDED_RESOURCE_TYPES:
        return False
    if resource_type and resource_type not in ALLOWED_RESOURCE_TYPES:
        return False
    if not _is_open_access(record):
        return False
    text = _normalize_text(record.evidence_text())
    if not text:
        return False
    if any(re.search(pattern, text) for pattern in STRONG_EM_PATTERNS):
        return True
    if re.search(r"\btem\b", text) and any(term in text for term in TEM_CONTEXT_TERMS):
        return True
    if re.search(r"\bsem\b", text) and any(term in text for term in SEM_CONTEXT_TERMS):
        return True
    if re.search(r"\bvem\b", text) and any(term in text for term in TEM_CONTEXT_TERMS + SEM_CONTEXT_TERMS):
        return True
    return False


def _is_open_access(record: DumpRecord) -> bool:
    if not record.rights:
        return True
    for item in record.rights:
        text = _normalize_text(item.get("rights"), item.get("rights_uri"), item.get("rights_identifier"))
        if "openaccess" in text or "open access" in text or "creativecommons" in text or "cc-by" in text:
            return True
    return False


def _candidate_from_zenodo_detail(detail: Any, dump_record: DumpRecord) -> Candidate:
    if not isinstance(detail, dict):
        raise ValueError(f"Zenodo record {dump_record.record_id} JSON detail is not an object")
    files = detail.get("files") if isinstance(detail.get("files"), list) else []
    metadata = detail.get("metadata") if isinstance(detail.get("metadata"), dict) else {}
    links = detail.get("links") if isinstance(detail.get("links"), dict) else {}
    title = metadata.get("title") or detail.get("title") or dump_record.title or dump_record.record_id
    landing_url = links.get("html") or dump_record.landing_url or f"https://zenodo.org/records/{dump_record.record_id}"
    api_url = links.get("self") or ZENODO_RECORD_API_URL.format(record_id=dump_record.record_id)
    dataset_doi = detail.get("doi") or metadata.get("doi") or dump_record.dataset_doi
    evidence = _evidence_text(
        [
            title,
            metadata.get("description"),
            dump_record.description,
            " ".join(str(item) for item in metadata.get("keywords") or []),
            " ".join(str(item) for item in metadata.get("subjects") or []),
            " ".join(_file_name(file_info) for file_info in files[:MAX_FILE_METADATA]),
            _resource_type_label(metadata.get("resource_type")),
        ]
    )
    raw_metadata = dict(detail)
    raw_metadata["dump_record"] = dump_record.to_metadata()
    raw_metadata["candidate_source"] = "zenodo_metadata_dump"
    return Candidate(
        source_name="zenodo",
        source_record_id=str(detail.get("id") or detail.get("recid") or dump_record.record_id),
        title=str(title),
        landing_url=str(landing_url) if landing_url else None,
        download_or_manifest_urls=[str(api_url)] if api_url else [],
        publication_doi=_publication_doi(metadata, dump_record, dataset_doi),
        dataset_doi=str(dataset_doi) if dataset_doi else None,
        dimensions_or_image_count=_file_count_summary(files, detail.get("size")),
        file_formats=_file_formats_from_files(files),
        license=_license_text(metadata, dump_record),
        raw_metadata=raw_metadata,
        evidence_text=evidence,
        discovered_at=metadata.get("publication_date") or detail.get("created") or detail.get("updated"),
    )


def _file_name(file_info: Any) -> str:
    if not isinstance(file_info, dict):
        return ""
    return str(file_info.get("key") or file_info.get("filename") or file_info.get("name") or "")


def _file_formats_from_files(files: list[Any]) -> list[str]:
    formats: list[str] = []
    for file_info in files:
        if not isinstance(file_info, dict):
            continue
        for key in ("key", "filename", "name"):
            name = file_info.get(key)
            if name:
                suffix = _suffix(str(name))
                if suffix:
                    formats.append(suffix.upper().lstrip("."))
                break
        for key in ("type", "mimetype"):
            value = file_info.get(key)
            if value:
                formats.append(str(value))
    return sorted(set(formats))


def _suffix(name: str) -> str:
    lower = name.lower()
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".zip"):
        if lower.endswith(suffix):
            return suffix
    return Path(lower).suffix


def _file_count_summary(files: list[Any], total_size: Any = None) -> str | None:
    count = len(files)
    if count and total_size:
        return f"{count} files; {total_size} bytes total"
    if count:
        return f"{count} files"
    if total_size:
        return f"{total_size} bytes total"
    return None


def _publication_doi(metadata: dict[str, Any], dump_record: DumpRecord, dataset_doi: Any) -> str | None:
    dataset_doi_text = str(dataset_doi or "").lower()
    related = metadata.get("related_identifiers")
    if isinstance(related, list):
        for item in related:
            if not isinstance(item, dict):
                continue
            doi = item.get("identifier")
            scheme = str(item.get("scheme") or item.get("identifier_type") or "").lower()
            relation = str(item.get("relation") or item.get("relation_type") or "").lower()
            if not doi or scheme != "doi":
                continue
            doi_text = str(doi).strip()
            if doi_text.lower() == dataset_doi_text or "zenodo" in doi_text.lower():
                continue
            if relation in {"isversionof", "hasversion", "ispartof", "haspart"}:
                continue
            return doi_text
    for item in dump_record.related_identifiers:
        doi = item.get("identifier")
        scheme = str(item.get("identifier_type") or "").lower()
        relation = str(item.get("relation_type") or "").lower()
        if doi and scheme == "doi" and str(doi).lower() != dataset_doi_text and "zenodo" not in str(doi).lower():
            if relation not in {"isversionof", "hasversion", "ispartof", "haspart"}:
                return str(doi)
    return None


def _license_text(metadata: dict[str, Any], dump_record: DumpRecord) -> str | None:
    license_info = metadata.get("license")
    if isinstance(license_info, dict):
        title = license_info.get("title") or license_info.get("id")
        url = license_info.get("url")
        if title and url:
            return f"{title} ({url})"
        if title or url:
            return str(title or url)
    for item in dump_record.rights:
        label = item.get("rights") or item.get("rights_identifier")
        uri = item.get("rights_uri")
        if label and uri:
            return f"{label} ({uri})"
        if label:
            return str(label)
    access_right = metadata.get("access_right")
    return str(access_right) if access_right else None


def _resource_type_label(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(str(part) for part in [value.get("type"), value.get("subtype"), value.get("title")] if part)
    return str(value) if value else ""


def _complete_cursor(cache_info: CacheInfo) -> dict[str, Any]:
    return {
        "adapter": "zenodo_dump",
        "candidate_offset": cache_info.candidate_count,
        "candidate_count": cache_info.candidate_count,
        "cache_path": str(cache_info.path),
        "dump_version_id": cache_info.dump_version_id,
        "complete": True,
    }


def _cursor_int(cursor: dict[str, Any] | None, key: str, default: int) -> int:
    if not isinstance(cursor, dict):
        return default
    try:
        return max(0, int(cursor.get(key, default)))
    except (TypeError, ValueError):
        return default


def _text(node: ET.Element | None) -> str | None:
    if node is None:
        return None
    text = " ".join(part.strip() for part in node.itertext() if part and part.strip())
    if not text:
        return None
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def _evidence_text(parts: Iterable[Any]) -> str:
    text = " ".join(str(part) for part in parts if part)
    text = re.sub(r"<[^>]+>", " ", html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


def _normalize_text(*parts: Any) -> str:
    return re.sub(r"\s+", " ", " ".join(str(part) for part in parts if part)).strip().lower()


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
