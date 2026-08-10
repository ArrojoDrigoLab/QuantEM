"""BossDB metadata scanner."""

from __future__ import annotations

import json
import re
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..http import HttpFetchError, get_json
from ..models import Candidate, ScannerError, clean_text
from .base import ScannerResult, safe_collect, unique_candidates
from .cursors import cursor_int, cursor_result


PROJECTS_API_URL = "https://api.metadata.bossdb.org/api/latest/projects"
PROJECT_SNAPSHOT_URL = "https://bossdb-metadata-snapshot.s3.amazonaws.com/mongo-data.json"
LANDING_URL_TEMPLATE = "https://bossdb.org/project/{project_id}"

PROJECT_URL_RE = re.compile(r"https?://(?:www\.)?bossdb\.org/project/([^/?#]+)", re.IGNORECASE)
BOSSDB_URI_RE = re.compile(r"\bbossdb://[^\s<>'\"`),;]+", re.IGNORECASE)
DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s,;\"'<>]+)", re.IGNORECASE)

MODALITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ATUM-SEM", re.compile(r"\bATUM[-_ ]?SEM\b|automatic tape.*\bSEM\b", re.IGNORECASE)),
    ("FIB-SEM", re.compile(r"\bFIB[-_ ]?SEM\b|focused ion beam", re.IGNORECASE)),
    ("SBF-SEM", re.compile(r"\bSBF[-_ ]?SEM\b|\bSBEM\b|serial block[- ]face", re.IGNORECASE)),
    ("AT-SEM", re.compile(r"\bAT[-_ ]?SEM\b|array tomography.*\bSEM\b", re.IGNORECASE)),
    ("ssTEM", re.compile(r"\bss[-_ ]?TEM\b|serial section transmission electron|serial-section TEM", re.IGNORECASE)),
    ("TEM", re.compile(r"\bTEM\b|transmission electron microscopy", re.IGNORECASE)),
    ("SEM", re.compile(r"\bSEM\b|scanning electron microscopy", re.IGNORECASE)),
    ("EM", re.compile(r"\bEM\b|electron microscopy", re.IGNORECASE)),
)


def scan_bossdb(
    root: str | Path = ".",
    since: str | None = None,
    query: str | None = None,
    limit: int = 100,
    cursor: dict[str, Any] | None = None,
    **_: Any,
) -> ScannerResult:
    """Scan public BossDB project metadata without requesting image cutouts."""

    return safe_collect("bossdb", "bossdb_metadata_api", _scan_bossdb, Path(root), since, query, limit, cursor)


def _scan_bossdb(root: Path, since: str | None, query: str | None, limit: int, cursor: dict[str, Any] | None) -> ScannerResult:
    if cursor and cursor.get("complete"):
        return ScannerResult(cursor=cursor, cursor_complete=True)
    records = _fetch_project_records()
    result = _candidates_from_project_records(records, since=since, query=query, limit=0)
    return _slice_candidate_cursor(result, limit=limit, cursor=cursor)


def _fetch_project_records() -> list[dict[str, Any]]:
    try:
        data = get_json(PROJECTS_API_URL)
        records = data.get("data", []) if isinstance(data, dict) else []
        if not isinstance(records, list):
            raise ValueError("BossDB projects API returned a non-list data field")
        return records
    except HttpFetchError:
        snapshot = get_json(PROJECT_SNAPSHOT_URL)
        if not isinstance(snapshot, dict):
            raise ValueError("BossDB metadata snapshot returned a non-object payload")
        return list(snapshot.values())


def _candidates_from_project_records(
    records: Iterable[Any],
    seeds_by_project_id: dict[str, list[Candidate]] | None = None,
    since: str | None = None,
    query: str | None = None,
    limit: int = 100,
) -> ScannerResult:
    seeds_by_project_id = seeds_by_project_id or {}
    seed_ids = set(seeds_by_project_id)
    result = ScannerResult()
    for record in records:
        try:
            attrs = _project_attrs(record)
            project_id = _project_id(attrs)
            seeds = seeds_by_project_id.get(project_id, [])
            if not _is_public_project(attrs) and project_id not in seed_ids:
                continue
            if since and _is_before_since(attrs, since):
                continue
            if not (_looks_like_em_project(attrs, seeds) or project_id in seed_ids):
                continue
            if not _matches_query(query, attrs, seeds):
                continue
            result.candidates.append(_candidate_from_project_attrs(attrs, seeds))
        except Exception as exc:  # noqa: BLE001 - malformed source rows should not abort the scan.
            result.errors.append(_parse_error(record, exc))

    result.candidates = unique_candidates(_sort_seeded_first(result.candidates, seed_ids))
    if limit and limit > 0:
        result.candidates = result.candidates[:limit]
    return result


def _slice_candidate_cursor(result: ScannerResult, *, limit: int, cursor: dict[str, Any] | None) -> ScannerResult:
    max_candidates = max(int(limit), 0)
    if max_candidates == 0:
        return cursor_result([], complete=True, cursor={"complete": True})
    offset = cursor_int(cursor, "candidate_offset", 0)
    candidates = result.candidates[offset : offset + max_candidates]
    next_offset = offset + len(candidates)
    complete = next_offset >= len(result.candidates)
    return ScannerResult(
        candidates=candidates,
        errors=result.errors,
        cursor={"complete": True}
        if complete
        else {
            "adapter": "bossdb_project_list",
            "candidate_offset": next_offset,
            "candidate_count": len(result.candidates),
            "complete": False,
        },
        cursor_complete=complete,
    )


def _candidate_from_project_attrs(attrs: dict[str, Any], seeds: list[Candidate] | None = None) -> Candidate:
    seeds = seeds or []
    project_id = _project_id(attrs)
    landing_url = _landing_url(attrs, project_id)
    title = (
        clean_text(attrs.get("Title"))
        or clean_text(attrs.get("Name"))
        or clean_text(attrs.get("ShortTitle"))
        or clean_text(attrs.get("title"))
        or project_id
    )
    modality = _modality(attrs, seeds)
    bossdb_uris = _bossdb_uris(attrs)
    evidence = _evidence_text(attrs, seeds, modality, bossdb_uris)
    seed = seeds[0] if seeds else None
    raw_metadata = {
        "project": attrs,
        "bossdb_project_id": project_id,
        "bossdb_uris": bossdb_uris,
        "comparator_seeds": [candidate.to_dict() for candidate in seeds],
    }
    return Candidate(
        source_name="bossdb",
        source_record_id=project_id,
        title=title,
        landing_url=landing_url,
        download_or_manifest_urls=bossdb_uris,
        publication_doi=_publication_doi(attrs) or (seed.publication_doi if seed else None),
        dataset_doi=_dataset_doi(attrs) or (seed.dataset_doi if seed else None),
        modality=modality or (seed.modality if seed else None),
        organism=_join_values(_get_any(attrs, "Species", "species")) or (seed.organism if seed else None),
        tissue_or_sample=seed.tissue_or_sample if seed else None,
        dimensions_or_image_count=_dimensions(attrs) or (seed.dimensions_or_image_count if seed else None),
        file_formats=[],
        license=_license(attrs) or (seed.license if seed else None),
        raw_metadata=raw_metadata,
        evidence_text=evidence,
        discovered_at=clean_text(_get_any(attrs, "DateModified", "date")) or clean_text(_get_any(attrs, "DateCreated")),
    )


def _project_attrs(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("BossDB project record is not an object")
    if isinstance(record.get("attributes"), dict):
        return dict(record["attributes"])
    if "attributes" in record:
        raise ValueError("BossDB project record has a non-object attributes field")
    if "ID" in record or "id" in record:
        return dict(record)
    raise ValueError("BossDB project record is missing attributes")


def _project_id(attrs: dict[str, Any]) -> str:
    project_id = clean_text(_get_any(attrs, "ID", "id"))
    if not project_id:
        raise ValueError("BossDB project record is missing ID")
    return project_id


def _project_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    match = PROJECT_URL_RE.search(url)
    if not match:
        return None
    return match.group(1).strip()


def _landing_url(attrs: dict[str, Any], project_id: str) -> str:
    for link in _as_list(_get_any(attrs, "Links", "links")):
        if not isinstance(link, dict):
            continue
        url = clean_text(_get_any(link, "URI", "url"))
        if url and _project_id_from_url(url):
            return url
    return LANDING_URL_TEMPLATE.format(project_id=project_id)


def _is_public_project(attrs: dict[str, Any]) -> bool:
    public = _get_any(attrs, "Public", "public")
    if isinstance(public, bool) and not public:
        return False
    if isinstance(public, str) and public.strip().lower() in {"false", "no", "0"}:
        return False
    status = clean_text(_get_any(attrs, "Status", "status"))
    if status and status.lower() not in {"live", "public"}:
        return False
    return True


def _looks_like_em_project(attrs: dict[str, Any], seeds: list[Candidate]) -> bool:
    text = _metadata_text(attrs, seeds)
    return any(pattern.search(text) for _, pattern in MODALITY_PATTERNS)


def _modality(attrs: dict[str, Any], seeds: list[Candidate]) -> str | None:
    text = _metadata_text(attrs, seeds)
    modalities = [label for label, pattern in MODALITY_PATTERNS if pattern.search(text)]
    if len(modalities) > 1 and "EM" in modalities:
        modalities.remove("EM")
    return " ; ".join(dict.fromkeys(modalities)) or None


def _metadata_text(attrs: dict[str, Any], seeds: list[Candidate]) -> str:
    values: list[Any] = [
        _get_any(attrs, "Title", "title"),
        _get_any(attrs, "Name", "name"),
        _get_any(attrs, "ShortTitle"),
        _get_any(attrs, "Description", "description"),
        _get_any(attrs, "ProjectPageDescription"),
        _get_any(attrs, "Keywords", "tags"),
        _get_any(attrs, "GeneralModality"),
        _get_any(attrs, "GeneralModalityOther"),
        _get_any(attrs, "ImagingModalities"),
        _get_any(attrs, "ImagingModalitySpecific"),
        _get_any(attrs, "Methods"),
        _get_any(attrs, "TechnicalInfo"),
        _get_any(attrs, "Publications", "publications"),
    ]
    for seed in seeds:
        values.extend([seed.title, seed.modality, seed.organism, seed.tissue_or_sample, seed.evidence_text])
    return " ".join(_flatten_text(values))


def _bossdb_uris(attrs: dict[str, Any]) -> list[str]:
    uris: set[str] = set()
    for value in [_get_any(attrs, "BossDBURI"), _get_any(attrs, "Collections", "collections")]:
        for text in _flatten_text([value]):
            uris.update(_extract_bossdb_uris(text))
    for location in _as_list(_get_any(attrs, "Locations", "locations")):
        if isinstance(location, dict):
            uri = _bossdb_uri_from_path(location.get("uri"))
            if uri:
                uris.add(uri)
            uris.update(_extract_bossdb_uris(" ".join(_flatten_text([location]))))
    for segmentation in _as_list(_get_any(attrs, "Segmentation", "segmentation")):
        if isinstance(segmentation, dict):
            uri = _bossdb_uri_from_path(segmentation.get("uri"))
            if uri:
                uris.add(uri)
    return sorted(uris)


def _extract_bossdb_uris(text: str) -> set[str]:
    return {match.group(0).rstrip("./") for match in BOSSDB_URI_RE.finditer(text)}


def _bossdb_uri_from_path(value: Any) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    if text.lower().startswith("bossdb://"):
        return text
    if "://" in text:
        return None
    parts = [part for part in text.strip("/").split("/") if part]
    if not parts:
        return None
    return "bossdb://" + "/".join(parts[:3])


def _publication_doi(attrs: dict[str, Any]) -> str | None:
    for publication in _as_list(_get_any(attrs, "Publications", "publications")):
        if not isinstance(publication, dict):
            continue
        doi = _normalize_doi(_get_any(publication, "RelatedIdentifier", "DOI", "doi"))
        if doi:
            return doi
    return _normalize_doi(_get_any(attrs, "PublicationDOI"))


def _dataset_doi(attrs: dict[str, Any]) -> str | None:
    return _normalize_doi(_get_any(attrs, "DOI", "doi"))


def _normalize_doi(value: Any) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    match = DOI_RE.search(text)
    if not match:
        return None
    return match.group(1).rstrip(".")


def _license(attrs: dict[str, Any]) -> str | None:
    license_value = _get_any(attrs, "License", "license")
    if isinstance(license_value, dict):
        return clean_text(_get_any(license_value, "Rights", "rights"))
    return clean_text(license_value)


def _dimensions(attrs: dict[str, Any]) -> str | None:
    for location in _as_list(_get_any(attrs, "Locations", "locations")):
        if not isinstance(location, dict):
            continue
        ranges = []
        for axis in ("xs", "ys", "zs"):
            values = location.get(axis)
            if isinstance(values, list) and len(values) == 2:
                ranges.append(f"{axis}: {values[0]}-{values[1]}")
        if ranges:
            return "BossDB display location " + ", ".join(ranges)
    return None


def _evidence_text(
    attrs: dict[str, Any],
    seeds: list[Candidate],
    modality: str | None,
    bossdb_uris: list[str],
) -> str:
    pieces = [
        f"BossDB project metadata ID: {_project_id(attrs)}",
        f"title: {clean_text(_get_any(attrs, 'Title', 'Name', 'title', 'name'))}"
        if clean_text(_get_any(attrs, "Title", "Name", "title", "name"))
        else None,
        f"modality terms: {modality}" if modality else None,
        f"species: {_join_values(_get_any(attrs, 'Species', 'species'))}"
        if _join_values(_get_any(attrs, "Species", "species"))
        else None,
        f"keywords: {_join_values(_get_any(attrs, 'Keywords', 'tags'))}"
        if _join_values(_get_any(attrs, "Keywords", "tags"))
        else None,
        f"BossDB identifiers: {', '.join(bossdb_uris[:5])}" if bossdb_uris else None,
    ]
    pieces.extend(f"comparator seed evidence: {seed.evidence_text}" for seed in seeds if seed.evidence_text)
    return " ; ".join(piece for piece in pieces if piece)


def _matches_query(query: str | None, attrs: dict[str, Any], seeds: list[Candidate]) -> bool:
    if not query:
        return True
    text = _metadata_text(attrs, seeds)
    tokens = [token.lower() for token in re.split(r"\s+", query.strip()) if token.strip()]
    lower_text = text.lower()
    return all(token in lower_text for token in tokens)


def _is_before_since(attrs: dict[str, Any], since: str) -> bool:
    date = clean_text(
        _get_any(
            attrs,
            "DateModified",
            "dateModified",
            "Updated",
            "updated",
            "DateCreated",
            "dateCreated",
            "date",
        )
    )
    return bool(date and date[:10] < since)


def _sort_seeded_first(candidates: Iterable[Candidate], seed_ids: set[str]) -> list[Candidate]:
    return sorted(candidates, key=lambda candidate: (candidate.source_record_id not in seed_ids, candidate.source_record_id.lower()))


def _parse_error(record: Any, exc: Exception) -> ScannerError:
    return ScannerError(
        source_name="bossdb",
        adapter="bossdb_project_parser",
        url=PROJECTS_API_URL,
        error_type=type(exc).__name__,
        message=str(exc),
        response_excerpt=_json_excerpt(record),
        reproduction_command="python -m catalog.scan --source bossdb",
        stack_trace=traceback.format_exc(),
    )


def _json_excerpt(value: Any, limit: int = 1000) -> str:
    try:
        text = json.dumps(value, sort_keys=True)
    except TypeError:
        text = repr(value)
    return text[:limit]


def _get_any(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _join_values(value: Any) -> str | None:
    text = " ; ".join(_flatten_text([value]))
    return clean_text(text)


def _flatten_text(values: Iterable[Any]) -> list[str]:
    flattened: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if text:
                flattened.append(text)
        elif isinstance(value, dict):
            flattened.extend(_flatten_text(value.values()))
        elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            flattened.extend(_flatten_text(value))
        else:
            text = str(value).strip()
            if text:
                flattened.append(text)
    return flattened
