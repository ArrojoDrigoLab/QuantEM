"""OpenOrganelle/CellMap scanner."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable
from urllib.parse import quote

from ..http import HttpFetchError, USER_AGENT, get_json
from ..models import Candidate, as_list, clean_text
from .base import ScannerResult, safe_collect, unique_candidates
from .cursors import cursor_int, cursor_result


METADATA_RAW_BASE = "https://raw.githubusercontent.com/janelia-cellmap/fibsem-metadata/stable"
METADATA_HTML_BASE = "https://github.com/janelia-cellmap/fibsem-metadata/blob/stable"
INDEX_URL = f"{METADATA_RAW_BASE}/api/index.json"
OPENORGANELLE_DATASET_BASE = "https://openorganelle.janelia.org/datasets"
SUPABASE_URL = "https://kvwjcggnkipomjlbjykf.supabase.co"

# This is the anonymous read-only key that OpenOrganelle publishes in its own
# public web frontend bundle; it grants exactly the access any visitor to
# openorganelle.janelia.org already has, and it belongs to Janelia rather than to
# this project. It is inlined so this scanner runs with no configuration, and is
# a published access token rather than a credential. Override with
# OPENORGANELLE_ANON_KEY if Janelia rotates it.
SUPABASE_ANON_KEY = os.environ.get("OPENORGANELLE_ANON_KEY") or (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt2d2pjZ2dua2lwb21qbGJqeWtmIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NjUxODgyMjksImV4cCI6MTk4MDc2NDIyOX0."
    "o_yLKX9erKbIrG3mwdwFkWYI8N9EjTNUnu9FWMngw9E"
)
SUPABASE_DATASET_SELECT = (
    "name,description,thumbnail_url,created_at,stage,segmentation_challenge,"
    "sample:sample(name,description,protocol,contributions,type,subtype,treatment,organism),"
    "image_acquisition:image_acquisition("
    "name,institution,start_date,grid_axes,grid_spacing,grid_dimensions,grid_spacing_unit,grid_dimensions_unit"
    "),"
    "images:image("
    "name,description,url,format,source,grid_scale,grid_translation,grid_dims,grid_units,"
    "display_settings,sample_type,content_type,institution,created_at,stage,image_stack,"
    "meshes:mesh(name,description,url,source,grid_scale,grid_translation,grid_dims,grid_units,created_at,format,stage,ids)"
    "),"
    "publications:publication(name,url,type,stage)"
)
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)


def scan_openorganelle(
    since: str | None = None,
    query: str | None = None,
    limit: int = 100,
    cursor: dict[str, Any] | None = None,
    **_: Any,
) -> ScannerResult:
    return safe_collect("openorganelle", "openorganelle_supabase", _scan_openorganelle, since, query, limit, cursor)


def _scan_openorganelle(since: str | None, query: str | None, limit: int, cursor: dict[str, Any] | None) -> ScannerResult:
    if cursor and cursor.get("complete"):
        return ScannerResult(cursor=cursor, cursor_complete=True)
    records = _dataset_records(_fetch_supabase_datasets())
    result = ScannerResult()
    max_candidates = max(int(limit), 0)
    if max_candidates == 0:
        return cursor_result([], complete=True, cursor={"complete": True})
    offset = cursor_int(cursor, "record_offset", 0)
    next_offset = offset
    for record_index, record in enumerate(records[offset:], start=offset):
        if len(result.candidates) >= max_candidates:
            break
        next_offset = record_index + 1
        candidate_result = safe_collect(
            "openorganelle",
            f"openorganelle_{record['record_type']}",
            _candidate_from_record,
            record,
            since,
            query,
        )
        result.extend(candidate_result)
    result.candidates = unique_candidates(result.candidates[:max_candidates])
    complete = next_offset >= len(records)
    result.cursor = {"complete": True} if complete else {
        "adapter": "openorganelle_supabase",
        "record_offset": next_offset,
        "record_count": len(records),
    }
    result.cursor_complete = complete
    if complete:
        result.cursor = {"complete": True}
    else:
        result.cursor["complete"] = False
    return result


def _fetch_supabase_datasets() -> list[dict[str, Any]]:
    params = {
        "select": SUPABASE_DATASET_SELECT,
        "stage": "eq.prod",
        "order": "name.asc",
        "limit": "10000",
    }
    url = f"{SUPABASE_URL}/rest/v1/dataset?{urllib.parse.urlencode(params, safe='(),:*')}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Prefer": "count=exact",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        excerpt = exc.read(1000).decode("utf-8", errors="replace")
        raise HttpFetchError(url, f"HTTP {exc.code}", exc.code, excerpt) from exc
    except urllib.error.URLError as exc:
        raise HttpFetchError(url, str(exc.reason)) from exc
    except json.JSONDecodeError as exc:
        raise HttpFetchError(url, f"invalid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("OpenOrganelle Supabase dataset response is not a JSON array")
    return [row for row in payload if isinstance(row, dict)]


def _candidate_from_record(
    record: dict[str, Any],
    since: str | None,
    query: str | None,
) -> list[Candidate]:
    record_type = record.get("record_type")
    if record_type == "supabase_dataset":
        payload = _as_dict(record.get("payload"))
        candidate = _candidate_from_supabase_dataset(payload)
    elif record_type == "manifest":
        dataset_id = str(record["dataset_id"])
        manifest_url = _manifest_url(str(record["metadata_path"]))
        return _candidate_from_manifest_url(dataset_id, manifest_url, since, query)
    else:
        raise ValueError(f"unsupported OpenOrganelle record type: {record_type}")

    if since and _is_before_since(candidate, payload, since):
        return []
    if query and not _matches_query(candidate, query):
        return []
    return [candidate]


def _candidate_from_manifest_url(
    dataset_id: str,
    manifest_url: str,
    since: str | None,
    query: str | None,
) -> list[Candidate]:
    payload = get_json(manifest_url)
    candidate = _candidate_from_manifest(dataset_id, payload, manifest_url=manifest_url)
    if since and _is_before_since(candidate, payload, since):
        return []
    if query and not _matches_query(candidate, query):
        return []
    return [candidate]


def _candidate_from_supabase_dataset(payload: dict[str, Any]) -> Candidate:
    if not isinstance(payload, dict):
        raise ValueError("OpenOrganelle Supabase dataset row is not a JSON object")

    record_id = clean_text(payload.get("name"))
    if not record_id:
        raise ValueError("OpenOrganelle Supabase dataset row is missing a name")

    sample = _as_dict(payload.get("sample"))
    acquisition = _as_dict(payload.get("image_acquisition"))
    images = [image for image in payload.get("images") or [] if isinstance(image, dict)]
    publications = [pub for pub in payload.get("publications") or [] if isinstance(pub, dict)]
    title = clean_text(payload.get("description")) or record_id
    publication_dois = _publication_dois(publications)
    dataset_doi = _dataset_doi(publications)
    raw_metadata = dict(payload)
    raw_metadata["_metadata_source"] = "openorganelle_supabase"
    raw_metadata["_metadata_url"] = _supabase_dataset_url(record_id)

    return Candidate(
        source_name="openorganelle",
        source_record_id=record_id,
        title=title,
        landing_url=f"{OPENORGANELLE_DATASET_BASE}/{quote(record_id, safe='-_.~')}",
        download_or_manifest_urls=_supabase_urls(images),
        publication_doi=publication_dois[0] if publication_dois else None,
        publication_dois=publication_dois,
        dataset_doi=dataset_doi,
        modality=_infer_supabase_modality(images, publications),
        organism=_join_values(sample.get("organism")),
        tissue_or_sample=_sample_text(sample),
        dimensions_or_image_count=_supabase_dimensions_text(acquisition, images),
        file_formats=_supabase_file_formats(images),
        license=None,
        raw_metadata=raw_metadata,
        evidence_text=_supabase_evidence_text(title, sample, acquisition, images, publications, dataset_doi),
        discovered_at=clean_text(payload.get("created_at")) or clean_text(acquisition.get("start_date")) or None,
    )


def _candidate_from_manifest(dataset_id: str, payload: dict[str, Any], manifest_url: str | None = None) -> Candidate:
    if not isinstance(payload, dict):
        raise ValueError(f"OpenOrganelle manifest for {dataset_id} is not a JSON object")

    metadata = _as_dict(payload.get("metadata"))
    record_id = clean_text(dataset_id) or clean_text(metadata.get("id")) or clean_text(payload.get("name"))
    if not record_id:
        raise ValueError("OpenOrganelle manifest is missing a dataset id")

    title = clean_text(metadata.get("title")) or clean_text(payload.get("name")) or record_id
    sample = _as_dict(metadata.get("sample"))
    imaging = _as_dict(metadata.get("imaging"))
    source_records = list(_iter_sources(_as_dict(payload.get("sources")).values()))
    source_urls = sorted({url for url in (_clean_url(source.get("url")) for source in source_records) if url})
    manifest_urls = [manifest_url] if manifest_url else []

    raw_metadata = dict(payload)
    if manifest_url:
        raw_metadata["_metadata_url"] = manifest_url
        raw_metadata["_metadata_html_url"] = _html_url_for_manifest(manifest_url)

    return Candidate(
        source_name="openorganelle",
        source_record_id=record_id,
        title=title,
        landing_url=f"{OPENORGANELLE_DATASET_BASE}/{quote(record_id, safe='-_.~')}",
        download_or_manifest_urls=manifest_urls + source_urls,
        publication_doi=_first_doi(metadata.get("publications")),
        dataset_doi=_first_doi(metadata.get("DOI")),
        modality=_infer_modality(source_records),
        organism=_join_values(sample.get("organism")),
        tissue_or_sample=_sample_text(sample),
        dimensions_or_image_count=_dimensions_text(imaging),
        file_formats=sorted({fmt for fmt in (_format_text(source.get("format")) for source in source_records) if fmt}),
        license=None,
        raw_metadata=raw_metadata,
        evidence_text=_evidence_text(title, metadata, sample, source_records, manifest_url),
    )


def _dataset_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return _dataset_records_from_supabase(payload)
    return _dataset_records_from_index(payload)


def _dataset_records_from_supabase(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        dataset_id = clean_text(row.get("name"))
        if not dataset_id:
            continue
        records.append({"dataset_id": dataset_id, "record_type": "supabase_dataset", "payload": row})
    if not records:
        raise ValueError("OpenOrganelle Supabase response did not contain any datasets")
    return sorted(records, key=lambda item: item["dataset_id"])


def _dataset_records_from_index(index: Any) -> list[dict[str, Any]]:
    if not isinstance(index, dict):
        raise ValueError("OpenOrganelle metadata index is not a JSON object")
    datasets = index.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("OpenOrganelle metadata index is missing a datasets object")
    records: list[dict[str, Any]] = []
    for dataset_id, metadata_path in datasets.items():
        clean_id = clean_text(dataset_id)
        if not clean_id:
            continue
        clean_path = clean_text(metadata_path) or f"api/{clean_id}"
        records.append({"dataset_id": clean_id, "metadata_path": clean_path, "record_type": "manifest"})
    if not records:
        raise ValueError("OpenOrganelle metadata index did not contain any datasets")
    return sorted(records, key=lambda item: item["dataset_id"])


def _supabase_dataset_url(dataset_id: str) -> str:
    params = {
        "select": SUPABASE_DATASET_SELECT,
        "stage": "eq.prod",
        "name": f"eq.{dataset_id}",
    }
    return f"{SUPABASE_URL}/rest/v1/dataset?{urllib.parse.urlencode(params, safe='(),:*')}"


def _manifest_url(metadata_path: str) -> str:
    path = metadata_path.strip()
    if path.startswith(("http://", "https://")):
        path = _github_blob_to_raw(path)
        return path if path.endswith(".json") else f"{path.rstrip('/')}/manifest.json"
    path = path.strip("/")
    if path.endswith(".json"):
        return f"{METADATA_RAW_BASE}/{path}"
    return f"{METADATA_RAW_BASE}/{path}/manifest.json"


def _github_blob_to_raw(url: str) -> str:
    return (
        url.replace("https://github.com/janelia-cosem/fibsem-metadata/blob/stable", METADATA_RAW_BASE)
        .replace("https://github.com/janelia-cellmap/fibsem-metadata/blob/stable", METADATA_RAW_BASE)
        .replace("https://raw.githubusercontent.com/janelia-cosem/fibsem-metadata/stable", METADATA_RAW_BASE)
    )


def _html_url_for_manifest(manifest_url: str) -> str | None:
    if manifest_url.startswith(METADATA_RAW_BASE):
        return manifest_url.replace(METADATA_RAW_BASE, METADATA_HTML_BASE)
    return None


def _iter_sources(values: Iterable[Any]) -> Iterable[dict[str, Any]]:
    for value in values:
        if not isinstance(value, dict):
            continue
        yield value
        for subsource in value.get("subsources") or []:
            if isinstance(subsource, dict):
                yield subsource


def _infer_modality(source_records: list[dict[str, Any]]) -> str | None:
    text = " ".join(
        str(value)
        for source in source_records
        for value in [source.get("name"), source.get("description"), source.get("url"), source.get("contentType")]
        if value
    ).lower()
    modalities: list[str] = []
    if "fib-sem" in text or "fibsem" in text or "fib_sem" in text:
        modalities.append("FIB-SEM")
    if re.search(r"\btem\b", text):
        modalities.append("TEM")
    if "electron tomography" in text:
        modalities.append("electron tomography")
    if not modalities and any(source.get("contentType") == "em" for source in source_records):
        modalities.append("volume EM")
    return " ; ".join(modalities) or None


def _infer_supabase_modality(images: list[dict[str, Any]], publications: list[dict[str, Any]]) -> str | None:
    text = " ".join(
        str(value)
        for image in images
        for value in [
            image.get("name"),
            image.get("description"),
            image.get("url"),
            image.get("content_type"),
            image.get("format"),
        ]
        if value
    )
    text = " ".join(
        [
            text,
            " ".join(
                str(value)
                for publication in publications
                for value in [publication.get("name"), publication.get("url"), publication.get("type")]
                if value
            ),
        ]
    ).lower()
    modalities: list[str] = []
    if "fib-sem" in text or "fibsem" in text or "fib_sem" in text:
        modalities.append("FIB-SEM")
    if re.search(r"\btem\b", text):
        modalities.append("TEM")
    if "electron tomography" in text:
        modalities.append("electron tomography")
    if "volume em" in text or "vem" in text:
        modalities.append("volume EM")
    if not modalities and any(image.get("content_type") == "em" for image in images):
        modalities.append("volume EM")
    return " ; ".join(dict.fromkeys(modalities)) or None


def _sample_text(sample: dict[str, Any]) -> str | None:
    values = [
        clean_text(sample.get("description")),
        _join_values(sample.get("type")),
        _join_values(sample.get("subtype")),
        _join_values(sample.get("treatment")),
    ]
    return " ; ".join(value for value in values if value)


def _dimensions_text(imaging: dict[str, Any]) -> str | None:
    values = [
        _axis_values_text("volume dimensions", imaging.get("dimensions")),
        _axis_values_text("voxel spacing", imaging.get("gridSpacing")),
    ]
    return " ; ".join(value for value in values if value) or None


def _supabase_dimensions_text(acquisition: dict[str, Any], images: list[dict[str, Any]]) -> str | None:
    axes = [str(value) for value in acquisition.get("grid_axes") or [] if value is not None]
    values = [
        _array_values_text(
            "volume dimensions",
            acquisition.get("grid_dimensions"),
            axes=axes,
            unit=clean_text(acquisition.get("grid_dimensions_unit")),
        ),
        _array_values_text(
            "voxel spacing",
            acquisition.get("grid_spacing"),
            axes=axes,
            unit=clean_text(acquisition.get("grid_spacing_unit")),
        ),
    ]
    if any(values):
        return " ; ".join(value for value in values if value)

    for image in images:
        image_axes = [str(value) for value in image.get("grid_dims") or [] if value is not None]
        values = [
            _array_values_text("volume dimensions", image.get("grid_scale"), axes=image_axes, unit=None),
            _array_values_text("voxel spacing", image.get("grid_scale"), axes=image_axes, unit=_common_unit(image.get("grid_units"))),
        ]
        if any(values):
            return " ; ".join(value for value in values if value)
    return None


def _axis_values_text(label: str, payload: Any) -> str | None:
    item = _as_dict(payload)
    values = _as_dict(item.get("values"))
    parts = [f"{axis}={values[axis]}" for axis in ("x", "y", "z") if axis in values]
    if not parts:
        return None
    unit = clean_text(item.get("unit"))
    suffix = f" {unit}" if unit else ""
    return f"{label} ({', '.join(parts)}){suffix}"


def _array_values_text(label: str, values: Any, *, axes: list[str], unit: str | None) -> str | None:
    if not isinstance(values, list) or not values:
        return None
    axis_names = axes if len(axes) == len(values) else [str(index) for index in range(len(values))]
    parts = [f"{axis_names[index]}={value}" for index, value in enumerate(values)]
    suffix = f" {unit}" if unit else ""
    return f"{label} ({', '.join(parts)}){suffix}"


def _common_unit(values: Any) -> str | None:
    units = [str(value) for value in values or [] if value is not None]
    return units[0] if units and all(unit == units[0] for unit in units) else None


def _supabase_urls(images: list[dict[str, Any]]) -> list[str]:
    urls: set[str] = set()
    for image in images:
        url = _clean_url(image.get("url"))
        if url:
            urls.add(url)
        for mesh in image.get("meshes") or []:
            if isinstance(mesh, dict):
                mesh_url = _clean_url(mesh.get("url"))
                if mesh_url:
                    urls.add(mesh_url)
    return sorted(urls)


def _supabase_file_formats(images: list[dict[str, Any]]) -> list[str]:
    formats: set[str] = set()
    for image in images:
        fmt = _format_text(image.get("format"))
        if fmt:
            formats.add(fmt)
        for mesh in image.get("meshes") or []:
            if isinstance(mesh, dict):
                mesh_fmt = _format_text(mesh.get("format"))
                if mesh_fmt:
                    formats.add(mesh_fmt)
    return sorted(formats)


def _evidence_text(
    title: str | None,
    metadata: dict[str, Any],
    sample: dict[str, Any],
    source_records: list[dict[str, Any]],
    manifest_url: str | None,
) -> str:
    source_bits: list[str] = []
    for source in source_records:
        name = clean_text(source.get("name"))
        description = clean_text(source.get("description"))
        content_type = clean_text(source.get("contentType"))
        if name or description or content_type:
            source_bits.append(" / ".join(value for value in [name, description, content_type] if value))
        if len(source_bits) >= 8:
            break
    values = [
        title,
        _sample_text(sample),
        f"organism: {_join_values(sample.get('organism'))}" if _join_values(sample.get("organism")) else None,
        f"dataset DOI: {_first_doi(metadata.get('DOI'))}" if _first_doi(metadata.get("DOI")) else None,
        f"publication DOI: {_first_doi(metadata.get('publications'))}" if _first_doi(metadata.get("publications")) else None,
        f"sources: {'; '.join(source_bits)}" if source_bits else None,
        f"metadata manifest: {manifest_url}" if manifest_url else None,
    ]
    return " ; ".join(value for value in values if value)


def _supabase_evidence_text(
    title: str | None,
    sample: dict[str, Any],
    acquisition: dict[str, Any],
    images: list[dict[str, Any]],
    publications: list[dict[str, Any]],
    dataset_doi: str | None,
) -> str:
    source_bits: list[str] = []
    for image in sorted(images, key=lambda item: 0 if item.get("content_type") == "em" else 1):
        name = clean_text(image.get("name"))
        description = clean_text(image.get("description"))
        content_type = clean_text(image.get("content_type"))
        if name or description or content_type:
            source_bits.append(" / ".join(value for value in [name, description, content_type] if value))
        if len(source_bits) >= 8:
            break
    values = [
        title,
        _sample_text(sample),
        f"organism: {_join_values(sample.get('organism'))}" if _join_values(sample.get("organism")) else None,
        f"acquisition institution: {clean_text(acquisition.get('institution'))}" if clean_text(acquisition.get("institution")) else None,
        f"dataset DOI: {dataset_doi}" if dataset_doi else None,
        f"publication DOI: {(_publication_dois(publications) or [None])[0]}" if _publication_dois(publications) else None,
        f"sources: {'; '.join(source_bits)}" if source_bits else None,
        "metadata source: OpenOrganelle Supabase dataset table",
    ]
    return " ; ".join(value for value in values if value)


def _first_doi(value: Any) -> str | None:
    for item in _as_iterable(value):
        if isinstance(item, dict):
            for field in ("href", "doi", "DOI", "url", "id"):
                doi = _extract_doi(item.get(field))
                if doi:
                    return doi
        else:
            doi = _extract_doi(item)
            if doi:
                return doi
    return None


def _publication_dois(publications: list[dict[str, Any]]) -> list[str]:
    dois: list[str] = []
    seen: set[str] = set()
    for publication in publications:
        if clean_text(publication.get("type")) != "paper":
            continue
        doi = _extract_doi(publication.get("url")) or _extract_doi(publication.get("name"))
        if doi and doi.lower() not in seen:
            seen.add(doi.lower())
            dois.append(doi)
    return dois


def _dataset_doi(publications: list[dict[str, Any]]) -> str | None:
    data_dois: list[tuple[int, str]] = []
    for index, publication in enumerate(publications):
        if clean_text(publication.get("type")) == "paper":
            continue
        doi = _extract_doi(publication.get("url")) or _extract_doi(publication.get("name"))
        if not doi:
            continue
        label = (clean_text(publication.get("name")) or "").lower()
        priority = 0 if any(term in label for term in ["em", "vem", "fib-sem", "reconstructed"]) else 1
        data_dois.append((priority * 10000 + index, doi))
    if not data_dois:
        return None
    return sorted(data_dois, key=lambda item: item[0])[0][1]


def _extract_doi(value: Any) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    match = _DOI_RE.search(text)
    if match:
        return match.group(0).rstrip(".,;)")
    lowered = text.lower()
    if lowered.startswith("doi:"):
        return text[4:].strip()
    if lowered.startswith("10."):
        return text.rstrip(".,;)")
    return None


def _join_values(value: Any) -> str | None:
    values = as_list(value)
    return " ; ".join(values) if values else None


def _format_text(value: Any) -> str | None:
    return clean_text(value)


def _clean_url(value: Any) -> str | None:
    return clean_text(value)


def _matches_query(candidate: Candidate, query: str) -> bool:
    terms = [term for term in re.split(r"\s+", query.strip().casefold()) if term]
    if not terms:
        return True
    haystack = " ".join(
        value
        for value in [
            candidate.source_record_id,
            candidate.title,
            candidate.modality,
            candidate.organism,
            candidate.tissue_or_sample,
            candidate.evidence_text,
            " ".join(candidate.file_formats),
        ]
        if value
    ).casefold()
    return all(term in haystack for term in terms)


def _is_before_since(candidate: Candidate, payload: dict[str, Any], since: str) -> bool:
    metadata = _as_dict(payload.get("metadata"))
    imaging = _as_dict(metadata.get("imaging"))
    date = (
        clean_text(payload.get("created_at"))
        or clean_text(metadata.get("updated"))
        or clean_text(metadata.get("modified"))
        or clean_text(metadata.get("releaseDate"))
        or clean_text(imaging.get("startDate"))
        or clean_text(candidate.discovered_at)
    )
    return bool(date and date[:10] < since)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_iterable(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
