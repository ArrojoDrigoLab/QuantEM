"""EMPIAR scanner."""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from typing import Any

from ..http import HttpFetchError, USER_AGENT, get_json, urlencode
from ..models import Candidate
from .base import ScannerResult, safe_collect, unique_candidates
from .cursors import cursor_int, cursor_result


DEFAULT_QUERY = '("FIB-SEM" OR "SBF-SEM" OR "serial block face" OR "TEM" OR "electron tomography" OR "cellular EM" OR mitochondria OR organelle)'
SEARCH_URL = "https://www.ebi.ac.uk/ebisearch/ws/rest/empiar"
ENTRY_API_URL = "https://www.ebi.ac.uk/empiar/api/entry/"
HOME_URL = "https://www.ebi.ac.uk/empiar/"
DEFAULT_FULL_HISTORY_START_ACCESSION = 10000
DEFAULT_FULL_HISTORY_MIN_END_ACCESSION = 15000
FULL_HISTORY_ACCESSION_HEADROOM = 2500
ENTRY_RANGE_BATCH_SIZE = 100


def scan_empiar(
    since: str | None = None,
    query: str | None = None,
    limit: int = 100,
    full_history: bool = False,
    cursor: dict[str, Any] | None = None,
    **_: Any,
) -> ScannerResult:
    return safe_collect(
        "empiar",
        "empiar_api",
        _scan_empiar,
        since,
        query or DEFAULT_QUERY,
        limit,
        full_history,
        cursor,
        query is not None,
    )


def _scan_empiar(
    since: str | None,
    query: str,
    limit: int,
    full_history: bool,
    cursor: dict[str, Any] | None,
    explicit_query: bool,
) -> ScannerResult:
    if cursor and cursor.get("complete"):
        return ScannerResult(cursor=cursor, cursor_complete=True)
    if full_history and not explicit_query:
        return _scan_empiar_entry_ranges(since=since, limit=limit, cursor=cursor)
    return _scan_empiar_search(since=since, query=query, limit=limit, cursor=cursor)


def _scan_empiar_search(
    *,
    since: str | None,
    query: str,
    limit: int,
    cursor: dict[str, Any] | None,
) -> ScannerResult:
    max_candidates = max(int(limit), 0)
    if max_candidates == 0:
        return ScannerResult(cursor={"complete": True}, cursor_complete=True)

    candidates: list[Candidate] = []
    page_size = max(1, min(max_candidates, 100))
    start = cursor_int(cursor, "start", 0)
    while len(candidates) < max_candidates:
        params = urlencode(
            {
                "query": query,
                "fields": "id,title,description",
                "format": "json",
                "size": page_size,
                "start": start,
            }
        )
        search = get_json(f"{SEARCH_URL}?{params}")
        entries = search.get("entries", []) if isinstance(search, dict) else []
        if not isinstance(entries, list):
            raise ValueError("EMPIAR search payload entries field is not a list")
        if not entries:
            return cursor_result(unique_candidates(candidates), complete=True, cursor={"complete": True})
        for entry in entries:
            accession = entry.get("id") if isinstance(entry, dict) else None
            if not accession:
                continue
            detail = get_json(f"{ENTRY_API_URL}{accession}/")
            payload = detail.get(accession, detail)
            if since and _is_before_since(payload, since):
                continue
            candidates.append(_candidate_from_entry(accession, payload))
            candidates = unique_candidates(candidates)
            if len(candidates) >= max_candidates:
                break
        start += len(entries)
        hit_count = _int_or_none(search.get("hitCount")) if isinstance(search, dict) else None
        complete = (hit_count is not None and start >= hit_count) or len(entries) < page_size
        if complete:
            return cursor_result(candidates[:max_candidates], complete=True, cursor={"complete": True})
        if len(candidates) >= max_candidates:
            return cursor_result(
                candidates[:max_candidates],
                complete=False,
                cursor={"adapter": "empiar_search", "start": start, "hit_count": hit_count},
            )
    return cursor_result(candidates[:max_candidates], complete=False, cursor={"adapter": "empiar_search", "start": start})


def _scan_empiar_entry_ranges(
    *,
    since: str | None,
    limit: int,
    cursor: dict[str, Any] | None,
) -> ScannerResult:
    max_candidates = max(int(limit), 0)
    if max_candidates == 0:
        return ScannerResult(cursor={"complete": True}, cursor_complete=True)

    next_accession = cursor_int(cursor, "next_accession", DEFAULT_FULL_HISTORY_START_ACCESSION)
    end_accession = cursor_int(cursor, "end_accession", 0) or _default_end_accession()
    expected_entry_count = cursor_int(cursor, "expected_entry_count", 0) or (_current_entry_count() or 0)
    seen_entry_count = cursor_int(cursor, "seen_entry_count", 0)
    candidates: list[Candidate] = []

    while next_accession <= end_accession and len(candidates) < max_candidates:
        remaining = max_candidates - len(candidates)
        stop_accession = min(end_accession, next_accession + min(ENTRY_RANGE_BATCH_SIZE, remaining) - 1)
        entries = _post_entry_range(next_accession, stop_accession)
        seen_entry_count += len(entries)
        for accession in sorted(entries, key=_accession_sort_key):
            payload = entries[accession]
            if not isinstance(payload, dict):
                continue
            if since and _is_before_since(payload, since):
                continue
            candidates.append(_candidate_from_entry(accession, payload))
        candidates = unique_candidates(candidates)
        next_accession = stop_accession + 1
        if expected_entry_count and seen_entry_count >= expected_entry_count:
            break

    complete = next_accession > end_accession or bool(expected_entry_count and seen_entry_count >= expected_entry_count)
    if complete:
        return cursor_result(candidates[:max_candidates], complete=True, cursor={"complete": True})
    return cursor_result(
        candidates[:max_candidates],
        complete=False,
        cursor={
            "adapter": "empiar_entry_range",
            "next_accession": next_accession,
            "end_accession": end_accession,
            "seen_entry_count": seen_entry_count,
            "expected_entry_count": expected_entry_count or None,
        },
    )


def _post_entry_range(start_accession: int, stop_accession: int) -> dict[str, Any]:
    body = str(start_accession) if start_accession == stop_accession else f"{start_accession}-{stop_accession}"
    request = urllib.request.Request(
        ENTRY_API_URL,
        data=body.encode("utf-8"),
        headers={"User-Agent": USER_AGENT, "Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - public metadata API.
            import json

            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        excerpt = exc.read(1000).decode("utf-8", errors="replace")
        raise HttpFetchError(ENTRY_API_URL, f"HTTP {exc.code}", exc.code, excerpt) from exc
    except urllib.error.URLError as exc:
        raise HttpFetchError(ENTRY_API_URL, str(exc.reason)) from exc
    if not isinstance(payload, dict):
        raise ValueError("EMPIAR entry range payload is not a JSON object")
    return payload


def _default_end_accession() -> int:
    entry_count = _current_entry_count()
    if entry_count:
        return max(
            DEFAULT_FULL_HISTORY_MIN_END_ACCESSION,
            DEFAULT_FULL_HISTORY_START_ACCESSION + entry_count + FULL_HISTORY_ACCESSION_HEADROOM,
        )
    return DEFAULT_FULL_HISTORY_MIN_END_ACCESSION


def _current_entry_count() -> int | None:
    try:
        html = _get_text(HOME_URL)
    except HttpFetchError:
        return None
    match = re.search(r'id="num_of_entries">\s*([0-9,]+)\s*<', html)
    if not match:
        return None
    return _int_or_none(match.group(1).replace(",", ""))


def _get_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - public metadata page.
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        excerpt = exc.read(1000).decode("utf-8", errors="replace")
        raise HttpFetchError(url, f"HTTP {exc.code}", exc.code, excerpt) from exc
    except urllib.error.URLError as exc:
        raise HttpFetchError(url, str(exc.reason)) from exc


def _accession_sort_key(accession: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", accession)
    return (int(match.group(1)) if match else 0, accession)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_before_since(payload: dict[str, Any], since: str) -> bool:
    date = payload.get("update_date") or payload.get("release_date") or payload.get("deposition_date")
    return bool(date and str(date)[:10] < since)


def _candidate_from_entry(accession: str, payload: dict[str, Any]) -> Candidate:
    imagesets = [item for item in payload.get("imagesets") or [] if isinstance(item, dict)]
    citation_titles: list[str] = []
    citation_details: list[str] = []
    formats: list[str] = []
    image_texts: list[str] = []
    segmentation_texts: list[str] = []
    counts: list[str] = []
    for image_set in imagesets:
        if image_set.get("data_format"):
            formats.append(str(image_set["data_format"]))
        for key in ("details", "name"):
            if image_set.get(key):
                image_texts.append(str(image_set[key]))
        segmentation_texts.extend(_segmentation_texts(image_set.get("segmentations")))
        if image_set.get("num_images_or_tilt_series"):
            counts.append(f"{image_set.get('num_images_or_tilt_series')} images/series")
    publication_doi = None
    for citation in payload.get("citation") or []:
        if not isinstance(citation, dict):
            continue
        if not publication_doi and citation.get("doi"):
            publication_doi = citation.get("doi")
        if citation.get("title"):
            citation_titles.append(str(citation["title"]))
        if citation.get("details"):
            citation_details.append(str(citation["details"]))
    evidence_sources = _unique_texts(
        [
            payload.get("title"),
            *citation_titles,
            *citation_details,
            payload.get("experiment_type"),
            *(f"scale: {payload['scale']}" for _ in [0] if payload.get("scale")),
            *image_texts,
            *(f"Segmentation: {text}" for text in segmentation_texts),
        ]
    )
    evidence = " ; ".join(evidence_sources)
    extraction_text = " ".join(
        _unique_texts(
            [
                payload.get("title"),
                *citation_titles,
                *citation_details,
                payload.get("experiment_type"),
                payload.get("scale"),
                *image_texts,
                *segmentation_texts,
            ]
        )
    )
    return Candidate(
        source_name="empiar",
        source_record_id=accession,
        title=payload.get("title"),
        landing_url=f"https://www.ebi.ac.uk/empiar/{accession}/",
        publication_doi=publication_doi,
        dataset_doi=payload.get("entry_doi"),
        modality=_extract_modality(payload, extraction_text),
        organism=_extract_organism(payload, extraction_text),
        tissue_or_sample=_extract_tissue_or_sample(payload, extraction_text),
        dimensions_or_image_count=" ; ".join(counts) or payload.get("dataset_size"),
        file_formats=sorted(set(formats)),
        license=None,
        raw_metadata=payload,
        evidence_text=evidence,
        discovered_at=payload.get("release_date") or payload.get("deposition_date"),
    )


def _segmentation_texts(segmentations: Any) -> list[str]:
    texts: list[str] = []
    if not isinstance(segmentations, list):
        return texts
    for segmentation in segmentations:
        if not isinstance(segmentation, dict):
            continue
        name = _clean_value(segmentation.get("name"))
        description = _clean_value(segmentation.get("description"))
        if name and description and name.lower() != description.lower():
            texts.append(f"{name}: {description}")
        elif description:
            texts.append(description)
        elif name:
            texts.append(name)
    return _unique_texts(texts)


def _extract_modality(payload: dict[str, Any], text: str) -> str | None:
    direct = _first_direct_value(
        payload,
        {
            "experiment_type",
            "modality",
            "microscopy_type",
            "imaging_method",
            "imaging_modality",
        },
    )
    modalities = [direct] if direct else []
    lowered = text.lower()
    for pattern, label in [
        (r"\bsbf[- ]?sem\b|serial block[- ]face", "SBF-SEM"),
        (r"\bfib[- ]?sem\b|focused ion beam", "FIB-SEM"),
        (r"\batum[- ]?sem\b", "ATUM-SEM"),
        (r"\bat[- ]?sem\b|array tomography sem", "AT-SEM"),
        (r"\bss[- ]?tem\b|serial section tem|serial-section tem", "ssTEM"),
        (r"transmission electron microscopy|\btem\b", "TEM"),
        (r"electron tomography|resin-section et", "electron tomography"),
    ]:
        if re.search(pattern, lowered):
            modalities.append(label)
    specific_modalities = [
        modality
        for modality in modalities
        if modality and modality.lower() not in {"em", "electron microscopy", "volume em", "volume electron microscopy"}
    ]
    has_volume_modality = any(
        modality and modality.lower() in {"volume em", "volume electron microscopy"} for modality in modalities
    ) or bool(re.search(r"volume electron microscopy|\bvolume em\b|\bvem\b", lowered))
    if specific_modalities:
        modalities = specific_modalities
    elif has_volume_modality:
        modalities = ["volume EM"]
    return " ; ".join(_unique_texts(modalities)) or None


def _extract_organism(payload: dict[str, Any], text: str) -> str | None:
    direct = _first_direct_value(
        payload,
        {
            "organism",
            "organism_name",
            "species",
            "taxon",
            "taxon_name",
            "scientific_name",
        },
    )
    if direct:
        normalized = _normalize_organism(direct, allow_unrecognized=True)
        if normalized:
            return normalized
    return _normalize_organism(text)


def _extract_tissue_or_sample(payload: dict[str, Any], text: str) -> str | None:
    direct = _first_direct_value(
        payload,
        {
            "tissue",
            "tissue_type",
            "sample",
            "sample_name",
            "sample_type",
            "sample_description",
            "specimen",
            "specimen_type",
            "cell_line",
            "cell_type",
            "biological_sample",
        },
    )
    if direct and not _is_vague_sample(direct):
        return direct
    return _extract_sample_from_text(text)


def _normalize_organism(text: str, allow_unrecognized: bool = False) -> str | None:
    lowered = text.lower()
    for pattern, label in [
        (r"\bhomo sapiens\b|\bhuman\b|\bhela\b", "Human"),
        (r"\bmus musculus\b|\bmouse\b|\bmurine\b", "Mouse"),
        (r"\brattus norvegicus\b|\brat\b", "Rat"),
        (r"\bdanio rerio\b|\bzebrafish\b", "Zebrafish"),
        (r"\barabidopsis thaliana\b|\barabidopsis\b", "Arabidopsis thaliana"),
        (r"\bdrosophila melanogaster\b|\bdrosophila\b", "Drosophila melanogaster"),
        (r"\bcaenorhabditis elegans\b|\bc\.?\s*elegans\b", "C. elegans"),
        (r"\bsaccharomyces cerevisiae\b|\byeast\b", "Saccharomyces cerevisiae"),
        (r"\bplasmodium berghei\b|\bp\.?\s*berghei\b", "Plasmodium berghei"),
    ]:
        if re.search(pattern, lowered):
            return label
    cleaned = _clean_value(text)
    if allow_unrecognized and cleaned and not _is_unknown(cleaned):
        return cleaned
    return None


def _extract_sample_from_text(text: str) -> str | None:
    normalized = re.sub(r"[_/-]+", " ", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    lowered = normalized.lower()
    if re.search(r"\bhela\s+cell\s+pellet\b", lowered):
        return "HeLa cell pellet"
    if re.search(r"\bhela\s+cells?\b|\bhela\b", lowered):
        return "HeLa cell"
    if "jugular vein" in lowered and "carotid artery" in lowered:
        return "jugular vein or carotid artery"
    if re.search(r"\bookinetes?\b", lowered):
        return "ookinete"

    for pattern, label in [
        (r"\bbrown adipose tissue\b", "brown adipose tissue"),
        (r"\badipose tissue\b", "adipose tissue"),
        (r"\bcochlear hair cells?\b", "cochlear hair cell"),
    ]:
        if re.search(pattern, lowered):
            return label

    organs = [
        "liver",
        "brain",
        "retina",
        "kidney",
        "breast",
        "tongue",
        "intestine",
        "cochlea",
        "heart",
        "lung",
        "muscle",
        "spleen",
        "pancreas",
        "skin",
        "root",
        "leaf",
    ]
    organ_pattern = "|".join(re.escape(organ) for organ in organs)
    for suffix in ["tissue sample", "tissue", "cells", "cell", "sample"]:
        match = re.search(rf"\b(?:human|mouse|murine|rat|zebrafish|arabidopsis)?\s*({organ_pattern})\s+{suffix}\b", lowered)
        if match:
            return f"{match.group(1)} {suffix}"
    match = re.search(rf"\b(?:human|mouse|murine|rat|zebrafish|arabidopsis)\s+({organ_pattern})\b", lowered)
    if match:
        organ = match.group(1)
        return "retina" if organ == "retina" else f"{organ} tissue"

    for pattern, label in [
        (r"\bhepatocytes?\b", "hepatocyte"),
        (r"\bcholangiocytes?\b", "cholangiocyte"),
        (r"\bendothelial cells?\b", "endothelial cell"),
        (r"\bneurons?\b", "neuron"),
    ]:
        if re.search(pattern, lowered):
            return label
    return None


def _first_direct_value(payload: dict[str, Any], keys: set[str]) -> str | None:
    for value in _collect_direct_values(payload, keys):
        cleaned = _clean_value(value)
        if cleaned and not _is_unknown(cleaned):
            return cleaned
    return None


def _collect_direct_values(value: Any, keys: set[str]) -> list[Any]:
    values: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = _normalize_key(key)
            if normalized_key in keys:
                values.extend(_direct_string_values(child))
            values.extend(_collect_direct_values(child, keys))
    elif isinstance(value, list):
        for item in value:
            values.extend(_collect_direct_values(item, keys))
    return values


def _direct_string_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return []
    if isinstance(value, list):
        return [item for item in value if not isinstance(item, (dict, list))]
    return [value]


def _normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def _clean_value(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip(" ;,")
    return text or None


def _is_unknown(value: str | None) -> bool:
    return not value or value.strip().lower() in {"unknown", "not known", "not specified", "n/a", "na", "none", "null", "/"}


def _is_vague_sample(value: str) -> bool:
    return value.strip().lower() in {"cell", "cells", "tissue", "sample", "specimen", "volume", "raw images", "benchmark data"}


def _unique_texts(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = _clean_value(value)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(text)
    return unique
