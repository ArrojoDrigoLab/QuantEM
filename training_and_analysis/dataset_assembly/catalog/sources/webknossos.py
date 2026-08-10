"""webKnossos public dataset scanner."""

from __future__ import annotations

import re
import traceback
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from ..http import HttpFetchError, USER_AGENT, get_json
from ..models import Candidate, ScannerError
from .base import ScannerResult, unique_candidates
from .cursors import cursor_int


SOURCE_NAME = "webknossos"
ADAPTER_NAME = "webknossos_public_dataset_api"
DEFAULT_BASE_URL = "https://webknossos.org"
DEFAULT_BASE_URLS = [
    DEFAULT_BASE_URL,
    "https://webknossos.tnw.tudelft.nl",
]
DEFAULT_SEED_URLS = [
    "https://webknossos.org/datasets/b2275d664e4c2a96/HuaLab-CBA_Ca-mouse-unexposed-M2",
]
DATASET_ID_RE = re.compile(r"^[0-9a-f]{24}$", re.IGNORECASE)
FULL_HISTORY_LIST_LIMIT = 5000
FetchJson = Callable[[str], Any]
FetchText = Callable[[str], str]


def scan_webknossos(
    since: str | None = None,
    query: str | None = None,
    limit: int = 100,
    full_history: bool = False,
    cursor: dict[str, Any] | None = None,
    seed_urls: list[str] | None = None,
    fetch_json: FetchJson = get_json,
    fetch_text: FetchText | None = None,
    base_urls: list[str] | None = None,
    **_: Any,
) -> ScannerResult:
    """Scan deterministic webKnossos seed URLs, the full public dataset listing, or a text search."""
    fetch_text = fetch_text or _get_text
    if cursor and cursor.get("complete"):
        return ScannerResult(cursor=cursor, cursor_complete=True)
    try:
        if seed_urls is not None:
            seeds = seed_urls
        elif full_history:
            seeds = _public_dataset_seed_urls(query, fetch_json, base_urls=base_urls or DEFAULT_BASE_URLS)
        else:
            seeds = _seed_urls_from_query(query)
            if not seeds and query:
                seeds = _search_seed_urls(query, limit, fetch_json, base_urls=base_urls or DEFAULT_BASE_URLS)
    except Exception as exc:  # noqa: BLE001 - scanners must queue source/API failures.
        return ScannerResult(errors=[_scanner_error(f"{DEFAULT_BASE_URL}/api/datasets", exc)])
    return _collect_seed_urls(seeds, since=since, limit=limit, cursor=cursor, fetch_json=fetch_json, fetch_text=fetch_text)


def _seed_urls_from_query(query: str | None) -> list[str]:
    if not query:
        return list(DEFAULT_SEED_URLS)
    urls = _extract_urls(query)
    return urls


def _extract_urls(text: str) -> list[str]:
    values = re.split(r"[\s,]+", text.strip())
    return [value for value in values if value.startswith(("http://", "https://")) or DATASET_ID_RE.match(value)]


def _collect_search(query: str, limit: int, result: ScannerResult, fetch_json: FetchJson, base_url: str = DEFAULT_BASE_URL) -> None:
    params = urllib.parse.urlencode({"isActive": "true", "limit": limit, "compact": "true", "searchQuery": query})
    url = f"{base_url}/api/datasets?{params}"
    try:
        payload = fetch_json(url)
        if not isinstance(payload, list):
            raise ValueError("expected webKnossos dataset search response to be a list")
        for item in payload:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            _collect_seed(f"{base_url}/api/datasets/{item['id']}", result, fetch_json, _get_text)
    except Exception as exc:  # noqa: BLE001 - scanners must queue source/API failures.
        result.errors.append(_scanner_error(url, exc))


def _search_seed_urls(
    query: str,
    limit: int,
    fetch_json: FetchJson,
    *,
    base_urls: list[str],
) -> list[str]:
    urls: list[str] = []
    for base_url in base_urls:
        params = urllib.parse.urlencode({"isActive": "true", "limit": limit, "compact": "true", "searchQuery": query})
        url = f"{base_url}/api/datasets?{params}"
        payload = fetch_json(url)
        if not isinstance(payload, list):
            raise ValueError("expected webKnossos dataset search response to be a list")
        urls.extend(_seed_urls_from_listing(payload, base_url=base_url))
    return sorted(dict.fromkeys(urls))


def _public_dataset_seed_urls(query: str | None, fetch_json: FetchJson, *, base_urls: list[str]) -> list[str]:
    urls: list[str] = []
    for base_url in base_urls:
        params = {
            "isActive": "true",
            "limit": FULL_HISTORY_LIST_LIMIT,
            "compact": "true",
        }
        if query:
            params["searchQuery"] = query
        url = f"{base_url}/api/datasets?{urllib.parse.urlencode(params)}"
        payload = fetch_json(url)
        if not isinstance(payload, list):
            raise ValueError("expected webKnossos public dataset response to be a list")
        urls.extend(_seed_urls_from_listing(payload, base_url=base_url))
    return sorted(dict.fromkeys(urls))


def _seed_urls_from_listing(payload: list[Any], *, base_url: str = DEFAULT_BASE_URL) -> list[str]:
    urls: list[str] = []
    for item in payload:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        urls.append(f"{base_url}/api/datasets/{item['id']}")
    return sorted(dict.fromkeys(urls))


def _collect_seed_urls(
    seeds: list[str],
    *,
    since: str | None,
    limit: int,
    cursor: dict[str, Any] | None,
    fetch_json: FetchJson,
    fetch_text: FetchText,
) -> ScannerResult:
    result = ScannerResult()
    max_candidates = max(int(limit), 0)
    if max_candidates == 0:
        return ScannerResult(cursor={"complete": True}, cursor_complete=True)
    offset = cursor_int(cursor, "seed_offset", 0)
    next_offset = offset
    for seed_index, seed_url in enumerate(seeds[offset:], start=offset):
        if len(result.candidates) >= max_candidates:
            break
        next_offset = seed_index + 1
        previous_count = len(result.candidates)
        _collect_seed(seed_url, result, fetch_json, fetch_text)
        if since and len(result.candidates) > previous_count:
            result.candidates = [
                *result.candidates[:previous_count],
                *(candidate for candidate in result.candidates[previous_count:] if not _is_before_since(candidate, since)),
            ]
    result.candidates = unique_candidates(result.candidates[:max_candidates])
    complete = next_offset >= len(seeds)
    return ScannerResult(
        candidates=result.candidates,
        errors=result.errors,
        cursor={"complete": True}
        if complete
        else {
            "adapter": "webknossos_seed_list",
            "seed_offset": next_offset,
            "seed_count": len(seeds),
            "complete": False,
        },
        cursor_complete=complete,
    )


def _collect_seed(seed_url: str, result: ScannerResult, fetch_json: FetchJson, fetch_text: FetchText) -> None:
    try:
        dataset_api_url = _dataset_api_url_from_seed(seed_url, fetch_json, fetch_text)
        payload = fetch_json(dataset_api_url)
        result.candidates.append(_candidate_from_dataset_payload(dataset_api_url, payload))
    except Exception as exc:  # noqa: BLE001 - keep other seeds collectible.
        result.errors.append(_scanner_error(seed_url, exc))


def _dataset_api_url_from_seed(seed_url: str, fetch_json: FetchJson, fetch_text: FetchText) -> str:
    if DATASET_ID_RE.match(seed_url):
        return f"{DEFAULT_BASE_URL}/api/datasets/{seed_url}"

    parsed = urllib.parse.urlparse(seed_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else DEFAULT_BASE_URL
    parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) >= 3 and parts[0] == "api" and parts[1] == "datasets" and DATASET_ID_RE.match(parts[2]):
        return f"{base_url}/api/datasets/{parts[2]}"
    if len(parts) >= 2 and parts[0] == "links":
        dataset_id = _dataset_id_from_link_page(seed_url, fetch_text)
        return f"{base_url}/api/datasets/{dataset_id}"
    if len(parts) >= 6 and parts[:3] == ["api", "datasets", "disambiguate"] and parts[-1] == "toId":
        resolved = _resolve_dataset_id(base_url, parts[3], parts[4], fetch_json)
        return f"{base_url}/api/datasets/{resolved}"
    if parts and parts[0] == "datasets":
        dataset_parts = _strip_dataset_suffixes(parts[1:])
        if len(dataset_parts) >= 2:
            organization, dataset_name = dataset_parts[0], dataset_parts[1]
            resolved = _resolve_dataset_id(base_url, organization, dataset_name, fetch_json)
            return f"{base_url}/api/datasets/{resolved}"
        if len(dataset_parts) == 1:
            dataset_name = dataset_parts[0]
            organization = _resolve_organization(base_url, dataset_name, fetch_json)
            resolved = _resolve_dataset_id(base_url, organization, dataset_name, fetch_json)
            return f"{base_url}/api/datasets/{resolved}"
    raise ValueError(f"unsupported webKnossos dataset seed URL: {seed_url}")


def _dataset_id_from_link_page(seed_url: str, fetch_text: FetchText) -> str:
    html = fetch_text(seed_url)
    matches = re.findall(r"/api/datasets/([0-9a-f]{24})(?:/|[?\"'&<])", html, flags=re.IGNORECASE)
    if not matches:
        matches = re.findall(r"api/datasets/([0-9a-f]{24})", html, flags=re.IGNORECASE)
    if not matches:
        raise ValueError(f"webKnossos link page did not expose a dataset id: {seed_url}")
    return matches[0]


def _get_text(url: str, timeout: int = 30, max_bytes: int = 500_000) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raise HttpFetchError(url, f"HTML response exceeded {max_bytes} bytes")
            charset = response.headers.get_content_charset() or "utf-8"
            return raw.decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        excerpt = exc.read(1000).decode("utf-8", errors="replace")
        raise HttpFetchError(url, f"HTTP {exc.code}", exc.code, excerpt) from exc
    except urllib.error.URLError as exc:
        raise HttpFetchError(url, str(exc.reason)) from exc


def _strip_dataset_suffixes(parts: list[str]) -> list[str]:
    suffixes = {"view", "edit", "sandbox", "createExplorative"}
    if len(parts) > 2 and parts[2] in suffixes:
        return parts[:2]
    if len(parts) > 1 and parts[1] in suffixes:
        return parts[:1]
    return parts


def _resolve_organization(base_url: str, dataset_name: str, fetch_json: Callable[[str], Any]) -> str:
    quoted_name = urllib.parse.quote(dataset_name, safe="")
    url = f"{base_url}/api/datasets/disambiguate/{quoted_name}/toNew"
    payload = fetch_json(url)
    if not isinstance(payload, dict) or not payload.get("organization"):
        raise ValueError("webKnossos disambiguation response did not include organization")
    return str(payload["organization"])


def _resolve_dataset_id(base_url: str, organization: str, dataset_name: str, fetch_json: Callable[[str], Any]) -> str:
    quoted_org = urllib.parse.quote(organization, safe="")
    quoted_name = urllib.parse.quote(dataset_name, safe="")
    url = f"{base_url}/api/datasets/disambiguate/{quoted_org}/{quoted_name}/toId"
    payload = fetch_json(url)
    if not isinstance(payload, dict) or not payload.get("id"):
        raise ValueError("webKnossos disambiguation response did not include dataset id")
    return str(payload["id"])


def _candidate_from_dataset_payload(api_url: str, payload: Any) -> Candidate:
    if not isinstance(payload, dict):
        raise ValueError("expected webKnossos dataset metadata object")
    dataset_id = _required_str(payload, "id")
    name = _required_str(payload, "name")
    organization = _required_str(payload, "owningOrganization")
    data_source = payload.get("dataSource")
    if not isinstance(data_source, dict):
        raise ValueError("webKnossos dataset metadata missing dataSource object")

    metadata = _metadata_map(payload.get("metadata"))
    layers = data_source.get("dataLayers") or []
    if not isinstance(layers, list):
        raise ValueError("webKnossos dataset metadata dataLayers is not a list")

    title = _first(metadata, "title", "datasetTitle") or name
    modality = _first(metadata, "modality", "acquisition", "microscopy", "technique", "imaging")
    organism = _first(metadata, "organism", "species", "organismName")
    tissue = _first(metadata, "tissue", "tissueOrSample", "sample", "sampleType", "organ", "brainRegion", "region")
    publication_doi = _first(metadata, "publicationDoi", "publicationDOI", "paperDoi", "paperDOI")
    dataset_doi = _first(metadata, "datasetDoi", "datasetDOI", "doi", "DOI")
    license_value = _first(metadata, "license", "licence")
    formats = sorted({str(layer.get("dataFormat")) for layer in layers if isinstance(layer, dict) and layer.get("dataFormat")})

    return Candidate(
        source_name=SOURCE_NAME,
        source_record_id=f"{organization}/{name}",
        title=title,
        landing_url=_landing_url(_base_url_from_api_url(api_url), organization, name),
        publication_doi=publication_doi,
        dataset_doi=dataset_doi,
        modality=modality,
        organism=organism,
        tissue_or_sample=tissue,
        dimensions_or_image_count=_dimensions_text(data_source),
        file_formats=formats,
        license=license_value,
        raw_metadata=payload,
        evidence_text=_evidence_text(payload, metadata, layers, dataset_id, api_url),
    )


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"webKnossos dataset metadata missing {key!r}")
    return str(value)


def _metadata_map(metadata: Any) -> dict[str, str]:
    values: dict[str, str] = {}
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if value is not None and str(value).strip():
                values[_normalize_key(str(key))] = str(value).strip()
        return values
    if isinstance(metadata, list):
        for item in metadata:
            if not isinstance(item, dict):
                continue
            key = item.get("key") or item.get("name")
            value = item.get("value")
            if key is not None and value is not None and str(value).strip():
                values[_normalize_key(str(key))] = str(value).strip()
    return values


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _first(metadata: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = metadata.get(_normalize_key(key))
        if value:
            return value
    return None


def _dimensions_text(data_source: dict[str, Any]) -> str | None:
    layers = data_source.get("dataLayers") or []
    bounding_box = None
    if isinstance(layers, list):
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            if layer.get("category") == "color" and isinstance(layer.get("boundingBox"), dict):
                bounding_box = layer["boundingBox"]
                break
        if bounding_box is None:
            for layer in layers:
                if isinstance(layer, dict) and isinstance(layer.get("boundingBox"), dict):
                    bounding_box = layer["boundingBox"]
                    break

    pieces: list[str] = []
    if bounding_box:
        width = bounding_box.get("width")
        height = bounding_box.get("height")
        depth = bounding_box.get("depth")
        if width and height and depth:
            pieces.append(f"bounding box {width}x{height}x{depth} voxels")
    scale = data_source.get("scale")
    if isinstance(scale, dict) and isinstance(scale.get("factor"), list):
        factor = "/".join(str(value) for value in scale["factor"])
        unit = scale.get("unit")
        pieces.append(f"voxel size {factor} {unit}".strip())
    return "; ".join(pieces) or None


def _evidence_text(
    payload: dict[str, Any],
    metadata: dict[str, str],
    layers: list[Any],
    dataset_id: str,
    api_url: str,
) -> str:
    layer_names = [
        str(layer.get("name"))
        for layer in layers
        if isinstance(layer, dict) and layer.get("name")
    ][:8]
    metadata_bits = [
        f"{key}={metadata[key]}"
        for key in sorted(metadata)
        if key in {"acquisition", "modality", "microscopy", "technique", "species", "organism", "brainregion", "tissue", "sample"}
    ]
    description = payload.get("description")
    evidence = [
        f"webKnossos dataset id {dataset_id}",
        *metadata_bits,
    ]
    if layer_names:
        evidence.append(f"layers: {', '.join(layer_names)}")
    if description:
        evidence.append(str(description))
    evidence.append(f"metadata endpoint: {api_url}")
    return " ; ".join(evidence)


def _landing_url(base_url: str, organization: str, name: str) -> str:
    quoted_org = urllib.parse.quote(organization, safe="")
    quoted_name = urllib.parse.quote(name, safe="")
    return f"{base_url}/datasets/{quoted_org}/{quoted_name}"


def _base_url_from_api_url(api_url: str) -> str:
    parsed = urllib.parse.urlparse(api_url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return DEFAULT_BASE_URL


def _is_before_since(candidate: Candidate, since: str) -> bool:
    raw = candidate.raw_metadata if isinstance(candidate.raw_metadata, dict) else {}
    value = raw.get("updated") or raw.get("created")
    if value is None:
        return False
    if isinstance(value, (int, float)):
        # webKnossos reports numeric created/updated timestamps as epoch milliseconds.
        from datetime import datetime, timezone

        date = datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc).date().isoformat()
    else:
        date = str(value)[:10]
    return bool(date and date < since)


def _scanner_error(seed_url: str, exc: Exception) -> ScannerError:
    status = exc.status if isinstance(exc, HttpFetchError) else None
    excerpt = exc.excerpt if isinstance(exc, HttpFetchError) else None
    error_url = exc.url if isinstance(exc, HttpFetchError) else seed_url
    return ScannerError(
        source_name=SOURCE_NAME,
        adapter=ADAPTER_NAME,
        url=error_url,
        response_status=status,
        response_excerpt=excerpt,
        error_type=type(exc).__name__,
        message=str(exc),
        stack_trace=traceback.format_exc(),
        reproduction_command="python scripts/scan_sources.py --source webknossos",
    )
