"""Generic metadata search adapters for public data portals."""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urlparse

from ..http import get_json, get_text, post_json, urlencode
from ..models import Candidate
from .base import ScannerResult, safe_collect, unique_candidates


HIGH_RECALL_TERMS = [
    "TEM",
    "FIB-SEM",
    "SBF-SEM",
    "serial block face",
    "ultrastructure",
    "organelle",
    "mitochondria",
    "ER",
    "nucleus",
    "electron microscopy dataset",
    "volume electron microscopy",
    "serial section TEM",
    "cellular electron microscopy",
]
DEFAULT_TERMS = " ".join(f'"{term}"' if " " in term else term for term in HIGH_RECALL_TERMS)
PORTAL_QUERY_GROUPS = [
    "electron microscopy dataset intracellular ultrastructure mitochondria",
    "FIB-SEM ultrastructure mitochondria",
    "SBF-SEM serial block face ultrastructure",
    "TEM organelle ultrastructure mitochondria",
    "electron microscopy ER nucleus organelle",
    "volume electron microscopy cellular dataset repository",
    "serial section TEM connectomics intracellular ultrastructure",
    "electron microscopy segmentation annotation organelle dataset",
    "cellular electron microscopy intracellular dataset repository",
]
FIGSHARE_SEARCH_URL = "https://api.figshare.com/v2/articles/search"
FIGSHARE_OAI_URL = "https://api.figshare.com/v2/oai"
FIGSHARE_OAI_METADATA_PREFIX = "oai_datacite"
DATACITE_DOIS_URL = "https://api.datacite.org/dois"
DATAVERSE_SEARCH_URL = "https://dataverse.harvard.edu/api/search"
MENDELEY_CLIENT_IDS = ("bl.mendeley", "elsevier.md")
MAX_FILE_METADATA = 25
OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "d": "http://datacite.org/schema/kernel-4",
    "dc": "http://purl.org/dc/elements/1.1/",
}
FIGSHARE_OAI_STRONG_EM_PATTERNS = [
    r"\belectron microscopy\b",
    r"\belectron micrographs?\b",
    r"\belectron microscope\b",
    r"\btransmission electron\b",
    r"\bscanning electron\b",
    r"\bvolume electron microscopy\b",
    r"\bvolume em\b",
    r"\bfocused ion beam\b",
    r"\bfib[- ]?sem\b",
    r"\bsbf[- ]?sem\b",
    r"\bsbem\b",
    r"\bserial block[- ]face\b",
    r"\bserial section(?:ing)?\b",
    r"\barray tomography\b",
    r"\bultrastructure\b",
]
FIGSHARE_OAI_TEM_CONTEXT_TERMS = [
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
FIGSHARE_OAI_SEM_CONTEXT_TERMS = [
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

SOURCE_CONFIG = {
    "zenodo": {
        "url": "https://zenodo.org/api/records",
        "params": {"q": DEFAULT_TERMS, "sort": "mostrecent", "size": 25},
        "query_param": "q",
        "limit_param": "size",
        "max_page_size": 25,
        "next_url_path": ("links", "next"),
        "items_path": ("hits", "hits"),
        "id": "id",
        "title": ("metadata", "title"),
        "landing_url": "links.html",
        "doi": "doi",
    },
    "huggingface": {
        "url": "https://huggingface.co/api/datasets",
        "params": {"search": "electron microscopy", "limit": 50},
        "query_param": "search",
        "limit_param": "limit",
        "max_page_size": 1000,
        "items_path": (),
        "id": "id",
        "title": "id",
        "landing_url_prefix": "https://huggingface.co/datasets/",
    },
    "biostudies": {
        "url": "https://www.ebi.ac.uk/biostudies/api/v1/search",
        "params": {"query": DEFAULT_TERMS, "pageSize": 50},
        "query_param": "query",
        "limit_param": "pageSize",
        "max_page_size": 100,
        "items_path": ("hits",),
        "id": "accession",
        "title": "title",
        "landing_url_prefix": "https://www.ebi.ac.uk/biostudies/studies/",
    },
    "dryad": {
        "url": "https://datadryad.org/api/v2/search",
        "params": {"q": DEFAULT_TERMS, "per_page": 50},
        "query_param": "q",
        "limit_param": "per_page",
        "max_page_size": 100,
        "next_url_path": ("_links", "next", "href"),
        "items_path": ("_embedded", "stash:datasets"),
        "id": "identifier",
        "title": "title",
        "landing_url": "_links.stash:datasets.href",
    },
}



def scan_generic_source(
    source: str,
    query: str | None = None,
    limit: int = 50,
    since: str | None = None,
    full_history: bool = False,
    cursor: dict[str, Any] | None = None,
    **_: Any,
) -> ScannerResult:
    return safe_collect(source, f"{source}_adapter", _scan_generic_source, source, query, limit, since, full_history, cursor)


def _scan_generic_source(
    source: str,
    query: str | None,
    limit: int,
    since: str | None,
    full_history: bool,
    cursor: dict[str, Any] | None,
) -> ScannerResult:
    active_since = None if full_history else since
    if cursor and cursor.get("complete"):
        return ScannerResult(cursor=cursor, cursor_complete=True)
    if source == "figshare":
        return _scan_figshare([query] if query else PORTAL_QUERY_GROUPS, limit, active_since, cursor)
    if source == "figshare_oai":
        return _scan_figshare_oai(limit, active_since, cursor)
    if source == "mendeley":
        return _scan_mendeley([query] if query else PORTAL_QUERY_GROUPS, limit, active_since, cursor)
    if source == "datacite":
        return _scan_datacite([query] if query else PORTAL_QUERY_GROUPS, limit, active_since, cursor)
    if source == "dataverse":
        return _scan_dataverse([query] if query else PORTAL_QUERY_GROUPS, limit, cursor)
    if source in SOURCE_CONFIG:
        config = dict(SOURCE_CONFIG[source])
        return _scan_configured_source(source, config, [query] if query else PORTAL_QUERY_GROUPS, limit, active_since, cursor)
    raise ValueError(f"no generic adapter for source {source}")


def _scan_configured_source(
    source: str,
    config: dict[str, Any],
    queries: list[str],
    limit: int,
    since: str | None = None,
    cursor: dict[str, Any] | None = None,
) -> ScannerResult:
    candidates: list[Candidate] = []
    limit_param = config.get("limit_param")
    query_param = config.get("query_param")
    page_size = max(1, min(limit, int(config.get("max_page_size") or limit or 1)))
    query_index = _cursor_int(cursor, "query_index", 0)
    next_url = str(cursor.get("next_url")) if isinstance(cursor, dict) and cursor.get("next_url") else None
    while query_index < len(queries):
        query = queries[query_index]
        page = _cursor_int(cursor, "page", 1) if not next_url else 1
        while len(candidates) < limit:
            if next_url:
                data = get_json(next_url)
            else:
                params = dict(config.get("params") or {})
                if query and query_param:
                    params[str(query_param)] = _query_with_since(source, query, since)
                if limit_param:
                    params[str(limit_param)] = page_size
                if config.get("page_param"):
                    params[str(config["page_param"])] = page
                data = get_json(f"{config['url']}?{urlencode(params)}")
            items = _get_path(data, config.get("items_path") or ()) or []
            if not isinstance(items, list):
                raise ValueError(f"{source} search payload items field is not a list")
            if not items:
                query_index += 1
                next_url = None
                break
            candidates.extend(_candidate_from_generic(source, config, item) for item in items)
            candidates = unique_candidates(candidates)
            next_url = _get_path(data, config.get("next_url_path")) if config.get("next_url_path") else None
            if len(candidates) >= limit:
                complete_after_page = not next_url and query_index + 1 >= len(queries)
                next_cursor = (
                    {
                        "adapter": "configured",
                        "source": source,
                        "query_index": query_index,
                        "page": page + 1,
                        "next_url": next_url,
                    }
                    if next_url
                    else {
                        "adapter": "configured",
                        "source": source,
                        "query_index": query_index + 1,
                        "page": 1,
                    }
                )
                return _cursor_result(
                    candidates[:limit],
                    complete=complete_after_page,
                    cursor=next_cursor,
                )
            if next_url:
                continue
            if len(items) < page_size:
                query_index += 1
                page = 1
                break
            page += 1
        else:
            continue
    return _cursor_result(unique_candidates(candidates)[:limit], complete=True, cursor={"complete": True})


def _scan_figshare(
    queries: list[str],
    limit: int,
    since: str | None = None,
    cursor: dict[str, Any] | None = None,
) -> ScannerResult:
    candidates: list[Candidate] = []
    page_size = max(1, min(limit, 100))
    query_index = _cursor_int(cursor, "query_index", 0)
    page = _cursor_int(cursor, "page", 1)
    while query_index < len(queries):
        query = queries[query_index]
        while len(candidates) < limit:
            search_payload = {
                "search_for": query,
                "item_type": 3,
                "page_size": page_size,
                "page": page,
            }
            if since:
                search_payload["published_since"] = since
            data = post_json(FIGSHARE_SEARCH_URL, search_payload)
            if not isinstance(data, list):
                raise ValueError("figshare search payload is not a list")
            if not data:
                query_index += 1
                page = 1
                break
            for item in data:
                if not isinstance(item, dict):
                    raise ValueError("figshare search result is not an object")
                record_id = item.get("id")
                if record_id is None:
                    raise ValueError("figshare search result missing id")
                detail_url = item.get("url_public_api") or f"https://api.figshare.com/v2/articles/{record_id}"
                detail = get_json(str(detail_url))
                if not isinstance(detail, dict):
                    raise ValueError(f"figshare article {record_id} payload is not an object")
                candidates.append(_candidate_from_figshare_article(item, detail))
                candidates = unique_candidates(candidates)
                if len(candidates) >= limit:
                    return _cursor_result(
                        candidates[:limit],
                        complete=False,
                        cursor={"adapter": "figshare", "query_index": query_index, "page": page + 1},
                    )
            if len(data) < page_size:
                query_index += 1
                page = 1
                break
            page += 1
        else:
            continue
    return _cursor_result(unique_candidates(candidates)[:limit], complete=True, cursor={"complete": True})


def _scan_figshare_oai(
    limit: int,
    since: str | None = None,
    cursor: dict[str, Any] | None = None,
) -> ScannerResult:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    candidates: list[Candidate] = []
    records_examined = 0
    metadata_prefix = str((cursor or {}).get("metadata_prefix") or FIGSHARE_OAI_METADATA_PREFIX)
    resumption_token = str((cursor or {}).get("resumption_token") or "") or None
    initial_from = str((cursor or {}).get("from") or since or "") or None
    last_response: dict[str, Any] | None = None

    while records_examined < limit:
        url = _figshare_oai_url(
            metadata_prefix=metadata_prefix,
            since=initial_from,
            resumption_token=resumption_token,
        )
        response = _parse_figshare_oai_response(get_text(url))
        last_response = response
        records = response["records"]
        resumption_token = response.get("resumption_token")
        if not records:
            break
        for record in records:
            records_examined += 1
            if record.get("deleted"):
                continue
            candidate = _candidate_from_figshare_oai_record(record)
            if candidate and _figshare_oai_record_is_relevant(record):
                candidates.append(candidate)
        if not resumption_token:
            break

    complete = not resumption_token
    cursor_payload = {
        "adapter": "figshare_oai",
        "metadata_prefix": metadata_prefix,
        "resumption_token": resumption_token,
        "from": initial_from,
        "records_examined_in_batch": records_examined,
        "response_date": (last_response or {}).get("response_date"),
        "resumption": (last_response or {}).get("resumption"),
    }
    return _cursor_result(unique_candidates(candidates), complete=complete, cursor=cursor_payload)


def _figshare_oai_url(
    *,
    metadata_prefix: str,
    since: str | None,
    resumption_token: str | None,
) -> str:
    params: dict[str, Any] = {"verb": "ListRecords"}
    if resumption_token:
        params["resumptionToken"] = resumption_token
    else:
        params["metadataPrefix"] = metadata_prefix
        if since:
            params["from"] = since
    return f"{FIGSHARE_OAI_URL}?{urlencode(params)}"


def _parse_figshare_oai_response(raw_xml: str) -> dict[str, Any]:
    root = ET.fromstring(raw_xml)
    error = root.find("oai:error", OAI_NS)
    if error is not None:
        code = error.get("code") or "unknown"
        message = _xml_text(error) or code
        if code == "noRecordsMatch":
            return {
                "records": [],
                "resumption_token": None,
                "resumption": None,
                "response_date": _xml_text(root.find("oai:responseDate", OAI_NS)),
            }
        raise ValueError(f"Figshare OAI error {code}: {message}")
    list_records = root.find("oai:ListRecords", OAI_NS)
    if list_records is None:
        raise ValueError("Figshare OAI response missing ListRecords")
    token_node = list_records.find("oai:resumptionToken", OAI_NS)
    token = _xml_text(token_node)
    return {
        "records": [_figshare_oai_record_from_node(node) for node in list_records.findall("oai:record", OAI_NS)],
        "resumption_token": token,
        "resumption": _figshare_oai_resumption_metadata(token_node),
        "response_date": _xml_text(root.find("oai:responseDate", OAI_NS)),
    }


def _figshare_oai_resumption_metadata(node: ET.Element | None) -> dict[str, str] | None:
    if node is None:
        return None
    metadata = {
        key: value
        for key, value in {
            "expiration_date": node.get("expirationDate"),
            "complete_list_size": node.get("completeListSize"),
            "cursor": node.get("cursor"),
        }.items()
        if value
    }
    return metadata or None


def _figshare_oai_record_from_node(record_node: ET.Element) -> dict[str, Any]:
    header = record_node.find("oai:header", OAI_NS)
    if header is None:
        raise ValueError("Figshare OAI record missing header")
    oai_identifier = _xml_text(header.find("oai:identifier", OAI_NS))
    metadata_node = record_node.find("oai:metadata", OAI_NS)
    record: dict[str, Any] = {
        "oai_identifier": oai_identifier,
        "article_id": _figshare_article_id(oai_identifier),
        "datestamp": _xml_text(header.find("oai:datestamp", OAI_NS)),
        "set_specs": [_xml_text(node) for node in header.findall("oai:setSpec", OAI_NS) if _xml_text(node)],
        "deleted": header.get("status") == "deleted",
        "metadata_prefix": None,
    }
    if record["deleted"] or metadata_node is None:
        return record
    datacite = metadata_node.find("d:resource", OAI_NS)
    if datacite is None:
        datacite = metadata_node.find(".//d:resource", OAI_NS)
    if datacite is not None:
        record.update(_figshare_oai_datacite_metadata(datacite))
        return record
    dc = _figshare_oai_dc_node(metadata_node)
    if dc is not None:
        record.update(_figshare_oai_dc_metadata(dc))
        return record
    raise ValueError(f"Figshare OAI record {oai_identifier or '<unknown>'} missing supported metadata")


def _figshare_oai_dc_node(metadata_node: ET.Element) -> ET.Element | None:
    for child in list(metadata_node):
        if child.find("dc:title", OAI_NS) is not None:
            return child
    return None


def _figshare_oai_datacite_metadata(resource: ET.Element) -> dict[str, Any]:
    titles = [_xml_text(node) for node in resource.findall("d:titles/d:title", OAI_NS)]
    descriptions = [_xml_text(node) for node in resource.findall("d:descriptions/d:description", OAI_NS)]
    subjects = [_xml_text(node) for node in resource.findall("d:subjects/d:subject", OAI_NS)]
    formats = [_xml_text(node) for node in resource.findall("d:formats/d:format", OAI_NS)]
    resource_type_node = resource.find("d:resourceType", OAI_NS)
    alternate_identifiers = [
        {
            "type": node.get("alternateIdentifierType"),
            "identifier": _xml_text(node),
        }
        for node in resource.findall("d:alternateIdentifiers/d:alternateIdentifier", OAI_NS)
    ]
    related_identifiers = [
        {
            "identifier": _xml_text(node),
            "identifier_type": node.get("relatedIdentifierType"),
            "relation_type": node.get("relationType"),
            "resource_type_general": node.get("resourceTypeGeneral"),
        }
        for node in resource.findall("d:relatedIdentifiers/d:relatedIdentifier", OAI_NS)
    ]
    rights = [
        {
            "rights": _xml_text(node),
            "rights_uri": node.get("rightsURI") or node.get("rightsUri"),
            "rights_identifier": node.get("rightsIdentifier"),
        }
        for node in resource.findall("d:rightsList/d:rights", OAI_NS)
    ]
    return {
        "metadata_prefix": "oai_datacite",
        "title": next((item for item in titles if item), None),
        "titles": [item for item in titles if item],
        "description": _evidence_text(descriptions),
        "descriptions": [item for item in descriptions if item],
        "subjects": [item for item in subjects if item],
        "formats": [item for item in formats if item],
        "dataset_doi": _xml_text(resource.find("d:identifier[@identifierType='DOI']", OAI_NS)),
        "alternate_identifiers": alternate_identifiers,
        "related_identifiers": related_identifiers,
        "landing_url": _figshare_oai_landing_url(alternate_identifiers),
        "rights": rights,
        "creators": [
            _xml_text(node.find("d:creatorName", OAI_NS))
            for node in resource.findall("d:creators/d:creator", OAI_NS)[:20]
            if _xml_text(node.find("d:creatorName", OAI_NS))
        ],
        "resource_type_general": resource_type_node.get("resourceTypeGeneral") if resource_type_node is not None else None,
        "resource_type": _xml_text(resource_type_node),
        "publisher": _xml_text(resource.find("d:publisher", OAI_NS)),
        "publication_year": _xml_text(resource.find("d:publicationYear", OAI_NS)),
        "dates": [
            {"date": _xml_text(node), "date_type": node.get("dateType")}
            for node in resource.findall("d:dates/d:date", OAI_NS)
        ],
    }


def _figshare_oai_dc_metadata(dc_node: ET.Element) -> dict[str, Any]:
    identifiers = [_xml_text(node) for node in dc_node.findall("dc:identifier", OAI_NS)]
    relations = [_xml_text(node) for node in dc_node.findall("dc:relation", OAI_NS)]
    rights = [{"rights": _xml_text(node), "rights_uri": None, "rights_identifier": None} for node in dc_node.findall("dc:rights", OAI_NS)]
    return {
        "metadata_prefix": "oai_dc",
        "title": _xml_text(dc_node.find("dc:title", OAI_NS)),
        "titles": [_xml_text(node) for node in dc_node.findall("dc:title", OAI_NS) if _xml_text(node)],
        "description": _evidence_text([_xml_text(node) for node in dc_node.findall("dc:description", OAI_NS)]),
        "descriptions": [_xml_text(node) for node in dc_node.findall("dc:description", OAI_NS) if _xml_text(node)],
        "subjects": [_xml_text(node) for node in dc_node.findall("dc:subject", OAI_NS) if _xml_text(node)],
        "formats": [_xml_text(node) for node in dc_node.findall("dc:format", OAI_NS) if _xml_text(node)],
        "dataset_doi": _first_doi(identifiers),
        "alternate_identifiers": [{"type": "URL", "identifier": value} for value in identifiers if value and value.startswith("http")],
        "related_identifiers": [{"identifier": value, "identifier_type": "URL", "relation_type": "Relation"} for value in relations if value],
        "landing_url": next((value for value in relations + identifiers if value and value.startswith("http")), None),
        "rights": rights,
        "creators": [_xml_text(node) for node in dc_node.findall("dc:creator", OAI_NS)[:20] if _xml_text(node)],
        "resource_type_general": None,
        "resource_type": " ".join(_xml_text(node) or "" for node in dc_node.findall("dc:type", OAI_NS)).strip() or None,
        "publisher": _xml_text(dc_node.find("dc:publisher", OAI_NS)),
        "publication_year": None,
        "dates": [{"date": _xml_text(node), "date_type": "DC"} for node in dc_node.findall("dc:date", OAI_NS)],
    }


def _candidate_from_figshare_oai_record(record: dict[str, Any]) -> Candidate | None:
    record_id = record.get("article_id") or record.get("oai_identifier") or record.get("dataset_doi")
    if not record_id:
        return None
    title = record.get("title") or record_id
    article_id = record.get("article_id")
    api_url = f"https://api.figshare.com/v2/articles/{article_id}" if article_id else None
    landing_url = record.get("landing_url") or (_doi_url(record.get("dataset_doi")) if record.get("dataset_doi") else None)
    related_file_urls = _figshare_oai_related_file_urls(record)
    return Candidate(
        source_name="figshare_oai",
        source_record_id=str(record_id),
        title=str(title),
        landing_url=str(landing_url) if landing_url else None,
        download_or_manifest_urls=[str(url) for url in [api_url] if url],
        publication_doi=_figshare_oai_publication_doi(record),
        dataset_doi=str(record["dataset_doi"]) if record.get("dataset_doi") else None,
        dimensions_or_image_count=f"{len(related_file_urls)} related file URLs" if related_file_urls else None,
        file_formats=_file_formats_from_files([], extra_formats=record.get("formats")),
        license=_figshare_oai_license(record.get("rights") or []),
        raw_metadata={"oai": _compact_figshare_oai_record(record), "candidate_source": "figshare_oai_datestamp_harvest"},
        evidence_text=_evidence_text(
            [
                title,
                record.get("description"),
                " ".join(str(subject) for subject in record.get("subjects") or []),
                " ".join(str(fmt) for fmt in record.get("formats") or []),
                record.get("resource_type_general"),
                record.get("resource_type"),
                record.get("publisher"),
                " ".join(str(item.get("identifier") or "") for item in record.get("related_identifiers") or [] if isinstance(item, dict)),
            ]
        ),
        discovered_at=_figshare_oai_discovered_at(record),
    )


def _figshare_oai_record_is_relevant(record: dict[str, Any]) -> bool:
    text = _normalize_text_for_matching(
        record.get("title"),
        record.get("description"),
        " ".join(str(item) for item in record.get("subjects") or []),
        " ".join(str(item) for item in record.get("formats") or []),
        record.get("resource_type_general"),
        record.get("resource_type"),
        record.get("publisher"),
        " ".join(str(item.get("identifier") or "") for item in record.get("related_identifiers") or [] if isinstance(item, dict)),
    )
    if not text:
        return False
    if any(re.search(pattern, text) for pattern in FIGSHARE_OAI_STRONG_EM_PATTERNS):
        return True
    if re.search(r"\btem\b", text) and any(term in text for term in FIGSHARE_OAI_TEM_CONTEXT_TERMS):
        return True
    if re.search(r"\bsem\b", text) and any(term in text for term in FIGSHARE_OAI_SEM_CONTEXT_TERMS):
        return True
    if re.search(r"\bvem\b", text) and any(term in text for term in FIGSHARE_OAI_TEM_CONTEXT_TERMS + FIGSHARE_OAI_SEM_CONTEXT_TERMS):
        return True
    return False


def _figshare_oai_landing_url(alternate_identifiers: list[dict[str, str | None]]) -> str | None:
    for item in alternate_identifiers:
        value = item.get("identifier")
        if value and "figshare.com/articles/" in value:
            return value
    for item in alternate_identifiers:
        value = item.get("identifier")
        if value and value.startswith("http"):
            return value
    return None


def _figshare_oai_publication_doi(record: dict[str, Any]) -> str | None:
    dataset_doi = str(record.get("dataset_doi") or "").lower()
    for item in record.get("related_identifiers") or []:
        if not isinstance(item, dict):
            continue
        identifier = item.get("identifier")
        if not identifier or str(item.get("identifier_type") or "").upper() != "DOI":
            continue
        doi = str(identifier).strip()
        relation = str(item.get("relation_type") or "")
        if doi.lower() == dataset_doi or "figshare" in doi.lower():
            continue
        if relation in {"IsSupplementTo", "References", "IsReferencedBy", "IsDerivedFrom", "Cites"}:
            return doi
    return None


def _figshare_oai_license(rights: list[Any]) -> str | None:
    for item in rights:
        if not isinstance(item, dict):
            continue
        label = item.get("rights") or item.get("rights_identifier")
        uri = item.get("rights_uri")
        if label and uri:
            return f"{label} ({uri})"
        if label:
            return str(label)
        if uri:
            return str(uri)
    return None


def _figshare_oai_related_file_urls(record: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for item in record.get("related_identifiers") or []:
        if not isinstance(item, dict):
            continue
        identifier = item.get("identifier")
        if not identifier or str(item.get("identifier_type") or "").upper() != "URL":
            continue
        relation = str(item.get("relation_type") or "")
        if relation in {"HasPart", "HasMetadata"} or "ndownloader.figshare.com/files/" in str(identifier):
            urls.append(str(identifier))
    return urls


def _figshare_oai_discovered_at(record: dict[str, Any]) -> str | None:
    for preferred in ("Updated", "Issued", "Created", "Available"):
        for item in record.get("dates") or []:
            if isinstance(item, dict) and item.get("date_type") == preferred and item.get("date"):
                return str(item["date"])
    for item in record.get("dates") or []:
        if isinstance(item, dict) and item.get("date"):
            return str(item["date"])
    return str(record["datestamp"]) if record.get("datestamp") else None


def _compact_figshare_oai_record(record: dict[str, Any]) -> dict[str, Any]:
    compact = dict(record)
    related = []
    for item in compact.get("related_identifiers") or []:
        if isinstance(item, dict):
            related.append(dict(item))
    compact["related_identifiers"] = related[:MAX_FILE_METADATA]
    if len(related) > MAX_FILE_METADATA:
        compact["related_identifiers"].append({"omitted_related_identifier_count": len(related) - MAX_FILE_METADATA})
    return compact


def _figshare_article_id(value: Any) -> str | None:
    if not value:
        return None
    match = re.search(r"(?:article/|articles/)(\d+)", str(value))
    return match.group(1) if match else None


def _first_doi(values: list[str | None]) -> str | None:
    for value in values:
        if not value:
            continue
        match = re.search(r"(10\.\d{4,9}/[^\s\"<>]+)", str(value), flags=re.IGNORECASE)
        if match:
            return match.group(1).rstrip(".,);]")
    return None


def _xml_text(node: ET.Element | None) -> str | None:
    if node is None:
        return None
    text = " ".join(part.strip() for part in node.itertext() if part and part.strip())
    if not text:
        return None
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def _normalize_text_for_matching(*parts: Any) -> str:
    return re.sub(r"\s+", " ", " ".join(_clean_fragment(part) for part in parts if part)).strip().lower()


def _scan_mendeley(
    queries: list[str],
    limit: int,
    since: str | None = None,
    cursor: dict[str, Any] | None = None,
) -> ScannerResult:
    candidates: list[Candidate] = []
    page_size = max(1, min(limit, 100))
    query_index = _cursor_int(cursor, "query_index", 0)
    client_index = _cursor_int(cursor, "client_index", 0)
    page = _cursor_int(cursor, "page", 1)
    while query_index < len(queries):
        query = queries[query_index]
        while client_index < len(MENDELEY_CLIENT_IDS):
            client_id = MENDELEY_CLIENT_IDS[client_index]
            while len(candidates) < limit:
                params = {
                    "client-id": client_id,
                    "query": query,
                    "resource-type-id": "dataset",
                    "page[number]": page,
                    "page[size]": page_size,
                    "detail": "true",
                }
                if since:
                    params["from-updated-date"] = since
                data = get_json(f"{DATACITE_DOIS_URL}?{urlencode(params)}")
                items = data.get("data") if isinstance(data, dict) else None
                if not isinstance(items, list):
                    raise ValueError("DataCite DOI search payload missing data list")
                if not items:
                    client_index += 1
                    page = 1
                    break
                for item in items:
                    if not isinstance(item, dict):
                        raise ValueError("DataCite DOI result is not an object")
                    public_detail = None
                    public_api_url = _mendeley_public_api_url(item)
                    if public_api_url:
                        detail = get_json(public_api_url)
                        if not isinstance(detail, dict):
                            raise ValueError(f"Mendeley public API payload is not an object for {public_api_url}")
                        public_detail = detail
                    candidates.append(_candidate_from_mendeley_datacite(item, public_detail, public_api_url))
                    candidates = unique_candidates(candidates)
                    if len(candidates) >= limit:
                        return _cursor_result(
                            candidates[:limit],
                            complete=False,
                            cursor={
                                "adapter": "mendeley",
                                "query_index": query_index,
                                "client_index": client_index,
                                "page": page + 1,
                            },
                        )
                if len(items) < page_size:
                    client_index += 1
                    page = 1
                    break
                page += 1
            else:
                continue
        query_index += 1
        client_index = 0
        page = 1
    return _cursor_result(unique_candidates(candidates)[:limit], complete=True, cursor={"complete": True})


def _scan_datacite(
    queries: list[str],
    limit: int,
    since: str | None = None,
    cursor: dict[str, Any] | None = None,
) -> ScannerResult:
    candidates: list[Candidate] = []
    page_size = max(1, min(limit, 1000))
    query_index = _cursor_int(cursor, "query_index", 0)
    page = _cursor_int(cursor, "page", 1)
    while query_index < len(queries):
        query = queries[query_index]
        while len(candidates) < limit:
            params = {
                "query": query,
                "resource-type-id": "dataset",
                "page[number]": page,
                "page[size]": page_size,
                "detail": "true",
            }
            if since:
                params["from-updated-date"] = since
            data = get_json(f"{DATACITE_DOIS_URL}?{urlencode(params)}")
            items = data.get("data") if isinstance(data, dict) else None
            if not isinstance(items, list):
                raise ValueError("DataCite DOI search payload missing data list")
            if not items:
                query_index += 1
                page = 1
                break
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("DataCite DOI result is not an object")
                candidates.append(_candidate_from_datacite_item(item))
                candidates = unique_candidates(candidates)
                if len(candidates) >= limit:
                    return _cursor_result(
                        candidates[:limit],
                        complete=False,
                        cursor={"adapter": "datacite", "query_index": query_index, "page": page + 1},
                    )
            if len(items) < page_size:
                query_index += 1
                page = 1
                break
            page += 1
        else:
            continue
    return _cursor_result(unique_candidates(candidates)[:limit], complete=True, cursor={"complete": True})


def _query_with_since(source: str, query: str, since: str | None) -> str:
    if not since:
        return query
    if source == "zenodo":
        return f"({query}) AND updated:[{since} TO *]"
    return query


def _scan_dataverse(queries: list[str], limit: int, cursor: dict[str, Any] | None = None) -> ScannerResult:
    candidates: list[Candidate] = []
    page_size = max(1, min(limit, 100))
    query_index = _cursor_int(cursor, "query_index", 0)
    start = _cursor_int(cursor, "start", 0)
    while query_index < len(queries):
        query = queries[query_index]
        while len(candidates) < limit:
            params = {
                "q": query,
                "type": "dataset",
                "per_page": page_size,
                "start": start,
            }
            data = get_json(f"{DATAVERSE_SEARCH_URL}?{urlencode(params)}")
            response = data.get("data") if isinstance(data, dict) else None
            items = response.get("items") if isinstance(response, dict) else None
            if not isinstance(items, list):
                raise ValueError("Dataverse search payload missing data.items list")
            if not items:
                query_index += 1
                start = 0
                break
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("Dataverse search result is not an object")
                candidates.append(_candidate_from_dataverse_item(item))
                candidates = unique_candidates(candidates)
                if len(candidates) >= limit:
                    return _cursor_result(
                        candidates[:limit],
                        complete=False,
                        cursor={"adapter": "dataverse", "query_index": query_index, "start": start + page_size},
                    )
            if len(items) < page_size:
                query_index += 1
                start = 0
                break
            start += page_size
        else:
            continue
    return _cursor_result(unique_candidates(candidates)[:limit], complete=True, cursor={"complete": True})


def _cursor_int(cursor: dict[str, Any] | None, key: str, default: int) -> int:
    if not isinstance(cursor, dict):
        return default
    try:
        return int(cursor.get(key, default))
    except (TypeError, ValueError):
        return default


def _cursor_result(candidates: list[Candidate], *, complete: bool, cursor: dict[str, Any]) -> ScannerResult:
    if complete:
        cursor = {"complete": True}
    else:
        cursor = {key: value for key, value in cursor.items() if value is not None}
        cursor["complete"] = False
    return ScannerResult(candidates=candidates, cursor=cursor, cursor_complete=complete)


def _candidate_from_generic(source: str, config: dict[str, Any], item: dict[str, Any]) -> Candidate:
    record_id = str(_get_path(item, config["id"]) or _get_path(item, "id") or "unknown")
    title = _get_path(item, config["title"]) if config.get("title") else None
    landing = _get_path(item, config.get("landing_url")) if config.get("landing_url") else None
    if not landing and config.get("landing_url_prefix"):
        landing = config["landing_url_prefix"] + record_id
    doi = _get_path(item, config.get("doi")) if config.get("doi") else None
    return Candidate(
        source_name=source,
        source_record_id=record_id,
        title=str(title) if title else record_id,
        landing_url=str(landing) if landing else None,
        dataset_doi=str(doi) if doi else None,
        raw_metadata=item,
        evidence_text=" ".join(str(v) for v in [title, item.get("description"), item.get("abstract")] if v),
    )


def _candidate_from_figshare_article(search_result: dict[str, Any], detail: dict[str, Any]) -> Candidate:
    record_id = detail.get("id") or search_result.get("id")
    if record_id is None:
        raise ValueError("figshare article missing id")
    files = detail.get("files") or []
    if not isinstance(files, list):
        raise ValueError(f"figshare article {record_id} files field is not a list")
    title = detail.get("title") or search_result.get("title") or str(record_id)
    landing_url = detail.get("figshare_url") or detail.get("url_public_html") or search_result.get("url_public_html")
    api_url = detail.get("url_public_api") or detail.get("url") or search_result.get("url_public_api")
    dataset_doi = detail.get("doi") or search_result.get("doi")
    publication_doi = detail.get("resource_doi") or search_result.get("resource_doi")
    evidence = _evidence_text(
        [
            title,
            detail.get("resource_title") or search_result.get("resource_title"),
            detail.get("description"),
            " ".join(str(tag) for tag in detail.get("tags") or []),
            " ".join(str(keyword) for keyword in detail.get("keywords") or []),
            " ".join(_title_from_mapping(category) for category in detail.get("categories") or []),
            " ".join(_file_name(file_info) for file_info in files[:MAX_FILE_METADATA]),
        ]
    )
    return Candidate(
        source_name="figshare",
        source_record_id=str(record_id),
        title=str(title),
        landing_url=str(landing_url) if landing_url else None,
        download_or_manifest_urls=[str(api_url)] if api_url else [],
        publication_doi=str(publication_doi) if publication_doi else None,
        dataset_doi=str(dataset_doi) if dataset_doi else None,
        dimensions_or_image_count=_file_count_summary(files, detail.get("size")),
        file_formats=_file_formats_from_files(files),
        license=_figshare_license(detail.get("license")),
        raw_metadata={
            "search_result": search_result,
            "article": _compact_figshare_article(detail),
        },
        evidence_text=evidence,
        discovered_at=detail.get("published_date") or search_result.get("published_date") or detail.get("created_date"),
    )


def _candidate_from_mendeley_datacite(
    item: dict[str, Any],
    public_detail: dict[str, Any] | None = None,
    public_api_url: str | None = None,
) -> Candidate:
    attributes = item.get("attributes")
    if not isinstance(attributes, dict):
        raise ValueError("DataCite DOI result missing attributes")
    doi = attributes.get("doi") or item.get("id")
    if not doi:
        raise ValueError("DataCite DOI result missing DOI")
    files = public_detail.get("files") if public_detail else []
    if files is None:
        files = []
    if not isinstance(files, list):
        raise ValueError(f"Mendeley public detail files field is not a list for {doi}")
    public_doi = _get_path(public_detail, ("doi", "id")) if public_detail else None
    dataset_doi = str(public_doi or doi)
    title = (public_detail or {}).get("name") or _datacite_title(attributes) or dataset_doi
    landing_url = (public_detail or {}).get("links", {}).get("view") or attributes.get("url")
    evidence = _evidence_text(
        [
            title,
            (public_detail or {}).get("description"),
            _datacite_descriptions(attributes),
            " ".join(_subject_text(subject) for subject in attributes.get("subjects") or []),
            " ".join(_title_from_mapping(category, key="label") for category in (public_detail or {}).get("categories") or []),
            " ".join(_file_name(file_info) for file_info in files[:MAX_FILE_METADATA]),
        ]
    )
    manifest_urls = [url for url in [public_api_url, landing_url] if url]
    return Candidate(
        source_name="mendeley",
        source_record_id=_mendeley_record_id(dataset_doi, public_detail, attributes),
        title=str(title),
        landing_url=str(landing_url) if landing_url else None,
        download_or_manifest_urls=[str(url) for url in manifest_urls],
        publication_doi=_mendeley_publication_doi(attributes, public_detail),
        dataset_doi=dataset_doi,
        dimensions_or_image_count=_file_count_summary(files, (public_detail or {}).get("size") if public_detail else None),
        file_formats=_file_formats_from_files(files, extra_formats=attributes.get("formats")),
        license=_mendeley_license(attributes, public_detail),
        raw_metadata={
            "datacite": _compact_datacite_item(item),
            "mendeley_public_api": _compact_mendeley_public_detail(public_detail) if public_detail else None,
        },
        evidence_text=evidence,
        discovered_at=(public_detail or {}).get("publish_date") or _datacite_issued_date(attributes) or attributes.get("created"),
    )


def _candidate_from_datacite_item(item: dict[str, Any]) -> Candidate:
    attributes = item.get("attributes")
    if not isinstance(attributes, dict):
        raise ValueError("DataCite DOI result missing attributes")
    doi = attributes.get("doi") or item.get("id")
    if not doi:
        raise ValueError("DataCite DOI result missing DOI")
    dataset_doi = str(doi)
    title = _datacite_title(attributes) or dataset_doi
    landing_url = attributes.get("url") or _doi_url(dataset_doi)
    evidence = _evidence_text(
        [
            title,
            _datacite_descriptions(attributes),
            " ".join(_subject_text(subject) for subject in attributes.get("subjects") or []),
            " ".join(str(value) for value in attributes.get("formats") or []),
            attributes.get("publisher"),
        ]
    )
    return Candidate(
        source_name="datacite",
        source_record_id=dataset_doi,
        title=str(title),
        landing_url=str(landing_url) if landing_url else None,
        download_or_manifest_urls=[str(landing_url)] if landing_url else [],
        publication_doi=_mendeley_publication_doi(attributes, None),
        dataset_doi=dataset_doi,
        file_formats=_file_formats_from_files([], extra_formats=attributes.get("formats")),
        license=_mendeley_license(attributes, None),
        raw_metadata={"datacite": _compact_datacite_item(item)},
        evidence_text=evidence,
        discovered_at=_datacite_issued_date(attributes) or attributes.get("created"),
    )


def _candidate_from_dataverse_item(item: dict[str, Any]) -> Candidate:
    persistent_url = item.get("persistentUrl")
    global_id = item.get("global_id")
    url = persistent_url or item.get("url")
    title = item.get("name") or item.get("title") or global_id or url or "Dataverse dataset"
    file_count = item.get("fileCount")
    size = item.get("size_in_bytes") or item.get("storageSize")
    return Candidate(
        source_name="dataverse",
        source_record_id=str(global_id or url or title),
        title=str(title),
        landing_url=str(url) if url else None,
        download_or_manifest_urls=[str(url)] if url else [],
        dataset_doi=_doi_from_dataverse_identifier(global_id) or _doi_from_dataverse_identifier(url),
        dimensions_or_image_count=_dataverse_file_count_summary(file_count, size),
        raw_metadata={"dataverse": item},
        evidence_text=_evidence_text(
            [
                title,
                item.get("description"),
                item.get("publisher"),
                " ".join(str(subject) for subject in item.get("subjects") or []),
                " ".join(str(author) for author in item.get("authors") or []),
            ]
        ),
        discovered_at=item.get("published_at") or item.get("createdAt"),
    )


def _get_path(data: Any, path: str | tuple[str, ...] | None) -> Any:
    if path is None:
        return None
    if isinstance(path, str):
        parts = path.split(".")
    else:
        parts = list(path)
    value = data
    for part in parts:
        if part == "":
            continue
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def _mendeley_public_api_url(item: dict[str, Any]) -> str | None:
    attributes = item.get("attributes") if isinstance(item, dict) else None
    url = attributes.get("url") if isinstance(attributes, dict) else None
    if not url:
        return None
    parsed = urlparse(str(url))
    if parsed.netloc != "data.mendeley.com":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2 or parts[0] != "datasets":
        return None
    return f"https://data.mendeley.com/public-api/datasets/{parts[1]}"


def _doi_url(doi: str | None) -> str | None:
    return f"https://doi.org/{doi}" if doi else None


def _doi_from_dataverse_identifier(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    match = re.search(r"(10\.\d{4,9}/[^\s\"<>]+)", text, flags=re.IGNORECASE)
    return match.group(1).rstrip(".,);]") if match else None


def _dataverse_file_count_summary(file_count: Any, total_size: Any = None) -> str | None:
    try:
        count = int(file_count)
    except (TypeError, ValueError):
        count = 0
    if count and total_size:
        return f"{count} files; {total_size} bytes total"
    if count:
        return f"{count} files"
    if total_size:
        return f"{total_size} bytes total"
    return None


def _figshare_license(value: Any) -> str | None:
    if isinstance(value, dict):
        name = value.get("name")
        url = value.get("url")
        if name and url:
            return f"{name} ({url})"
        return str(name or url) if (name or url) else None
    return str(value) if value else None


def _mendeley_license(attributes: dict[str, Any], public_detail: dict[str, Any] | None) -> str | None:
    licence = (public_detail or {}).get("data_licence") if public_detail else None
    if isinstance(licence, dict):
        short_name = licence.get("short_name")
        full_name = licence.get("full_name")
        url = licence.get("url")
        label = short_name or full_name
        if label and url:
            return f"{label} ({url})"
        if label:
            return str(label)
    for rights in attributes.get("rightsList") or []:
        if not isinstance(rights, dict):
            continue
        label = rights.get("rightsIdentifier") or rights.get("rights")
        uri = rights.get("rightsUri")
        if label and uri:
            return f"{label} ({uri})"
        if label:
            return str(label)
    return None


def _mendeley_record_id(dataset_doi: str, public_detail: dict[str, Any] | None, attributes: dict[str, Any]) -> str:
    if public_detail and public_detail.get("id") and public_detail.get("version"):
        return f"{public_detail['id']}:v{public_detail['version']}"
    url = attributes.get("url")
    if url:
        parsed = urlparse(str(url))
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] == "datasets":
            if len(parts) >= 3:
                return f"{parts[1]}:v{parts[2]}"
            return str(parts[1])
    return dataset_doi


def _mendeley_publication_doi(attributes: dict[str, Any], public_detail: dict[str, Any] | None) -> str | None:
    for article in (public_detail or {}).get("articles") or []:
        if isinstance(article, dict) and article.get("doi"):
            return str(article["doi"])
    for related in attributes.get("relatedIdentifiers") or []:
        if not isinstance(related, dict):
            continue
        if str(related.get("relatedIdentifierType", "")).upper() != "DOI":
            continue
        if related.get("resourceTypeGeneral") == "Dataset":
            continue
        relation = str(related.get("relationType") or "")
        if relation in {"IsSupplementTo", "References", "IsReferencedBy", "IsDerivedFrom"}:
            return str(related.get("relatedIdentifier"))
    return None


def _datacite_title(attributes: dict[str, Any]) -> str | None:
    for title in attributes.get("titles") or []:
        if isinstance(title, dict) and title.get("title"):
            return str(title["title"])
    return None


def _datacite_descriptions(attributes: dict[str, Any]) -> str:
    return " ".join(
        str(description.get("description"))
        for description in attributes.get("descriptions") or []
        if isinstance(description, dict) and description.get("description")
    )


def _datacite_issued_date(attributes: dict[str, Any]) -> str | None:
    for date in attributes.get("dates") or []:
        if isinstance(date, dict) and date.get("dateType") == "Issued" and date.get("date"):
            return str(date["date"])
    year = attributes.get("publicationYear")
    return str(year) if year else None


def _file_formats_from_files(files: list[Any], extra_formats: Any = None) -> list[str]:
    formats: set[str] = set()
    for file_info in files:
        if not isinstance(file_info, dict):
            continue
        name = _file_name(file_info)
        if "." in name:
            formats.add(_normalize_format(name.rsplit(".", 1)[1]))
        content_type = _get_path(file_info, ("content_details", "content_type")) or file_info.get("mimetype") or file_info.get("content_type")
        if content_type:
            formats.add(str(content_type))
    for value in extra_formats or []:
        if value:
            formats.add(_normalize_format(str(value)))
    return sorted(formats)


def _normalize_format(value: str) -> str:
    cleaned = value.strip().lstrip(".")
    mapping = {
        "tif": "TIFF",
        "tiff": "TIFF",
        "jpg": "JPEG",
        "jpeg": "JPEG",
        "png": "PNG",
        "mrc": "MRC",
        "dm3": "DM3",
        "dm4": "DM4",
        "h5": "HDF5",
        "hdf5": "HDF5",
        "zarr": "Zarr",
        "n5": "N5",
        "zip": "ZIP",
    }
    return mapping.get(cleaned.lower(), cleaned)


def _file_count_summary(files: list[Any], total_size: Any = None) -> str | None:
    count = len(files)
    if count and total_size:
        return f"{count} files; {total_size} bytes total"
    if count:
        return f"{count} files"
    if total_size:
        return f"{total_size} bytes total"
    return None


def _file_name(file_info: Any) -> str:
    if not isinstance(file_info, dict):
        return ""
    return str(file_info.get("name") or file_info.get("filename") or "")


def _subject_text(subject: Any) -> str:
    if isinstance(subject, dict):
        return str(subject.get("subject") or "")
    return str(subject or "")


def _title_from_mapping(value: Any, key: str = "title") -> str:
    if isinstance(value, dict):
        return str(value.get(key) or value.get("title") or "")
    return str(value or "")


_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _clean_fragment(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = _TAG_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def _evidence_text(parts: list[Any]) -> str | None:
    fragments = [_clean_fragment(part) for part in parts]
    evidence = " ; ".join(fragment for fragment in fragments if fragment)
    return evidence[:4000] if evidence else None


def _compact_figshare_article(detail: dict[str, Any]) -> dict[str, Any]:
    compact = {key: value for key, value in detail.items() if key not in {"references", "related_materials", "files"}}
    compact["files"] = _compact_files(detail.get("files") or [])
    return compact


def _compact_datacite_item(item: dict[str, Any]) -> dict[str, Any]:
    compact = dict(item)
    attributes = dict(compact.get("attributes") or {})
    attributes.pop("xml", None)
    compact["attributes"] = attributes
    return compact


def _compact_mendeley_public_detail(detail: dict[str, Any] | None) -> dict[str, Any] | None:
    if not detail:
        return None
    compact = {key: value for key, value in detail.items() if key != "files"}
    compact["files"] = _compact_files(detail.get("files") or [])
    return compact


def _compact_files(files: list[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for file_info in files[:MAX_FILE_METADATA]:
        if not isinstance(file_info, dict):
            continue
        content_details = file_info.get("content_details") or {}
        compact.append(
            {
                "name": file_info.get("name") or file_info.get("filename"),
                "size": file_info.get("size") or content_details.get("size"),
                "mimetype": file_info.get("mimetype") or content_details.get("content_type") or file_info.get("content_type"),
                "is_link_only": file_info.get("is_link_only"),
                "status": file_info.get("status"),
            }
        )
    if len(files) > MAX_FILE_METADATA:
        compact.append({"omitted_file_count": len(files) - MAX_FILE_METADATA})
    return compact
