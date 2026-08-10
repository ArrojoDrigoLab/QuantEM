"""Literature discovery scanner for article-linked public EM data."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from catalog.http import get_json, urlencode
from catalog.models import Candidate
from catalog.sources.base import ScannerResult, safe_collect, unique_candidates


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PUBMED_ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "sources.json"
DEFAULT_LITERATURE_QUERY = '("electron microscopy" OR ultrastructure OR "FIB-SEM" OR "SBF-SEM") AND ("data availability" OR dataset OR repository OR "source data")'
LITERATURE_SOURCES = ("openalex", "crossref", "europepmc", "pubmed")
MAX_SOURCE_DATA_LINKS = 25

DATA_LINK_HINTS = (
    "source data",
    "source-data",
    "source_data",
    "data availability",
    "dataset",
    "repository",
    "supplement",
    "supplementary",
    "figshare",
    "zenodo",
    "dryad",
    "biostudies",
    "empiar",
    "openorganelle",
    "bossdb",
    "webknossos",
    "dataverse",
    "osf.io",
    "github.com",
    "data.mendeley.com",
)


def scan_literature(
    since: str | None = None,
    query: str | None = None,
    limit: int = 50,
    config_path: str | Path | None = None,
    sources: Iterable[str] | str | None = None,
    full_history: bool = False,
    **_: Any,
) -> ScannerResult:
    config = load_literature_config(config_path)
    source_names = _requested_sources(sources)
    result = ScannerResult()
    adapters: dict[str, Callable[[str | None, str | None, int, dict[str, Any]], list[Candidate]]] = {
        "openalex": _scan_openalex,
        "crossref": _scan_crossref,
        "europepmc": _scan_europepmc,
        "pubmed": _scan_pubmed,
    }
    for source in source_names:
        source_since = since or (None if full_history else _default_since(config, source))
        source_limit = _source_limit(config, source, limit, full_history=full_history)
        result.extend(
            safe_collect(
                "literature",
                f"{source}_literature",
                adapters[source],
                source_since,
                query,
                source_limit,
                config,
            )
        )
    result.candidates = _round_robin_limit(unique_candidates(result.candidates), limit)
    return result


def load_literature_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"literature config {path} is not an object")
    return config


def _scan_openalex(since: str | None, query: str | None, limit: int, config: dict[str, Any] | None = None) -> list[Candidate]:
    config = config or {}
    filters = []
    if since:
        filters.append(f"from_publication_date:{since}")
    params = {
        "search": query or _default_query(config),
        "per-page": min(limit, 200),
        "filter": ",".join(filters) if filters else None,
        "select": "id,doi,ids,title,publication_year,primary_location,locations,abstract_inverted_index",
    }
    data = get_json(f"{OPENALEX_WORKS_URL}?{urlencode(params)}")
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("OpenAlex literature payload missing results list")
    candidates: list[Candidate] = []
    for item in data["results"]:
        if not isinstance(item, dict):
            raise ValueError("OpenAlex literature result is not an object")
        candidate = _candidate_from_openalex(item, config)
        if _candidate_has_literature_relevance(candidate, config):
            candidates.append(candidate)
    return candidates[:limit]


def _scan_crossref(since: str | None, query: str | None, limit: int, config: dict[str, Any] | None = None) -> list[Candidate]:
    config = config or {}
    candidates: list[Candidate] = []
    for search_query in _query_strings(config, query, "crossref"):
        params = {
            "query.bibliographic": search_query,
            "rows": max(1, min(limit, 1000)),
            "sort": "published",
            "order": "desc",
            "filter": _crossref_filter(since),
            "select": "DOI,title,container-title,published-print,published-online,published,created,URL,abstract,link,relation,ISSN",
        }
        data = get_json(f"{CROSSREF_WORKS_URL}?{urlencode(params)}")
        if not isinstance(data, dict):
            raise ValueError("Crossref literature payload is not an object")
        message = data.get("message")
        items = message.get("items") if isinstance(message, dict) else None
        if not isinstance(items, list):
            raise ValueError("Crossref literature payload missing message.items list")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Crossref literature result is not an object")
            candidate = _candidate_from_crossref(item, config)
            if not _candidate_has_literature_relevance(candidate, config):
                continue
            candidates.append(candidate)
            candidates = unique_candidates(candidates)
            if len(candidates) >= limit:
                return candidates[:limit]
    return unique_candidates(candidates)[:limit]


def _scan_europepmc(since: str | None, query: str | None, limit: int, config: dict[str, Any] | None = None) -> list[Candidate]:
    config = config or {}
    candidates: list[Candidate] = []
    for search_query in _query_strings(config, query, "europepmc"):
        params = {
            "query": _europepmc_query(search_query, since),
            "format": "json",
            "pageSize": max(1, min(limit, 1000)),
            "resultType": "core",
        }
        data = get_json(f"{EUROPEPMC_SEARCH_URL}?{urlencode(params)}")
        if not isinstance(data, dict):
            raise ValueError("Europe PMC literature payload is not an object")
        result_list = data.get("resultList")
        items = result_list.get("result") if isinstance(result_list, dict) else None
        if not isinstance(items, list):
            raise ValueError("Europe PMC literature payload missing resultList.result list")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Europe PMC literature result is not an object")
            candidate = _candidate_from_europepmc(item, config)
            if not _candidate_has_literature_relevance(candidate, config):
                continue
            candidates.append(candidate)
            candidates = unique_candidates(candidates)
            if len(candidates) >= limit:
                return candidates[:limit]
    return unique_candidates(candidates)[:limit]


def _scan_pubmed(since: str | None, query: str | None, limit: int, config: dict[str, Any] | None = None) -> list[Candidate]:
    config = config or {}
    term = query or _default_query(config)
    search_params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": max(1, min(limit, 1000)),
        "sort": "pub date",
        "datetype": "pdat" if since else None,
        "mindate": since,
        "tool": "quantem_catalog",
    }
    search_data = get_json(f"{PUBMED_ESEARCH_URL}?{urlencode(search_params)}")
    ids = _pubmed_esearch_ids(search_data)
    if not ids:
        return []

    joined_ids = ",".join(ids[:limit])
    summary_params = {
        "db": "pubmed",
        "id": joined_ids,
        "retmode": "json",
        "tool": "quantem_catalog",
    }
    summary_data = get_json(f"{PUBMED_ESUMMARY_URL}?{urlencode(summary_params)}")
    summary_items = _pubmed_summary_items(summary_data)

    elink_params = {
        "dbfrom": "pubmed",
        "id": joined_ids,
        "cmd": "llinks",
        "retmode": "json",
        "tool": "quantem_catalog",
    }
    elink_data = get_json(f"{PUBMED_ELINK_URL}?{urlencode(elink_params)}")
    links_by_uid = _pubmed_links_by_uid(elink_data)

    candidates: list[Candidate] = []
    for uid in ids:
        item = summary_items.get(uid)
        if not item:
            continue
        candidate = _candidate_from_pubmed(uid, item, links_by_uid.get(uid, []), config)
        if not _candidate_has_literature_relevance(candidate, config):
            continue
        candidates.append(candidate)
        candidates = unique_candidates(candidates)
        if len(candidates) >= limit:
            return candidates[:limit]
    return candidates[:limit]


def _candidate_from_openalex(item: dict[str, Any], config: dict[str, Any]) -> Candidate:
    ids = item.get("ids") if isinstance(item.get("ids"), dict) else {}
    doi = _clean_doi(item.get("doi") or ids.get("doi"))
    pmid = _clean_pubmed_id(ids.get("pmid"))
    pmcid = _clean_pmcid(ids.get("pmcid"))
    title = _first_text(item.get("title")) or doi or item.get("id") or "OpenAlex literature lead"
    abstract = _openalex_abstract(item.get("abstract_inverted_index"))
    primary = item.get("primary_location") if isinstance(item.get("primary_location"), dict) else {}
    source = primary.get("source") if isinstance(primary.get("source"), dict) else {}
    journal = source.get("display_name")
    year = item.get("publication_year")
    landing_url = primary.get("landing_page_url") or _doi_url(doi) or item.get("id")
    source_links = _source_data_links_from_openalex(item)
    matched_terms = _matched_terms(_evidence_text([title, abstract, journal, " ".join(source_links)]), config)
    return Candidate(
        source_name="literature_openalex",
        source_record_id=str(item.get("id") or doi or pmid or title),
        title=str(title),
        landing_url=str(landing_url) if landing_url else None,
        download_or_manifest_urls=source_links,
        publication_doi=doi,
        raw_metadata={
            "source": "openalex",
            "doi": doi,
            "pmid": pmid,
            "pmcid": pmcid,
            "journal": journal,
            "year": str(year) if year else None,
            "url": landing_url,
            "abstract": abstract,
            "matched_evidence_terms": matched_terms,
            "source_data_links": source_links,
            "record": item,
        },
        evidence_text=_evidence_text([title, journal, year, abstract, "matched terms: " + ", ".join(matched_terms) if matched_terms else None]),
    )


def _candidate_from_crossref(item: dict[str, Any], config: dict[str, Any]) -> Candidate:
    doi = _clean_doi(item.get("DOI"))
    title = _first_text(item.get("title")) or doi or "Crossref literature lead"
    journal = _first_text(item.get("container-title"))
    year = _crossref_year(item)
    abstract = _clean_fragment(item.get("abstract"))
    landing_url = item.get("URL") or _doi_url(doi)
    source_links = _source_data_links_from_crossref(item)
    matched_terms = _matched_terms(_evidence_text([title, journal, year, abstract, " ".join(source_links)]), config)
    return Candidate(
        source_name="literature_crossref",
        source_record_id=doi or str(item.get("URL") or title),
        title=str(title),
        landing_url=str(landing_url) if landing_url else None,
        download_or_manifest_urls=source_links,
        publication_doi=doi,
        raw_metadata={
            "source": "crossref",
            "doi": doi,
            "pmid": None,
            "pmcid": None,
            "journal": journal,
            "year": year,
            "url": landing_url,
            "abstract": abstract,
            "matched_evidence_terms": matched_terms,
            "source_data_links": source_links,
            "record": _compact_crossref_record(item),
        },
        evidence_text=_evidence_text([title, journal, year, abstract, "matched terms: " + ", ".join(matched_terms) if matched_terms else None]),
    )


def _candidate_from_europepmc(item: dict[str, Any], config: dict[str, Any]) -> Candidate:
    doi = _clean_doi(item.get("doi"))
    pmid = _clean_pubmed_id(item.get("pmid") or item.get("id") if item.get("source") in {None, "MED"} else item.get("pmid"))
    pmcid = _clean_pmcid(item.get("pmcid"))
    title = _clean_fragment(item.get("title")) or doi or pmid or pmcid or "Europe PMC literature lead"
    journal = _clean_fragment(item.get("journalTitle") or _europepmc_journal_title(item))
    year = str(item.get("pubYear")) if item.get("pubYear") else None
    abstract = _clean_fragment(item.get("abstractText"))
    landing_url = _doi_url(doi) or _pmcid_url(pmcid) or _pubmed_url(pmid)
    source_links = _source_data_links_from_europepmc(item)
    matched_terms = _matched_terms(_evidence_text([title, journal, year, abstract, " ".join(source_links)]), config)
    return Candidate(
        source_name="literature_europepmc",
        source_record_id=doi or pmcid or pmid or str(item.get("id") or title),
        title=str(title),
        landing_url=landing_url,
        download_or_manifest_urls=source_links,
        publication_doi=doi,
        raw_metadata={
            "source": "europepmc",
            "doi": doi,
            "pmid": pmid,
            "pmcid": pmcid,
            "journal": journal,
            "year": year,
            "url": landing_url,
            "abstract": abstract,
            "matched_evidence_terms": matched_terms,
            "source_data_links": source_links,
            "record": item,
        },
        evidence_text=_evidence_text([title, journal, year, abstract, "matched terms: " + ", ".join(matched_terms) if matched_terms else None]),
    )


def _europepmc_journal_title(item: dict[str, Any]) -> Any:
    journal_info = item.get("journalInfo")
    if not isinstance(journal_info, dict):
        return None
    journal = journal_info.get("journal")
    if not isinstance(journal, dict):
        return None
    return journal.get("title")


def _candidate_from_pubmed(uid: str, item: dict[str, Any], elink_urls: list[dict[str, Any]], config: dict[str, Any]) -> Candidate:
    doi = _clean_doi(_pubmed_article_id(item, "doi") or _doi_from_elocation(item.get("elocationid")))
    pmid = _clean_pubmed_id(_pubmed_article_id(item, "pubmed") or uid)
    pmcid = _clean_pmcid(_pubmed_article_id(item, "pmc"))
    title = _clean_fragment(item.get("title")) or doi or pmid or "PubMed literature lead"
    journal = _clean_fragment(item.get("fulljournalname") or item.get("source"))
    year = _pubmed_year(item)
    source_links = _source_data_links_from_pubmed(elink_urls)
    snippet = _evidence_text([item.get("summary"), item.get("availablefromurl"), " ".join(source_links)])
    matched_terms = _matched_terms(_evidence_text([title, journal, year, snippet, " ".join(source_links)]), config)
    landing_url = _doi_url(doi) or _pubmed_url(pmid) or _pmcid_url(pmcid)
    return Candidate(
        source_name="literature_pubmed",
        source_record_id=pmid or doi or pmcid or uid,
        title=str(title),
        landing_url=landing_url,
        download_or_manifest_urls=source_links,
        publication_doi=doi,
        raw_metadata={
            "source": "pubmed",
            "doi": doi,
            "pmid": pmid,
            "pmcid": pmcid,
            "journal": journal,
            "year": year,
            "url": landing_url,
            "snippet": snippet,
            "matched_evidence_terms": matched_terms,
            "source_data_links": source_links,
            "summary": item,
            "elink_urls": elink_urls,
        },
        evidence_text=_evidence_text([title, journal, year, snippet, "matched terms: " + ", ".join(matched_terms) if matched_terms else None]),
    )


def _requested_sources(sources: Iterable[str] | str | None) -> list[str]:
    if sources is None:
        return list(LITERATURE_SOURCES)
    if isinstance(sources, str):
        raw_sources = [source.strip() for source in sources.split(",")]
    else:
        raw_sources = [str(source).strip() for source in sources]
    selected = [source for source in raw_sources if source]
    unknown = [source for source in selected if source not in LITERATURE_SOURCES]
    if unknown:
        raise ValueError(f"unknown literature source(s): {', '.join(unknown)}")
    return selected


def _source_limit(
    config: dict[str, Any],
    source: str,
    cli_limit: int,
    *,
    full_history: bool = False,
) -> int:
    source_config = (config.get("per_source") or {}).get(source) or {}
    configured = source_config.get("limit")
    if configured is None or full_history:
        return max(1, cli_limit)
    return max(1, min(cli_limit, int(configured)))


def _default_since(config: dict[str, Any], source: str) -> str | None:
    source_config = (config.get("per_source") or {}).get(source) or {}
    days = source_config.get("date_window_days")
    if not days:
        return None
    start = datetime.now(timezone.utc).date() - timedelta(days=int(days))
    return start.isoformat()


def _default_query(config: dict[str, Any]) -> str:
    groups = config.get("query_groups")
    if isinstance(groups, list) and groups:
        first = groups[0]
        if isinstance(first, dict) and first.get("query"):
            return str(first["query"])
    return DEFAULT_LITERATURE_QUERY


def _query_strings(config: dict[str, Any], query: str | None, source: str) -> list[str]:
    if query:
        return [query]
    source_config = (config.get("per_source") or {}).get(source) or {}
    max_groups = int(source_config.get("max_query_groups") or 2)
    queries: list[str] = []
    source_key = f"{source}_query"
    for group in config.get("query_groups") or []:
        if isinstance(group, dict) and (group.get(source_key) or group.get("query")):
            queries.append(str(group.get(source_key) or group["query"]))
        if len(queries) >= max_groups:
            break
    return queries or [DEFAULT_LITERATURE_QUERY]


def _crossref_filter(since: str | None) -> str:
    filters = ["type:journal-article"]
    if since:
        filters.append(f"from-pub-date:{since}")
    return ",".join(filters)


def _europepmc_query(query: str, since: str | None) -> str:
    if not since:
        return query
    return f"({query}) AND FIRST_PDATE:[{since} TO 9999-12-31]"


def _pubmed_esearch_ids(data: Any) -> list[str]:
    if not isinstance(data, dict):
        raise ValueError("PubMed ESearch payload is not an object")
    result = data.get("esearchresult")
    ids = result.get("idlist") if isinstance(result, dict) else None
    if not isinstance(ids, list):
        raise ValueError("PubMed ESearch payload missing esearchresult.idlist")
    return [str(uid) for uid in ids if str(uid).strip()]


def _pubmed_summary_items(data: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(data, dict):
        raise ValueError("PubMed ESummary payload is not an object")
    result = data.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("uids"), list):
        raise ValueError("PubMed ESummary payload missing result.uids list")
    items: dict[str, dict[str, Any]] = {}
    for uid in result["uids"]:
        item = result.get(str(uid))
        if not isinstance(item, dict):
            raise ValueError(f"PubMed ESummary payload missing object for uid {uid}")
        items[str(uid)] = item
    return items


def _pubmed_links_by_uid(data: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(data, dict):
        raise ValueError("PubMed ELink payload is not an object")
    linksets = data.get("linksets")
    if not isinstance(linksets, list):
        raise ValueError("PubMed ELink payload missing linksets list")
    links_by_uid: dict[str, list[dict[str, Any]]] = {}
    for linkset in linksets:
        if not isinstance(linkset, dict):
            raise ValueError("PubMed ELink linkset is not an object")
        ids = [str(uid) for uid in linkset.get("ids") or [] if str(uid).strip()]
        for idurl in linkset.get("idurls") or []:
            if not isinstance(idurl, dict):
                continue
            idurl_ids = [str(idurl.get("id"))] if idurl.get("id") else ids
            for objurl in idurl.get("objurls") or []:
                if not isinstance(objurl, dict):
                    continue
                for uid in idurl_ids:
                    links_by_uid.setdefault(uid, []).append(objurl)
    return links_by_uid


def _source_data_links_from_openalex(item: dict[str, Any]) -> list[str]:
    links: list[str] = []
    for location in [item.get("primary_location"), *(item.get("locations") or [])]:
        if not isinstance(location, dict):
            continue
        for key in ("landing_page_url", "pdf_url"):
            url = location.get(key)
            if _looks_like_source_data_link(url, label=key):
                links.append(str(url))
    return _unique_strings(links)[:MAX_SOURCE_DATA_LINKS]


def _source_data_links_from_crossref(item: dict[str, Any]) -> list[str]:
    links: list[str] = []
    for link in item.get("link") or []:
        if not isinstance(link, dict):
            continue
        url = link.get("URL") or link.get("url")
        label = " ".join(str(link.get(key) or "") for key in ("content-type", "content-version", "intended-application"))
        if _looks_like_source_data_link(url, label=label):
            links.append(str(url))
    relation = item.get("relation")
    if isinstance(relation, dict):
        for relation_type, values in relation.items():
            if not _has_data_hint(relation_type):
                continue
            for value in values or []:
                if not isinstance(value, dict):
                    continue
                if str(value.get("id-type") or "").lower() == "doi" and value.get("id"):
                    links.append(_doi_url(str(value["id"])) or str(value["id"]))
                elif _looks_like_source_data_link(value.get("id"), label=relation_type):
                    links.append(str(value["id"]))
    return _unique_strings(links)[:MAX_SOURCE_DATA_LINKS]


def _source_data_links_from_europepmc(item: dict[str, Any]) -> list[str]:
    links: list[str] = []
    full_text_list = item.get("fullTextUrlList")
    urls = full_text_list.get("fullTextUrl") if isinstance(full_text_list, dict) else []
    for link in urls or []:
        if not isinstance(link, dict):
            continue
        label = " ".join(str(link.get(key) or "") for key in ("documentStyle", "availability", "site", "provider"))
        if _looks_like_source_data_link(link.get("url"), label=label):
            links.append(str(link["url"]))
    data_links = item.get("dataLinksTagsList")
    if isinstance(data_links, dict):
        for value in data_links.get("dataLinkstag") or []:
            if _looks_like_source_data_link(value, label="dataLinksTagsList"):
                links.append(str(value))
    return _unique_strings(links)[:MAX_SOURCE_DATA_LINKS]


def _source_data_links_from_pubmed(elink_urls: list[dict[str, Any]]) -> list[str]:
    links: list[str] = []
    for objurl in elink_urls:
        url_value = objurl.get("url")
        if isinstance(url_value, dict):
            url = url_value.get("value")
        else:
            url = url_value
        provider = objurl.get("provider") if isinstance(objurl.get("provider"), dict) else {}
        attributes = objurl.get("attributes") or []
        label = " ".join(
            [
                str(provider.get("name") or ""),
                " ".join(str(attribute) for attribute in attributes),
                str(objurl.get("subjecttype") or ""),
            ]
        )
        if _looks_like_source_data_link(url, label=label):
            links.append(str(url))
    return _unique_strings(links)[:MAX_SOURCE_DATA_LINKS]


def _looks_like_source_data_link(url: Any, label: str = "") -> bool:
    if not url:
        return False
    text = f"{url} {label}".lower()
    if ".pdf" in text or "application/pdf" in text:
        return False
    return _has_data_hint(text)


def _has_data_hint(text: Any) -> bool:
    lowered = str(text or "").lower()
    return any(hint in lowered for hint in DATA_LINK_HINTS)


def _matched_terms(text: str | None, config: dict[str, Any]) -> list[str]:
    if not text:
        return []
    terms: list[str] = []
    query_terms = config.get("query_terms")
    if isinstance(query_terms, dict):
        for values in query_terms.values():
            if isinstance(values, list):
                terms.extend(str(value).strip('"') for value in values)
    if not terms:
        terms = ["TEM", "FIB-SEM", "SBF-SEM", "volume EM", "ultrastructure", "organelle", "source data", "repository", "dataset"]
    matched: list[str] = []
    for term in terms:
        if not term:
            continue
        pattern = r"(?<![A-Za-z0-9])" + re.escape(term.lower()) + r"(?![A-Za-z0-9])"
        if re.search(pattern, text.lower()):
            matched.append(term)
    return _unique_strings(matched)


def _candidate_has_literature_relevance(candidate: Candidate, config: dict[str, Any]) -> bool:
    matched_terms = candidate.raw_metadata.get("matched_evidence_terms")
    if not isinstance(matched_terms, list) or not matched_terms:
        return False
    relevance_terms = {term.lower() for term in _configured_relevance_terms(config)}
    return any(str(term).lower() in relevance_terms for term in matched_terms)


def _configured_relevance_terms(config: dict[str, Any]) -> list[str]:
    query_terms = config.get("query_terms")
    relevant_categories = ("tem", "sbf_sem", "fib_sem", "volume_em", "ultrastructure", "organelle")
    terms: list[str] = []
    if isinstance(query_terms, dict):
        for category in relevant_categories:
            values = query_terms.get(category)
            if isinstance(values, list):
                terms.extend(str(value).strip('"') for value in values)
    return terms or ["TEM", "FIB-SEM", "SBF-SEM", "volume EM", "volume electron microscopy", "ultrastructure", "organelle", "mitochondria", "endoplasmic reticulum", "nucleus"]


def _crossref_year(item: dict[str, Any]) -> str | None:
    for key in ("published-print", "published-online", "published", "created"):
        value = item.get(key)
        if not isinstance(value, dict):
            continue
        parts = value.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            return str(parts[0][0])
    return None


def _pubmed_article_id(item: dict[str, Any], id_type: str) -> str | None:
    for article_id in item.get("articleids") or []:
        if not isinstance(article_id, dict):
            continue
        if str(article_id.get("idtype") or "").lower() == id_type.lower() and article_id.get("value"):
            return str(article_id["value"])
    return None


def _pubmed_year(item: dict[str, Any]) -> str | None:
    for key in ("sortpubdate", "pubdate", "epubdate"):
        value = item.get(key)
        if not value:
            continue
        match = re.search(r"\b(19|20)\d{2}\b", str(value))
        if match:
            return match.group(0)
    return None


def _doi_from_elocation(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    text = re.sub(r"^doi:\s*", "", text, flags=re.IGNORECASE)
    return text if text else None


def _openalex_abstract(inverted_index: Any) -> str | None:
    if not isinstance(inverted_index, dict):
        return None
    positions: list[tuple[int, str]] = []
    for word, indexes in inverted_index.items():
        if not isinstance(indexes, list):
            continue
        for index in indexes:
            if isinstance(index, int):
                positions.append((index, str(word)))
    if not positions:
        return None
    return " ".join(word for _, word in sorted(positions))[:4000]


def _compact_crossref_record(item: dict[str, Any]) -> dict[str, Any]:
    return {key: item.get(key) for key in ("DOI", "title", "container-title", "published-print", "published-online", "published", "created", "URL", "abstract", "link", "relation", "ISSN") if key in item}


def _round_robin_limit(candidates: list[Candidate], limit: int) -> list[Candidate]:
    if len(candidates) <= limit:
        return candidates
    grouped: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.source_name, []).append(candidate)
    limited: list[Candidate] = []
    while len(limited) < limit and any(grouped.values()):
        for source in list(grouped):
            if grouped[source]:
                limited.append(grouped[source].pop(0))
                if len(limited) >= limit:
                    break
    return limited


def _first_text(value: Any) -> str | None:
    if isinstance(value, list):
        for item in value:
            cleaned = _clean_fragment(item)
            if cleaned:
                return cleaned
        return None
    return _clean_fragment(value)


_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _clean_fragment(value: Any) -> str | None:
    if value is None:
        return None
    text = html.unescape(str(value))
    text = _TAG_RE.sub(" ", text)
    cleaned = _SPACE_RE.sub(" ", text).strip()
    return cleaned or None


def _evidence_text(parts: Iterable[Any]) -> str | None:
    fragments = [_clean_fragment(part) for part in parts]
    evidence = " ; ".join(fragment for fragment in fragments if fragment)
    return evidence[:4000] if evidence else None


def _clean_doi(value: Any) -> str | None:
    if not value:
        return None
    doi = str(value).strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    doi = doi.strip()
    return doi or None


def _clean_pubmed_id(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    text = re.sub(r"^https?://pubmed\.ncbi\.nlm\.nih\.gov/", "", text, flags=re.IGNORECASE).strip("/")
    text = re.sub(r"^pmid:\s*", "", text, flags=re.IGNORECASE)
    return text or None


def _clean_pmcid(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    text = re.sub(r"^https?://pmc\.ncbi\.nlm\.nih\.gov/articles/", "", text, flags=re.IGNORECASE).strip("/")
    text = re.sub(r"^pmcid:\s*", "", text, flags=re.IGNORECASE)
    if text and not text.upper().startswith("PMC"):
        text = f"PMC{text}"
    return text or None


def _doi_url(doi: str | None) -> str | None:
    return f"https://doi.org/{doi}" if doi else None


def _pubmed_url(pmid: str | None) -> str | None:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None


def _pmcid_url(pmcid: str | None) -> str | None:
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else None


def _unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique
