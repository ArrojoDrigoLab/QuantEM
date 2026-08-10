"""Repository scanners.

One module per public data repository, each querying that repository's API and
returning a `ScannerResult` of `Candidate` records. Scanners apply no eligibility
filtering beyond what the API supports; that is decided later by
`catalog.eligibility` and `catalog.classify`.

`portals` covers repositories that share a records-and-metadata shape: FigShare
(search and OAI-PMH), Zenodo, BioStudies, Dryad, and Mendeley via DataCite. It
also carries handlers for DataCite at large, Dataverse and Hugging Face, which
were searched but contributed no datasets to the published corpus.

`zenodo_dump` is a separate full-history scanner that prefilters Zenodo's
published metadata dump before fetching details, for coverage the search API
does not reach.
"""
from __future__ import annotations

from .base import ScannerResult
from .bossdb import scan_bossdb
from .empiar import scan_empiar
from .openorganelle import scan_openorganelle
from .portals import scan_generic_source
from .webknossos import scan_webknossos
from .zenodo_dump import scan_zenodo_dump

SOURCE_NAMES = [
    "empiar",
    "openorganelle",
    "bossdb",
    "webknossos",
    "biostudies",
    "zenodo",
    "zenodo_dump",
    "figshare",
    "figshare_oai",
    "mendeley",
    "datacite",
    "dataverse",
    "dryad",
    "huggingface",
]


def run_scanner(source: str, **kwargs) -> ScannerResult:
    if source == "empiar":
        return scan_empiar(**kwargs)
    if source == "openorganelle":
        return scan_openorganelle(**kwargs)
    if source == "bossdb":
        return scan_bossdb(**kwargs)
    if source == "webknossos":
        return scan_webknossos(**kwargs)
    if source == "zenodo_dump":
        return scan_zenodo_dump(**kwargs)
    if source in SOURCE_NAMES:
        return scan_generic_source(source, **kwargs)
    raise ValueError(f"unknown source {source!r}; choose one of {', '.join(SOURCE_NAMES)}")


__all__ = ["SOURCE_NAMES", "ScannerResult", "run_scanner"]
