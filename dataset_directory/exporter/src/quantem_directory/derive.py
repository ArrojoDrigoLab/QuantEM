"""The derivation rules that decide what the export reports.

Four of these are not incidental formatting choices — they are the definitions
behind the published numbers, and changing one changes what the site reports:

* :func:`is_three_dimensional` reconciles three disagreeing sources of
  dimensionality into the published 2D / 3D split. The raw ``dimensionality``
  tag alone does not reproduce it.
* :func:`reference_url` reproduces the dataset link column of the published
  dataset inventory: a dataset's own DOI wins, then its source URL, then its
  experiment's.
* :func:`repository_of` names the repository that actually serves a dataset
  today, derived from that link. It is deliberately *not* the corpus's internal
  ``catalog_source_key``, which records how a dataset was originally discovered
  and is empty for everything that was not machine-scraped.
* :func:`resolution_band` buckets in-plane resolution on the log-spaced scale
  the corpus composition figures use, with unresolved values kept as an explicit
  band rather than silently dropped or coerced to zero.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

# --------------------------------------------------------------------------
# Dimensionality
# --------------------------------------------------------------------------

_NM_AXIS = re.compile(r"(\d+(?:\.\d+)?)\s*nm", re.IGNORECASE)


def is_three_dimensional(
    *, dimensionality_tags: Sequence[str], resolution_field: str, depth: int | None
) -> bool:
    """Decide whether an asset is a 3D acquisition.

    An asset counts as 3D if it is tagged ``3D``, *or* its resolution string
    names three or more axis extents in nanometres, *or* it has more than one
    plane. The tag alone is insufficient: a substantial number of assets carry
    ``Mixed`` or no tag at all, and only the reconciled rule reproduces the
    published counts.
    """
    if any(t.strip().upper() == "3D" for t in dimensionality_tags):
        return True
    if len(_NM_AXIS.findall(resolution_field or "")) >= 3:
        return True
    return bool(depth and depth > 1)


# --------------------------------------------------------------------------
# Dataset reference link
# --------------------------------------------------------------------------


def reference_url(
    *, dataset_doi: str = "", source_url: str = "", experiment_doi: str = ""
) -> str | None:
    """Return the public link for a dataset, or ``None`` if it has none.

    Preference order is the dataset's own DOI, then a direct source URL, then
    the DOI of its parent experiment. A dataset with none of these has not been
    deposited yet; the site renders that state explicitly rather than showing a
    dead link.
    """
    dataset_doi = (dataset_doi or "").strip()
    source_url = (source_url or "").strip()
    experiment_doi = (experiment_doi or "").strip()
    if dataset_doi:
        return "https://doi.org/" + dataset_doi
    if source_url:
        return source_url
    if experiment_doi:
        return "https://doi.org/" + experiment_doi
    return None


# --------------------------------------------------------------------------
# Repository
# --------------------------------------------------------------------------

NOT_DEPOSITED = "Not yet deposited"
OTHER_REPOSITORY = "Other"

# Ordered because EBI issues one DOI prefix for two repositories, so the
# accession shape has to be inspected before the prefix is trusted.
_REPOSITORY_RULES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"doi\.org/10\.6019/empiar", re.I), "EMPIAR"),
    (re.compile(r"doi\.org/10\.6019/s-biad", re.I), "BioImage Archive"),
    (re.compile(r"doi\.org/10\.17867/", re.I), "BioImage Archive"),
    (re.compile(r"ebi\.ac\.uk/(biostudies|bioimage)", re.I), "BioImage Archive"),
    (re.compile(r"ebi\.ac\.uk/empiar", re.I), "EMPIAR"),
    (re.compile(r"doi\.org/10\.5281/|(?:^|//)(?:www\.)?zenodo\.org", re.I), "Zenodo"),
    (re.compile(r"doi\.org/10\.25378/|openorganelle\.janelia\.org", re.I), "OpenOrganelle"),
    (re.compile(r"doi\.org/10\.60533/|bossdb\.org", re.I), "BossDB"),
    (re.compile(r"webknossos\.org", re.I), "WEBKNOSSOS"),
    (re.compile(r"doi\.org/10\.5061/|datadryad\.org", re.I), "Dryad"),
    (re.compile(r"doi\.org/10\.6084/|figshare\.com", re.I), "figshare"),
    (re.compile(r"idr\.openmicroscopy\.org", re.I), "IDR"),
    (re.compile(r"doi\.org/10\.17632/|data\.mendeley\.com", re.I), "Mendeley Data"),
    (re.compile(r"doi\.org/10\.57760/|scidb\.cn", re.I), "Science Data Bank"),
    (re.compile(r"open\.quiltdata\.com", re.I), "Quilt"),
    (re.compile(r"doi\.org/10\.26275/", re.I), "SPARC"),
    # Institutional and university repositories, individually tiny.
    (re.compile(r"doi\.org/10\.(25740|5258|6075)/", re.I), "Institutional repository"),
    (re.compile(r"epfl\.ch", re.I), "Institutional repository"),
)


def repository_of(url: str | None) -> str:
    """Name the repository serving ``url``.

    Falls back to :data:`OTHER_REPOSITORY` rather than inventing a label from
    the hostname, so an unrecognised host shows up in the export's own audit
    output instead of quietly becoming a one-dataset facet value.
    """
    if not url or not url.strip():
        return NOT_DEPOSITED
    for pattern, name in _REPOSITORY_RULES:
        if pattern.search(url):
            return name
    return OTHER_REPOSITORY


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

#: ``(label, exclusive upper bound in nm/px)``. The final band is unbounded and
#: the unknown band is appended separately so it always sorts last.
RESOLUTION_BANDS: tuple[tuple[str, float | None], ...] = (
    ("< 1 nm/px", 1.0),
    ("1 – 4 nm/px", 4.0),
    ("4 – 8 nm/px", 8.0),
    ("8 – 32 nm/px", 32.0),
    ("≥ 32 nm/px", None),
)

RESOLUTION_UNKNOWN = "Unknown"


def resolution_band(nm: float | None) -> str:
    """Bucket an in-plane resolution, keeping unresolved values addressable.

    Roughly a third of assets have no parsable in-plane resolution. They get a
    real band rather than being excluded, because otherwise selecting any
    resolution silently hides them — including the corpus's single largest
    dataset.
    """
    if nm is None:
        return RESOLUTION_UNKNOWN
    for label, upper in RESOLUTION_BANDS:
        if upper is None or nm < upper:
            return label
    return RESOLUTION_BANDS[-1][0]


def resolution_band_labels() -> list[str]:
    """Band labels in display order, unknown last."""
    return [label for label, _ in RESOLUTION_BANDS] + [RESOLUTION_UNKNOWN]


# --------------------------------------------------------------------------
# Display helpers
# --------------------------------------------------------------------------


def format_dimensions(
    width: int | None, height: int | None, depth: int | None
) -> str | None:
    """Render pixel extents as ``2048x2119`` or ``2048x2119x310``."""
    if not width or not height:
        return None
    if depth and depth > 1:
        return f"{width}×{height}×{depth}"
    return f"{width}×{height}"


def parse_int(value: object) -> int | None:
    """Parse an integer from extract CSV text, tolerating blanks and floats."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(value: object) -> float | None:
    """Parse a float from extract CSV text, tolerating blanks."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
