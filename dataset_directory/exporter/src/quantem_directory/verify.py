"""Gate the published artifacts before anything is deployed.

Three kinds of check, all of which fail the build rather than warn:

* **Structure.** The columnar arrays have to agree with each other and with the
  dataset rows, or the site renders silent nonsense rather than an error.
* **Privacy.** Every published byte is scanned for the patterns the allow-list
  is meant to exclude. The allow-list is the control; this is the assertion
  that it worked.
* **Exclusions.** Corpus entries deliberately kept out stay out.

Counts are deliberately not pinned here. The corpus grows, and the site reports
what it actually contains; an expected-count file can be passed to ``verify``
for a one-off check, but none is committed.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import allowlist


class VerificationFailed(Exception):
    """Raised with every failure found, not just the first."""


def _load(data_dir: Path, name: str) -> dict:
    return json.loads((data_dir / name).read_text(encoding="utf-8"))


def _check_invariants(manifest: dict, expected: dict, failures: list[str]) -> None:
    actual = manifest.get("counts", {})
    for key, want in expected.items():
        if key.startswith("_"):
            continue
        got = actual.get(key)
        if got != want:
            failures.append(f"invariant {key}: expected {want}, got {got}")


def _check_structure(assets: dict, datasets: dict, failures: list[str]) -> None:
    n = assets["n"]
    for name, column in assets["columns"].items():
        if len(column) != n:
            failures.append(f"column {name!r} has {len(column)} entries, expected {n}")
    for name, column in assets["single"].items():
        if len(column) != n:
            failures.append(f"single-valued column {name!r} has {len(column)} entries, expected {n}")
    for name, csr in assets["multi"].items():
        offsets = csr["offsets"]
        if len(offsets) != n + 1:
            failures.append(f"multi column {name!r} has {len(offsets)} offsets, expected {n + 1}")
        elif offsets[-1] != len(csr["values"]):
            failures.append(
                f"multi column {name!r} final offset {offsets[-1]} != {len(csr['values'])} values"
            )

    rows = datasets["rows"]
    known_ids = set(assets["columns"]["id"])
    # strict=False deliberately: a length mismatch between these two columns is
    # already reported by name in the loops above, and raising here would abort
    # the run before the rest of the failures were collected.
    thumbed_ids = {
        asset_id
        for asset_id, flag in zip(
            assets["columns"]["id"], assets["single"]["thumb"], strict=False
        )
        if flag
    }
    for row in rows:
        if row["n2d"] + row["n3d"] != row["n"]:
            failures.append(f"dataset {row['name']!r}: {row['n2d']}+{row['n3d']} != {row['n']}")
        if row["n"] == 0:
            failures.append(f"dataset {row['name']!r} has no assets")
        for hero in row["hero"]:
            # A hero the site cannot resolve renders as a broken card image.
            if hero not in known_ids:
                failures.append(f"dataset {row['name']!r} hero {hero!r} is not a known asset")
            elif hero not in thumbed_ids:
                failures.append(f"dataset {row['name']!r} hero {hero!r} has no thumbnail")

    if sum(r["n"] for r in rows) != n:
        failures.append(f"dataset asset counts sum to {sum(r['n'] for r in rows)}, expected {n}")

    for index in assets["columns"]["dataset"]:
        if not 0 <= index < len(rows):
            failures.append(f"asset references dataset index {index}, out of range")
            break

    thumbed = sum(assets["single"]["thumb"])
    if thumbed and thumbed > n:
        failures.append(f"thumbnail flag set on {thumbed} of {n} assets")


def _check_privacy(data_dir: Path, facets: dict, failures: list[str], notes: list[str]) -> None:
    """Scan every published byte, and hold facet vocabularies to a higher bar.

    Dataset and asset names are the depositors' own published titles and are
    reproduced verbatim, so they are only checked against the patterns that are
    unacceptable anywhere. Facet labels are our editorial choice and are checked
    against the stricter set as well.
    """
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        failures.extend(allowlist.scan_bytes(text, source=path.name))

    published_groups = set()
    vocabulary_size = 0
    for name, values in facets["dictionaries"].items():
        published_groups.add(name)
        vocabulary_size += len(values)
        failures.extend(allowlist.scan_vocabulary(values, source=f"facets.json/{name}"))

    unauthorised = published_groups - allowlist.PUBLISHED_TAG_GROUPS - allowlist.DERIVED_DICTIONARIES
    if unauthorised:
        failures.append(f"facet vocabulary published without authorisation: {sorted(unauthorised)}")

    notes.append(f"privacy scan clean over {vocabulary_size} facet values")


def _check_exclusions(datasets: dict, excluded: dict, failures: list[str], notes: list[str]) -> None:
    substrings = excluded.get("url_substrings") or []
    names = {n.strip() for n in excluded.get("dataset_names") or []}
    for row in datasets["rows"]:
        if row["name"].strip() in names:
            failures.append(f"excluded dataset {row['name']!r} is present in the export")
        for needle in substrings:
            if needle and needle in (row["url"] or ""):
                failures.append(
                    f"dataset {row['name']!r} links to excluded resource {needle!r}"
                )
    if substrings or names:
        notes.append(f"exclusions hold: {len(names)} name(s), {len(substrings)} URL pattern(s)")


def verify(
    data_dir: Path,
    *,
    expected_counts: Path | None = None,
    thumbs_dir: Path | None = None,
    excluded: dict | None = None,
) -> list[str]:
    """Run every check. Returns notes; raises :class:`VerificationFailed` on any failure."""
    data_dir = Path(data_dir)
    failures: list[str] = []
    notes: list[str] = []

    manifest = _load(data_dir, "manifest.json")
    datasets = _load(data_dir, "datasets.json")
    assets = _load(data_dir, "assets.json")
    facets = _load(data_dir, "facets.json")

    if expected_counts and Path(expected_counts).exists():
        expected = json.loads(Path(expected_counts).read_text(encoding="utf-8"))
        _check_invariants(manifest, expected, failures)
        notes.append(f"counts match the supplied expectations ({Path(expected_counts).name})")
    else:
        notes.append("no expected counts supplied — the export reports what the corpus holds")

    _check_structure(assets, datasets, failures)
    _check_privacy(data_dir, facets, failures, notes)
    _check_exclusions(datasets, excluded or {}, failures, notes)

    if thumbs_dir:
        thumbs_dir = Path(thumbs_dir)
        missing = 0
        # strict=False for the same reason as in _check_structure: any length
        # disagreement is already a recorded failure, not a reason to raise.
        for hex_id, flag in zip(
            assets["columns"]["id"], assets["single"]["thumb"], strict=False
        ):
            if flag and not (thumbs_dir / hex_id[:2] / f"{hex_id}.webp").exists():
                missing += 1
        if missing:
            failures.append(f"{missing} asset(s) flagged as having a thumbnail have no file")
        else:
            notes.append(f"all {sum(assets['single']['thumb'])} flagged thumbnails present on disk")

    if failures:
        raise VerificationFailed("\n".join(f"  - {f}" for f in failures))
    return notes
