"""Present saved exports using the current spreadsheet columns, without rewriting them."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import tempfile
from pathlib import Path

from .notes import analysis_notes

REMOVED_CSV_COLUMNS = frozenset({"caveats", "n_caveats"})


def spreadsheet_columns(columns: list[str]) -> list[int]:
    return [index for index, name in enumerate(columns) if name not in REMOVED_CSV_COLUMNS]


def _csv_export(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        columns = next(reader, [])
        keep = spreadsheet_columns(columns)
        if len(keep) == len(columns):
            return path.open("rb")
        output = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b")  # noqa: SIM115 -- the caller owns this stream.
        try:
            text = io.TextIOWrapper(output, encoding="utf-8-sig", newline="")
            writer = csv.writer(text)
            writer.writerow([columns[index] for index in keep])
            for row in reader:
                if row:
                    writer.writerow([row[index] if index < len(row) else "" for index in keep])
            text.flush()
            text.detach()
            output.seek(0)
            return output
        except Exception:
            output.close()
            raise


def _manifest_export(path: Path, result: dict | None):
    data = json.loads(path.read_text(encoding="utf-8"))
    original = copy.deepcopy(data)

    def update_notes(value):
        if isinstance(value, list):
            for item in value:
                update_notes(item)
        elif isinstance(value, dict):
            if isinstance(value.get("caveats"), list):
                value["caveats"] = analysis_notes({**(result or {}), "caveats": value["caveats"]})
            for item in value.values():
                update_notes(item)

    update_notes(data)
    outputs = data.get("outputs") or {}
    changed = False
    for entry in outputs.get("files", []):
        columns = entry.get("columns") or []
        if not REMOVED_CSV_COLUMNS.intersection(columns):
            continue
        filename = entry.get("filename", "")
        candidate = (path.parent / filename).resolve()
        if not candidate.is_relative_to(path.parent.resolve()) or candidate.suffix != ".csv" or not candidate.is_file():
            continue
        with _csv_export(candidate) as content:
            digest = hashlib.file_digest(content, "sha256").hexdigest()
            size = content.seek(0, io.SEEK_END)
        entry.update(sha256=digest, size_bytes=size, columns=[columns[i] for i in spreadsheet_columns(columns)])
        if filename == "image_summary.csv":
            entry["contents"] = "one row per image"
        changed = True
    if changed:
        outputs["encoding"] = "UTF-8 with a byte-order mark. Read CSV files as utf-8-sig."
        outputs["note"] = "Join objects.csv to image_summary.csv on image_key. Notes are shown in the analysis summary."
    if data == original:
        return path.open("rb")
    return io.BytesIO(json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))


def open_export(path: Path, *, result: dict | None = None):
    """Return a binary stream owned by the caller (including its close)."""
    if path.suffix.lower() == ".csv":
        return _csv_export(path)
    if path.name == "manifest.json":
        return _manifest_export(path, result)
    return path.open("rb")
