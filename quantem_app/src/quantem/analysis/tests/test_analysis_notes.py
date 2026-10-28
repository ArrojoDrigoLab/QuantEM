import csv
import hashlib
import io
import json

import pytest
from rest_framework.test import APIClient

from quantem.analysis.models import AnalysisRun
from quantem.analysis.notes import analysis_notes

TISSUE = "No tissue mask was supplied: fractions are relative to the whole image, including any empty resin or background."
REVIEW = "No area of this image is recorded as reviewed, so the bundle cannot say how much of it a person went through. Every count and fraction covers the whole image regardless."
SCALE = "The scale inference ran at in compartment 'mito' is unknown, not native: all 52 of its confirmed objects were drawn by hand, so no model output survives in it."
PARTIAL = "1 of 21 metrics is measured on fewer than the 52 confirmed objects (mean_prob). It carries its own n and the reason for it in the summary table; it is not a whole-image number."


@pytest.fixture
def saved_results():
    return {
        "caveats": [TISSUE, REVIEW, PARTIAL, SCALE],
        "objects": {
            "n": 52,
            "summary": {
                **{f"metric_{index}": {"n": 52} for index in range(20)},
                "mean_prob": {"n": 0, "n_missing": 52, "missing_by_source": {"manual": 52}},
            },
        },
    }


def test_saved_notes_remove_unwanted_messages_and_explain_partial_metrics(saved_results):
    notes = analysis_notes(saved_results)
    assert notes == [TISSUE]
    assert len(saved_results["caveats"]) == 4


def test_other_notes_and_actual_missing_metric_reasons_are_retained(saved_results):
    saved_results["objects"]["summary"]["area_px"] = {
        "n": 50,
        "n_missing": 2,
        "note": "Measured on 50 of 52 confirmed objects; 2 carry no stored value. Missing sources: 2 model objects.",
    }
    other = "The scale inference ran at in compartment 'mito' is unknown: an unrecognised model produced these objects."
    saved_results["caveats"].append(other)
    notes = analysis_notes(saved_results)
    assert "2 carry no stored value. Missing sources: 2 model objects." in notes[1]
    assert other not in notes
    assert all("summary table" not in note for note in notes)


def test_identical_missing_metric_reasons_are_not_repeated(saved_results):
    stats = {
        "n": 51,
        "n_missing": 1,
        "note": "Measured on 51 of 52 confirmed objects; 1 object has no saved measurements.",
    }
    saved_results["objects"]["summary"] = {"area_px": stats, "perimeter_px": stats}
    note = analysis_notes(saved_results)[1]
    assert "2 metrics: 1 object has no saved measurements." in note
    assert note.count("1 object has no saved measurements.") == 1


@pytest.mark.django_db
def test_saved_csv_preview_download_and_manifest_use_the_same_columns(tmp_path, saved_results):
    path = tmp_path / "image_summary.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_key", "caveats", "area_um2", "n_caveats"])
        writer.writerow(
            ["9007199254740993", " | ".join(saved_results["caveats"]), "0.00000012345", "4"]
        )
    original = path.read_bytes()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "images": [{"caveats": saved_results["caveats"]}],
                "outputs": {
                    "files": [
                        {
                            "filename": path.name,
                            "sha256": hashlib.sha256(original).hexdigest(),
                            "size_bytes": len(original),
                            "columns": ["image_key", "caveats", "area_um2", "n_caveats"],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    original_manifest = manifest_path.read_bytes()
    run = AnalysisRun.objects.create(
        status="SUCCESS", export_dir=str(tmp_path), results=saved_results
    )
    client = APIClient()
    preview = client.get(f"/api/analysis/{run.id}/table/image_summary.csv").data
    assert preview["columns"] == ["image_key", "area (µm²)"]
    assert preview["rows"] == [["9007199254740993", "0.00000012345"]]
    response = client.get(f"/api/analysis/{run.id}/export/image_summary.csv")
    downloaded = b"".join(response.streaming_content)
    # The test client's iterator closes the response with its signal guards.
    # Closing it again ends PostgreSQL's surrounding test transaction.
    assert response.closed
    assert list(csv.reader(io.StringIO(downloaded.decode("utf-8-sig")))) == [
        preview["columns"],
        *preview["rows"],
    ]
    response = client.get(f"/api/analysis/{run.id}/export/manifest.json")
    manifest = json.loads(b"".join(response.streaming_content))
    assert response.closed
    entry = manifest["outputs"]["files"][0]
    assert entry["sha256"] == hashlib.sha256(downloaded).hexdigest()
    assert entry["size_bytes"] == len(downloaded)
    assert entry["columns"] == preview["columns"]
    assert manifest["images"][0]["caveats"] == analysis_notes(saved_results)
    assert client.get(f"/api/analysis/{run.id}/").data["caveats"] == analysis_notes(saved_results)
    run.refresh_from_db()
    assert run.results == saved_results
    assert path.read_bytes() == original
    assert manifest_path.read_bytes() == original_manifest
