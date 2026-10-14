"""Saved spreadsheet previews preserve the export and bound each response."""

import csv

import pytest
from rest_framework.test import APIClient, APIRequestFactory

from quantem.analysis.models import AnalysisRun
from quantem.segmentation.api_views.analysis import AnalysisRunTableView

pytestmark = pytest.mark.django_db


@pytest.fixture
def saved_table(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    path = directory / "objects.csv"
    rows = [
        ["9007199254740993", "0.00000012345", 'Quoted, "object"\nsecond line'],
        ["2", "", "µm²"],
        ["3", "1.2345678901234567", "last"],
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["object_id", "area_um2", "note"])
        writer.writerows(rows)
    run = AnalysisRun.objects.create(status="SUCCESS", export_dir=str(directory))
    return run, path, rows


def test_preview_pages_exact_saved_values_without_changing_the_export(saved_table):
    run, path, rows = saved_table
    original = path.read_bytes()
    url = f"/api/analysis/{run.id}/table/objects.csv"
    first = APIClient().get(url, {"limit": 2})
    assert first.status_code == 200
    assert first.data == {
        "name": "objects.csv", "columns": ["object_id", "area_um2", "note"],
        "rows": rows[:2], "offset": 0, "limit": 2, "has_more": True,
    }
    last = APIClient().get(url, {"offset": 2, "limit": 2})
    assert last.data["rows"] == rows[2:]
    assert last.data["has_more"] is False
    assert path.read_bytes() == original


def test_preview_stops_reading_after_one_page_and_one_lookahead(saved_table):
    run, path, _rows = saved_table
    with path.open("a", encoding="utf-8") as stream:
        stream.write('"broken CSV after the first page')
    first = APIClient().get(f"/api/analysis/{run.id}/table/objects.csv", {"limit": 2})
    assert first.status_code == 200
    assert len(first.data["rows"]) == 2
    assert first.data["has_more"] is True


def test_preview_of_an_empty_table_keeps_its_headers(saved_table):
    run, path, _rows = saved_table
    path.write_text("object_id,area_um2\n", encoding="utf-8-sig")
    response = APIClient().get(f"/api/analysis/{run.id}/table/objects.csv")
    assert response.status_code == 200
    assert response.data["columns"] == ["object_id", "area_um2"]
    assert response.data["rows"] == []
    assert response.data["has_more"] is False


@pytest.mark.parametrize("query", [
    {"offset": -1}, {"offset": "bad"}, {"limit": 0}, {"limit": 501}, {"limit": "1.5"},
])
def test_preview_rejects_invalid_pagination(saved_table, query):
    run, _path, _rows = saved_table
    response = APIClient().get(f"/api/analysis/{run.id}/table/objects.csv", query)
    assert response.status_code == 400


def test_preview_keeps_export_path_guards_and_only_reads_csv(saved_table):
    run, path, _rows = saved_table
    outside = path.parent.parent / "outside.csv"
    outside.write_text("private", encoding="utf-8")
    request = APIRequestFactory().get("/")
    for name in ("../outside.csv", str(outside)):
        response = AnalysisRunTableView.as_view()(request, run_id=run.id, name=name)
        assert response.status_code == 400
    (path.parent / "manifest.json").write_text("{}", encoding="utf-8")
    assert APIClient().get(f"/api/analysis/{run.id}/table/manifest.json").status_code == 400
    assert APIClient().get(f"/api/analysis/{run.id}/table/missing.csv").status_code == 404
    run.export_dir = ""
    run.save(update_fields=["export_dir"])
    assert APIClient().get(f"/api/analysis/{run.id}/table/objects.csv").status_code == 404


def test_preview_handles_global_area_tables(saved_table):
    run, path, _rows = saved_table
    (path.parent / "area.csv").write_text(
        "name,foreground_percent\nWhole image,14.5\n", encoding="utf-8-sig"
    )
    response = APIClient().get(f"/api/analysis/{run.id}/table/area.csv")
    assert response.status_code == 200
    assert response.data["rows"] == [["Whole image", "14.5"]]
