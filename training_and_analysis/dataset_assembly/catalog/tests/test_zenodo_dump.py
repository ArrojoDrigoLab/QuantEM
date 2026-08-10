from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from catalog.sources import zenodo_dump


class ZenodoDumpScannerTests(unittest.TestCase):
    def test_dump_scanner_prefilters_dump_and_enriches_file_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dump_path = tmp_path / "records-xml.tar.gz"
            _write_dump(
                dump_path,
                {
                    "1001.xml": _datacite_xml(
                        "1001",
                        title="FIB-SEM mitochondria source data",
                        description="Open cellular ultrastructure image dataset.",
                        resource_type="Dataset",
                    ),
                    "1002.xml": _datacite_xml(
                        "1002",
                        title="FIB-SEM measurements",
                        description="Electron microscopy metadata with CSV tables only.",
                        resource_type="Dataset",
                    ),
                    "1003.xml": _datacite_xml(
                        "1003",
                        title="Climate model software",
                        description="No microscopy data.",
                        resource_type="Software",
                    ),
                    "1004.xml": _datacite_xml(
                        "1004",
                        title="TEM lung ultrastructure JPEG images",
                        description="Transmission electron microscopy images of alveolar epithelium.",
                        resource_type="Dataset",
                    ),
                },
            )

            detail_calls: list[str] = []

            def fake_get_json(url: str):
                detail_calls.append(url)
                if url.endswith("/1001"):
                    return _detail("1001", [{"key": "cell_001.tif", "size": 2048, "links": {"self": "https://files/1"}}])
                if url.endswith("/1002"):
                    return _detail("1002", [{"key": "measurements.csv", "size": 128, "links": {"self": "https://files/2"}}])
                if url.endswith("/1004"):
                    return _detail("1004", [{"key": "montage.jpg", "size": 4096, "links": {"self": "https://files/4"}}])
                raise AssertionError(f"unexpected detail URL {url}")

            with patch.object(zenodo_dump, "get_json", side_effect=fake_get_json):
                result = zenodo_dump.scan_zenodo_dump(
                    root=tmp_path,
                    limit=10,
                    dump_path=dump_path,
                    candidate_cache_path=tmp_path / "cache" / "candidates.jsonl",
                )

            self.assertEqual(result.errors, [])
            self.assertTrue(result.cursor_complete)
            self.assertEqual([candidate.source_record_id for candidate in result.candidates], ["1001", "1002", "1004"])
            candidate = result.candidates[0]
            self.assertEqual(candidate.source_name, "zenodo")
            self.assertEqual(candidate.dataset_doi, "10.5281/zenodo.1001")
            self.assertIn("TIF", candidate.file_formats)
            self.assertIn("dump_record", candidate.raw_metadata)
            self.assertIn("JPG", result.candidates[2].file_formats)
            self.assertEqual(len(detail_calls), 3)

            meta = json.loads((tmp_path / "cache" / "candidates.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["candidate_count"], 3)
            self.assertEqual(meta["stats"]["metadata_prefilter_matches"], 3)

    def test_dump_scanner_pages_from_candidate_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_path = tmp_path / "cache" / "candidates.jsonl"
            cache_path.parent.mkdir(parents=True)
            rows = [
                _candidate_row("1001"),
                _candidate_row("1002"),
            ]
            cache_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            (tmp_path / "cache" / "candidates.meta.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "candidate_cache_filter_version": zenodo_dump.CANDIDATE_CACHE_FILTER_VERSION,
                        "dump_version_id": f"local:{(tmp_path / 'dump.tar.gz').resolve()}:0",
                        "candidate_count": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            dump_path = tmp_path / "dump.tar.gz"
            dump_path.write_bytes(b"")

            first = zenodo_dump.scan_zenodo_dump(
                root=tmp_path,
                limit=1,
                dump_path=dump_path,
                candidate_cache_path=cache_path,
            )
            second = zenodo_dump.scan_zenodo_dump(
                root=tmp_path,
                limit=1,
                cursor=first.cursor,
                dump_path=dump_path,
                candidate_cache_path=cache_path,
            )

        self.assertEqual(first.errors, [])
        self.assertFalse(first.cursor_complete)
        self.assertEqual(first.candidates[0].source_record_id, "1001")
        self.assertEqual(second.candidates[0].source_record_id, "1002")
        self.assertTrue(second.cursor_complete)

    def test_dump_scanner_rebuilds_stale_filter_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_path = tmp_path / "cache" / "candidates.jsonl"
            cache_path.parent.mkdir(parents=True)
            cache_path.write_text(json.dumps(_candidate_row("old")) + "\n", encoding="utf-8")
            (tmp_path / "cache" / "candidates.meta.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "candidate_cache_filter_version": zenodo_dump.CANDIDATE_CACHE_FILTER_VERSION - 1,
                        "dump_version_id": f"local:{(tmp_path / 'records-xml.tar.gz').resolve()}:0",
                        "candidate_count": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            dump_path = tmp_path / "records-xml.tar.gz"
            _write_dump(
                dump_path,
                {
                    "1001.xml": _datacite_xml(
                        "1001",
                        title="TEM lung ultrastructure JPEG images",
                        description="Transmission electron microscopy images of alveolar epithelium.",
                        resource_type="Dataset",
                    )
                },
            )

            with patch.object(
                zenodo_dump,
                "get_json",
                return_value=_detail("1001", [{"key": "montage.jpg", "size": 4096, "links": {"self": "https://files/1"}}]),
            ):
                result = zenodo_dump.scan_zenodo_dump(
                    root=tmp_path,
                    limit=10,
                    dump_path=dump_path,
                    candidate_cache_path=cache_path,
                )

        self.assertEqual([candidate.source_record_id for candidate in result.candidates], ["1001"])


def _write_dump(path: Path, members: dict[str, str]) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for name, text in members.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _datacite_xml(record_id: str, *, title: str, description: str, resource_type: str) -> str:
    return f"""<?xml version='1.0' encoding='UTF-8'?>
<oai_datacite xmlns="http://schema.datacite.org/oai/oai-1.1/">
  <payload>
    <resource xmlns="http://datacite.org/schema/kernel-4">
      <alternateIdentifiers>
        <alternateIdentifier alternateIdentifierType="URL">https://zenodo.org/records/{record_id}</alternateIdentifier>
      </alternateIdentifiers>
      <titles><title>{title}</title></titles>
      <publisher>Zenodo</publisher>
      <publicationYear>2026</publicationYear>
      <resourceType resourceTypeGeneral="{resource_type}">{resource_type}</resourceType>
      <identifier identifierType="DOI">10.5281/zenodo.{record_id}</identifier>
      <rightsList>
        <rights rightsURI="info:eu-repo/semantics/openAccess">Open Access</rights>
      </rightsList>
      <descriptions><description descriptionType="Abstract">{description}</description></descriptions>
    </resource>
  </payload>
</oai_datacite>"""


def _detail(record_id: str, files: list[dict]) -> dict:
    return {
        "id": record_id,
        "doi": f"10.5281/zenodo.{record_id}",
        "links": {"html": f"https://zenodo.org/records/{record_id}", "self": f"https://zenodo.org/api/records/{record_id}"},
        "metadata": {
            "title": f"FIB-SEM record {record_id}",
            "description": "Cellular ultrastructure source images.",
            "publication_date": "2026-01-01",
            "resource_type": {"type": "dataset", "title": "Dataset"},
            "access_right": "open",
            "license": {"id": "cc-by-4.0", "title": "Creative Commons Attribution 4.0"},
        },
        "files": files,
    }


def _candidate_row(record_id: str) -> dict:
    return {
        "source_name": "zenodo",
        "source_record_id": record_id,
        "title": f"FIB-SEM record {record_id}",
        "landing_url": f"https://zenodo.org/records/{record_id}",
        "download_or_manifest_urls": [f"https://zenodo.org/api/records/{record_id}"],
        "publication_doi": None,
        "dataset_doi": f"10.5281/zenodo.{record_id}",
        "modality": None,
        "organism": None,
        "tissue_or_sample": None,
        "dimensions_or_image_count": "1 files",
        "file_formats": ["TIF"],
        "license": "cc-by-4.0",
        "raw_metadata": {"files": [{"key": "cell.tif"}]},
        "evidence_text": "FIB-SEM cellular ultrastructure",
        "discovered_at": "2026-01-01",
    }


if __name__ == "__main__":
    unittest.main()
