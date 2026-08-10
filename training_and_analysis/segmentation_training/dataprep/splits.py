"""Read the split CSVs and resolve every crop to its EM + label paths and manifest entry.

Resolution recipe:
  * collection == "gt":          em = ROOT/<dataset>/<image_path>;  label = em.replace(_em,_label)
  * collection == "openOrganelle": image_path = "openOrganelle/<ds>/<crop>/raw_xy.tif|raw_xz.tif";
    em (xy) = ROOT/<first>;  seg = <crop_dir>/seg_<organelle>.tif
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from ..constants import SPLIT_CSV
from .io import read_json


@dataclass
class CropRow:
    collection: str
    dataset: str
    crop_id: str
    image_path: str
    split: str
    subgroup: str
    modality: str
    scale_band: str
    tissue_context: str
    species_group: str

    @property
    def is_oo(self) -> bool:
        return self.collection == "openOrganelle"


def load_split_rows(corpus_root: str | Path, organelle: str) -> list[CropRow]:
    """Parse ``<corpus_root>/splits/<SPLIT_CSV[organelle]>`` into CropRow records."""
    csv_path = Path(corpus_root) / "splits" / SPLIT_CSV[organelle]
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")
    rows: list[CropRow] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(
                CropRow(
                    collection=r["collection"],
                    dataset=r["dataset"],
                    crop_id=r["crop_id"],
                    image_path=r["image_path"],
                    split=r["split"],
                    subgroup=r.get("subgroup", "") or "",
                    modality=r.get("modality", "") or "",
                    scale_band=r.get("scale_band", "") or "",
                    tissue_context=r.get("tissue_context", "") or "",
                    species_group=r.get("species_group", "") or "",
                )
            )
    return rows


def resolve_gt_paths(corpus_root: str | Path, row: CropRow) -> tuple[Path, Path]:
    """(em_path, label_path) for a gt-collection crop."""
    em = Path(corpus_root) / row.dataset / row.image_path
    label = Path(str(em).replace("_em.tif", "_label.tif"))
    return em, label


def resolve_oo_paths(corpus_root: str | Path, row: CropRow, organelle: str) -> tuple[Path, Path, Path]:
    """(raw_xy, raw_xz, seg_<organelle>) for an openOrganelle crop."""
    first = row.image_path.split("|")[0]
    em_xy = Path(corpus_root) / first
    crop_dir = em_xy.parent
    em_xz = crop_dir / "raw_xz.tif"
    seg = crop_dir / f"seg_{organelle}.tif"
    return em_xy, em_xz, seg


@dataclass
class ManifestCache:
    """Lazily loads + indexes each dataset's ``manifest.json`` crops by crop_id (gt collection)."""

    corpus_root: Path
    _cache: dict[str, dict] = field(default_factory=dict)

    def crop_entry(self, dataset: str, crop_id: str) -> dict | None:
        idx = self._cache.get(dataset)
        if idx is None:
            mpath = self.corpus_root / dataset / "manifest.json"
            idx = {}
            if mpath.exists():
                man = read_json(mpath)
                for c in man.get("crops", []):
                    idx[c.get("crop_id")] = c
            self._cache[dataset] = idx
        return idx.get(crop_id)


def sanitize_id(s: str) -> str:
    """Make a crop_id (which may contain '/' for OO) filesystem/key safe."""
    return s.replace("/", "__").replace("\\", "__").replace("|", "_")


def make_cache(corpus_root: str | Path) -> ManifestCache:
    return ManifestCache(Path(corpus_root))
