"""Synthetic mini-corpus and mock-checkpoint builders, shared by the segmentation_training tests and
the experiment smoke checks.

No pytest / network / real-corpus / real-weights dependency. Builds a tiny ``segmentations``-shaped
tree exercising the corpus edge cases (full and sparse coverage; instance vs multi-class labels;
a null / unknown-scale 2D-TEM crop for the drop-branch; OpenOrganelle dual orientation) plus the
group2 split CSVs and a ``crops_metadata.csv`` carrying voxel_x_nm / scale_band, and writes a
randomly-initialised DINOv3 checkpoint + ``checkpoint_index.json``. Tiles are 64px for speed (the
sparse multi-class crop sits on a 96px canvas so its valid region is a strict inset).

Module top level is numpy + tifffile only; ``write_mock_checkpoint`` imports torch, dinov3 and
em_ssl lazily. Runs without a GPU.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tifffile

from .constants import CROPS_METADATA_COLUMNS, SPLIT_COLUMNS


def _write_tif(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr)


def _gt_dataset(root: Path, name: str, *, em, label, coverage, organelles_present, valid_region,
                annotation_bbox, voxel_x, modality, dimensionality, label_encoding,
                organelle_classes, meta_rows: list, split_rows: dict):
    """Write a one-crop gt dataset (manifest.json + crops/<id>_em.tif + _label.tif) and record its
    crops_metadata + any split-CSV rows. ``voxel_x`` may be None (unknown-scale)."""
    cid = f"{name}_00000"
    em_rel = f"crops/{cid}_em.tif"
    _write_tif(root / name / em_rel, em.astype(np.uint8))
    _write_tif(root / name / "crops" / f"{cid}_label.tif", label)
    voxel = {"x": voxel_x, "y": voxel_x}
    manifest = {
        "dataset": {
            "name": name,
            "modality": modality,
            "dimensionality": dimensionality,
            "label_encoding": label_encoding,
            "organelle_classes": organelle_classes,
            "coverage_tier": coverage,
        },
        "n_crops": 1,
        "tile_size": int(max(em.shape)),
        "crops": [{
            "crop_id": cid,
            "em_file": em_rel,
            "label_file": f"crops/{cid}_label.tif",
            "valid_region_in_canvas_xyxy": list(valid_region),
            "annotation_bbox_in_canvas_xyxy": list(annotation_bbox),
            "coverage_tier": coverage,
            "organelles_present": organelles_present,
            "voxel_size_nm": voxel,
        }],
    }
    (root / name / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    scale_band = "unknown" if voxel_x is None else ("2-6" if voxel_x < 6 else "6-15")
    meta_rows.append({
        "collection": "gt", "dataset": name, "crop_id": cid, "image_path": em_rel,
        "modality": modality, "dimensionality": dimensionality,
        "voxel_x_nm": ("" if voxel_x is None else voxel_x), "scale_band": scale_band,
        "tissue_context": "cultured_cell", "species_group": "human", "in_situ_status": "in_vitro",
        "external_annotation": "no", "organelles": ";".join(organelle_classes.values()),
        "coverage_tier": coverage, "official_split": "", "n_tiles": 1,
    })
    return cid


def _oo_crop(root: Path, dataset: str, crop: str):
    """Write one OpenOrganelle crop (matched 2nm res; 32^3 seg, 64px raw, 32px annotation window)."""
    base = root / "openOrganelle" / dataset / crop
    base.mkdir(parents=True, exist_ok=True)
    raw = (np.random.default_rng(1).integers(0, 255, (4, 64, 64))).astype(np.uint8)
    tifffile.imwrite(str(base / "raw_xy.tif"), raw, photometric="minisblack")
    tifffile.imwrite(str(base / "raw_xz.tif"), raw.copy(), photometric="minisblack")
    seg_mito = np.zeros((32, 32, 32), np.uint8)
    seg_mito[:, :, 16:32] = 7  # instance id 7, x in [16,32)
    seg_er = np.zeros((32, 32, 32), np.uint8)
    seg_er[:, :, 0:16] = 1  # ER, x in [0,16)
    seg_er[0:4, 0:4, 0:4] = 255  # unknown -> ignore
    _write_tif(base / "seg_mito.tif", seg_mito)
    _write_tif(base / "seg_er.tif", seg_er)
    res = [2.0, 2.0, 2.0]
    seg_common = {"resolution_nm_zyx": res, "physical_origin_nm_zyx": [0.0, 0.0, 0.0],
                  "shape_zyx": [32, 32, 32]}
    man = {
        "crop_id": f"{dataset}/{crop}", "dataset": dataset,
        "original_image": {"resolution_nm_zyx": res},
        "segmentations": [{"class_name": "mito", "file": "seg_mito.tif", "annotation_type": "instance_segmentation", **seg_common},
                          {"class_name": "er", "file": "seg_er.tif", "annotation_type": "semantic_segmentation", **seg_common}],
        "raw_xy": {"file": "raw_xy.tif", "sample_axis": "z", "tile_axes_rows_cols": ["y", "x"],
                   "shape_planes_rows_cols": [4, 64, 64], "plane_physical_nm": [0.0, 2.0, 4.0, 6.0],
                   "annotation_bbox_in_tile_px": {"y": [16, 48], "x": [16, 48]}},
        "raw_xz": {"file": "raw_xz.tif", "sample_axis": "y", "tile_axes_rows_cols": ["z", "x"],
                   "shape_planes_rows_cols": [4, 64, 64], "plane_physical_nm": [0.0, 2.0, 4.0, 6.0],
                   "annotation_bbox_in_tile_px": {"z": [16, 48], "x": [16, 48]}},
    }
    (base / "crop_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    return f"{dataset}/{crop}"


def build_synthetic_corpus(root) -> dict:
    """Write the synthetic segmentations tree + group2 split CSVs + crops_metadata.csv under ``root``."""
    root = Path(root)
    rng = np.random.default_rng(0)
    meta_rows: list[dict] = []

    # (1) full-coverage instance mito, calibrated 4 nm (FIB-SEM 3D).
    em_a = rng.integers(0, 255, (64, 64)).astype(np.uint8)
    lab_a = np.zeros((64, 64), np.uint8)
    lab_a[20:40, 20:40] = 9  # instance id 9
    _gt_dataset(root, "ds_full_inst", em=em_a, label=lab_a, coverage="full",
                organelles_present={"mitochondria": {"instances": 1, "area_px": 400}},
                valid_region=[0, 0, 64, 64], annotation_bbox=[0, 0, 64, 64],
                voxel_x=4.0, modality="FIB-SEM", dimensionality="3D",
                label_encoding="instance (nonzero = object id)", organelle_classes={"9": "mito"},
                meta_rows=meta_rows, split_rows={})

    # (2) sparse multi-class (er=1, endosome=2, mito=3), calibrated 2.5 nm (TEM 2D).
    em_b = np.zeros((96, 96), np.uint8)
    em_b[16:80, 16:80] = rng.integers(30, 220, (64, 64)).astype(np.uint8)
    lab_b = np.zeros((96, 96), np.uint8)
    lab_b[30:50, 30:50] = 3  # mito
    lab_b[55:70, 55:70] = 1  # er
    _gt_dataset(root, "ds_partial_mc", em=em_b, label=lab_b, coverage="sparse",
                organelles_present={"er": {"value": 1, "area_px": 225},
                                    "endosome": {"value": 2, "area_px": 100},
                                    "mitochondria": {"value": 3, "area_px": 400}},
                valid_region=[16, 16, 80, 80], annotation_bbox=[24, 24, 76, 76],
                voxel_x=2.5, modality="TEM", dimensionality="2D",
                label_encoding="1=ER 2=endosome 3=mitochondria",
                organelle_classes={"1": "er", "2": "endosome", "3": "mito"},
                meta_rows=meta_rows, split_rows={})

    # (3) full-coverage multi-class with mito renumbered to 2, calibrated 8 nm (SBEM 3D).
    em_c = rng.integers(0, 255, (64, 64)).astype(np.uint8)
    lab_c = np.zeros((64, 64), np.uint8)
    lab_c[10:30, 10:50] = 2  # mito value renumbered to 2
    _gt_dataset(root, "ds_full_mc", em=em_c, label=lab_c, coverage="full",
                organelles_present={"cell": {"value": 1, "area_px": 100},
                                    "mitochondria": {"value": 2, "area_px": 800}},
                valid_region=[0, 0, 64, 64], annotation_bbox=[0, 0, 64, 64],
                voxel_x=8.0, modality="SBEM", dimensionality="3D",
                label_encoding="1=cell 2=mitochondria", organelle_classes={"1": "cell", "2": "mito"},
                meta_rows=meta_rows, split_rows={})

    # (4) unknown-scale 2D-TEM ER crop (voxel_x=None) — exercises the null-scale drop branch.
    em_d = rng.integers(0, 255, (64, 64)).astype(np.uint8)
    lab_d = np.zeros((64, 64), np.uint8)
    lab_d[8:56, 24:40] = 1  # binary ER
    _gt_dataset(root, "ds_null_scale", em=em_d, label=lab_d, coverage="sparse",
                organelles_present={"er": {"value": 1, "area_px": 768}},
                valid_region=[0, 0, 64, 64], annotation_bbox=[8, 24, 40, 56],
                voxel_x=None, modality="TEM", dimensionality="2D",
                label_encoding="semantic binary (1 = endoplasmic reticulum)",
                organelle_classes={"1": "er"}, meta_rows=meta_rows, split_rows={})

    oo_id = _oo_crop(root, "jrc_synthetic", "crop0")
    meta_rows.append({
        "collection": "openOrganelle", "dataset": "jrc_synthetic", "crop_id": oo_id,
        "image_path": f"openOrganelle/{oo_id}/raw_xy.tif|raw_xz.tif", "modality": "FIB-SEM",
        "dimensionality": "3D_iso", "voxel_x_nm": 2.0, "scale_band": "0.5-2",
        "tissue_context": "cultured_cell", "species_group": "human", "in_situ_status": "in_vitro",
        "external_annotation": "yes", "organelles": "mito;er;ld", "coverage_tier": "full",
        "official_split": "", "n_tiles": 1,
    })

    # --- split CSVs (group2 = the segmentation held-out-source benchmark) + crops_metadata.csv ---------------
    splits_dir = root / "splits"
    splits_dir.mkdir(exist_ok=True)
    cols = list(SPLIT_COLUMNS)

    def _row(coll, ds, cid, ip, split, sub="", mod="FIB-SEM", sb="2-6", tc="cultured_cell", sp="human"):
        return dict(zip(cols, [coll, ds, cid, ip, split, sub, mod, sb, tc, sp]))

    mito_rows = [
        _row("gt", "ds_full_inst", "ds_full_inst_00000", "crops/ds_full_inst_00000_em.tif",
             "train_pool", sub="FIB-SEM | cultured | 4nm"),
        _row("gt", "ds_full_mc", "ds_full_mc_00000", "crops/ds_full_mc_00000_em.tif",
             "train_pool", sub="SBEM | cultured | 8nm"),
        _row("gt", "ds_partial_mc", "ds_partial_mc_00000", "crops/ds_partial_mc_00000_em.tif",
             "test", sub="TEM | cultured | 2-6nm"),
        _row("openOrganelle", "jrc_synthetic", oo_id,
             f"openOrganelle/{oo_id}/raw_xy.tif|raw_xz.tif", "test", sub="FIB | neuronal | 2nm", sb="0.5-2"),
    ]
    er_rows = [
        _row("gt", "ds_partial_mc", "ds_partial_mc_00000", "crops/ds_partial_mc_00000_em.tif",
             "train_pool", sub="TEM | cultured | 2-6nm"),
        _row("gt", "ds_null_scale", "ds_null_scale_00000", "crops/ds_null_scale_00000_em.tif",
             "test", sub="TEM | cultured | unknown", sb="unknown"),
        _row("openOrganelle", "jrc_synthetic", oo_id,
             f"openOrganelle/{oo_id}/raw_xy.tif|raw_xz.tif", "test", sub="FIB | neuronal | 2nm", sb="0.5-2"),
        _row("gt", "ds_not_on_disk", "ghost_00000", "crops/ghost_00000_em.tif", "test"),  # skip-with-warning
    ]
    for fname, rows in [("group2_mito.csv", mito_rows), ("group2_er.csv", er_rows)]:
        with open(splits_dir / fname, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    with open(root / "crops_metadata.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CROPS_METADATA_COLUMNS))
        w.writeheader()
        w.writerows(meta_rows)

    return {"root": root, "oo_id": oo_id}


def write_mock_checkpoint(run_dir, framework: str = "dinov3", arch: str = "vit_small", depth: int = 12,
                          embedding_dim: int = 384, step: int = 100):
    """Build a randomly-initialised in_chans=1 DINOv3 ViT backbone, save it in the on-disk format the
    loader expects, and write a ``checkpoint_index.json`` pointing at it. ``framework`` labels the
    manifest; the backbone built is the DINOv3 ViT in every case. Returns run_dir, giving the tests a
    frozen encoder on CPU without real weights."""
    import importlib

    import torch

    from em_ssl.utils.checkpoint_index import (
        CheckpointIndex, EncoderManifest, dinov3_feature_entry_point,
    )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    vits = importlib.import_module("dinov3.models.vision_transformer")
    model = vits.__dict__[arch](patch_size=16, in_chans=1)
    model.init_weights()
    sd = {f"backbone.{k}": v for k, v in model.state_dict().items()}
    ckpt_dir = run_dir / "eval" / str(step)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "teacher_checkpoint.pth"
    torch.save({"teacher": sd}, ckpt_path)
    fep, kind = dinov3_feature_entry_point(arch, depth), "teacher"
    manifest = EncoderManifest(
        run_id=f"mock_{framework}", framework=framework, objective=framework, arch=arch,
        patch_size=16, embedding_dim=embedding_dim, depth=depth, input_channels=1,
        feature_entry_point=fep,
    )
    idx = CheckpointIndex(run_dir, manifest)
    idx.add(step=step, kind=kind, path=str(ckpt_path), crop_size=64)
    idx.add(step=step * 2, kind=kind, path=str(ckpt_path), crop_size=64)
    idx.save()
    return run_dir
