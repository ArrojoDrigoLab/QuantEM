"""Reproducible run-manifest provenance dumped at the start of every training run.

Writes into ``<run_dir>``:
    resolved_config.yaml            # the normalized ExperimentSpec actually used
    experiment_config.source.yaml   # verbatim copy of the config file, when the spec came from one
    git_commit.txt                  # repo commit, branch, dirty flag, and the DINOv3 pin
    environment.txt                 # interpreter + package versions + pip freeze
    system_info.json                # GPUs/CPU/RAM/disk/SLURM/distributed env
    dataset_fingerprint.json        # snapshot copied from the data bundle (if present)
    shard_index_snapshot.json       # snapshot of the shard index
    source_distribution_snapshot.csv
    tile_intensity_stats.json       # corpus intensity stats, copied from the same bundle
    run_extra.json                  # caller-supplied ``extra`` fields, when given
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from ..config.schema import ExperimentSpec
from ..utils.reproducibility import dump_environment, get_git_commit
from ..utils.system_probe import probe_system

def _bundle_manifests_dir(spec: ExperimentSpec) -> Path | None:
    """Locate the data-bundle manifests/ dir from the shard_dir layout."""
    if not spec.data.shard_dir:
        return None
    sd = Path(spec.data.shard_dir)
    parts = sd.parts
    if "shards" in parts:
        idx = parts.index("shards")
        cand = Path(*parts[:idx]) / "manifests"
        if cand.is_dir():
            return cand
    # Fallbacks: manifests/ beside the shard dir, manifests/ one level above it, then data_prep/.
    for cand in (sd.parent / "manifests", sd.parent.parent / "manifests", sd.parent.parent / "data_prep"):
        if cand.is_dir():
            return cand
    return None

def dump_run_manifest(run_dir: str | Path, spec: ExperimentSpec, extra: dict[str, Any] | None = None) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # resolved spec
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    if spec.config_path:
        try:
            (run_dir / "experiment_config.source.yaml").write_text(
                Path(spec.config_path).read_text(encoding="utf-8"), encoding="utf-8"
            )
        except Exception:
            pass

    # git + env + system
    (run_dir / "git_commit.txt").write_text(json.dumps(get_git_commit(), indent=2), encoding="utf-8")
    try:
        dump_environment(run_dir / "environment.txt")
    except Exception:
        pass
    try:
        (run_dir / "system_info.json").write_text(json.dumps(probe_system(), indent=2, default=str), encoding="utf-8")
    except Exception:
        pass

    # snapshots from the data bundle
    md = _bundle_manifests_dir(spec)
    if md is not None:
        for src_name, dst_name in (
            ("dataset_fingerprint.json", "dataset_fingerprint.json"),
            ("shard_index.json", "shard_index_snapshot.json"),
            ("source_distribution.csv", "source_distribution_snapshot.csv"),
            ("tile_intensity_stats.json", "tile_intensity_stats.json"),
        ):
            src = md / src_name
            if src.exists():
                try:
                    shutil.copyfile(src, run_dir / dst_name)
                except Exception:
                    pass

    if extra:
        (run_dir / "run_extra.json").write_text(json.dumps(extra, indent=2, default=str), encoding="utf-8")
