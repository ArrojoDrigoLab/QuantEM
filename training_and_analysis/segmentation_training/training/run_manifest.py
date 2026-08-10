"""Run provenance for a segmentation arm — mirrors em_ssl's run_manifest conventions.

Writes, into the arm's run dir: ``resolved_config.yaml`` (the fully-resolved SegConfig), a
``run.json`` marker (the discovery key the aggregator scans for), ``git_commit.txt`` and
``environment.txt`` (reusing em_ssl.utils.reproducibility so env/commit capture is identical to the
SSL runs). Per-step training progress is not written here: the training loop in
``harness/train.py`` mirrors it to ``progress.json`` in the same run dir, and ``harness/run_seg.py``
writes the evaluation outputs (``results.json``, ``results_per_crop.json``, ``results.csv``) there too.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def dump_run_manifest(run_dir, cfg, extra: dict | None = None) -> Path:
    """Write the resolved config + provenance + the run.json discovery marker. Returns run_dir."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = cfg.to_dict()
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    git = {}
    try:
        from em_ssl.utils.reproducibility import get_git_commit
        git = get_git_commit()
    except Exception:
        git = {"commit": None}
    (run_dir / "git_commit.txt").write_text(json.dumps(git, indent=2), encoding="utf-8")
    try:
        from em_ssl.utils.reproducibility import dump_environment
        dump_environment(run_dir / "environment.txt")
    except Exception:
        pass

    marker = {
        "name": cfg.name,
        "organelle": cfg.data.organelle,
        "canonical_nm": cfg.data.resolved_canonical_nm(),
        "neck": cfg.neck.type,
        "decoder": cfg.decoder.type,
        "loss": [t.type for t in cfg.loss.terms],
        "task": cfg.data.task,
        "encoder_run_dir": cfg.encoder.run_dir,
        "git_commit": git.get("commit"),
        "config_path": cfg.config_path,
        **(extra or {}),
    }
    (run_dir / "run.json").write_text(json.dumps(marker, indent=2, default=str), encoding="utf-8")
    return run_dir
