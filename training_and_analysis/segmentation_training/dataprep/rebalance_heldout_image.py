"""Rebalance a segmentation_training manifest to add a held-out-image split (``test_image``): a
split-label reshuffle only, with no image regeneration (``split`` is a manifest field; files load via
``em_path``).

Held-out-image measures generalization to unseen images of training sources (same appearance and same
annotation convention), which decomposes the val->test gap:
    val->test_image gap   = train-image overfitting (small, expected)
    test_image->test gap  = cross-source appearance shift (the large, addressable target)

For each training source a spatially disjoint band is held out — a contiguous slice by z-position for
OpenOrganelle volumes, else by stable crop ordering — so the measurement is held-out-image
generalization rather than memorization of the crops adjacent to training crops. Bands are taken from
every sufficiently large training source, giving a representative mix rather than cultured cells
alone. val / test (held-out-source) are untouched.

Usage:
    python -m segmentation_training.dataprep.rebalance_heldout_image \
        --manifest <supplied at launch> \
        --out <supplied at launch> --organelle mito --holdout-frac 0.15

Then run the decomposition with ``data.manifest_name: manifest_heldimg.jsonl`` (the harness also scores
the ``test_image`` split automatically when present).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _volume_key(r: dict):
    """Group crops into a 'volume' whose members are spatially contiguous: OpenOrganelle registers per
    orientation, so use (dataset, orientation); other sources use (dataset,)."""
    ds = r.get("dataset", "?")
    orient = r.get("orientation")
    return (ds, orient) if orient else (ds,)


def _spatial_key(r: dict):
    """Within-volume sort key so the held-out band is a contiguous spatial slice (disjoint from the
    retained training crops): z-position for OpenOrganelle, else the stable crop id."""
    z = r.get("plane_z_nm")
    if z is not None:
        return (0, float(z), "")
    return (1, 0.0, str(r.get("crop_id", r.get("sample_id", ""))))


def rebalance(records, organelle, holdout_frac=0.15, min_crops=20, sources=None,
              src_split="train", new_split="test_image"):
    """Return ``(new_records, stats)``: sort each eligible training volume by ``_spatial_key`` and move
    the contiguous band at the high end of that key (fraction ``holdout_frac``) to ``new_split``.
    ``sources`` (dataset-name prefixes) restricts which sources are held from; None = every training
    source with >= ``min_crops`` crops (representative)."""
    by_vol: dict = defaultdict(list)
    src_total: dict = defaultdict(int)
    for r in records:
        if r.get("organelle") != organelle or r.get("split") != src_split:
            continue
        ds = r.get("dataset", "?")
        src_total[ds] += 1
        if sources and not any(ds.startswith(s) for s in sources):
            continue
        by_vol[_volume_key(r)].append(r)

    held: set = set()
    per_src: dict = defaultdict(int)
    for vol, crops in by_vol.items():
        if len(crops) < min_crops:
            continue
        crops.sort(key=_spatial_key)
        k = max(1, int(round(holdout_frac * len(crops))))
        for r in crops[-k:]:
            held.add(r.get("sample_id"))
        per_src[vol[0]] += k

    out = []
    for r in records:
        if r.get("sample_id") in held and r.get("split") == src_split:
            r = {**r, "split": new_split}
        out.append(r)
    stats = {"n_held": len(held),
             "per_source": {ds: (per_src[ds], src_total[ds]) for ds in sorted(per_src)}}
    return out, stats


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Add a held-out-image (test_image) split to a segmentation_training manifest.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--organelle", required=True, choices=["er", "mito"])
    p.add_argument("--holdout-frac", type=float, default=0.15)
    p.add_argument("--min-crops", type=int, default=20)
    p.add_argument("--sources", nargs="*", default=None,
                   help="Dataset-name prefixes to hold from (default: all training sources >= min-crops).")
    p.add_argument("--new-split", default="test_image")
    a = p.parse_args(argv)

    records = [json.loads(line) for line in open(a.manifest, encoding="utf-8") if line.strip()]
    out, stats = rebalance(records, a.organelle, a.holdout_frac, a.min_crops, a.sources,
                           new_split=a.new_split)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"[rebalance] {a.organelle}: held {stats['n_held']} crops -> '{a.new_split}' across "
          f"{len(stats['per_source'])} sources -> {a.out}")
    for ds, (h, t) in stats["per_source"].items():
        print(f"    {ds:34s} held {h:4d} / {t:4d} train crops")


if __name__ == "__main__":
    main()
