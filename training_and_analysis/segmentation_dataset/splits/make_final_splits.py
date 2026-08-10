#!/usr/bin/env python
"""Build the FINAL train/val/test dataset splits (per organelle).

Design:
  * Splits are PER ORGANELLE and independent — a crop may be TEST for one organelle
    and TRAIN for another (this is expected and correct; the test benchmark is per
    organelle).
  * TEST = the held-out benchmark crops, reusing the exact TESTSETS definition from
    make_benchmark_splits.py (single source of truth for what "test" means).
  * TRAIN/VAL = every OTHER crop that has that organelle, split ~80/20 purely by
    NUMBER OF CROPS (n_tiles ignored for balancing). No dataset-level holdouts and no
    designed image-level holdouts inside train/val.
  * The ONLY thing removed from a given organelle's train/val pool is that organelle's
    own test crops PLUS any non-test crop that shares a source image with one of that
    organelle's test crops (within-organelle test-leak guard). Crops that merely live
    on a specimen held out for a DIFFERENT organelle stay in the pool (per-organelle
    rule).

Emits under <corpus root>/splits/:
  final_mito.csv final_er.csv final_nucleus.csv final_ld.csv
      (collection,dataset,crop_id,image_path,split,subgroup,modality,scale_band,
       tissue_context,species_group,organelle,source_image,split_role)
  final_dataset.csv   long/consolidated: one row per (crop x organelle) with split.

Deterministic: val = first ceil(0.20 * pool) crops by a stable sha1 hash of
"organelle/dataset/crop_id". Re-run after any re-ingest. Pure stdlib.
"""
import csv, os, math, collections, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_benchmark_splits import (FAMILY, TESTSETS, SUBGROUP, orgset, fams,
                                   load_srcmap, h01)

ROOT = os.environ.get("SEG_CORPUS_ROOT", ".")
SPLITS = os.path.join(ROOT, "splits"); os.makedirs(SPLITS, exist_ok=True)
VAL_FRAC = 0.20
ORGS = ("mito", "er", "nucleus", "ld")

FIELDS = ["collection", "dataset", "crop_id", "image_path", "split", "subgroup",
          "modality", "scale_band", "tissue_context", "species_group",
          "organelle", "source_image", "split_role"]


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "crops_metadata.csv"))))
    srcmap, msplit = load_srcmap()

    # ---- 1) TEST crops per organelle + that organelle's held-out source images ----
    test = collections.defaultdict(dict)          # org -> {(ds,cid): (row, subgroup, src, role)}
    test_src = collections.defaultdict(set)        # org -> {(ds, source_image)}
    for r in rows:
        ds, cid = r["dataset"], r["crop_id"]
        if ds not in TESTSETS:
            continue
        designated, cfilt, adopt = TESTSETS[ds]
        sp = (r.get("official_split") or "").strip() or msplit.get((ds, cid), "")
        if not cfilt(r, sp):
            continue
        si = srcmap.get((ds, cid), "")
        role = "test" + (f"(official={sp})" if adopt else "")
        for org in designated:
            if org in fams(orgset(r)):
                test[org][(ds, cid)] = (r, SUBGROUP.get((ds, org), ""), si, role)
                test_src[org].add((ds, si))

    def wrow(w, r, split, sg, org, si, role):
        w.writerow([r["collection"], r["dataset"], r["crop_id"], r["image_path"], split,
                    sg, r["modality"], r["scale_band"], r["tissue_context"],
                    r["species_group"], org, si, role])

    consolidated = []   # (dataset, crop_id, organelle, split, source_image, split_role)
    summary = {}
    for org in ORGS:
        toks = FAMILY[org]
        has = [r for r in rows if orgset(r) & toks]
        testkeys = test[org]

        pool, leak_guarded = [], 0
        for r in has:
            key = (r["dataset"], r["crop_id"])
            if key in testkeys:
                continue
            si = srcmap.get(key, "")
            if (r["dataset"], si) in test_src[org]:   # within-org test-image guard
                leak_guarded += 1
                continue
            pool.append(r)

        # deterministic ~80/20 by CROP COUNT (each crop weight 1)
        pool_sorted = sorted(pool, key=lambda r: h01(f"{org}/{r['dataset']}/{r['crop_id']}"))
        n_val = math.ceil(VAL_FRAC * len(pool_sorted))
        split_of = {}
        for i, r in enumerate(pool_sorted):
            split_of[id(r)] = "val" if i < n_val else "train"

        counts = collections.Counter()
        with open(os.path.join(SPLITS, f"final_{org}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            for (ds, cid), (r, sg, si, role) in sorted(testkeys.items()):
                wrow(w, r, "test", sg, org, si, role)
                counts["test"] += 1
                consolidated.append((ds, cid, org, "test", si, role))
            for r in pool_sorted:
                sp = split_of[id(r)]
                si = srcmap.get((r["dataset"], r["crop_id"]), "")
                wrow(w, r, sp, "", org, si, sp)
                counts[sp] += 1
                consolidated.append((r["dataset"], r["crop_id"], org, sp, si, sp))

        summary[org] = (len(has), counts["test"], counts["train"], counts["val"], leak_guarded)
        tot_tv = counts["train"] + counts["val"] or 1
        print(f"final_{org}.csv: has={len(has)} test={counts['test']} "
              f"train={counts['train']} val={counts['val']} "
              f"({100*counts['val']/tot_tv:.1f}% val by crops; guard-dropped={leak_guarded})")

    # ---- consolidated long-format file ----
    with open(os.path.join(SPLITS, "final_dataset.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "crop_id", "organelle", "split", "source_image", "split_role"])
        for row in sorted(consolidated):
            w.writerow(row)
    print(f"\nfinal_dataset.csv: {len(consolidated)} (crop x organelle) rows")
    print("  per-organelle (test/train/val/guard):")
    for org in ORGS:
        h, te, tr, va, g = summary[org]
        print(f"    {org:8} test={te:4} train={tr:4} val={va:4}  (pool={tr+va}, has={h}, guard={g})")


if __name__ == "__main__":
    main()
