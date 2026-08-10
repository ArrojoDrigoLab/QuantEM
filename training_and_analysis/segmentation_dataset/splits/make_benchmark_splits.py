#!/usr/bin/env python
"""Build the held-out benchmark split CSVs from crops_metadata.csv.

The benchmark is a distinct dataset configuration from group1/group2. The datasets below are the
per-organelle held-out TEST sets; EVERYTHING ELSE with that organelle goes to train/val, split
~80/20 at the tile level (deterministic, no dataset/source-level holdouts). Held-out TEST *images*
(source images that contribute any test crop) are excluded from ALL organelles' train/val (whole
image-level holdout) so a test specimen never appears in training.

Emits under <corpus root>/splits/:
  benchmark_mito.csv benchmark_er.csv benchmark_nucleus.csv benchmark_ld.csv
       (collection,dataset,crop_id,image_path,split,subgroup,modality,scale_band,tissue_context,
        species_group,organelle,source_image,split_role)
  benchmark_test.csv   consolidated TEST rows (every test crop x organelle; split_role marks split-adopt)

Split-adopt datasets (orgsegnet, deeppi) contribute only their official 'test' split to TEST;
their train/val split stays in the train pool. Re-run after re-ingest. Pure stdlib.
"""
import csv, json, os, glob, collections, hashlib

ROOT = os.environ.get("SEG_CORPUS_ROOT", ".")
SPLITS = os.path.join(ROOT, "splits"); os.makedirs(SPLITS, exist_ok=True)
VAL_FRAC = 0.20

FAMILY = {
    "mito":    {"mito"},
    "er":      {"er", "er_sheets", "er_tubules", "peripheral_er", "er_other_cells",
                "er_at_the_close_proximity_to_chromosomes_(at_320_nm_distance)"},
    "nucleus": {"nucleus", "nuclei"},
    "ld":      {"ld", "lipid_droplets", "lipid_droplet_above_nucleus",
                "lipid_droplet_below_nucleus", "lipid_droplet_touching_nucleus"},
}
def fams(oset): return {k for k, toks in FAMILY.items() if oset & toks}

ALL   = lambda r, sp: True
TEST  = lambda r, sp: sp == "test"
MOSSY = lambda r, sp: r["crop_id"].startswith("ME2-Mossy")
def LDF(r, sp): return bool(orgset(r) & FAMILY["ld"])

# dataset -> (designated organelles, crop_filter, split_adopt)
TESTSETS = {
    "zenodo_mitoem2":                 (("mito",),           MOSSY, False),
    "empiar_10982_mitonet_benchmark": (("mito",),           ALL,   False),
    "orgsegnet_plant":                (("mito", "nucleus"), TEST,  True),
    "deeppi_em_skeletal_muscle":      (("mito",),           TEST,  True),
    "deepcontact_tem":                (("mito", "er"),      ALL,   False),
    "empiar_10994_hela_sbfsem":       (("er",),             ALL,   False),
    "empiar_13156_hela_stard3_er":    (("er",),             ALL,   False),
    "lab_islet_liver_er":             (("er",),             ALL,   False),
    "empiar_12885_aive":              (("er", "ld"),        ALL,   False),
    "zenodo_3675220_platynereis":     (("nucleus",),        ALL,   False),
    "sbiad2822_nuclei":               (("nucleus",),        ALL,   False),
    "segapp_islet_nucleus":           (("nucleus",),        ALL,   False),
    "empiar_13420_macrophage_a431":   (("ld",),             LDF,   False),
    "deepcontact_cell":               (("ld",),             LDF,   False),
    # empiar_10791 (Parlakgul liver ER) is NOT a test set: its volumes are part of the SSL
    #   pretraining corpus, so it is an ER TRAIN source; empiar_12885 (AIVE, absent from the
    #   pretraining corpus) takes the ER test slot.
}
SUBGROUP = {  # short test-subgroup label per (dataset, organelle)
    ("zenodo_mitoem2","mito"): "vEM/FIB | cerebellum (ME2-Mossy)",
    ("empiar_10982_mitonet_benchmark","mito"): "TEM+FIB+SBF | MitoNet multi-tissue benchmark",
    ("orgsegnet_plant","mito"): "TEM | plant [underrep kingdom]",
    ("orgsegnet_plant","nucleus"): "TEM | plant [underrep kingdom]",
    ("deeppi_em_skeletal_muscle","mito"): "TEM | mouse skeletal muscle [underrep]",
    ("deepcontact_tem","mito"): "TEM | COS-7 cultured",
    ("deepcontact_tem","er"): "TEM | COS-7 cultured",
    ("empiar_10994_hela_sbfsem","er"): "SBF-SEM | HeLa (REEP3/4)",
    ("empiar_13156_hela_stard3_er","er"): "FIB | HeLa STARD3",
    ("lab_islet_liver_er","er"): "SEM | in-house islet+liver ER (sparse)",
    ("empiar_12885_aive","er"): "FIB | AIVE cultured/muscle (ER)",
    ("zenodo_3675220_platynereis","nucleus"): "SBEM | annelid whole-body [underrep]",
    ("sbiad2822_nuclei","nucleus"): "AT+FIB | mixed (instance nucleus)",
    ("segapp_islet_nucleus","nucleus"): "SEM | in-house pancreatic islet (3 blocks)",
    ("empiar_13420_macrophage_a431","ld"): "FIB/SBF | macrophage/A431 (LD)",
    ("deepcontact_cell","ld"): "SEM | U-2 OS (DeepContact LD)",
    ("empiar_12885_aive","ld"): "FIB | AIVE cultured (LD)",
}

def orgset(r):
    s = r.get("organelles", "") or ""
    return set(t for t in s.replace("|", ";").split(";") if t)

def load_srcmap():
    src, msplit = {}, {}
    for mf in glob.glob(os.path.join(ROOT, "*", "manifest.json")):
        folder = os.path.basename(os.path.dirname(mf))
        try: m = json.load(open(mf, encoding="utf-8"))
        except Exception: continue
        for c in m.get("crops", []):
            cid = c.get("crop_id", "")
            src[(folder, cid)] = c.get("source_image") or c.get("em_file") or ""
            if "split" in c: msplit[(folder, cid)] = c.get("split")
    return src, msplit

def h01(s):  # stable [0,1) hash
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "crops_metadata.csv"))))
    srcmap, msplit = load_srcmap()
    def nt(r): return int(r.get("n_tiles") or 0)

    # 1) TEST crops per organelle + the held-out test-image set (dataset, source_image)
    test = collections.defaultdict(dict)   # org -> {crop_id_key: (row, subgroup, source, split_role)}
    test_images = set()
    consolidated = []
    for r in rows:
        ds, cid = r["dataset"], r["crop_id"]
        if ds not in TESTSETS: continue
        designated, cfilt, adopt = TESTSETS[ds]
        sp = (r.get("official_split") or "").strip() or msplit.get((ds, cid), "")
        if not cfilt(r, sp): continue
        si = srcmap.get((ds, cid), "")
        test_images.add((ds, si))
        role = "test" + (f"(official={sp})" if adopt else "")
        for org in designated:
            if org in fams(orgset(r)):
                sg = SUBGROUP.get((ds, org), "")
                test[org][(ds, cid)] = (r, sg, si, role)
                consolidated.append((ds, cid, org, si, role, sg))

    # 2) build per-organelle CSVs: test + train/val (80/20 by tiles) over the rest
    FIELDS = ["collection","dataset","crop_id","image_path","split","subgroup","modality",
              "scale_band","tissue_context","species_group","organelle","source_image","split_role"]
    def wrow(w, r, split, sg, org, si, role):
        w.writerow([r["collection"], r["dataset"], r["crop_id"], r["image_path"], split, sg,
                    r["modality"], r["scale_band"], r["tissue_context"], r["species_group"],
                    org, si, role])

    for org in ("mito", "er", "nucleus", "ld"):
        toks = FAMILY[org]
        has = [r for r in rows if orgset(r) & toks]
        testkeys = test[org]
        # train/val pool = has-organelle crops that are NOT test crops and whose source image is NOT held out
        pool = []
        for r in has:
            key = (r["dataset"], r["crop_id"])
            if key in testkeys: continue
            si = srcmap.get(key, "")
            if (r["dataset"], si) in test_images: continue   # whole-image holdout
            pool.append(r)
        # deterministic 80/20 by cumulative tiles
        pool_sorted = sorted(pool, key=lambda r: h01(r["dataset"] + "/" + r["crop_id"]))
        total_tiles = sum(nt(r) for r in pool_sorted) or 1
        val_target = VAL_FRAC * total_tiles
        assign = {}; acc = 0
        for r in pool_sorted:
            if acc < val_target: assign[id(r)] = "val"; acc += nt(r)
            else: assign[id(r)] = "train"
        counts = collections.Counter()
        with open(os.path.join(SPLITS, f"benchmark_{org}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(FIELDS)
            for (ds, cid), (r, sg, si, role) in sorted(testkeys.items()):
                wrow(w, r, "test", sg, org, si, role); counts["test"] += 1
            for r in pool_sorted:
                sp = assign[id(r)]; wrow(w, r, sp, "", org, srcmap.get((r["dataset"], r["crop_id"]), ""), sp)
                counts[sp] += 1
        tt = sum(nt(r) for r in pool_sorted if assign[id(r)] == "train")
        vt = sum(nt(r) for r in pool_sorted if assign[id(r)] == "val")
        print(f"benchmark_{org}.csv: test={counts['test']} train={counts['train']} val={counts['val']} "
              f"| train/val tiles={tt}/{vt} ({100*vt/(tt+vt or 1):.1f}% val)")

    # 3) consolidated benchmark_test.csv
    with open(os.path.join(SPLITS, "benchmark_test.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset","crop_id","organelle","source_image","split_role","subgroup"])
        for row in sorted(consolidated): w.writerow(row)
    print(f"\nbenchmark_test.csv: {len(consolidated)} (crop x organelle) test rows; "
          f"{len(test_images)} distinct held-out test images")
    per = collections.Counter(o for _,_,o,_,_,_ in consolidated)
    print("  test crops per organelle:", dict(per))

if __name__ == "__main__":
    main()
