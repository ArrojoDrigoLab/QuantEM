#!/usr/bin/env python
"""Write ESTIMATED per-crop nm/px into the canonical dataset manifests for the crops
whose acquisition resolution was never recorded, flagging them estimated so future
crops_metadata rebuilds inherit the value + flag.

Datasets patched:
  * zenodo_17068504_cardiomyocyte : all 35 crops -> 8.0 nm/px.
        All crops are 3500x TEM, native-tiled into the 4096 canvas (no rescale), so they
        share one pixel size. Estimated from cardiac sarcomere periodicity (Z-Z ~2 um
        measured at ~250 stored px across cm_00004/28/32) + cristae-bearing mito short axis
        (~170 px ~1.2 um); consistent with 3500x + ~2x storage downsample.
  * orgsegnet_plant : the 98 crops with no recorded nm/px.
        Estimated by plant-organelle feature scale (mito 0.5-1 um, chloroplast 3-8 um,
        cell wall, starch) from fixed-scale contact sheets, snapped to the dataset's
        37040/magnification (Hitachi H-7650) grid. Per source-name series (same series ==
        same imaging session == same magnification).
  * deeppi_em_skeletal_muscle : all crops -> 5.0 nm/px.
        The files embed no pixel size; estimated from the acquisition description in the
        DeepPI-EM publication (mouse skeletal-muscle TEM).

deeppi/cardiomyocyte crops are single-value; orgsegnet is per-source. Idempotent: only fills
crops with no numeric voxel_size_nm.x. Run then re-run splits/build_crops_metadata.py to refresh
crops_metadata.csv. Pure stdlib.
"""
import json, os, re, sys, collections

ROOT = os.environ.get("SEG_CORPUS_ROOT", ".")

CARDIO_NM = 8.0
CARDIO_NOTE = ("estimated ~8 nm/px from cardiac sarcomere Z-Z periodicity (~2 um ~250 stored px) "
               "+ cristae-bearing mito short axis; 3500x TEM native-tiled (no rescale), "
               "consistent with ~2x storage downsample. acquisition nm/px not published.")

ORGSEG_NOTE = ("estimated from plant-organelle feature scale (mito/chloroplast/cell-wall/starch) "
               "on fixed-scale contact sheets, snapped to the dataset 37040/magnification grid; "
               "no recorded acquisition nm/px for this source.")


def orgseg_est(src):
    """Return estimated nm/px for an orgsegnet source image, or None if unclassified."""
    b = os.path.basename(src).lower()
    b = re.sub(r"\.(tif|tiff|png|jpg|jpeg)$", "", b)
    if b.startswith("protoplasma_original"):       # whole-tissue, many cells -> lowest mag
        return 18.51
    if re.match(r"figure[ _]*\d+_original", b):     # "Figure N_Original" whole-cell panels -> low mag
        return 14.81
    if re.match(r"z6\d\d$", b):                     # z600..z609: dense cytoplasm, mito+cristae
        return 2.47
    if b.startswith("n22-miao"):                    # large mito, high mag
        return 2.47
    if b.startswith("xc10"):                        # chloroplast + large starch grains
        return 6.17
    if b.startswith("xc0"):                         # chloroplast + vacuole cell view
        return 4.63
    if b.startswith("cs10"):                        # whole cell, chloroplast w/ starch
        return 6.17
    if b.startswith("wh5"):                         # big chloroplast + vacuole
        return 4.63
    if re.match(r"c60\d$", b):                      # C601..C609: mito ~350-450 px, cell wall
        return 3.09
    if b.startswith("a_60"):                        # A_601: chloroplast + mito
        return 3.09
    return None


def has_numeric_x(c):
    x = (c.get("voxel_size_nm") or {}).get("x")
    return isinstance(x, (int, float))


def patch_cardio(apply):
    p = os.path.join(ROOT, "zenodo_17068504_cardiomyocyte", "manifest.json")
    m = json.load(open(p, encoding="utf-8"))
    n = 0
    for c in m["crops"]:
        if has_numeric_x(c):
            continue
        old = c.get("voxel_size_nm") or {}
        note = CARDIO_NOTE + ((" prior: " + old["note"]) if old.get("note") else "")
        c["voxel_size_nm"] = {"x": CARDIO_NM, "y": CARDIO_NM, "z": None,
                              "estimated": True, "note": note}
        n += 1
    if apply:
        json.dump(m, open(p, "w", encoding="utf-8"), indent=2)
    return n, len(m["crops"])


def patch_orgseg(apply):
    p = os.path.join(ROOT, "orgsegnet_plant", "manifest.json")
    m = json.load(open(p, encoding="utf-8"))
    n = 0
    unmatched = collections.Counter()
    by_val = collections.Counter()
    for c in m["crops"]:
        if has_numeric_x(c):
            continue
        val = orgseg_est(c.get("source_image", ""))
        if val is None:
            unmatched[c.get("source_image", "")] += 1
            continue
        old = c.get("voxel_size_nm") or {}
        c["voxel_size_nm"] = {"x": val, "y": val, "z": old.get("z"),
                              "estimated": True, "note": ORGSEG_NOTE}
        by_val[val] += 1
        n += 1
    if unmatched:
        print("  unmatched orgsegnet sources (not patched):")
        for s, k in unmatched.most_common():
            print(f"       {k:2d}  {s}")
    print("  orgsegnet estimated nm/px distribution:", dict(by_val))
    if apply and not unmatched:
        json.dump(m, open(p, "w", encoding="utf-8"), indent=2)
    elif apply and unmatched:
        print("  orgsegnet manifest not written: the sources above have no nm/px classification in orgseg_est().")
    return n, len(m["crops"]), len(unmatched)


DEEPPI_NM = 5.0
DEEPPI_NOTE = ("estimated 5.0 nm/px from the DeepPI-EM acquisition description "
               "(mouse skeletal-muscle TEM); nm/px not embedded in the shipped files.")


def patch_deeppi(apply):
    p = os.path.join(ROOT, "deeppi_em_skeletal_muscle", "manifest.json")
    m = json.load(open(p, encoding="utf-8"))
    n = 0
    for c in m["crops"]:
        if has_numeric_x(c):
            continue
        old = c.get("voxel_size_nm") or {}
        note = DEEPPI_NOTE + ((" prior: " + old["note"]) if old.get("note") else "")
        c["voxel_size_nm"] = {"x": DEEPPI_NM, "y": DEEPPI_NM, "z": None,
                              "estimated": True, "note": note}
        n += 1
    if apply:
        json.dump(m, open(p, "w", encoding="utf-8"), indent=2)
    return n, len(m["crops"])


def main():
    apply = "--apply" in sys.argv
    print(f"{'APPLYING' if apply else 'DRY-RUN'} estimated-resolution patch\n")
    print("cardiomyocyte:")
    cn, ct = patch_cardio(apply)
    print(f"  patched {cn}/{ct} crops -> {CARDIO_NM} nm/px")
    print("orgsegnet:")
    on, ot, un = patch_orgseg(apply)
    print(f"  patched {on}/{ot} empty crops ({un} unmatched sources)")
    print("deeppi:")
    dn, dt = patch_deeppi(apply)
    print(f"  patched {dn}/{dt} crops -> {DEEPPI_NM} nm/px")
    if not apply:
        print("\n(dry-run; re-run with --apply to write manifests)")
    else:
        print("\nWROTE manifests. Now run: python splits/build_crops_metadata.py")


if __name__ == "__main__":
    main()
