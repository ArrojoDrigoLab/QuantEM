#!/usr/bin/env python
"""Build benchmark split-definition CSVs from crops_metadata.csv.

Emits, under <corpus root>/splits/:
  group1_mito.csv, group1_er.csv          (encoder-comparison benchmarks; one per organelle)
  group2_<organelle>.csv  (mito/nucleus/er/ld)   (segmentation-training splits)

Each row: collection,dataset,crop_id,image_path,split,subgroup,modality,scale_band,tissue,species.
Dataset-level holdouts give clean, dataset-disjoint test subgroups.
Re-run after re-ingesting datasets.
"""
import csv, os, collections
ROOT=os.environ.get("SEG_CORPUS_ROOT",".")
SPLITS=os.path.join(ROOT,"splits"); os.makedirs(SPLITS,exist_ok=True)
rows=list(csv.DictReader(open(os.path.join(ROOT,"crops_metadata.csv"))))
def orgs(r):
    s=r['organelles'] or ''; return set(s.replace('|',';').split(';')) if s else set()
def val_sample(crop_rows,n):  # deterministic tiny val
    return set(id(r) for r in sorted(crop_rows,key=lambda r:(r['dataset'],r['crop_id']))[:n])

import re as _re
def _lab_img(cid):  # source image = crop_id minus trailing _NNNNN tile index
    return _re.sub(r'_\d{5}$','',cid)
# In-house lab data (all SEM): Group-2 image-level (specimen-disjoint) TEST holdouts,
# ~1/5 of each sub-dataset's source images. Group 1 is train-only for lab
# data (falls through below). secretory_granule has no group (not in mito/er/ld/nucleus) -> stays
# catalog-only. Each held-out source image is whole (all its tiles -> test) so no intra-image leak.
LAB_G2_TEST={
  "C57_August_CD3_Islet1_5nm":"SEM | mouse islet (in-house C57) [held-out specimen]",
  "MLIV_HFD1_ROI10_fullsize_sem":"SEM | mouse liver mito (in-house, HFD) [held-out specimen]",
  "MLIV_HFD2_ROI15_fullsize_sem":"SEM | mouse liver mito (in-house, HFD) [held-out specimen]",
  "MLIV_HFD3_ROI15_fullsize_sem":"SEM | mouse liver mito (in-house, HFD) [held-out specimen]",
  "MLIV_HFD3_ROI7_fullsize_sem":"SEM | mouse liver mito (in-house, HFD) [held-out specimen]",
  "R400_48H_Islet2":"SEM | human islet ER (in-house nPOD R-donor) [held-out specimen]",
}

FIELDS=["collection","dataset","crop_id","image_path","split","subgroup",
        "modality","scale_band","tissue_context","species_group"]
def write(name,assign):
    """assign: list of (row, split, subgroup)."""
    p=os.path.join(SPLITS,name)
    with open(p,"w",newline="") as f:
        w=csv.writer(f); w.writerow(FIELDS)
        for r,split,sg in assign:
            w.writerow([r['collection'],r['dataset'],r['crop_id'],r['image_path'],split,sg,
                        r['modality'],r['scale_band'],r['tissue_context'],r['species_group']])
    c=collections.Counter(s for _,s,_ in assign)
    sg=collections.Counter((s,g) for _,s,g in assign if s=='test')
    print(f"\n# {name}: "+", ".join(f"{k}={v}" for k,v in sorted(c.items())))
    for (s,g),n in sorted(sg.items()): print(f"    test subgroup [{g}]: {n}")

# ----------------------------------------------------------------- GROUP 1 MITO
def group1_mito():
    has=[r for r in rows if 'mito' in orgs(r)]
    # SPLIT RULE: per image/volume. Tiles from the SAME source image/volume must share one split
    # (no intra-volume train+test). Different images/volumes of one dataset may land
    # in different splits. A dataset that is itself another model's training set -> copy its split.
    #
    # dataset-level whole holdouts -> subgroup label. Belongs here when the dataset is effectively a
    # SINGLE image/volume (all its tiles must share a split) or a deposit held out wholesale.
    test_whole={
      "empiar_11746_u2os_fibsem":"FIB-SEM | cultured U2OS | 2-6nm",
      "zenodo_15602048_tem_breast_mito":"TEM | breast tumor | high-res",
      "webknossos_fastem_mito":"FAST-EM | pancreas+breast | 4nm  [underrep modality]",
      "jrc_mus-liver":"FIB-SEM | liver | 6-15nm",
    }
    # published-split copies: these ship an image/volume-disjoint official split, copied
    # verbatim -- published 'test'/'eval' rows -> test, everything else (train+val) -> train.
    test_official={
      "deeppi_em_skeletal_muscle":("test","TEM | skeletal muscle | [underrep tissue]"),
      "guay_platelet":("eval","SBF-SEM | platelet/blood | [underrep tissue]"),
      "orgsegnet_plant":("test","TEM | plant | [underrep kingdom]"),
    }
    assign=[]; train_pool=[]
    for r in has:
        ds=r['dataset']
        if ds in test_whole: assign.append((r,'test',test_whole[ds])); continue
        if ds in test_official:
            osp,sg=test_official[ds]
            if r['official_split']==osp: assign.append((r,'test',sg))
            else: train_pool.append(r)   # published train/val -> train (the full published split is copied)
            continue
        train_pool.append(r)
    val=val_sample([r for r in train_pool if r['dataset']=="zenodo_mitoem2"],12)
    for r in train_pool: assign.append((r,'val' if id(r) in val else 'train',''))
    write("group1_mito.csv",assign)

# ----------------------------------------------------------------- GROUP 1 ER
def group1_er():
    has=[r for r in rows if 'er' in orgs(r)]
    test_whole={
      "empiar_11746_u2os_fibsem":"FIB-SEM | cultured U2OS | 2.5nm",
      "empiar_10791_liver_er_manual":"FIB-SEM | liver (+sheet/tubule) | 8nm",
      "empiar_10994_hela_sbfsem":"SBF-SEM | cultured HeLa | 15nm  [underrep modality for ER]",
      "deepcontact_tem":"TEM | COS-7 cultured | 4.68nm  [fills TEM modality]",
    }
    test_oo_neuronal={"jrc_fly-mb-1a","jrc_fly-vnc-1"}  # FIB neuronal
    # deepcontact_sem ER crops fall through to train (SEM-ER train source)
    # in-house islet TEM-ER (de Boer human nPOD + Wynn mouse): fills TEM modality + secretory
    # tissue. Specimen-disjoint holdout (2 de Boer human + 1 Wynn mouse) -> test; rest -> train.
    islet_er_test={"deboer_00007","deboer_00008","wynn_00004"}
    islet_er_sg="TEM | islet secretory (human+mouse) | 2.4-2.5nm  [fills TEM modality + secretory tissue]"
    assign=[]; train_pool=[]
    for r in has:
        ds=r['dataset']
        if ds=="segapp_islet_er":
            if r['crop_id'] in islet_er_test: assign.append((r,'test',islet_er_sg))
            else: train_pool.append(r)
            continue
        if ds in test_whole: assign.append((r,'test',test_whole[ds])); continue
        if ds in test_oo_neuronal: assign.append((r,'test',"FIB-SEM | neuronal (fly) | 4nm")); continue
        train_pool.append(r)
    val=val_sample([r for r in train_pool if r['dataset']=="empiar_13156_hela_stard3_er"],10)
    for r in train_pool: assign.append((r,'val' if id(r) in val else 'train',''))
    write("group1_er.csv",assign)

# ----------------------------------------------------------------- GROUP 2 (per organelle)
def group2(org,test_datasets):
    """test_datasets: dict dataset-> (subgroup, official_split_or_None)."""
    has=[r for r in rows if org in orgs(r)]
    assign=[]; pool=[]
    for r in has:
        ds=r['dataset']
        if ds.startswith('lab_'):  # in-house: image-level holdout; else -> train (diversity)
            sg=LAB_G2_TEST.get(_lab_img(r['crop_id']))
            if sg: assign.append((r,'test',sg))
            else: pool.append(r)
            continue
        if ds in test_datasets:
            sg,osp=test_datasets[ds]
            if osp is None or r['official_split']==osp: assign.append((r,'test',sg)); continue
        pool.append(r)
    val=val_sample(pool,8)
    for r in pool: assign.append((r,'val' if id(r) in val else 'train_pool',''))
    write(f"group2_{org}.csv",assign)

group1_mito(); group1_er()
group2("mito",{
  "empiar_13420_macrophage_a431":("FIB/SBF | cultured (multi-organelle anchor)",None),
  "zenodo_15602048_tem_breast_mito":("TEM | breast tumor",None),
  "webknossos_fastem_mito":("FAST-EM/SEM | pancreas+breast",None),
  "orgsegnet_plant":("TEM | plant [underrep]","test"),
  "segapp_islet_mito":("SEM | pancreatic islet (in-house, 2 imgs)",None),
})
group2("nucleus",{
  "empiar_13420_macrophage_a431":("FIB/SBF | cultured (multi-organelle anchor)",None),
  "sbiad2822_nuclei":("AT+FIB | mixed (instance nucleus)",None),
  "zenodo_3675220_platynereis":("SBEM | whole-organism annelid | 40+nm [underrep]",None),
  "zenodo_17068504_cardiomyocyte":("TEM | cardiac muscle",None),
  "segapp_islet_nucleus":("SEM | pancreatic islet (in-house, 3 imgs)",None),
})
group2("er",{
  "empiar_13420_macrophage_a431":("FIB/SBF | cultured (multi-organelle anchor)",None),
  "empiar_10791_liver_er_manual":("FIB-SEM | liver (+sheet/tubule)",None),
  "empiar_10994_hela_sbfsem":("SBF-SEM | cultured HeLa | 15nm",None),
  "deepcontact_tem":("TEM | COS-7 cultured | 4.68nm  [TEM modality]",None),
  "empiar_13156_hela_stard3_er":("FIB-SEM | cultured HeLa | 5-8nm",None),
})
group2("ld",{
  "empiar_13420_macrophage_a431":("FIB/SBF | cultured (multi-organelle anchor)",None),
  "jrc_mus-liver":("FIB-SEM | liver | 8nm",None),
  "empiar_12885_aive":("FIB-SEM | cultured+muscle (multi-organelle)",None),
  "empiar_11746_u2os_fibsem":("FIB-SEM | U2OS | 2.5nm high-res",None),
  "deepcontact_cell":("SEM | cultured (DeepContact) [non-FIB LD]",None),
})
print("\nDONE -> splits/")
