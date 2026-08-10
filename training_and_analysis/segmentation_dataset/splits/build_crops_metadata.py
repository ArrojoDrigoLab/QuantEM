"""Build a unified per-crop metadata index (crops_metadata.csv) across BOTH
collections, adding tissue_context / species_group / modality / scale_band /
in_situ_status / external_annotation by a curated dataset (+subfolder) mapping.
Pure stdlib."""
import json, os, glob, csv, statistics
ROOT = os.environ.get("SEG_CORPUS_ROOT", ".")

CANON = {
 "mito":"mito","mitos":"mito","mitochondria":"mito","er":"er","endoplasmic_reticulum":"er",
 "nuc":"nucleus","nucleus":"nucleus","ld":"ld","lds":"ld","lipid_droplet":"ld","golgi":"golgi",
 "vesc":"vesicle","vesicles":"vesicle","synaptic_vesicle":"synaptic_vesicle",
 "nuclear_pore":"nuclear_pore","nuclear_envelope":"nuclear_envelope","lysosome":"lysosome",
 "peroxisome":"peroxisome","chromosomes":"chromatin","chromatin":"chromatin","cilium":"cilium",
 "earlyendo":"endosome","lateendo":"endosome","endosome":"endosome","clathrin_coated_pits":"ccp",
 "nucleolus":"nucleolus","centrioles":"centriole",
 "plasma_membrane":"plasma_membrane","plasma membrane":"plasma_membrane",  # structural (non-organelle) class
}
def canon(n):
    n=n.strip().lower()
    return CANON.get(n,n)
def band(v):
    if v is None: return "unknown"
    if v<2: return "0.5-2"
    if v<6: return "2-6"
    if v<15: return "6-15"
    if v<40: return "15-40"
    return "40+"

# dataset-level defaults: (modality, dim, tissue, species, in_situ)
DS = {
 "empiar_10994_hela_sbfsem":("SBF-SEM","3D","cultured_cell","human","in_vitro"),
 "empiar_11746_u2os_fibsem":("FIB-SEM","3D","cultured_cell","human","in_vitro"),
 "empiar_13420_macrophage_a431":("FIB/SBF-SEM","3D","cultured_cell","human","in_vitro"),
 "empiar_12885_aive":("FIB-SEM","3D","mixed_cultured_muscle","mammal","mixed"),
 "sbiad2822_nuclei":("AT+FIB-SEM","2D+3D","mixed","mixed","mixed"),
 "zenodo_15602048_tem_breast_mito":("TEM","2D","breast_tumor","human","in_situ"),
 "zenodo_17068504_cardiomyocyte":("TEM","2D","cardiac_muscle","mouse","in_situ"),
 "zenodo_3675220_platynereis":("SBEM","3D","whole_organism","invertebrate_annelid","in_situ"),
 "webknossos_fastem_mito":("FAST-EM","3D","_perfolder_","_perfolder_","_perfolder_"),
 "empiar_10982_mitonet_benchmark":("volume-EM/TEM","mixed","_perfolder_","_perfolder_","_perfolder_"),
 "zenodo_mitoem2":("volume-EM","3D","_perfolder_","_perfolder_","_perfolder_"),
 "deeppi_em_skeletal_muscle":("TEM","2D","skeletal_muscle","mouse","in_situ"),
 "orgsegnet_plant":("TEM","2D","plant_cell","plant","in_situ"),
 "empiar_10791_liver_er_manual":("FIB-SEM","3D","liver","mouse","in_situ"),
 "empiar_13156_hela_stard3_er":("FIB-SEM","3D","cultured_cell","human","in_vitro"),
 "guay_platelet":("SBF-SEM","3D","platelet_blood","human","ex_vivo"),
 # in-house exports
 "segapp_islet_mito":("SEM","2D","pancreatic_islet","human","ex_vivo"),
 "segapp_islet_nucleus":("SEM","2D","pancreatic_islet","human","ex_vivo"),
 "segapp_islet_er":("TEM","2D","pancreatic_islet","mixed","ex_vivo"),  # de Boer human nPOD + Wynn mouse islet TEM-ER (species per-source via SUB)
 # DeepContact modality+species+tissue from Liu 2022 JCB M&M (PMC9361564):
 "deepcontact_tem":("TEM","2D","cultured_cell","monkey","in_vitro"),       # COS-7, FEI Tecnai Spirit, 4.68nm
 "deepcontact_sem":("SEM","2D","testis_seminiferous","mouse","in_situ"),   # Sertoli tissue, Helios BSE, 10nm
 "deepcontact_cell":("SEM","2D","cultured_cell","human","in_vitro"),       # U-2 OS, Helios BSE, 5nm — has LD
 "asem_incasem":("FIB-SEM","3D","cultured_cell","mammal","in_vitro"),      # incasem CF/HEK cells, 4-5nm
}
INHOUSE = {"segapp_islet_mito","segapp_islet_nucleus","segapp_islet_er"}  # external_annotation = "no" (in-house manual GT)
# subfolder overrides: dataset -> { subfolder_substr: (tissue,species,in_situ[,modality]) }
SUB = {
 "empiar_10982_mitonet_benchmark":{
   "c_elegans":("whole_organism","invertebrate_celegans","in_situ","SBF-SEM"),
   "fly_brain":("neuronal","invertebrate_fly","in_situ","FIB-SEM"),
   "glycolytic_muscle":("skeletal_muscle","mammal","in_situ","SBF-SEM"),
   "hela_cell":("cultured_cell","human","in_vitro","FIB-SEM"),
   "lucchi_pp":("neuronal_hippocampus","mouse","in_situ","FIB-SEM"),
   "salivary_gland":("secretory_gland","invertebrate_fly","in_situ","FIB-SEM"),
   "tem":("mixed","mixed","mixed","TEM"),
 },
 "zenodo_mitoem2":{
   "ME2-Beta":("pancreatic_islet_betacell","mammal","in_situ"),
   "ME2-Jurkat":("cultured_cell","human","in_vitro"),
   "ME2-Macro":("cultured_cell","mammal","in_vitro"),
   "ME2-Mossy":("neuronal_cerebellum","rodent","in_situ"),
   "ME2-Podo":("kidney_podocyte","mammal","in_situ"),
   "ME2-Pyra":("neuronal_cortex","rodent","in_situ"),
   "ME2-Sperm":("reproductive_sperm","mammal","in_situ"),
   "ME2-Stem":("stem_cell","mammal","in_vitro"),
 },
 "webknossos_fastem_mito":{
   "Rat_Pancreas":("pancreas","rat","in_situ"),
   "MCF7":("cultured_cell_breast","human","in_vitro"),
 },
 "segapp_islet_er":{
   "wynn":("pancreatic_islet","mouse","in_situ","TEM"),
   "deboer":("pancreatic_islet","human","ex_vivo","TEM"),
 },
}
OO = {  # dataset -> (tissue, species, in_situ)
 "jrc_mus-liver":("liver","mouse","in_situ"),
 "jrc_mus-liver-zon-1":("liver","mouse","in_situ"),
 "jrc_mus-liver-zon-2":("liver","mouse","in_situ"),
 "jrc_mus-kidney":("kidney","mouse","in_situ"),
 "jrc_mus-nacc-1":("neuronal_nacc","mouse","in_situ"),
 "jrc_hela-2":("cultured_cell","human","in_vitro"),
 "jrc_hela-3":("cultured_cell","human","in_vitro"),
 "jrc_jurkat-1":("cultured_cell","human","in_vitro"),
 "jrc_macrophage-2":("cultured_cell","human","in_vitro"),
 "jrc_sum159-1":("cultured_cell_breast","human","in_vitro"),
 "jrc_sum159-4":("cultured_cell_breast","human","in_vitro"),
 "jrc_ut21-1413-003":("kidney_renal_carcinoma","human","in_situ"),
 "jrc_cos7-1a":("cultured_cell","monkey","in_vitro"),
 "jrc_cos7-1b":("cultured_cell","monkey","in_vitro"),
 "jrc_fly-mb-1a":("neuronal","invertebrate_fly","in_situ"),
 "jrc_fly-vnc-1":("neuronal","invertebrate_fly","in_situ"),
 "jrc_ctl-id8-1":("cultured_immune_cancer","mixed","in_vitro"),
 "jrc_zf-cardiac-1":("cardiac_muscle","zebrafish","in_situ"),
}

rows=[]
# GT collection
for mf in sorted(glob.glob(os.path.join(ROOT,"*","manifest.json"))):
    folder=os.path.basename(os.path.dirname(mf))
    if folder=="openOrganelle": continue
    m=json.load(open(mf,encoding="utf-8"))
    crops=m.get("crops",[])
    if not crops or "organelles_present" not in crops[0]: continue
    base=DS.get(folder,("?","?","?","?","?"))
    for c in crops:
        em=c.get("em_file",""); src=c.get("source_image","") or em
        tissue,species,insitu=base[2],base[3],base[4]; mod=base[0]
        if folder in SUB:
            for key,ov in SUB[folder].items():
                if key in em or key in src:
                    tissue,species,insitu=ov[0],ov[1],ov[2]
                    if len(ov)>3: mod=ov[3]
                    break
        vx=(c.get("voxel_size_nm") or {}).get("x")
        vest=bool((c.get("voxel_size_nm") or {}).get("estimated"))  # flagged estimated in manifest
        orgs=sorted({canon(k) for k,v in c.get("organelles_present",{}).items()
                     if not (isinstance(v,dict) and not v.get("is_organelle",True))})
        rows.append({"collection":"gt","dataset":folder,"crop_id":c.get("crop_id",""),
            "image_path":em,"modality":mod,"dimensionality":base[1],
            "voxel_x_nm":vx,"voxel_estimated":vest,"scale_band":band(vx),"tissue_context":tissue,
            "species_group":species,"in_situ_status":insitu,
            "external_annotation":"no" if folder in INHOUSE else "yes",
            "organelles":";".join(orgs),"coverage_tier":c.get("coverage_tier",""),
            "official_split":c.get("split",""),"n_tiles":1})
# OO collection
oo=json.load(open(os.path.join(ROOT,"openOrganelle","manifest.json"),encoding="utf-8"))
for c in oo.get("crops",[]):
    d=c["dataset"]; t,s,i=OO.get(d,("?","?","in_vitro"))
    vx=(c.get("original_image",{}).get("resolution_nm_zyx") or [None,None,None])[2]
    orgs=sorted({canon(o) for o in c.get("organelles_target_present",[])})
    rows.append({"collection":"openOrganelle","dataset":d,"crop_id":c.get("crop_id",""),
        "image_path":f"openOrganelle/{c.get('crop_id','')}/raw_xy.tif|raw_xz.tif",
        "modality":"FIB-SEM","dimensionality":"3D_iso","voxel_x_nm":vx,"voxel_estimated":False,
        "scale_band":band(vx),
        "tissue_context":t,"species_group":s,"in_situ_status":i,"external_annotation":"yes",
        "organelles":";".join(orgs),"coverage_tier":"","official_split":"","n_tiles":2})

cols=["collection","dataset","crop_id","image_path","modality","dimensionality",
      "voxel_x_nm","voxel_estimated","scale_band","tissue_context","species_group","in_situ_status",
      "external_annotation","organelles","coverage_tier","official_split","n_tiles"]
out=os.path.join(ROOT,"crops_metadata.csv")
with open(out,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
print(f"Wrote {len(rows)} rows -> {out}")

# quick pivots
from collections import Counter
def pivot(field):
    c=Counter(r[field] for r in rows);
    return dict(c.most_common())
print("\nBy tissue_context:"); [print(f"  {k:34s}{v}") for k,v in pivot("tissue_context").items()]
print("\nBy species_group:"); [print(f"  {k:24s}{v}") for k,v in pivot("species_group").items()]
print("\nBy modality:"); [print(f"  {k:20s}{v}") for k,v in pivot("modality").items()]
print("\nBy scale_band:"); [print(f"  {k:10s}{v}") for k,v in pivot("scale_band").items()]
# crops carrying mito and ER (the benchmark targets) by tissue
print("\nMITO crops by tissue:")
mt=Counter(r["tissue_context"] for r in rows if "mito" in r["organelles"].split(";"))
[print(f"  {k:34s}{v}") for k,v in mt.most_common()]
print("\nER crops by tissue:")
er=Counter(r["tissue_context"] for r in rows if "er" in r["organelles"].split(";"))
[print(f"  {k:34s}{v}") for k,v in er.most_common()]
print("\nER crops by (scale_band, modality):")
erb=Counter((r["scale_band"],r["modality"]) for r in rows if "er" in r["organelles"].split(";"))
[print(f"  {k}  {v}") for k,v in erb.most_common()]
