"""ASEM / incasem FIB-SEM organelle GT (zarr + s3fs).
s3://asem-project (anonymous). Reads manual GT label ROIs (mito/ER/Golgi/CCP/nuclear-pore) +
matching raw_equalized_0.02 EM, tiles to 4096, keeps tiles with organelle. CC BY-SA 4.0.

Usage: python run_asem_zarr.py [--cells 2E]
"""
import os, sys, json, time
import numpy as np, zarr, s3fs
sys.path.insert(0, os.path.dirname(__file__))
import seg_crop as sc

S3 = "asem-project"
OUT = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "asem_incasem")
CHUNK = 128
fs = s3fs.S3FileSystem(anon=True)

GT = {"1E":["ccp"], "2E":["ccp","er","mito"], "3A":["mito","np","np_bottom"],
      "46":["er","golgi","mito"], "58":["er","golgi","mito"], "61":["er","golgi","mito"],
      "64":["er","golgi","mito"]}
ORG_NAME = {"ccp":"clathrin_coated_pit","er":"endoplasmic_reticulum","mito":"mitochondria",
            "golgi":"golgi","np":"nuclear_pore","np_bottom":"nuclear_pore"}
META = {
  "name":"asem_incasem",
  "source_repo":"AWS Open Data (s3://asem-project) / incasem (Kirchhausen lab)",
  "accession":"ASEM CF cells 1E,2E,3A,46,58,61,64","doi":"10.1083/jcb.202208005",
  "license":"CC-BY-SA-4.0",
  "paper":"Gallusser et al., JCB 2023 — incasem organelle segmentation",
  "gt_provenance":"manual annotation in VAST + Ilastik carving (proofread); GT only (predictions excluded)",
  "modality":"FIB-SEM (3D)","dimensionality":"3D",
  "label_encoding":"per-organelle (nonzero = organelle); volumes/labels/<org>",
  "organelle_classes":sorted(set(ORG_NAME.values())),
  "alignment":"label+raw_equalized_0.02 share full-cell grid; ROI from populated chunks; z>=400nm",
  "source_url":f"s3://{S3}","ingested_via":"run_asem_zarr.py (zarr+s3fs)",
}

def _retry(fn, tries=5):
    for t in range(tries):
        try: return fn()
        except Exception:
            if t == tries-1: raise
            time.sleep(1.5*(t+1))   # transient s3 errors
def open_arr(path):
    return _retry(lambda: zarr.open(f"s3://{S3}/{path}", mode="r", storage_options={"anon":True}))
def rd(arr, sl):
    return _retry(lambda: np.asarray(arr[sl]))
def jget(path):
    with fs.open(f"{S3}/{path}") as f: return json.load(f)
def roi_from_chunks(arr_path):
    cs=[]
    for k in fs.ls(f"{S3}/{arr_path}", detail=False):
        tail=k.split("/")[-1]; parts=tail.split(".")
        if len(parts)==3 and all(p.isdigit() for p in parts):
            cs.append(tuple(int(p) for p in parts))
    if not cs: return None
    zs,ys,xs=zip(*cs)
    return (min(zs)*CHUNK,(max(zs)+1)*CHUNK, min(ys)*CHUNK,(max(ys)+1)*CHUNK, min(xs)*CHUNK,(max(xs)+1)*CHUNK)

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--cells",default=""); a=ap.parse_args()
    cells=[c for c in a.cells.split(",") if c] or list(GT)
    ds=sc.Dataset(OUT, META, fresh=not a.cells)
    for cell in cells:
        base=f"datasets/{cell}/{cell}.zarr"; rawpath=f"{base}/volumes/raw_equalized_0.02"
        try: raw=open_arr(rawpath)
        except Exception as e: print(f"{cell}: raw open failed {e}"); continue
        SZ,SY,SX=raw.shape
        for org in GT[cell]:
            lp=f"{base}/volumes/labels/{org}"
            try:
                za=jget(f"{lp}/.zattrs"); res=za.get("resolution",[5,5,5]); off_nm=za.get("offset",[0,0,0])
            except Exception: res,off_nm=[5,5,5],[0,0,0]
            zres=res[0]; zstep=sc.zstep_for_spacing(zres)
            try: lab=open_arr(lp)
            except Exception as e: print(f"  {cell}/{org}: label open failed {e}"); continue
            LZ,LY,LX=lab.shape
            offv=[int(round(o/r)) for o,r in zip(off_nm,res)]
            if offv[0]>=SZ or offv[1]>=SY or offv[2]>=SX:
                print(f"  {cell}/{org}: offset {offv} outside raw {raw.shape} - SKIP"); continue
            roi=roi_from_chunks(lp)
            if roi is None: print(f"  {cell}/{org}: no chunks"); continue
            lzmin,lzmax,lymin,lymax,lxmin,lxmax=roi
            lzmax,lymax,lxmax=min(lzmax,LZ),min(lymax,LY),min(lxmax,LX)
            vx={"x":res[2],"y":res[1],"z":res[0]}; name=ORG_NAME[org]
            gz0_roi,gz1_roi=offv[0]+lzmin, offv[0]+lzmax
            kept=[z for z in range(0,SZ,zstep) if gz0_roi<=z<gz1_roi]
            print(f"  {cell}/{org}->{name}: Lsh={lab.shape} off={offv} ROIz[{lzmin}:{lzmax}] res{res} zstep{zstep} planes{len(kept)}", flush=True)
            slabs={}
            for z in kept:
                lz=z-offv[0]; slabs.setdefault(lz//CHUNK,[]).append((z,lz))
            for sidx,zl in sorted(slabs.items()):
              try:
                lz0=sidx*CHUNK; lz1=min(lz0+CHUNK,lzmax)
                lab_slab=rd(lab, (slice(lz0,lz1), slice(lymin,lymax), slice(lxmin,lxmax)))
                if not lab_slab.any(): continue
                gy0,gx0=offv[1]+lymin, offv[2]+lxmin; gz0=offv[0]+lz0
                ry1=min(gy0+(lymax-lymin),SY); rx1=min(gx0+(lxmax-lxmin),SX); rz1=min(gz0+(lz1-lz0),SZ)
                raw_slab=rd(raw, (slice(gz0,rz1), slice(gy0,ry1), slice(gx0,rx1)))
                for z,lz in zl:
                    li=lz-lz0
                    if li>=raw_slab.shape[0] or li>=lab_slab.shape[0]: continue
                    lpl=lab_slab[li]
                    if not lpl.any(): continue
                    rpl=raw_slab[li]; hh=min(lpl.shape[0],rpl.shape[0]); ww=min(lpl.shape[1],rpl.shape[1])
                    ds.add_plane(rpl[:hh,:ww], lpl[:hh,:ww], source_image=f"{cell}.zarr:labels/{org}",
                                 source_shape_xy=(SX,SY), z_index=int(z), z_physical_nm=float(z*zres),
                                 voxel_size_nm=vx, label_kind="instance_single", organelle_name=name,
                                 subdir=f"{cell}_{org}", id_prefix=f"{cell}{org}_", origin_xy=(gx0,gy0))
              except Exception as e:
                print(f"  {cell}/{org} slab {sidx}: skipped ({type(e).__name__}: {e})", flush=True)
    path,n=ds.write_manifest(); print(f"DONE: {n} crops -> {path}")

if __name__=="__main__": main()
