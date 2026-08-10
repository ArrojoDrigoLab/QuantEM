"""Batch runner: analyze a set of OpenOrganelle datasets by folder name.

Usage:
  python oo_gt_batch.py scan                 # fast GT-presence scan, all datasets
  python oo_gt_batch.py run <folder> [...]   # full analysis for given folders
  python oo_gt_batch.py run-all              # full analysis, all datasets

Full results are written to oo_gt_out/<folder>.json and summarized to stdout.
"""
import sys, os, json
from oo_gt_lib import analyze, s3_list, to_https, key_of

DATASETS = {d["folder"]: d for d in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "oo_datasets.json")))}
OUT = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "_work", "oo_gt_out")
os.makedirs(OUT, exist_ok=True)


def gt_presence(folder):
    url = DATASETS[folder]["url"]
    https = to_https(url).rstrip("/")
    import re
    m = re.search(r"^(.*?)/em/", https)
    recon_root = m.group(1) if m else https.rsplit("/", 2)[0]
    crops, _, st = s3_list(key_of(recon_root) + "/labels/groundtruth/")
    return len(crops or []), st


def fmt_gb(b):
    return f"{b/1e9:.2f}GB"


def run_one(folder):
    url = DATASETS[folder]["url"]
    res = analyze(url)
    res["folder"] = folder
    res["experiment"] = DATASETS[folder]["exp"]
    json.dump(res, open(f"{OUT}/{folder}.json", "w"), indent=1)
    sh = res.get("em_shape_zyx")
    line = (f"{folder:<32} store={res.get('store_type'):<11} "
            f"shape_zyx={sh} crops={res.get('n_crops')} "
            f"annot_z={res.get('annotated_z_slices')} "
            f"full={fmt_gb(res.get('em_full_bytes_8bit',0))} "
            f"gt_planes={fmt_gb(res.get('storage_bytes_8bit',0))}")
    if res.get("error"):
        line += f" ERROR={res['error']}"
    if res.get("note"):
        line += f" NOTE={res['note']}"
    return line


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if mode == "scan":
        for f in sorted(DATASETS):
            n, st = gt_presence(f)
            print(f"{f:<34} crops={n:<4} status={st}")
    elif mode == "run-all":
        for f in sorted(DATASETS):
            print(run_one(f), flush=True)
    elif mode == "run":
        for f in sys.argv[2:]:
            print(run_one(f), flush=True)
