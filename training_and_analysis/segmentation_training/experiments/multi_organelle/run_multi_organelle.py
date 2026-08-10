"""Multi-organelle versus single-organelle decoders: runner and config generator.

Three arms:
  * ``shared-dodnet``      — one adapted encoder plus a single DoDNet organelle-conditioned head,
                             trained on the mixed mitochondria + ER dataset and evaluated per organelle
                             by task code.
  * ``per-organelle-lora`` — a per-organelle specialist with its own adapter set, trained under the same
                             matched template.
  * ``specialist``         — an already-trained single-organelle head, evaluated without retraining.

All three are matched on steps, seed, adaptation and evaluation settings through the shared template in
``config_templates``, so any difference between them is the arm's.

Subcommands
-----------
  * ``gen-configs``   — write the matched configs for all three arms plus a data-root map.
  * ``dodnet``        — train and evaluate the shared DoDNet model; one report per organelle.
  * ``per-organelle`` — train and evaluate one organelle's own-adapter head.
  * ``specialist``    — evaluate an existing single-organelle head.
  * ``gen-capacity``  — write the DoDNet capacity-sweep configs.
  * ``compare``       — assemble the shared-versus-specialist verdict across seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.config_templates import base_config, with_overrides, write_config

# ``--data-roots`` takes a JSON object mapping organelle to its derived dataset root, or the path to a
# ``data_roots.json`` written by ``gen-configs``. Each root holds that organelle's own manifest.


def gen_configs(out_dir: str | Path, organelles=("mito", "er"), data_roots: dict | None = None,
                *, mid_channels: int = 8, n_dynamic: int = 3, mechanism: str = "dynamic") -> dict:
    """Write the three-arm baseline-matched configs (per organelle) + a data_roots.json map. Returns the map."""
    out_dir = Path(out_dir)
    # One entry per organelle, so the returned map always names what each arm needs even when
    # the roots themselves are supplied at launch.
    data_roots = {o: (data_roots or {}).get(o) for o in organelles}
    k = len(organelles)

    # Shared DoDNet arm: one config with the dodnet decoder + all organelles' data (mixed at train time).
    base = base_config(organelles[0], name="multi_dodnet")
    dodnet = with_overrides(
        base, name="multi_dodnet",
        **{"decoder.type": "dodnet",
           "decoder.params": {"n_organelles": k, "mid_channels": mid_channels, "n_dynamic": n_dynamic,
                              "mechanism": mechanism},
           "data.num_classes": 2,
           "notes": (f"Shared DoDNet: adapted encoder + dynamic organelle-conditioned head over "
                     f"{list(organelles)} (mixed dataset; the task code selects the organelle).")})
    write_config(dodnet, out_dir / "multi_dodnet.yaml")

    # The per-organelle-adapter and specialist arms share one config per organelle; the specialist arm
    # loads an already-trained head instead of retraining.
    for org in organelles:
        b = base_config(org, name=f"multi_perorg_{org}")
        cfg = with_overrides(b, name=f"multi_perorg_{org}", **{
            "notes": f"Per-organelle adapter / specialist arm: {org} own-adapter head, matched to the baseline."})
        write_config(cfg, out_dir / f"multi_perorg_{org}.yaml")

    (out_dir / "data_roots.json").write_text(json.dumps(data_roots, indent=2), encoding="utf-8")
    print(f"[gen] wrote multi_dodnet.yaml + {len(organelles)} per-organelle configs + data_roots.json "
          f"-> {out_dir}")
    return data_roots


def run_dodnet(organelles, *, data_roots: dict, run_dir, device: str = "cuda", seed: int = 0,
               max_steps=None, mid_channels: int = 8, n_dynamic: int = 3, mechanism: str = "dynamic",
               balance: str = "raw", eval_splits=("test_image", "test"), out_dir=None) -> dict:
    """Train the shared DoDNet model on the mixed dataset + eval per organelle; write one report per organelle."""
    from ..common.eval_report import assemble_report, write_report
    from .mixed_dataset import load_per_organelle
    from .train_multi import evaluate_multi, train_multi
    from ...config.schema import SegConfig
    from ...harness.run_seg import resolve_device, resolve_encoder

    k = len(organelles)
    cfg = SegConfig.from_dict(base_config(organelles[0], name="multi_dodnet", seed=seed))
    cfg.encoder.run_dir = str(run_dir)
    cfg.data.num_classes = 2
    if max_steps:
        cfg.optim.max_steps = int(max_steps)
    device = resolve_device(device)
    enc, _ = resolve_encoder(cfg, device); enc.to(device)

    train_per_org = load_per_organelle({o: data_roots[o] for o in organelles}, split=cfg.data.train_split)
    model = train_multi(cfg, enc, train_per_org, device=device, n_organelles=k,
                        mid_channels=mid_channels, n_dynamic=n_dynamic, mechanism=mechanism, balance=balance)
    train_stats = getattr(model, "_train_stats", None)      # per-organelle step allocation and dataset imbalance

    reports = {}
    for sp in eval_splits:
        eval_per_org = load_per_organelle({o: data_roots[o] for o in organelles}, split=sp)
        per_org_out = evaluate_multi(model, eval_per_org, cfg, device, enc.image_mean, enc.image_std,
                                     n_organelles=k)
        for org, out in per_org_out.items():
            reports.setdefault(org, {})[sp] = out
    assembled = {}
    for org, split_results in reports.items():
        rep = assemble_report(f"dodnet_{org}", org, split_results,
                              extra={"arm": "shared-dodnet", "mechanism": mechanism, "n_organelles": k,
                                     "mid_channels": mid_channels, "balance": balance,
                                     "organelles": list(organelles), "seed": seed, "train_stats": train_stats})
        if out_dir:
            write_report(rep, out_dir, arm=f"dodnet_{org}_mid{mid_channels}_{balance}_s{seed}")
        assembled[org] = rep
    return assembled


def run_per_organelle(organelle: str, *, data_root: str, run_dir, device: str = "cuda", seed: int = 0,
                      max_steps=None, eval_splits=("test_image", "test"), out_dir=None) -> dict:
    """Train + eval one organelle's own-LoRA head (the per-organelle-LoRA specialist arm)."""
    from ..common.eval_report import assemble_report, write_report
    from .per_organelle_lora import (
        evaluate_per_organelle, train_per_organelle_lora)
    from ...config.schema import SegConfig
    from ...harness.dataset import load_manifest
    from ...harness.run_seg import resolve_device, resolve_encoder

    cfg = SegConfig.from_dict(base_config(organelle, name=f"perorg_{organelle}", seed=seed))
    cfg.encoder.run_dir = str(run_dir)
    if max_steps:
        cfg.optim.max_steps = int(max_steps)
    device = resolve_device(device)
    enc, _ = resolve_encoder(cfg, device); enc.to(device)
    group = cfg.data.resolved_group()
    train_recs = load_manifest(data_root, group, cfg.data.train_split)
    model = train_per_organelle_lora(cfg, enc, train_recs, data_root, device, organelle=organelle)
    split_results = {}
    for sp in eval_splits:
        recs = load_manifest(data_root, group, sp)
        if recs and len(recs) > 300:   # cap evaluation cost; stratified and seeded
            from ...harness.dataset import subset_fraction
            recs = subset_fraction(recs, 300 / len(recs), seed=int(getattr(cfg.optim, "seed", 0) or 0))
        if recs:
            split_results[sp] = evaluate_per_organelle(model, recs, cfg, data_root, device, enc.image_mean,
                                                       enc.image_std, organelle=organelle)
    rep = assemble_report(f"perorg_lora_{organelle}", organelle, split_results,
                          extra={"arm": "per-organelle-lora", "seed": seed})
    if out_dir:
        write_report(rep, out_dir, arm=f"perorg_lora_{organelle}_s{seed}")
    return rep


def run_specialist(organelle: str, *, data_root: str, device: str = "cuda", run_dir=None,
                   split: str = "test", out_dir=None) -> dict:
    """Evaluate an existing single-organelle head as the specialist baseline arm."""
    from ..common.eval_report import assemble_report, write_report
    from .per_organelle_lora import run_specialist_eval

    split_results = run_specialist_eval(organelle, data_root=data_root, device=device, split=split,
                                        run_dir=run_dir)
    rep = assemble_report(f"specialist_{organelle}", organelle, split_results,
                          extra={"arm": "specialist", "note": "existing single-organelle head; no new training"})
    if out_dir:
        write_report(rep, out_dir, arm=f"specialist_{organelle}")
    return rep


def gen_capacity_configs(out_dir, organelles=("mito", "er"), mid_channels=(8, 16, 32), n_dynamic=3) -> dict:
    """Capacity sweep — the shared DoDNet at several dynamic-head widths.

    A loss caused by too thin a head (``mid_channels=8``) is indistinguishable from a loss caused by
    sharing, and the two support opposite conclusions, so the sweep is what separates them.
    """
    out_dir = Path(out_dir)
    k = len(organelles)
    manifest = {}
    for mid in mid_channels:
        name = f"multi_dodnet_mid{mid}"
        base = base_config(organelles[0], name=name)
        cfg = with_overrides(base, name=name, **{
            "decoder.type": "dodnet", "data.num_classes": 2,
            "decoder.params": {"n_organelles": k, "mid_channels": int(mid), "n_dynamic": n_dynamic,
                               "mechanism": "dynamic"},
            "notes": f"DoDNet capacity sweep: mid_channels={mid}, separating a thin-head loss from a sharing loss."})
        write_config(cfg, out_dir / f"{name}.yaml")
        manifest[name] = {"mid_channels": int(mid), "organelles": list(organelles)}
    (out_dir / "capacity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[gen] {len(manifest)} capacity configs (mid_channels {list(mid_channels)}) -> {out_dir}")
    return manifest


def compare(runs_root, out_dir=None, tie_k: float = 1.0) -> dict:
    """Across-seed help/hurt/wash verdict: shared-DoDNet (per capacity/balance variant) against the specialist
    arm — or, where no specialist report is present, against per-organelle-LoRA — per organelle, on both the
    semantic and the instance metric, with mito and LD read primarily off instance. Reports
    the per-organelle step allocation and dataset imbalance alongside, since those are confounds that lie
    outside the across-seed comparison."""
    from ..common.seed_stats import compare_arms
    runs_root = Path(runs_root)
    recs, train_stats = {}, {}
    for f in runs_root.rglob("report_*.json"):
        try:
            rep = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        org, ex = rep.get("organelle"), rep.get("extra", {})
        kind = ex.get("arm")
        if not org or kind not in ("shared-dodnet", "per-organelle-lora", "specialist"):
            continue
        variant = f"mid{ex.get('mid_channels')}_{ex.get('balance')}" if kind == "shared-dodnet" else "base"
        macro = rep.get("splits", {}).get("test", {}).get("macro", {})
        recs.setdefault((org, kind, variant), {})[int(ex.get("seed", 0))] = macro
        if kind == "shared-dodnet" and ex.get("train_stats"):
            train_stats[(org, variant)] = ex["train_stats"]

    METRICS = {"semantic": "dice", "instance": "inst_pq"}
    report = {}
    for org in sorted({k[0] for k in recs}):
        spec = recs.get((org, "specialist", "base"), {})
        perorg = recs.get((org, "per-organelle-lora", "base"), {})
        baseline, baseline_name = (spec, "specialist") if spec else (perorg, "per-organelle-lora")
        variants = {v: recs[(org, "shared-dodnet", v)] for (o, kk, v) in recs
                    if o == org and kk == "shared-dodnet"}
        primary = "instance" if org in ("mito", "ld") else "semantic"
        vout = {}
        for v, seedmap in variants.items():
            per_metric = {}
            for mname, mkey in METRICS.items():
                a = [m.get(mkey) for m in seedmap.values()]
                b = [m.get(mkey) for m in baseline.values()]
                if any(x is not None for x in a) and any(x is not None for x in b):
                    per_metric[mname] = compare_arms(a, b, tie_k=tie_k, lower_better=False)
            vout[v] = {"metrics": per_metric, "train_stats": train_stats.get((org, v))}
        report[org] = {
            "baseline": baseline_name, "primary_metric": primary, "variants": vout,
            "caveats": [
                f"The {org} verdict is read primarily off the {primary} metric.",
                ("shared-DoDNet is semantic-only (no dynamic instance head), so its mito inst_pq is the "
                 "semantic connected-component metric rather than true instance; the mito instance sharing "
                 "question requires a dynamic instance head. " if org in ("mito", "ld") else ""),
                "'wash' is the default outcome a delta must escape (band = tie_k × across-seed SD); multi-task "
                "rarely moves much on two related dense tasks.",
                "Step-allocation and imbalance confounds are recorded in train_stats: under raw balance the "
                "majority organelle dominates gradients, so 'sharing hurts mito' can be a step or imbalance "
                "artifact rather than sharing. The raw and balanced variants separate the two, and the verdict "
                "can differ between them.",
                "Capacity (thin head): the mid8/16/32 sweep separates a thin-head loss from a sharing loss.",
            ],
            "headline_framing": "The primary result is the extensibility recommendation (shared-conditioned "
                                "against per-organelle-LoRA for adding LD or nucleus); whether sharing helps the "
                                "primary organelles is the secondary question.",
        }
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "multi_organelle_verdict.json").write_text(json.dumps(report, indent=2, default=str),
                                                                    encoding="utf-8")
        print(f"[compare] -> {out_dir}/multi_organelle_verdict.json")
    for org, R in report.items():
        print(f"  {org} (primary={R['primary_metric']}, baseline={R['baseline']}):")
        for v, vo in R["variants"].items():
            pm = vo["metrics"].get(R["primary_metric"], {})
            print(f"    {v}: {R['primary_metric']} verdict={pm.get('verdict')} "
                  f"(delta={pm.get('delta')}, band={pm.get('band')}, seeds={pm.get('min_seeds')})")
            ts = vo.get("train_stats")
            if ts:
                print(f"      steps/organelle: {ts.get('per_organelle_step_fraction')} | ratio {ts.get('dataset_ratio', {}).get('max_over_min_ratio')}")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description="Multi-organelle versus single-organelle decoders.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-configs", help="Write the three matched arm configs and the data-root map.")
    g.add_argument("--out-dir", default=str(Path(__file__).parent / "configs"))
    g.add_argument("--organelles", nargs="+", default=["mito", "er"])
    g.add_argument("--mechanism", default="dynamic", choices=["dynamic", "film_moe"])

    d = sub.add_parser("dodnet", help="Train and evaluate the shared DoDNet model on the mixed dataset.")
    d.add_argument("--organelles", nargs="+", default=["mito", "er"])
    d.add_argument("--data-roots", required=True, help="JSON: {organelle: data_root} or path to data_roots.json")
    d.add_argument("--run-dir", required=True, help="Encoder run directory.")
    d.add_argument("--device", default="cuda")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--max-steps", type=int, default=None)
    d.add_argument("--mechanism", default="dynamic", choices=["dynamic", "film_moe"])
    d.add_argument("--mid-channels", type=int, default=8, help="dynamic-head width (capacity sweep: 8/16/32).")
    d.add_argument("--balance", default="raw", choices=["raw", "balanced"],
                   help="raw=matched-total-steps (majority organelle dominates); balanced=matched-per-organelle.")
    d.add_argument("--out-dir", default=None)

    gcap = sub.add_parser("gen-capacity", help="Capacity-sweep configs (mid_channels 8/16/32).")
    gcap.add_argument("--out-dir", default=str(Path(__file__).parent / "configs_capacity"))
    gcap.add_argument("--organelles", nargs="+", default=["mito", "er"])
    gcap.add_argument("--mid-channels", nargs="+", type=int, default=[8, 16, 32])

    cmp = sub.add_parser("compare", help="Shared-DoDNet versus specialist verdict across seeds.")
    cmp.add_argument("--runs-root", required=True)
    cmp.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    cmp.add_argument("--tie-k", type=float, default=1.0)

    po = sub.add_parser("per-organelle", help="Train and evaluate one organelle's own-adapter head.")
    po.add_argument("--organelle", required=True, choices=["mito", "er", "ld", "nucleus"])
    po.add_argument("--data-root", required=True)
    po.add_argument("--run-dir", required=True, help="Encoder run directory.")
    po.add_argument("--device", default="cuda")
    po.add_argument("--seed", type=int, default=0)
    po.add_argument("--max-steps", type=int, default=None)
    po.add_argument("--out-dir", default=None)

    sp = sub.add_parser("specialist", help="Evaluate an existing single-organelle head.")
    sp.add_argument("--organelle", required=True, choices=["mito", "er", "ld", "nucleus"])
    sp.add_argument("--data-root", required=True)
    sp.add_argument("--run-dir", required=True, help="Encoder run directory.")
    sp.add_argument("--device", default="cuda")
    sp.add_argument("--split", default="test")
    sp.add_argument("--out-dir", default=None)

    a = p.parse_args(argv)

    def _load_roots(s: str) -> dict:
        pth = Path(s)
        if pth.exists():
            return json.loads(pth.read_text(encoding="utf-8"))
        return json.loads(s)

    if a.cmd == "gen-configs":
        gen_configs(a.out_dir, organelles=tuple(a.organelles), mechanism=a.mechanism)
    elif a.cmd == "dodnet":
        r = run_dodnet(tuple(a.organelles), data_roots=_load_roots(a.data_roots), device=a.device,
                       run_dir=a.run_dir, seed=a.seed, max_steps=a.max_steps, mechanism=a.mechanism,
                       mid_channels=a.mid_channels, balance=a.balance, out_dir=a.out_dir)
        print(json.dumps({o: rep.get("splits", {}) for o, rep in r.items()}, indent=2, default=str)[:1500])
        for o, rep in r.items():
            ts = rep.get("extra", {}).get("train_stats") or {}
            print(f"[{o}] step allocation: {ts.get('per_organelle_step_fraction')} ({ts.get('note')})")
    elif a.cmd == "gen-capacity":
        gen_capacity_configs(a.out_dir, organelles=tuple(a.organelles), mid_channels=tuple(a.mid_channels))
    elif a.cmd == "compare":
        compare(a.runs_root, out_dir=a.out_dir, tie_k=a.tie_k)
    elif a.cmd == "per-organelle":
        r = run_per_organelle(a.organelle, data_root=a.data_root, device=a.device, run_dir=a.run_dir,
                              seed=a.seed, max_steps=a.max_steps, out_dir=a.out_dir)
        print(json.dumps(r.get("splits", {}), indent=2, default=str)[:2000])
    elif a.cmd == "specialist":
        r = run_specialist(a.organelle, data_root=a.data_root, device=a.device, run_dir=a.run_dir,
                           split=a.split, out_dir=a.out_dir)
        print(json.dumps(r.get("splits", {}), indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
