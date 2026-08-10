"""Emit a complete pretraining configuration.

The files under ``configs/`` are experiment definitions: only what this project sets, with upstream
DINOv3's defaults left to upstream. This expands one of them into every value the run actually saw. It
performs the two steps training performs — translate the experiment definition into DINOv3 keys, then
merge them under the upstream DINOv3 default — and writes the result, so a configuration is one file
rather than something to reconstruct by reading two codebases at once. The output is mostly upstream
defaults, so it is written wherever the caller asks rather than kept under ``configs/``.

    python -m em_ssl.config.resolve --config configs/<group>/<arm>.yaml \
        --out <dest>/<arm>.resolved.yaml --world-size 2

A multi-stage experiment yields one configuration per crop stage; select it with ``--stage-index``.

One value is not knowable from the merge alone: the optimizer applies ``scaling_rule`` to the base
learning rate using the global batch size, so the effective rate depends on world size.
``--world-size`` records it in the header.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

def _require_dinov3():
    try:
        from dinov3.configs import get_default_config  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Upstream DINOv3 is not importable.\n"
            "Obtain it from Meta at the pinned commit, then install it:\n"
            "  third_party/fetch_dinov3.sh <dest> && pip install <dest>\n"
            "third_party/dinov3.pin names the required repository, branch and commit.\n"
            f"({exc})"
        )

def pinned_commit(pin_path: Path) -> str | None:
    if not pin_path.exists():
        return None
    for line in pin_path.read_text(encoding="utf8").splitlines():
        if line.startswith("DINOV3_COMMIT="):
            return line.split("=", 1)[1].strip()
    return None

def upstream_commit() -> str | None:
    """The commit of the installed dinov3, when it was installed from git."""
    try:
        import dinov3
    except ImportError:
        return None
    root = Path(dinov3.__file__).resolve().parent.parent
    head = root / ".git" / "HEAD"
    if head.exists():
        ref = head.read_text(encoding="utf8").strip()
        if not ref.startswith("ref:"):
            return ref
        target = root / ".git" / ref.split(" ", 1)[1]
        if target.exists():
            return target.read_text(encoding="utf8").strip()
    for meta in root.glob("dinov3-*.dist-info/direct_url.json"):
        try:
            return json.loads(meta.read_text(encoding="utf8")).get(
                "vcs_info", {}).get("commit_id")
        except Exception:
            return None
    return None

def resolve(experiment_yaml: Path, *, world_size: int = 1, stage_index: int = 0):
    """Translate an experiment file to DINOv3 overrides and merge them under the upstream default.

    This is the same two steps training performs — ``config_translation.translate_stage`` followed by
    ``OmegaConf.merge(get_default_config(), overrides)`` — so the emitted file is the configuration a
    run of this experiment sees, not a reconstruction of it.

    Returns ``(cfg, provenance)``.
    """
    _require_dinov3()
    from omegaconf import OmegaConf
    from dinov3.configs import get_default_config

    from ..integration import config_translation as ct
    from .schema import load_experiment

    spec = load_experiment(str(experiment_yaml))
    n_stages = len(spec.crops.schedule)
    if not 0 <= stage_index < n_stages:
        raise SystemExit(f"{experiment_yaml.name} has {n_stages} crop stage(s); "
                         f"--stage-index {stage_index} is out of range.")

    overrides = OmegaConf.create(
        ct.translate_stage(spec, stage_index, output_dir="<run dir>", seed=spec.train.seed))
    default = get_default_config()
    cfg = OmegaConf.merge(default, overrides)

    n_default = len(_leaves(OmegaConf.to_container(default, resolve=False)))
    n_set = len(_leaves(OmegaConf.to_container(overrides, resolve=False)))

    provenance = {
        "overrides_file": experiment_yaml.name,
        "experiment_name": spec.name,
        "stage_index": stage_index,
        "n_stages": n_stages,
        "upstream_leaf_keys": n_default,
        "keys_set_by_this_experiment": n_set,
        "dinov3_commit_installed": upstream_commit(),
        "world_size_assumed": world_size,
    }

    # The optimizer scales the base LR by global batch size; record the result so the
    # effective value is not left implicit.
    try:
        base_lr = float(cfg.optim.lr)
        rule = str(getattr(cfg.optim, "scaling_rule", ""))
        per_gpu = int(cfg.train.batch_size_per_gpu)
        global_bs = per_gpu * max(1, world_size)
        if rule == "sqrt_wrt_1024":
            provenance["effective_lr"] = base_lr * math.sqrt(global_bs / 1024.0)
        elif rule == "linear_wrt_256":
            provenance["effective_lr"] = base_lr * global_bs / 256.0
        else:
            provenance["effective_lr"] = base_lr
        provenance["scaling_rule"] = rule or "none"
        provenance["global_batch_size"] = global_bs
    except Exception:
        pass

    return cfg, provenance

def _leaves(obj, prefix=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        out.append(prefix)
    else:
        out.append(prefix)
    return out

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", required=True, help="experiment YAML from configs/")
    ap.add_argument("--out", required=True, help="resolved YAML to write")
    ap.add_argument("--world-size", type=int, default=2,
                    help="GPUs the run used; sets the effective learning rate")
    ap.add_argument("--stage-index", type=int, default=0,
                    help="which crop stage to resolve, for multi-stage experiments")
    args = ap.parse_args(argv)

    from omegaconf import OmegaConf

    src = Path(args.config).expanduser()
    cfg, prov = resolve(src, world_size=args.world_size, stage_index=args.stage_index)

    pin = Path(__file__).resolve().parents[2] / "third_party" / "dinov3.pin"
    want = pinned_commit(pin)
    got = prov.get("dinov3_commit_installed")
    if want and got and want != got:
        print(f"WARNING: installed dinov3 is {got}, pin expects {want}. "
              f"The resolved values may not match the published recipe.", file=sys.stderr)

    header = "\n".join(
        [f"# {prov['experiment_name']}"
         + (f" — crop stage {prov['stage_index']} of {prov['n_stages']}."
            if prov["n_stages"] > 1 else "."),
         f"# Defaults are upstream DINOv3 at commit {want or 'unpinned'}; the values below are the "
         f"complete configuration this arm ran under.",
         f"# `optim.lr` is the base rate: {prov.get('scaling_rule')} scales it to "
         f"{prov.get('effective_lr')} at global batch {prov.get('global_batch_size')}.",
         ""])

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + OmegaConf.to_yaml(cfg), encoding="utf8")

    print(f"{prov['keys_set_by_this_experiment']} overrides merged under "
          f"{prov['upstream_leaf_keys']} defaults -> {out}")
    if "effective_lr" in prov:
        print(f"effective LR {prov['effective_lr']:.3g} "
              f"({prov['scaling_rule']}, global batch {prov['global_batch_size']})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
