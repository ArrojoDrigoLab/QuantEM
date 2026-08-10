"""``python -m quantem_em.weights`` — inspect, pre-seed and verify the model cache.

Also what a lab admin or an installer uses to populate a shared read-only cache:

    python -m quantem_em.weights download --all
    QUANTEM_MODEL_DIR=/srv/quantem python -m quantem_em.weights verify
"""

from __future__ import annotations

import argparse
import sys

from ..registry import REGISTRY, get_model_spec
from . import fetch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="quantem-weights", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="show every artifact and whether it is cached")
    p_list.set_defaults(func=_list)

    p_dl = sub.add_parser("download", help="fetch artifacts")
    g = p_dl.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="every released model")
    g.add_argument("--model", action="append", help="a model id, repeatable (e.g. quantem/mito)")
    p_dl.add_argument(
        "--to",
        metavar="DIR",
        help="also write plainly-named copies into DIR, ready for QUANTEM_MODEL_DIR. The hub "
        "cache stores blobs under content hashes, so copying it to an offline machine does "
        "not work; this is what does.",
    )
    p_dl.set_defaults(func=_download)

    p_v = sub.add_parser("verify", help="re-check the sha256 of everything cached")
    p_v.set_defaults(func=_verify)

    p_p = sub.add_parser("path", help="print the local path of an artifact")
    p_p.add_argument("artifact")
    p_p.set_defaults(func=_path)

    args = ap.parse_args(argv)
    return args.func(args) or 0


def _specs(args):
    if getattr(args, "all", False):
        return list(REGISTRY.values())
    return [get_model_spec(m) for m in (args.model or [])]


def _list(args) -> int:
    plan = fetch.download_plan(list(REGISTRY.values()))
    print(f"{'artifact':24s} {'cached':7s} {'size':>10s}  file")
    for a in plan["artifacts"]:
        print(
            f"{a['name']:24s} {'yes' if a['cached'] else 'no':7s} "
            f"{fetch.format_bytes(a['bytes']):>10s}  {a['filename']}"
        )
    print(
        f"\nnot cached: {len(plan['missing'])} artifact(s), "
        f"{fetch.format_bytes(plan['download_bytes'])}"
    )
    return 0


def _download(args) -> int:
    specs = _specs(args)
    plan = fetch.download_plan(specs)
    names = []
    for s in specs:
        for a in fetch.artifacts_for(s):
            if a not in names:
                names.append(a)

    dest = getattr(args, "to", None)
    if plan["all_present"] and not dest:
        print("everything needed is already cached")
        return 0
    if not plan["all_present"]:
        print(
            f"downloading {len(plan['missing'])} artifact(s), "
            f"{fetch.format_bytes(plan['download_bytes'])}"
        )

    if dest:
        # Always export, even when everything is already cached: the point of --to is to produce
        # the flat directory, and "already cached" only means the hub cache has it under a
        # content-hashed blob name that QUANTEM_MODEL_DIR cannot read.
        paths = fetch.export_flat(names, dest)
        print(f"exported {len(paths)} file(s) to {dest}")
        print(f"  set QUANTEM_MODEL_DIR={dest} on the offline machine")
    else:
        paths = fetch.ensure(names)
    for n, p in paths.items():
        print(f"  {n:24s} {p}")
    return 0


def _verify(args) -> int:
    bad = 0
    for name in fetch.load_registry()["artifacts"]:
        p = fetch.cached_path(name)
        if p is None:
            continue
        try:
            fetch._verify(name, p, fetch.load_registry())
            print(f"  OK      {name}")
        except fetch.WeightsCorruptError as e:
            bad += 1
            print(f"  CORRUPT {name}: {e}", file=sys.stderr)
    return 1 if bad else 0


def _path(args) -> int:
    p = fetch.cached_path(args.artifact)
    if p is None:
        print(f"{args.artifact} is not cached", file=sys.stderr)
        return 1
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
