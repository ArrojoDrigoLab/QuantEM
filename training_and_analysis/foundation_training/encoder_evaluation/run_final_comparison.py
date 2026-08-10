"""Final encoder comparison: one fixed decoder across QuantEM and the public baselines, with a
context sweep for the RoPE encoders.

Assembles the encoder run-dir list — QuantEM (``--quantem-run-dir``) plus the baselines registered
under ``--weights-root`` — and drives ``encoder_evaluation.harness.run_probe`` with the fixed decoder,
taking each encoder's final checkpoint (``--n-checkpoints 1``).

Context sweep: for each tile in ``--context-tiles`` (default 512/768/1024) the RoPE encoders read that
context window while the decoder and scoring stay on the common compare region (``compare_tile`` in the
config). Learned-position baselines cannot be swept, so they run once at the smallest tile. Results land
in ``<output-dir>/ctx<tile>/``.

Usage:
    python -m encoder_evaluation.run_final_comparison \
        --quantem-run-dir <released encoder run dir> \
        --weights-root <encoder weights> --derived-root <ground-truth tiles> \
        --organelles mito er --output-dir <results dir>/final_comparison --context-tiles 512 768 1024
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from foundation_baselines.external_vit import REGISTRY

from .constants import DEFAULT_DERIVED_ROOT, VALID_ORGANELLES
from .harness import run_probe

# The external public baselines; QuantEM is prepended at runtime.
EXTERNAL_ENCODERS = ["emcf_mae_vitb", "dinov2_l_base", "dinov3_meta_vitl", "omniem_emdino_vitl"]

def _external_encoders(weights_root: Path, include=None):
    """Registered external encoders as ``(name, run_dir, context_sweepable)``."""
    out = []
    for name in EXTERNAL_ENCODERS:
        d = weights_root / name
        if not (d / "checkpoint_index.json").exists():
            warnings.warn(
                f"[skip external] {name}: no checkpoint_index.json under {d}. Run "
                f"`python -m foundation_baselines.register_external_encoders --weights-root {weights_root}` first.")
            continue
        if include is not None and name not in include:
            continue
        sweepable = bool(REGISTRY[name].context_sweepable) if name in REGISTRY else False
        out.append((name, str(d), sweepable))
    return out

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Final encoder comparison: fixed decoder across encoders, plus a context sweep.")
    p.add_argument("--quantem-run-dir", required=True,
                   help="The QuantEM encoder run dir, the one holding checkpoint_index.json.")
    p.add_argument("--weights-root", default="foundation_weights",
                   help="Dir holding the registered external baseline encoders.")
    p.add_argument("--derived-root", default=DEFAULT_DERIVED_ROOT,
                   help="Derived dataset root. Use a build whose regions are large enough that the widest "
                        "context window still sees real EM rather than padding.")
    p.add_argument("--config", default="encoder_evaluation/configs/final_comparison_upernet.yaml",
                   help="Probe config: the fixed decoder (…_upernet.yaml or …_unet.yaml). It sets compare_tile.")
    p.add_argument("--organelles", nargs="+", default=list(VALID_ORGANELLES))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--context-tiles", type=int, nargs="+", default=[512, 768, 1024],
                   help="Context windows to sweep for the RoPE encoders. The smallest is the baseline "
                        "where every encoder runs and equals the compare region; learned-position "
                        "baselines run only there, at their native tile.")
    p.add_argument("--device", default=None)
    p.add_argument("--include-externals", nargs="*", default=None,
                   help="Restrict to these external encoder names (default: all present).")
    args = p.parse_args(argv)

    weights_root = Path(args.weights_root)
    ext = _external_encoders(weights_root, args.include_externals)
    tiles = sorted(set(int(t) for t in args.context_tiles))
    baseline = min(tiles)  # equals the compare region: where every encoder is on equal footing

    for tile in tiles:
        run_dirs = [args.quantem_run_dir]  # the RoPE encoder is always swept
        for name, d, sweepable in ext:
            if tile == baseline or sweepable:  # non-sweepable baselines run only at the baseline tile
                run_dirs.append(d)
        out = str(Path(args.output_dir) / f"ctx{tile}")
        print(f"[ctx {tile}] encoders: {[Path(d).name for d in run_dirs]} -> {out}")
        probe_argv = [
            "--run-dir", *run_dirs,
            "--derived-root", args.derived_root,
            "--config", args.config,
            "--organelles", *args.organelles,
            "--n-checkpoints", "1",
            "--output-dir", out,
            "--context-tile", str(tile),
        ]
        if args.device:
            probe_argv += ["--device", args.device]
        run_probe.main(probe_argv)

if __name__ == "__main__":
    main()
