"""Build-time export: turn a rebuilt encoder into a self-contained artifact.

This is what moves a pack from tier (c) or (b) to tier (a) in
:mod:`quantem.inference.encoders`. Run it once, on a machine that has whatever
the pack's encoder needs; afterwards the pack loads from a TorchScript file and
the app needs neither ``dinov3`` nor ``timm`` for it.

Why per pack and not per encoder
--------------------------------
The obvious thing to export is the shared base encoder, and it would be wrong.
None of the eight packs runs the base encoder unmodified:

* the four OmniEM packs install LoRA adapters as **forward hooks inside the
  transformer blocks** -- a scripted module has no ``.blocks`` to hook;
* ``quantem:mito`` / ``:ld`` / ``:nucleus`` (``adapt: last_n``) replace the
  weights of the last four blocks from the head;
* ``quantem:er`` (``adapt: full``) replaces the entire backbone.

So the export happens **after** the pack's encoder-side tensors are applied, and
the artifact lands beside that pack's ``head.pt``. The neck and decoder are not
exported: they are our own code, they are already in this package, and keeping
them eager means a pack's head can still be inspected and fine-tuned.

What is verified
----------------
An export that silently differs from the eager model is worse than no export, so
:func:`export_pack` traces, reloads the traced file, and compares it against the
eager encoder on random input. The default tolerance is 1e-4 max absolute
difference. It also *probes* whether the traced graph generalises to other
spatial sizes and records the answer honestly in the embedded metadata rather
than assuming it: ViTs with a learned position embedding resample it at a size
the tracer may bake in as a constant. In practice this does not restrict the
app -- ``tiling.pad_for_tiling`` guarantees every window is exactly
``spec.tile_size`` -- but a mismatched request must fail loudly, and
:class:`~quantem.inference.encoders.ExportedEncoder` uses this flag to do that.

Two entry points
----------------
:func:`export_pack` exports a pack that is **installed in this machine's model
cache**, writing the artifact beside its weights. That is the developer's
command and what the CLI below drives.

:func:`export_encoder_files` takes the four source files directly and writes the
artifact wherever it is told. It exists because the release builder
(:mod:`quantem.registry.release`) has to export *into a bundle it is
assembling*, from the maintainer's raw training outputs, without first
installing anything into a cache. Both share one implementation, so the artifact
a user downloads is produced by exactly the code the developer's command
verifies.

CLI::

    python -m quantem.inference.export omniem:mito
    python -m quantem.inference.export --all
    python -m quantem.inference.export --all --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .encoders import EXPORT_META_FILE, EXPORTED_ENCODER_NAME
from .specs import MODEL_SPECS, ModelSpec

logger = logging.getLogger(__name__)

#: Max absolute difference tolerated between the exported and eager encoders.
DEFAULT_TOLERANCE = 1e-4


class ExportError(RuntimeError):
    """An encoder could not be exported, or the export did not reproduce the original."""


@dataclass(frozen=True)
class ExportResult:
    pack_id: str
    path: Path
    max_abs_diff: float
    dynamic_spatial: bool
    traced_tile: int
    size_bytes: int
    source_tier: str


@dataclass(frozen=True)
class EncoderSources:
    """The files one export reads, wherever they came from.

    The registry cache produces these for :func:`export_pack`; the release
    builder produces them straight from the maintainer's training outputs. The
    export itself does not care which, and must not: an artifact built at
    release time has to be bit-identical to one a developer builds from an
    install of the same weights.
    """

    head_path: Path
    config_path: Path
    index_path: Path | None = None
    #: The shared foundation encoder blob. None only for a pack whose head
    #: embeds a fully fine-tuned encoder (``quantem:er``, ``adapt: full``).
    encoder_path: Path | None = None


class _TapModule(nn.Module):
    """Trace target: image in, the pack's feature taps out.

    A tuple return, not a list -- ``torch.jit.trace`` treats a returned list as
    mutable state and refuses it in strict mode, and turning strict off to allow
    a list would also disable the checks that catch real tracing hazards.
    """

    def __init__(self, encoder: nn.Module, layers: list[int]) -> None:
        super().__init__()
        self.encoder: Any = encoder
        self.layers = [int(i) for i in layers]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.encoder.features(x, self.layers))


def _max_abs_diff(a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...]) -> float:
    if len(a) != len(b):
        raise ExportError(f"exported encoder returned {len(b)} taps, eager returned {len(a)}")
    worst = 0.0
    for x, y in zip(a, b, strict=True):
        if x.shape != y.shape:
            raise ExportError(f"tap shape drift: eager {tuple(x.shape)} vs exported {tuple(y.shape)}")
        worst = max(worst, float((x.float() - y.float()).abs().max()))
    return worst


def export_pack(
    pack_id: str,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    device: str = "cpu",
    output: str | Path | None = None,
    overwrite: bool = False,
) -> ExportResult:
    """Export an **installed** pack's encoder to TorchScript beside its weights.

    Args:
        pack_id: e.g. ``"omniem:mito"``.
        tolerance: max absolute difference allowed against the eager encoder.
        device: where to build and verify. CPU is the honest default -- the
            artifact is device-independent and CPU float32 has no autocast to
            mask a discrepancy.
        output: destination; defaults to ``<pack dir>/encoder_ts.pt``.
        overwrite: replace an existing artifact.

    Raises:
        ExportError: if the pack is not installed, the eager encoder cannot be
            built, or the traced module does not reproduce it.
    """
    if pack_id not in MODEL_SPECS:
        raise ExportError(f"unknown pack id {pack_id!r}; known: {sorted(MODEL_SPECS)}")

    from .engine import resolve_model_files

    files = resolve_model_files(pack_id)
    if files.config_path is None or not files.config_path.exists():
        raise ExportError(f"{pack_id}: no resolved_config.yaml beside the weights.")

    return export_encoder_files(
        pack_id,
        EncoderSources(
            head_path=files.head_path,
            config_path=files.config_path,
            index_path=files.index_path,
            encoder_path=files.encoder_path,
        ),
        output=Path(output) if output else files.head_path.parent / EXPORTED_ENCODER_NAME,
        tolerance=tolerance,
        device=device,
        overwrite=overwrite,
    )


def export_encoder_files(
    pack_id: str,
    sources: EncoderSources,
    *,
    output: str | Path,
    tolerance: float = DEFAULT_TOLERANCE,
    device: str = "cpu",
    overwrite: bool = False,
) -> ExportResult:
    """Export a pack's encoder from explicit source files to ``output``.

    The pack does not have to be installed anywhere. This is what the release
    builder calls: it has the maintainer's raw training outputs and needs the
    artifact written into the bundle directory it is assembling.

    Args:
        pack_id: e.g. ``"omniem:mito"``.
        sources: the head, its ``resolved_config.yaml``, the encoder family's
            ``checkpoint_index.json`` and the shared encoder weights.
        output: where to write the TorchScript artifact.
        tolerance: max absolute difference allowed against the eager encoder.
        device: where to build and verify.
        overwrite: replace an existing artifact.

    Raises:
        ExportError: if a source file is missing, the eager encoder cannot be
            built, or the traced module does not reproduce it.
    """
    spec = MODEL_SPECS.get(pack_id)
    if spec is None:
        raise ExportError(f"unknown pack id {pack_id!r}; known: {sorted(MODEL_SPECS)}")

    from ._fig3.load_head import build_and_load_head
    from ._fig3.schema import load_head_config
    from .encoders import EncoderManifest, EncoderUnavailable, build_encoder

    head_path = Path(sources.head_path)
    config_path = Path(sources.config_path)
    for path, what in ((head_path, "head.pt"), (config_path, "resolved_config.yaml")):
        if not path.is_file():
            raise ExportError(f"{pack_id}: no {what} at {path}")

    target = Path(output)
    if target.exists() and not overwrite:
        raise ExportError(f"{target} already exists; pass --overwrite to replace it.")

    cfg = load_head_config(config_path)
    index_path = Path(sources.index_path) if sources.index_path else None
    manifest = (
        EncoderManifest.from_index(index_path)
        if index_path is not None and index_path.exists()
        else None
    )
    encoder_path = Path(sources.encoder_path) if sources.encoder_path else None

    # A pack adapted with `full` carries its whole fine-tuned backbone in the
    # head, so the module *shape* can be recovered from the head's own parameter
    # names when the shared encoder blob is not to hand. The head overwrites the
    # weights immediately afterwards either way; this only decides the skeleton.
    skeleton_state = None
    if spec.embeds_encoder and encoder_path is None:
        skeleton_state = torch.load(
            str(head_path), map_location="cpu", weights_only=False
        ).get("encoder_trainable")

    # Build eagerly on purpose: exporting an already-exported encoder would just
    # copy it, and the point is to capture the eager graph.
    try:
        bundle = build_encoder(
            manifest=manifest,
            encoder_path=encoder_path,
            export_path=None,
            apply_encoder_norm=cfg.encoder.apply_encoder_norm,
            device=device,
            skeleton_state=skeleton_state,
        )
    except EncoderUnavailable as exc:
        raise ExportError(f"{pack_id}: {exc}") from exc

    encoder = bundle.module
    # Load the head so the encoder-side tensors this pack owns -- LoRA adapters,
    # replaced blocks -- are in place before tracing. Tracing the bare encoder
    # would export a model this pack never runs.
    model, info = build_and_load_head(cfg, encoder, head_path, device=device)
    layers = list(model.layers)
    del info

    return export_built_encoder(
        pack_id,
        encoder,
        layers,
        adapt=cfg.encoder.adapt,
        apply_encoder_norm=bool(cfg.encoder.apply_encoder_norm),
        source_tier=bundle.contract.tier,
        depth=int(bundle.contract.depth),
        embedding_dim=int(bundle.contract.embedding_dim),
        patch_size=int(bundle.contract.patch_size),
        input_mean=float(bundle.contract.input_mean),
        input_std=float(bundle.contract.input_std),
        output=target,
        tolerance=tolerance,
        device=device,
    )


def export_built_encoder(
    pack_id: str,
    encoder: nn.Module,
    layers: list[int],
    *,
    adapt: str,
    apply_encoder_norm: bool,
    source_tier: str,
    depth: int,
    embedding_dim: int,
    patch_size: int,
    input_mean: float,
    input_std: float,
    output: str | Path,
    tolerance: float = DEFAULT_TOLERANCE,
    device: str = "cpu",
) -> ExportResult:
    """Trace, verify and atomically write an **already-built** pack encoder.

    The tail of :func:`export_encoder_files`, split out so the engine's
    fallback repair (:func:`quantem.inference.engine.build_module`) can rewrite
    a missing ``encoder_ts.pt`` from the eager encoder it *just built and
    loaded the head into* -- paying only the trace and verification, not a
    second full build. The encoder must already carry the pack's encoder-side
    tensors (LoRA adapters, replaced blocks); tracing a bare trunk would
    export a model the pack never runs, which is why callers hand this the
    post-``build_and_load_head`` module.

    Same guarantees as the build-time export: the traced module is verified
    against the eager one *from the bytes on disk*, the dynamic-shape claim is
    probed rather than assumed, and the write is tmp-then-rename so a crash or
    a failed verification never leaves a half-written artifact where the
    engine would load it.
    """
    spec = MODEL_SPECS.get(pack_id)
    if spec is None:
        raise ExportError(f"unknown pack id {pack_id!r}; known: {sorted(MODEL_SPECS)}")
    target = Path(output)
    layers = [int(i) for i in layers]

    tap = _TapModule(encoder, layers).to(device).eval()
    tile = int(spec.tile_size)
    example = torch.randn(1, 1, tile, tile, device=device)

    with torch.no_grad():
        eager = tap(example)
        traced = torch.jit.trace(tap, example)
        traced = torch.jit.freeze(traced.eval())

    meta = {
        "pack_id": pack_id,
        "family": spec.family,
        "organelle": spec.organelle,
        "encoder_run": spec.encoder,
        "layers": layers,
        "depth": int(depth),
        "embedding_dim": int(embedding_dim),
        "patch_size": int(patch_size),
        "input_mean": float(input_mean),
        "input_std": float(input_std),
        "traced_tile": tile,
        "adapt": adapt,
        "apply_encoder_norm": bool(apply_encoder_norm),
        "source_tier": source_tier,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "torch_version": torch.__version__,
        "tolerance": float(tolerance),
    }

    # Probe other spatial sizes before committing to a claim about them.
    dynamic = _probe_dynamic(tap, traced, tile, spec.patch_size, device, tolerance)
    meta["dynamic_spatial"] = dynamic

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".partial")
    torch.jit.save(traced, str(tmp), _extra_files={EXPORT_META_FILE: json.dumps(meta).encode()})

    # Verify what actually landed on disk, not the in-memory traced object: a
    # serialisation bug would otherwise pass a check against the wrong thing.
    extra = {EXPORT_META_FILE: b""}
    reloaded = torch.jit.load(str(tmp), map_location=device, _extra_files=extra)
    reloaded.eval()
    with torch.no_grad():
        got = tuple(reloaded(example))
    diff = _max_abs_diff(eager, got)
    if diff > tolerance:
        tmp.unlink(missing_ok=True)
        raise ExportError(
            f"{pack_id}: exported encoder differs from the eager one by {diff:.3e} "
            f"(tolerance {tolerance:.1e}). Not writing {target.name} -- an artifact that "
            "does not reproduce the published model is worse than none."
        )
    tmp.replace(target)

    logger.info(
        "Exported %s from tier %s: max|diff|=%.3e, dynamic=%s, taps=%s, adapt=%s",
        pack_id, source_tier, diff, dynamic, layers, adapt,
    )
    return ExportResult(
        pack_id=pack_id,
        path=target,
        max_abs_diff=diff,
        dynamic_spatial=dynamic,
        traced_tile=tile,
        size_bytes=target.stat().st_size,
        source_tier=source_tier,
    )


def _probe_dynamic(
    eager: nn.Module,
    traced: torch.jit.ScriptModule,
    tile: int,
    patch: int,
    device: str,
    tolerance: float,
) -> bool:
    """Does the traced graph still match the eager one at a different tile?

    Recorded rather than assumed. ``torch.jit.trace`` constant-folds anything
    that depended on the example's shape, which for a ViT with a learned
    position embedding can include the pos-embed resample grid. The app only
    ever feeds ``spec.tile_size`` windows, so a False here is not a defect --
    but claiming dynamic shapes that do not hold would be.
    """
    probe = tile + patch
    try:
        x = torch.randn(1, 1, probe, probe, device=device)
        with torch.no_grad():
            want = eager(x)
            got = tuple(traced(x))
        return _max_abs_diff(want, got) <= tolerance
    except Exception as exc:
        logger.debug("dynamic-shape probe at %d failed: %r", probe, exc)
        return False


def export_all(
    *,
    pack_ids: list[str] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    device: str = "cpu",
    overwrite: bool = False,
    skip_unavailable: bool = True,
) -> tuple[list[ExportResult], dict[str, str]]:
    """Export several packs. Returns ``(results, {pack_id: failure reason})``."""
    results: list[ExportResult] = []
    failures: dict[str, str] = {}
    for pack_id in pack_ids or sorted(MODEL_SPECS):
        try:
            results.append(
                export_pack(pack_id, tolerance=tolerance, device=device, overwrite=overwrite)
            )
        except Exception as exc:
            if not skip_unavailable:
                raise
            failures[pack_id] = str(exc)
            logger.warning("export failed for %s: %s", pack_id, exc)
    return results, failures


def _plan(spec: ModelSpec) -> str:
    return (
        f"{spec.pack_id:18s} encoder={spec.encoder:20s} tile={spec.tile_size} "
        f"patch={spec.patch_size} adapt={spec.adapt}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m quantem.inference.export",
        description="Export a pack's encoder to TorchScript beside its weights.",
    )
    p.add_argument("packs", nargs="*", help="pack ids, e.g. omniem:mito")
    p.add_argument("--all", action="store_true", help="export every released pack")
    p.add_argument("--device", default="cpu", help="build/verify device (default cpu)")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print what would be exported")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    pack_ids = sorted(MODEL_SPECS) if args.all else args.packs
    if not pack_ids:
        print("nothing to do: pass pack ids or --all", file=sys.stderr)
        return 2

    if args.dry_run:
        for pack_id in pack_ids:
            spec = MODEL_SPECS.get(pack_id)
            print(_plan(spec) if spec else f"{pack_id}: unknown pack")
        return 0

    results, failures = export_all(
        pack_ids=pack_ids,
        tolerance=args.tolerance,
        device=args.device,
        overwrite=args.overwrite,
        skip_unavailable=len(pack_ids) > 1,
    )
    for r in results:
        print(
            f"{r.pack_id:18s} -> {r.path.name}  {r.size_bytes / 1e6:8.1f} MB  "
            f"max|diff|={r.max_abs_diff:.2e}  tile={r.traced_tile}  "
            f"dynamic={r.dynamic_spatial}  from={r.source_tier}"
        )
    for pack_id, reason in failures.items():
        print(f"{pack_id:18s} FAILED: {reason}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
