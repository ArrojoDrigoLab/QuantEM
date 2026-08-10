"""FiLM-liveness gate — determines whether the conditioning path is active or a no-op.

A FiLM-routed arm depends on the FiLM path being active. Identity-initialised generators that never
activate, or a style code that carries no gradient, collapse every conditioned arm onto the unconditioned
baseline, which is indistinguishable from a genuine conditioning-null. This module establishes that the
path is live before a FiLM-routed arm is run or interpreted:

  * LINK1 — variation of the style codes across sources (cross-source cosine spread of the produced codes).
  * LINK2 — effect of swapping the code on the output. Two different codes are fed through FiLM on a fixed
            image, and the measurement is the max |Δ fg-prob| plus the fraction of pixels crossing a
            decision-relevant threshold, together with how far the produced γ/β depart from identity (1, 0).

Verdict: a conditioned arm ≈ the unconditioned baseline (or pooled-global FiLM ≈ the unconditioned baseline)
is interpretable as "conditioning does not help" only when LINK2 shows the code measurably moves the output.
If LINK2 is ~0 the null reflects an inactive mechanism rather than the conditioning itself, and the arm's
result is not interpretable. (Observed behaviour: path active but modulation negligible on the adapted base
— a genuine "conditioning inert once adapted" result; this module makes that reproducible.)

Torch-only, CPU safe. Run as `python -m segmentation_training.harness.film_liveness --config ...
--head ... --run-dir ... --data-root ...`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def gamma_beta_departure(conditioner, code: torch.Tensor) -> dict:
    """How far the FiLM γ/β produced by ``code`` depart from identity (γ=1, β=0), pooled over injection
    points. A live-but-trained path can still be near-identity — that is itself the diagnostic."""
    if conditioner is None or conditioner.film is None:
        return {"n_points": 0, "gamma_dev": None, "beta_abs": None}
    gdev, babs, n = 0.0, 0.0, 0
    for key, head in conditioner.film.heads.items():
        g, b = head(code)
        gdev += float((g - 1.0).abs().mean())
        babs += float(b.abs().mean())
        n += 1
    return {"n_points": n, "gamma_dev": (gdev / n if n else None), "beta_abs": (babs / n if n else None)}


@torch.no_grad()
def output_sensitivity(model, image: torch.Tensor, code_a: torch.Tensor, code_b: torch.Tensor,
                       thresholds=(0.05,)) -> dict:
    """LINK2: swap ``code_a`` vs ``code_b`` on a fixed image via the FiLM preset and measure the output shift.

    Returns max/mean |Δ P(fg)|, the fraction of pixels whose |Δ| exceeds each threshold, and whether any
    pixel crosses the 0.5 decision boundary — the functional test the weight-norm view misses."""
    cond = model.conditioner
    if cond is None or cond.film is None:
        raise ValueError("output_sensitivity needs a FiLM conditioner.")
    model.eval()
    cond.set_context(preset_code=code_a.view(1, -1).to(image.device))
    pa = model(image).softmax(1)[:, 1]
    cond.set_context(preset_code=code_b.view(1, -1).to(image.device))
    pb = model(image).softmax(1)[:, 1]
    cond.set_context(preset_code=None)
    d = (pa - pb).abs()
    out = {"max_delta": float(d.max()), "mean_delta": float(d.mean()),
           "decision_flips": int(((pa >= 0.5) != (pb >= 0.5)).sum())}
    for t in thresholds:
        out[f"frac_gt_{t}"] = float((d > t).float().mean())
    return out


def gradient_reaches_film(model, image: torch.Tensor, target: torch.Tensor, code: torch.Tensor) -> dict:
    """Training-side check: after one backward through a conditioned forward, whether the FiLM γ/β generator
    params receive non-zero gradient. (A zero grad means the code never influences the loss, i.e. an
    inactive path.)"""
    cond = model.conditioner
    if cond is None or cond.film is None:
        raise ValueError("gradient_reaches_film needs a FiLM conditioner.")
    model.train()
    model.encoder.backbone.eval()
    cond.set_context(preset_code=code.view(1, -1).to(image.device))
    for p in cond.film.parameters():
        p.grad = None
    logits = model(image)
    loss = F.cross_entropy(logits, target.to(image.device), ignore_index=255)
    loss.backward()
    grads = [float(p.grad.abs().sum()) for p in cond.film.parameters() if p.grad is not None]
    cond.set_context(preset_code=None)
    total = sum(grads)
    return {"n_film_params_with_grad": len(grads), "total_film_grad": total, "reaches": total > 0}


def code_diversity(codes: torch.Tensor) -> dict:
    """LINK1: cross-source cosine spread of the produced style codes ([N, d]). Low min-cosine => the codes
    are genuinely source-specific; ~1 everywhere => the style encoder collapsed to a near-constant code."""
    if codes.shape[0] < 2:
        return {"n": int(codes.shape[0]), "cos_min": None, "cos_mean": None}
    c = F.normalize(codes, dim=1)
    sim = c @ c.t()
    off = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
    return {"n": int(codes.shape[0]), "cos_min": float(off.min()), "cos_mean": float(off.mean())}


def verdict(link2: dict, min_max_delta: float = 0.02) -> str:
    """LIVE if the code measurably moves the output, INERT if it does not, NO_FILM if there is no
    FiLM path to measure."""
    md = link2.get("max_delta", 0.0)
    if md is None:
        return "NO_FILM"
    if md >= min_max_delta or link2.get("decision_flips", 0) > 0:
        return "LIVE"
    return "INERT"  # path wired but modulation negligible -> a conditioned-arm null is a genuine result


def liveness_report(model, image: torch.Tensor, source_codes: dict | None = None) -> dict:
    """Full LINK1+LINK2 report. ``source_codes`` = {source: code} (e.g. from precompute_source_codes) drives
    LINK1 + supplies two near-orthogonal codes for LINK2; else two random codes are used."""
    cond = model.conditioner
    dev = image.device
    report: dict = {"has_film": bool(cond is not None and cond.film is not None)}
    if not report["has_film"]:
        return {**report, "verdict": "NO_FILM"}
    d = cond.style_dim
    if source_codes and len(source_codes) >= 2:
        bank = torch.stack([c.to(dev) for c in source_codes.values()])
        report["link1_code_diversity"] = code_diversity(bank)
        cs = F.normalize(bank, dim=1) @ F.normalize(bank, dim=1).t()
        i, j = divmod(int(cs.argmin()), cs.shape[1])  # the two most-different source codes
        code_a, code_b = bank[i], bank[j]
    else:
        g = torch.Generator(device="cpu").manual_seed(0)
        code_a = torch.randn(d, generator=g).to(dev)
        code_b = torch.randn(d, generator=g).to(dev)
        report["link1_code_diversity"] = {"note": "random codes (no source bank supplied)"}
    report["link2_output_sensitivity"] = output_sensitivity(model, image, code_a, code_b)
    report["gamma_beta_departure"] = gamma_beta_departure(cond, code_a.view(1, -1))
    report["verdict"] = verdict(report["link2_output_sensitivity"])
    return report


def main(argv=None) -> None:
    import argparse
    import json

    from ..config.schema import load_seg_config
    from .conditioning_eval import _center_tile, precompute_source_codes
    from .dataset import load_manifest, load_sample, normalize_em
    from .load_adapted import build_and_load_head
    from .run_seg import resolve_device, resolve_encoder

    p = argparse.ArgumentParser(description="FiLM-liveness gate on a trained conditioned head.")
    p.add_argument("--config", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    cfg = load_seg_config(a.config)
    cfg.encoder.run_dir = a.run_dir
    device = resolve_device(a.device)
    enc, _ = resolve_encoder(cfg, device)
    enc.to(device)
    model, _, _ = build_and_load_head(cfg, enc, a.head, device=device)
    recs = load_manifest(a.data_root, cfg.data.resolved_group(), a.split,
                         manifest_name=getattr(cfg.data, "manifest_name", "manifest.jsonl"))
    patch = int(getattr(model.encoder, "patch_size", 16))
    tile = ((int(cfg.encoder.tile_size) + patch - 1) // patch) * patch
    em, _, _ = load_sample(recs[0], a.data_root, with_inst=False)
    x = torch.from_numpy(normalize_em(_center_tile(em, tile), enc.image_mean, enc.image_std))
    x = x.view(1, 1, tile, tile).float().to(device)
    codes = None
    if model.conditioner is not None and getattr(model.conditioner, "style_encoder", None) is not None:
        # LINK1 uses per-source codes estimated from unlabelled tiles (no labels; the deployment scope).
        codes = precompute_source_codes(model, recs, cfg, a.data_root, enc.image_mean, enc.image_std, device)
    rep = liveness_report(model, x, source_codes=codes)
    print(json.dumps(rep, indent=2, default=str))
    print(f"\nVERDICT: {rep['verdict']}  "
          f"(LIVE=code moves output; INERT=path wired but negligible => a conditioned null is genuine; "
          f"NO_FILM=there is no conditioning path, so FiLM-routed arms are not interpretable)")


if __name__ == "__main__":
    main()
