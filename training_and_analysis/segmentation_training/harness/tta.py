"""Test-time support: adapt a trained head to an unseen source without its labels.

All arms operate on a trained head (loaded via ``load_adapted``) and run episodically per region:
seeds, prototypes and appearance codes are derived from that region alone and no adaptation state is
carried between records, so one region's — and therefore one source's — adaptation never leaks into
another.

* ``b2_support`` — pixel matching. A first pass gives confident foreground; support prototypes are
  built from it (or from manual annotations) and every pixel is re-scored by feature cosine
  similarity, with positional debiasing applied first.
* ``support`` — the same mechanism with the support source and the combination rule varied.
* ``b4_b2film`` — pooled global. The support features are pooled into a single appearance code that
  conditions the decoder by FiLM before re-segmenting.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .conditioning_eval import _center_tile
from .dataset import load_sample, normalize_em
from .evaluate import evaluate_head
from .metrics import aggregate
from ..constants import IGNORE_INDEX


def _center_window(arr: np.ndarray, size: int, *, mask: bool = False) -> np.ndarray:
    """Symmetric center crop-or-pad to ``size``x``size`` (content stays centered regardless of region size).

    Unlike ``_center_tile`` (bottom/right pad only), this keeps the real content centered so the scored
    central base-tile window is identical across field sizes. EM is 0-padded (centered); a mask is padded
    with IGNORE so padded pixels are never scored (per_crop_metrics excludes ignore). For H>=size in a
    dim this is a plain symmetric center-crop (no padding)."""
    H, W = arr.shape
    ph, pw = max(size - H, 0), max(size - W, 0)
    if ph or pw:
        t, l = ph // 2, pw // 2
        pad = ((t, ph - t), (l, pw - l))
        if mask:
            arr = np.pad(arr, pad, mode="constant", constant_values=IGNORE_INDEX)
        else:
            arr = np.pad(arr, pad, mode="constant")  # 0-pad, not reflect: the border is never fabricated tissue
        H, W = arr.shape
    y0, x0 = (H - size) // 2, (W - size) // 2
    return arr[y0:y0 + size, x0:x0 + size]


# --------------------------------------------------------------------------- #
# pixel-matching support: support-source ablation
# --------------------------------------------------------------------------- #
def run_b2_support(model, records, cfg, data_root, device, mean, std) -> dict:
    """Support-source ablation. Build the fg/bg support prototypes from the GT mask
    (``cond.support_source=gt`` — oracle, verified true positives) vs the model's own confident predictions
    (``inferred`` = the standard setting), and optionally as ``cond.n_support`` single-patch seeds (>1) to expose
    per-object seed variance. This separates the behaviour of the propagation mechanism (where the GT oracle
    also underperforms) from that of support selection (where GT helps and inferred does not). Scoring is at
    tile scale."""
    import torch.nn.functional as F

    from ..models.conditioning.positional_debias import PositionalDebias, SelfSupportHead

    patch = int(getattr(model.encoder, "patch_size", 16))
    tile = ((int(cfg.encoder.tile_size) + patch - 1) // patch) * patch
    layer = cfg.encoder.resolved_layers(model.encoder.depth)[-1]
    debias = PositionalDebias(model.encoder, layer=layer, svd_components=int(cfg.cond.tta_debias_svd))
    debias.build_basis(tile, device)
    ss = SelfSupportHead()
    src = str(getattr(cfg.cond, "support_source", "inferred"))
    K = max(1, int(getattr(cfg.cond, "n_support", 1)))
    gen = torch.Generator(device="cpu").manual_seed(int(getattr(cfg.cond, "support_seed", 0)))
    fg_thr = float(cfg.eval.fg_threshold)

    per_crop = []
    for r in records:
        em, mask, _ = load_sample(r, data_root, with_inst=False)
        emc = _center_window(em, tile)
        mc = _center_window(mask.astype(np.int32), tile, mask=True) if mask is not None else None
        gt = (mc if mc is not None else np.zeros((tile, tile), np.int32)).astype(np.uint8)
        x = torch.from_numpy(normalize_em(emc, mean, std)).view(1, 1, tile, tile).float().to(device)
        with torch.no_grad():
            if model.conditioner is not None:
                model.set_record_context(r, device)
            prob0 = model(x).softmax(1)[:, 1]                               # [1,H,W]
            fmap = model.encoder.features(x, [layer], grad=False)[0]        # [1,C,h,w]
        feat = debias.debias(fmap)
        h, w = feat.shape[-2:]
        if src == "gt" and mc is not None:                                 # oracle: prototypes from GT fg
            supp = torch.from_numpy((mc == 1).astype(np.float32)).to(device).view(1, 1, tile, tile)
            supp = F.interpolate(supp, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        else:                                                              # inferred: from confident preds
            supp = F.interpolate(prob0[None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]

        if K <= 1:                                                         # pooled prototype (standard SSP)
            refined = ss.predict(feat, supp[None]).softmax(1)[:, 1]
            pf = F.interpolate(refined[None], size=(tile, tile), mode="bilinear", align_corners=False)[0, 0]
            per_crop.append(_score_tile((pf.cpu().numpy() >= fg_thr), gt, r, cfg))
            continue
        # N single-patch seeds -> N prototypes -> per-seed recall spread; per_crop uses the mean map.
        c = feat.shape[1]
        flat = feat.view(c, -1)                                           # [C, hw]
        fg_ids = (supp.reshape(-1) > 0.5).nonzero(as_tuple=True)[0]
        bg_ids = (supp.reshape(-1) < 0.5).nonzero(as_tuple=True)[0]
        if fg_ids.numel() == 0:                                           # no seed available (under-caller)
            per_crop.append(_score_tile(np.zeros((tile, tile), bool), gt, r, cfg))
            continue
        bg_proto = flat[:, bg_ids].mean(-1, keepdim=True) if bg_ids.numel() else flat.mean(-1, keepdim=True)
        picks = fg_ids[torch.randperm(fg_ids.numel(), generator=gen)[:K]]
        maps, recalls = [], []
        for pid in picks:
            sf = F.cosine_similarity(flat, flat[:, pid:pid + 1], dim=0)
            sb = F.cosine_similarity(flat, bg_proto, dim=0)
            p = torch.softmax(torch.stack([sb, sf]) * ss.temp, dim=0)[1].view(1, 1, h, w)
            pf = F.interpolate(p, size=(tile, tile), mode="bilinear", align_corners=False)[0, 0]
            maps.append(pf)
            recalls.append(_score_tile((pf.cpu().numpy() >= fg_thr), gt, r, cfg).get("recall"))
        m = _score_tile((torch.stack(maps).mean(0).cpu().numpy() >= fg_thr), gt, r, cfg)
        rr = [v for v in recalls if isinstance(v, (int, float))]
        m["seed_recall_mean"] = float(np.mean(rr)) if rr else None
        m["seed_recall_std"] = float(np.std(rr)) if rr else None
        per_crop.append(m)

    summary = aggregate(per_crop, bootstrap_n=int(cfg.eval.bootstrap_n), seed=int(cfg.optim.seed))
    summary["support_source"], summary["n_support"] = src, K
    if K > 1:
        stds = [c.get("seed_recall_std") for c in per_crop if c.get("seed_recall_std") is not None]
        summary["mean_seed_recall_std"] = float(np.mean(stds)) if stds else None
    print(f"[support src={src} K={K}] scored {len(per_crop)} regions"
          + (f"; mean per-object seed-recall std={summary.get('mean_seed_recall_std')}" if K > 1 else ""))
    return {"summary": summary, "per_crop": per_crop}


# --------------------------------------------------------------------------- #
# Support-prototype family: Axis-1 seed source x Axis-2 combination x Axis-3 K/gating
# --------------------------------------------------------------------------- #
def _select_seed_mask(src, prob0_hw, gt_hw, cfg, gen) -> np.ndarray:
    """Axis 1 — boolean fg-seed mask at image resolution [H,W] (seed quality is the binding constraint).

    gt (oracle/ceiling) | few_shot (the k largest GT instances, as a user would verify them) |
    interactive (few GT clicks) | inferred_gated (hard-conf + connected-component size + morphological
    opening -> reject specks, the key realizable arm) | inferred(_raw), the fallthrough (confident pred)."""
    from scipy import ndimage as ndi  # seed selection is host-side labelling (scipy.ndimage only)

    H, W = prob0_hw.shape
    if src == "gt":
        return gt_hw == 1
    if src == "few_shot":  # accept k user-verified GT instances (connected components) as seeds
        fg = gt_hw == 1
        lab, n = ndi.label(fg)
        if n == 0:
            return np.zeros((H, W), bool)
        k = min(max(1, int(cfg.cond.n_shots)), n)
        sizes = ndi.sum(np.ones_like(lab, dtype=np.float32), lab, index=np.arange(1, n + 1))
        keep = np.arange(1, n + 1)[np.argsort(-sizes)[:k]]         # the k largest instances (a user verifies these)
        return np.isin(lab, keep)
    if src == "interactive":
        fg = gt_hw == 1
        ys, xs = np.where(fg)
        if not len(ys):
            return np.zeros((H, W), bool)
        n = min(int(cfg.cond.interactive_clicks), len(ys))
        idx = torch.randperm(len(ys), generator=gen)[:n].tolist()
        seed = np.zeros((H, W), bool)
        rad = int(cfg.cond.interactive_radius)
        for i in idx:
            y, x = int(ys[i]), int(xs[i])
            seed[max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1] = True
        return seed & fg  # a real click lands on the organelle (GT-quality but sparse)
    conf = float(cfg.cond.support_conf) if src == "inferred_gated" else float(cfg.cond.confident_thresh)
    m = prob0_hw >= conf
    if src == "inferred_gated":
        if int(cfg.cond.support_open) > 0:
            m = ndi.binary_opening(m, iterations=int(cfg.cond.support_open))  # spatial coherence
        lab, n = ndi.label(m)
        if n:
            sizes = ndi.sum(np.ones_like(lab, dtype=np.float32), lab, index=np.arange(1, n + 1))
            keep = np.arange(1, n + 1)[sizes >= int(cfg.cond.support_min_size)]  # reject specks
            m = np.isin(lab, keep) if len(keep) else np.zeros_like(m, bool)
    return m


def _corrupt_seed(seed, bg, drop_frac, false_frac, gen):
    """Degrade a seed by dropping ``drop_frac`` of true seed pixels (lowers recall) and injecting
    ``false_frac``x#true false seed pixels drawn from the true background ``bg`` (=``gt==0``, excluding
    ignore), which lowers precision. Deterministic per ``gen``."""
    seed = seed.copy()
    ys, xs = np.where(seed)
    n = len(ys)
    if n == 0:
        return seed
    if drop_frac > 0:
        ndrop = int(round(drop_frac * n))
        if ndrop:
            drop = torch.randperm(n, generator=gen)[:ndrop].tolist()
            seed[ys[drop], xs[drop]] = False
    if false_frac > 0:
        bys, bxs = np.where(bg)                                    # false seeds from real background (gt==0), not ignore
        if len(bys):
            nf = min(int(round(false_frac * n)), len(bys))
            add = torch.randperm(len(bys), generator=gen)[:nf].tolist()
            seed[bys[add], bxs[add]] = True
    return seed


def _mean_key(per_crop, key):
    vals = [c[key] for c in per_crop if c.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def _seed_quality(seed, gt_fg) -> dict:
    """Realized seed precision/recall vs GT (the x-axis of the Dice-vs-seed-quality curve)."""
    s = int(seed.sum())
    tp = int((seed & gt_fg).sum())
    g = int(gt_fg.sum())
    return {"seed_precision": (tp / s if s else None), "seed_recall": (tp / g if g else None),
            "seed_px": s}


def _build_protos(feat, seed_fg_hw, cfg):
    """Axis 3 — fg prototype(s) [K,C] + a bg prototype [C] from the (debiased, L2-normed) feature grid.
    K>1 clusters the seed pixels into appearance modes; ``proto_gate`` drops small/noisy modes."""
    import torch.nn.functional as F

    from ..models.conditioning.pooling import torch_kmeans

    C, h, w = feat.shape[1], feat.shape[2], feat.shape[3]
    fmap = F.normalize(feat, dim=1)[0].permute(1, 2, 0).reshape(-1, C)  # [hw, C]
    seed = F.interpolate(torch.from_numpy(seed_fg_hw.astype(np.float32))[None, None].to(feat.device),
                         size=(h, w), mode="bilinear", align_corners=False)[0, 0].reshape(-1) > 0.5
    if int(seed.sum()) == 0:
        return None, None
    fg, bg = fmap[seed], (fmap[~seed] if int((~seed).sum()) else fmap)
    bg_proto = F.normalize(bg.mean(0), dim=0)
    K = max(1, int(cfg.cond.n_support))
    if K == 1 or fg.shape[0] <= K:
        return F.normalize(fg.mean(0), dim=0)[None], bg_proto
    labels, _ = torch_kmeans(fg, K, iters=15, seed=int(cfg.cond.support_seed))
    protos = []
    for k in range(K):
        cl = fg[labels == k]
        if cl.shape[0] == 0:
            continue
        if bool(cfg.cond.proto_gate) and cl.shape[0] < float(cfg.cond.proto_min_frac) * fg.shape[0]:
            continue  # per-prototype confidence gate: drop a small/noisy appearance mode
        protos.append(F.normalize(cl.mean(0), dim=0))
    return (torch.stack(protos) if protos else F.normalize(fg.mean(0), dim=0)[None]), bg_proto


def _support_sim(feat, fg_protos, bg_proto, temp: float = 10.0):
    """Prototype-similarity fg prob [1,h,w] = softmax(max_k cos(feat, fg_k) vs cos(feat, bg)) * temp."""
    import torch.nn.functional as F

    fmap = F.normalize(feat, dim=1)
    C = fmap.shape[1]
    sf = torch.stack([torch.cosine_similarity(fmap, p.view(1, C, 1, 1), dim=1) for p in fg_protos]).max(0).values
    sb = torch.cosine_similarity(fmap, bg_proto.view(1, C, 1, 1), dim=1)
    return torch.softmax(torch.stack([sb, sf], 1) * temp, 1)[:, 1]


def _combine_support(mode, prob0, support, cfg):
    """Axis 2 — combine head prob ``prob0`` with prototype ``support`` (both [H,W], [0,1]).

    none/off (return the head unchanged) | replace (override; the negative reference, which degrades a
    strong mitochondria base) | residual (nudge toward support only where head and support disagree,
    protecting confident-correct pixels) | uncertainty_gated (support only where the head is unconfident
    — structurally cannot hurt a confident strong base). The ``early`` (FiLM) route is handled by
    ``run_support`` before it reaches this function."""
    if mode in ("none", "off"):
        return prob0                          # no-support tile-scale baseline: scores the base head prob0 in
        #   the same tile frame and record set, so support arms are compared against doing nothing as well as
        #   against the replace negative reference.
    if mode == "replace":
        return support
    if mode == "residual":
        a = float(cfg.cond.support_alpha)
        disagree = (prob0 >= 0.5) != (support >= 0.5)
        return torch.where(disagree, (1 - a) * prob0 + a * support, prob0)
    if mode == "uncertainty_gated":
        unconf = (prob0 - 0.5).abs() < float(cfg.cond.support_uncertain_margin)
        return torch.where(unconf, support, prob0)
    raise ValueError(f"unknown support_combine {mode!r} (none|replace|residual|uncertainty_gated|early)")


def run_support(model, records, cfg, data_root, device, mean, std) -> dict:
    """The support family: Axis-1 seed source x Axis-2 combination x Axis-3 K/gating, one config each.

    First pass -> select seeds -> debiased prototypes -> combine with the head
     -> re-score. The early route goes through FiLM and is liveness-gated.
    The GT seed source is an upper bound rather than a deployable setting, and the report shows the
    inferred-to-GT gap. The success metric is precision recovery at comparable recall, since the
    mechanism targets the high-recall/low-precision regime."""
    from ..models.conditioning.positional_debias import PositionalDebias, matchability

    cond = model.conditioner
    combine = str(cfg.cond.support_combine)
    src = str(cfg.cond.support_source)
    tile, patch = _tile_of(model, cfg)

    if combine == "early":  # exemplar-as-conditioning (FiLM): the seed's appearance code drives the decoder
        if cond is None or getattr(cond, "confident_style", None) is None:
            raise ValueError("the early route is FiLM-routed: it needs a confident_feature head, and is subject "
                             "to the liveness gate.")
        gen = torch.Generator(device="cpu").manual_seed(int(cfg.cond.support_seed))
        fg_thr = float(cfg.eval.fg_threshold)
        film_gate = bool(cfg.cond.film_gate)                              # safety gate
        drop, false = float(cfg.cond.seed_drop_frac), float(cfg.cond.seed_false_frac)  # seed corruption
        per_crop, collapses, n_gt = [], 0, 0
        for r in records:
            em, mask, _ = load_sample(r, data_root, with_inst=False)
            emc = _center_window(em, tile)
            gt = (_center_window(mask.astype(np.int32), tile, mask=True) if mask is not None
                  else np.full((tile, tile), IGNORE_INDEX, np.int32)).astype(np.uint8)
            gt_fg = gt == 1
            x = torch.from_numpy(normalize_em(emc, mean, std)).view(1, 1, tile, tile).float().to(device)
            model.set_record_context(r, device)
            with torch.no_grad():
                cond.set_context(fg_mask=None)                            # pass 1: base code (unconditioned)
                prob0 = model(x).softmax(1)[:, 1]
                seed = _select_seed_mask(src, prob0[0].cpu().numpy(), gt, cfg, gen)  # Axis-1 seed
                seed = _corrupt_seed(seed, gt == 0, drop, false, gen)     # false seeds from true bg (not ignore)
                fgm = torch.from_numpy(seed.astype(np.float32))[None, None].to(device)
                cond.set_context(fg_mask=fgm)                            # pass 2: appearance code from the seed
                prob1 = model(x).softmax(1)[:, 1]
            # Uncertainty-gate the conditioned pass against the base, so a poor code cannot override
            # confident pixels.
            pred_prob = _combine_support("uncertainty_gated", prob0[0], prob1[0], cfg) if film_gate else prob1[0]
            # collapse detection (the global-FiLM failure mode): conditioning craters recall on a region the
            # base handled, which is what the gate prevents (reported for both gated and ungated runs).
            if int(gt_fg.sum()):
                n_gt += 1
                rc_base = float(((prob0[0].cpu().numpy() >= fg_thr) & gt_fg).sum()) / int(gt_fg.sum())
                rc_cond = float(((prob1[0].cpu().numpy() >= fg_thr) & gt_fg).sum()) / int(gt_fg.sum())
                if rc_base > 0.5 and rc_cond < 0.2:
                    collapses += 1
            m = _score_tile((pred_prob.cpu().numpy() >= fg_thr), gt, r, cfg)
            m.update(_seed_quality(seed, gt_fg))
            m["is_ceiling"] = src in ("gt",)
            per_crop.append(m)
        summary = aggregate(per_crop, bootstrap_n=int(cfg.eval.bootstrap_n), seed=int(cfg.optim.seed))
        summary["support"] = {"source": src, "combine": "early", "film_gate": film_gate,
                              "is_ceiling": src == "gt", "n_shots": int(cfg.cond.n_shots),
                              "seed_drop_frac": drop, "seed_false_frac": false,
                              "ungated_collapse_frequency": collapses / max(n_gt, 1),
                              "mean_seed_precision": _mean_key(per_crop, "seed_precision"),
                              "mean_seed_recall": _mean_key(per_crop, "seed_recall")}
        print(f"[support src={src} combine=early(FiLM) film_gate={film_gate}] {len(per_crop)} regions; "
              f"ungated collapse on {collapses}/{n_gt} regions; liveness-gated.")
        return {"summary": summary, "per_crop": per_crop}

    layer = cfg.encoder.resolved_layers(model.encoder.depth)[-1]
    debias = PositionalDebias(model.encoder, layer=layer, svd_components=int(cfg.cond.tta_debias_svd))
    debias.build_basis(tile, device)
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.cond.support_seed))
    fg_thr = float(cfg.eval.fg_threshold)
    is_ceiling = src == "gt"
    per_crop, match_diag, no_seed = [], None, 0
    for r in records:
        em, mask, _ = load_sample(r, data_root, with_inst=False)
        emc = _center_window(em, tile)
        gt = (_center_window(mask.astype(np.int32), tile, mask=True) if mask is not None
              else np.full((tile, tile), IGNORE_INDEX, np.int32)).astype(np.uint8)
        x = torch.from_numpy(normalize_em(emc, mean, std)).view(1, 1, tile, tile).float().to(device)
        with torch.no_grad():
            if cond is not None:
                model.set_record_context(r, device)
            prob0 = model(x).softmax(1)[:, 1]                              # [1,tile,tile]
            fmap = model.encoder.features(x, [layer], grad=False)[0]
        if match_diag is None:
            match_diag = matchability(fmap, debias)
        feat = debias.debias(fmap)
        seed = _select_seed_mask(src, prob0[0].cpu().numpy(), gt, cfg, gen)
        seed = _corrupt_seed(seed, gt == 0, float(cfg.cond.seed_drop_frac),
                             float(cfg.cond.seed_false_frac), gen)         # false seeds from true bg (not ignore)
        fg_protos, bg_proto = _build_protos(feat, seed, cfg)
        if fg_protos is None:                                             # no seed (e.g. under-caller) -> head
            no_seed += 1
            pred_prob = prob0[0]
        else:
            sim = _support_sim(feat, fg_protos, bg_proto)                 # [1,h,w]
            sim = torch.nn.functional.interpolate(sim[None], size=(tile, tile), mode="bilinear",
                                                  align_corners=False)[0, 0]
            pred_prob = _combine_support(combine, prob0[0], sim, cfg)
        m = _score_tile((pred_prob.cpu().numpy() >= fg_thr), gt, r, cfg)
        m.update(_seed_quality(seed, gt == 1))                            # realized seed precision/recall
        m["is_ceiling"] = is_ceiling
        per_crop.append(m)
    summary = aggregate(per_crop, bootstrap_n=int(cfg.eval.bootstrap_n), seed=int(cfg.optim.seed))
    summary["support"] = {"source": src, "combine": combine, "K": int(cfg.cond.n_support),
                          "proto_gate": bool(cfg.cond.proto_gate), "is_ceiling": is_ceiling,
                          "n_shots": int(cfg.cond.n_shots), "seed_drop_frac": float(cfg.cond.seed_drop_frac),
                          "seed_false_frac": float(cfg.cond.seed_false_frac),
                          "mean_seed_precision": _mean_key(per_crop, "seed_precision"),
                          "mean_seed_recall": _mean_key(per_crop, "seed_recall"),
                          "matchability": match_diag, "no_seed_fraction": no_seed / max(len(records), 1)}
    prov = ("GT oracle (upper bound; not available at deployment)" if src == "gt" else
            "GT clicks/instances (verified user input at deployment)" if src in ("interactive", "few_shot")
            else "unlabelled confident predictions only")
    print(f"[support src={src} combine={combine} K={cfg.cond.n_support}] {len(per_crop)} regions; "
          f"seeds from: {prov}; matchability nn_coherence={match_diag.get('nn_coherence') if match_diag else None}; "
          f"no-seed on {summary['support']['no_seed_fraction']:.2f} of regions.")
    return {"summary": summary, "per_crop": per_crop}


# --------------------------------------------------------------------------- #
# shared tile-scale scoring (used by the seed-dependent two-pass arms)
# --------------------------------------------------------------------------- #
def _score_tile(pred_bin, gt_u8, r, cfg):
    from .metrics import per_crop_metrics

    m = per_crop_metrics(pred_bin, gt_u8, organelle=cfg.data.organelle)
    m["subgroup"] = r.get("subgroup", "") or "(none)"
    for f in ("dataset", "modality", "scale_band"):
        m[f] = r.get(f)
    return m


def _tile_of(model, cfg):
    patch = int(getattr(model.encoder, "patch_size", 16))
    return ((int(cfg.encoder.tile_size) + patch - 1) // patch) * patch, patch


# --------------------------------------------------------------------------- #
# pooled global — confident-instance-conditioned FiLM (two-pass; seed-dependent)
# --------------------------------------------------------------------------- #
def run_b4(model, records, cfg, data_root, device, mean, std) -> dict:
    """First pass (global-fallback code) -> select confident organelle regions -> pool their encoder
    features into a per-image appearance code -> FiLM-recalibrate -> re-segment. Requires a
    confident_feature conditioner. Seed-dependent: reports over- vs under-caller behavior
    (expected to help over-callers, do little for the near-zero-recall FAST-EM under-caller)."""
    cond = model.conditioner
    if cond is None or getattr(cond, "confident_style", None) is None:
        raise ValueError("pooled-global support needs a confident_feature conditioner (train with cond.style_source="
                         "confident_feature).")
    tile, _ = _tile_of(model, cfg)
    thr = float(cfg.cond.confident_thresh)
    fg_thr = float(cfg.eval.fg_threshold)
    per_crop = []
    for r in records:
        em, mask, _ = load_sample(r, data_root, with_inst=False)
        emc = _center_window(em, tile)
        gt = _center_window(mask.astype(np.int32), tile, mask=True).astype(np.uint8) if mask is not None \
            else np.zeros((tile, tile), np.uint8)
        x = torch.from_numpy(normalize_em(emc, mean, std)).view(1, 1, tile, tile).float().to(device)
        model.set_record_context(r, device)
        with torch.no_grad():
            cond.set_context(fg_mask=None)                       # pass 1: global-fallback code
            prob0 = model(x).softmax(1)[:, 1]
            # support-source ablation (the same ground-truth-vs-inferred test): pool the appearance code from
            # the GT foreground (oracle, verified TP) vs the model's own confident predictions (inferred).
            if str(getattr(cfg.cond, "support_source", "inferred")) == "gt":
                fg_mask = torch.from_numpy((gt == 1).astype(np.float32)).to(device).view(1, 1, tile, tile)
            else:
                fg_mask = (prob0 > thr)[:, None].float()         # confident organelle seeds
            cond.set_context(fg_mask=fg_mask)                    # pass 2: appearance code from those feats
            prob1 = model(x).softmax(1)[:, 1]
        per_crop.append(_score_tile((prob1[0].cpu().numpy() >= fg_thr), gt, r, cfg))
    summary = aggregate(per_crop, bootstrap_n=int(cfg.eval.bootstrap_n), seed=int(cfg.optim.seed))
    return {"summary": summary, "per_crop": per_crop}


# --------------------------------------------------------------------------- #
# dispatch and command-line entry point
# --------------------------------------------------------------------------- #
def run_tta(model, records, cfg, data_root, device, mean, std) -> dict:
    """Dispatch on ``cfg.cond.tta``."""
    mode = str(cfg.cond.tta)
    dispatch = {
        "b2_support": lambda: run_b2_support(model, records, cfg, data_root, device, mean, std),
        "support": lambda: run_support(model, records, cfg, data_root, device, mean, std),
        "b4_b2film": lambda: run_b4(model, records, cfg, data_root, device, mean, std),
    }
    if mode not in dispatch:
        raise ValueError(f"unknown tta mode {mode!r} (b2_support|support|b4_b2film)")
    return dispatch[mode]()


def main(argv=None) -> None:
    """segmentation_training-tta — apply a test-time-support arm to a trained head + write per-split metrics."""
    import argparse
    import json
    from pathlib import Path

    from ..config.schema import load_seg_config
    from .dataset import load_manifest
    from .load_adapted import build_and_load_head
    from .run_seg import resolve_device, resolve_encoder

    p = argparse.ArgumentParser(description="Apply a test-time-support arm to a trained head.")
    p.add_argument("--config", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--run-dir", required=True, help="Encoder run dir (checkpoint_index.json).")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tta", default=None, help="Override cfg.cond.tta (b2_support|support|b4_b2film).")
    p.add_argument("--split", default="test", help="Which split to adapt+score (default: test = held-out-source).")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    cfg = load_seg_config(a.config)
    cfg.encoder.run_dir = a.run_dir
    if a.tta:
        cfg.cond.tta = a.tta
    device = resolve_device(a.device)
    enc, _ = resolve_encoder(cfg, device)
    enc.to(device)
    model, vocab, _ = build_and_load_head(cfg, enc, a.head, device=device)
    group = cfg.data.resolved_group()
    recs = load_manifest(a.data_root, group, a.split,
                         manifest_name=getattr(cfg.data, "manifest_name", "manifest.jsonl"))
    out = run_tta(model, recs, cfg, a.data_root, device, mean=enc.image_mean, std=enc.image_std)
    outdir = Path(a.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"tta_{cfg.cond.tta}_{a.split}.json").write_text(
        json.dumps({"arm": cfg.name, "tta": cfg.cond.tta, "split": a.split,
                    "summary": out["summary"]}, indent=2, default=str), encoding="utf-8")
    (outdir / f"tta_{cfg.cond.tta}_{a.split}_per_crop.json").write_text(
        json.dumps(out["per_crop"], default=str), encoding="utf-8")
    print(f"[tta:{cfg.cond.tta}] {cfg.name} {a.split} -> {outdir}")


if __name__ == "__main__":
    main()
