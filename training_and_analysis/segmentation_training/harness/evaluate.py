"""Segmentation evaluation: sliding-window inference over the full annotated region, then scoring.

Val/test crops are scored at their stored (canonical nm/px) resolution — the derived data is already
resampled to the organelle's canonical scale, so no resampling occurs here (physical nm/px is
meaningful and fixed by data-prep). Each region is tiled into ``cfg.encoder.tile_size`` windows with
``cfg.eval.overlap``, per-window probabilities are cosine (Hann)-blended, thresholded once over the
whole region, and scored only on valid (non-ignore) pixels.

Design choices:
  * The model is the assembled ``SegModel`` (encoder+neck+decoder) driven by ``model(image) -> logits``
    rather than a separate (encoder, decoder) pair. mean/std are passed in explicitly (read off the
    encoder manifest by the caller) rather than pulled off the encoder object.
  * Config is nested (``cfg.encoder`` / ``cfg.eval`` / ``cfg.data``).
  * Instance metrics run whenever ``cfg.data.task == 'instance'``, using the stored per-crop instance
    ids when present, else connected-components of the binary GT.

Runs without a GPU: numpy, torch and — through ``.metrics`` — scipy.ndimage are the only imports at
module load. sklearn, skimage and matplotlib are not imported at load time; ``.metrics`` imports
skimage lazily, inside clDice only.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..constants import BACKGROUND, IGNORE_INDEX
from .dataset import load_sample, normalize_em
from .metrics import aggregate, per_crop_metrics


def _hann2d(t: int) -> np.ndarray:
    """2-D separable Hann window with a small floor so single-window region corners stay non-zero."""
    w = np.hanning(t).astype(np.float32)
    win = np.outer(w, w)
    return win + 1e-3


def _window_starts(length: int, tile: int, stride: int) -> list[int]:
    """Tile start offsets covering ``[0, length)`` with the last window flush to the edge."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


@torch.no_grad()
def predict_region(model, em_uint8: np.ndarray, cfg, mean: float, std: float, device, collect_aux: bool = False):
    """Sliding-window foreground probability map over a full EM region (cosine-blended, softmax).

    Args:
        model:     an assembled ``SegModel`` (or anything with ``forward(image)->[B,K,H,W]`` logits).
        em_uint8:  the raw EM region, uint8 ``[H, W]`` (already at canonical nm/px; not resampled).
        cfg:       ``SegConfig`` (uses ``cfg.encoder.tile_size`` + ``cfg.eval.overlap``).
        mean,std:  EM normalisation stats (from the encoder manifest, not ImageNet).
        device:    torch device to run the windows on.
        collect_aux: also accumulate the decoder's ``aux_logits`` over the same windows.

    Returns a float32 ``[H, W]`` foreground probability map over the original region size, or
    ``(prob, aux)`` when ``collect_aux`` is set, where ``aux`` is one full-region ``[C_i, H, W]``
    float32 array per ``aux_logits`` entry (what ``instance_eval.dual_instance_metrics`` consumes).

    Binary-foreground policy (num_classes==2): ``P(class 1)``; num_classes==1 is the single channel
    as-is. For num_classes>2 the max over the non-background classes ``1..K-1`` is returned as a
    single 'any-foreground' prob (the semantic metrics are binary-per-organelle).
    """
    patch = int(getattr(model.encoder, "patch_size", 16))
    t = _round_up(int(cfg.encoder.tile_size), patch)   # tile = whole number of encoder patches (14 or 16)
    H0, W0 = em_uint8.shape

    # Pad so (a) the region is at least one tile and (b) H,W are /patch multiples (the encoder needs a
    # whole number of patches; the tile itself is already a /patch multiple).
    ph = max(t - H0, 0)
    pw = max(t - W0, 0)
    Ht, Wt = H0 + ph, W0 + pw
    ph += _round_up(Ht, patch) - Ht
    pw += _round_up(Wt, patch) - Wt
    pad_mode = "constant"  # 0-padding rather than reflection, so no tissue is fabricated at the border
    em_p = np.pad(em_uint8, ((0, ph), (0, pw)), mode=pad_mode)
    H, W = em_p.shape

    overlap = float(cfg.eval.overlap)
    stride = max(1, int(round(t * (1.0 - overlap))))
    win = _hann2d(t)
    xnorm = normalize_em(em_p, mean, std)

    K = int(cfg.data.num_classes)
    acc = np.zeros((K, H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)
    aux_acc = None  # list of [C_i, H, W] buffers for the decoder's aux_logits (true-instance evaluation)
    # Most aux channels (affinities, center heatmap) are position-invariant local predictions, so
    # Hann-blended overlap-averaging is valid. A decoder may instead declare position-dependent aux via
    # AUX_POSITION_DEPENDENT (panoptic offset -> within-window centroid): averaging those across seams
    # corrupts differently-clipped instances. For those, each pixel takes its value from the window
    # where it is most central (max Hann weight) rather than an average.
    pos_dep = (set(getattr(getattr(model, "decoder", None), "AUX_POSITION_DEPENDENT", ()) or ())
               if collect_aux else set())
    aux_bestw = np.full((H, W), -1.0, dtype=np.float32) if pos_dep else None

    for y in _window_starts(H, t, stride):
        for x0 in _window_starts(W, t, stride):
            patch = xnorm[y:y + t, x0:x0 + t]
            xt = torch.from_numpy(np.ascontiguousarray(patch))[None, None].to(device)
            logits = model(xt)                                   # [1, K, t, t]
            probs = torch.softmax(logits[0].float(), dim=0).cpu().numpy()  # [K, t, t]
            acc[:, y:y + t, x0:x0 + t] += probs * win[None]
            wsum[y:y + t, x0:x0 + t] += win
            if collect_aux:
                auxs = getattr(model, "aux_logits", None) or []
                if aux_acc is None:
                    aux_acc = [np.zeros((int(a.shape[1]), H, W), dtype=np.float32) for a in auxs]
                better = None
                if pos_dep:
                    sub = aux_bestw[y:y + t, x0:x0 + t]          # view; holds prior best weights here
                    better = win > sub
                    sub[better] = win[better]                    # in-place -> updates aux_bestw
                for k, a in enumerate(auxs):
                    av = a[0].float().cpu().numpy()
                    if k in pos_dep:
                        vslice = aux_acc[k][:, y:y + t, x0:x0 + t]
                        vslice[:, better] = av[:, better]        # dominant-window value (no averaging)
                    else:
                        aux_acc[k][:, y:y + t, x0:x0 + t] += av * win[None]

    probs = acc / np.maximum(wsum, 1e-6)[None]                   # [K, H, W]
    probs = probs[:, :H0, :W0]
    if K <= 2:
        fg = probs[1] if K == 2 else probs[0]
    else:
        # any-foreground = max over non-background classes (binary-per-organelle metrics).
        fg = probs[BACKGROUND + 1:].max(axis=0)
    fg = fg.astype(np.float32)
    if collect_aux:
        wn = np.maximum(wsum[:H0, :W0], 1e-6)[None]
        aux_full = []
        for k, a in enumerate(aux_acc or []):
            if k in pos_dep:
                aux_full.append(a[:, :H0, :W0].astype(np.float32))         # dominant-window, already un-weighted
            else:
                aux_full.append((a[:, :H0, :W0] / wn).astype(np.float32))  # Hann-average
        return fg, aux_full
    return fg


_STRAT_FIELDS = (
    "dataset", "subgroup", "modality", "scale_band", "tissue_context", "species_group",
    "organelle", "split", "collection", "crop_id", "orientation", "plane_k", "coverage_tier",
    "bucket", "canonical_nm", "gt_is_instance",
)


def _cap_region(em, mask, inst, max_px: int, tile: int):
    """Central-crop a region to <= ``max_px`` pixels (side rounded to a /16 multiple, >= ``tile``) so
    eval on pathologically-huge crops (e.g. 100M+px upsampled SBF-SEM) stays bounded. ``max_px<=0`` or an
    already-small-enough region -> returned unchanged. Applied uniformly to every arm, so the ranking is
    fair; the chosen configuration is re-scored full-region (max_region_px=0) for the headline metric."""
    import math

    if max_px <= 0:
        return em, mask, inst
    H, W = em.shape
    if H * W <= max_px:
        return em, mask, inst
    side = max(int(tile), (int(math.isqrt(int(max_px))) // 16) * 16)
    h, w = min(H, side), min(W, side)
    y0, x0 = (H - h) // 2, (W - w) // 2
    em = em[y0:y0 + h, x0:x0 + w]
    mask = mask[y0:y0 + h, x0:x0 + w]
    if inst is not None:
        inst = inst[y0:y0 + h, x0:x0 + w]
    return em, mask, inst


def evaluate_head(model, records, cfg, data_root, device, mean: float, std: float,
                  on_crop=None, preprocess=None, do_aggregate: bool = True) -> dict:
    """Sliding-window eval over every record, then aggregation (macro + micro + subgroup + bootstrap).

    For each record: load the full EM + mask (+ instance ids if present), predict a foreground
    probability map, threshold at ``cfg.eval.fg_threshold``, score with
    ``segmentation_training.harness.metrics.per_crop_metrics``; when ``cfg.data.task == 'instance'`` also score
    ``segmentation_training.harness.instance_metrics.instance_metrics`` (GT instances from the stored inst.tif, else a
    connected-components labelling of the binary GT). Each per-crop dict carries the record's
    stratification fields so a per-image CSV can be written directly.

    ``on_crop(i, n)`` (optional) is called after each crop for progress reporting.

    Returns ``{"summary": <aggregate dict>, "per_crop": [<per-crop metric dicts>]}``.
    """
    from .instance_eval import dual_instance_metrics, has_native_instance

    model = model.to(device).eval()
    ev = cfg.eval
    organelle = cfg.data.organelle
    is_instance = getattr(cfg.data, "task", "semantic") == "instance"
    # dual-metric: if the decoder exposes a native instance post-proc (mutex-watershed / panoptic grouping),
    # also report true-instance PQ/AP (prefix ``inst_``) alongside the connected-components semantic-map metric.
    has_native = is_instance and has_native_instance(model)

    # image-style conditioning at eval: dataset-scope estimation produces one style code per source from
    # that source's unlabelled tiles (unsupervised, the deployment scenario) and injects it for every
    # window; the style arms set per-record metadata / per-window tile style. Arms without image-style
    # conditioning (conditioner is None) are unaffected.
    conditioner = getattr(model, "conditioner", None)
    source_codes = None
    if conditioner is not None and getattr(conditioner, "style_scope", "tile") == "dataset" \
            and getattr(conditioner, "style_encoder", None) is not None:
        override = getattr(conditioner, "source_style_override", None)
        if override is not None:  # a retrieval arm supplied a pooled/retrieval-snapped per-source code bank
            source_codes = override
            print(f"[eval] using externally-supplied per-source style codes ({len(source_codes)} sources; "
                  f"retrieval).")
        else:
            from .conditioning_eval import precompute_source_codes
            source_codes = precompute_source_codes(model, records, cfg, data_root, mean, std, device)
            print(f"[eval] dataset-scope: style estimated from unlabelled tiles of {len(source_codes)} "
                  f"sources (unsupervised, no labels used).")

    per_crop: list[dict] = []
    n = len(records)
    max_region_px = int(getattr(ev, "max_region_px", 0) or 0)
    # Eval progress telemetry: sliding-window eval over the full (optionally capped) region is the slow
    # tail and the runner is otherwise silent here, so a windowed crop/s + ETA is printed to stdout
    # (captured per-arm by the caller). Cheap: a print every 25 crops.
    _ev_t0 = time.time()
    for i, r in enumerate(records):
        em, mask, inst = load_sample(r, data_root)  # inst = stored instance ids or None
        if preprocess is not None:  # global appearance normalisation applied to the raw EM region
            em = preprocess(em)
        em, mask, inst = _cap_region(em, mask, inst, max_region_px, int(cfg.encoder.tile_size))
        if conditioner is not None:
            model.set_record_context(r, device)  # per-record metadata / source id (the style arms)
            if source_codes is not None:
                conditioner.set_context(preset_code=source_codes.get(str(r.get("dataset"))))
        aux = None
        if has_native:
            prob, aux = predict_region(model, em, cfg, mean, std, device, collect_aux=True)
        else:
            prob = predict_region(model, em, cfg, mean, std, device)
        pred_bin = prob >= float(ev.fg_threshold)

        m = per_crop_metrics(
            pred_bin, mask, prob=prob, organelle=organelle,
            theta_frac=ev.boundary_theta_frac, dilation_ratio=ev.boundary_dilation_ratio,
            auprc_bins=ev.auprc_bins, hd95_pct=ev.hd95_pct,
        )

        if is_instance and not m["excluded"]:
            valid = mask != IGNORE_INDEX
            gt_inst = inst
            if gt_inst is None:
                # No stored instance ids -> pseudo-instances = connected components of the binary GT
                # (scipy imported lazily; no GPU needed).
                from scipy import ndimage as ndi
                gt_inst, _ = ndi.label((mask == 1) & valid)
            # Dual metric (semantic-CC pq/ap + true-instance inst_pq/inst_ap) via the shared
            # instance-eval boundary. See harness/instance_eval.py.
            m.update(dual_instance_metrics(getattr(model, "decoder", None), aux, prob, gt_inst, valid,
                                           fg_threshold=float(ev.fg_threshold), min_size=int(ev.instance_min_size)))

        for f in _STRAT_FIELDS:
            m[f] = r.get(f)
        m["subgroup"] = r.get("subgroup", "") or "(none)"
        m["sample_id"] = r.get("sample_id")
        per_crop.append(m)
        if on_crop is not None:
            on_crop(i + 1, n)
        if (i + 1) % 25 == 0 or (i + 1) == n:
            _el = time.time() - _ev_t0
            _rate = (i + 1) / max(_el, 1e-6)
            _eta = (n - i - 1) / max(_rate, 1e-9)
            print(f"[{getattr(cfg, 'name', '?')}] eval {i + 1}/{n} crops | {_el:.0f}s | "
                  f"{_rate:.2f} crop/s | ETA {_eta / 60:.1f}min", flush=True)

    if not do_aggregate:
        # Shard-worker path (segmentation_training.harness.eval_parallel): return raw per-crop dicts so the parent can
        # merge shards in original record order and aggregate once, matching the serial summary
        # (the seeded bootstrap depends on per-crop order, which the ordered merge preserves).
        return {"per_crop": per_crop}
    summary = aggregate(per_crop, bootstrap_n=int(ev.bootstrap_n), ci=float(ev.bootstrap_ci),
                        seed=int(cfg.optim.seed))
    return {"summary": summary, "per_crop": per_crop}
