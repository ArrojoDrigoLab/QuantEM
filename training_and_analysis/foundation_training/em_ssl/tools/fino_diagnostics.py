"""Post-checkpoint representation diagnostics for FINO runs (no segmentation labels needed).

For a configurable sample of tiles, extracts frozen features from a run's teacher
checkpoint (via the run's ``checkpoint_index.json`` feature entry-point) and reports whether
the representation became more or less decodable for each metadata factor — the interpretability
signal for whether M+/M- changed the representation in the intended direction. These are not the
success metric; the downstream mito/ER decoder evaluations are. They are the diagnostics required
to interpret a run.

Computes:
  * linear/shallow-probe predictability of ``modality`` and ``organ`` (the two discrete guide
    objectives) and of ``tissue``, ``source_id`` and ``dataset_id`` (diagnostics — high
    ``source_id`` / ``dataset_id`` values flag provenance leakage);
  * a continuous probe (R²/MAE) for ``log(effective_nm_per_px)``;
  * PCA (+ t-SNE, + UMAP if available) 2-D embeddings coloured by each factor;
  * nearest-neighbour retrieval examples across scale / modality / organ.

Writes under ``<run_dir>/diagnostics/``: ``metadata_probe_results.{csv,json}``,
``embedding_pca_by_metadata.png``, ``embedding_tsne_by_metadata.png``,
``embedding_umap_by_metadata.png`` (if ``umap`` present), ``nearest_neighbors/``.

    python -m em_ssl.tools.fino_diagnostics --run-dir <encoder run dir> --data-root <BUNDLE> \\
        --num-tiles 2000

Feature extraction needs torch + the pinned DINOv3; probes/embeddings need scikit-learn and
matplotlib. Each capability degrades gracefully (skips + logs) when its dependency is absent.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# Factors probed. Objectives + provenance diagnostics; continuous handled separately.
DISCRETE_PROBE_FIELDS = ("modality", "organ", "tissue", "source_id", "dataset_id")
CONTINUOUS_PROBE = "log_effective_nm_per_px"

# --------------------------------------------------------------------------- #
# Pure analysis (testable on synthetic features; numpy + scikit-learn)
# --------------------------------------------------------------------------- #
def probe_predictability(
    features: np.ndarray,
    labels: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    continuous: np.ndarray | None = None,
    continuous_valid: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Shallow-probe predictability of each factor from frozen features.

    Discrete: logistic-regression cross-val accuracy + balanced accuracy + majority baseline.
    Continuous: ridge-regression cross-val R² + MAE. Returns a per-factor result dict; entries
    are skipped (with a reason) when scikit-learn is missing or a factor has too few valid
    samples / classes.
    """
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
        from sklearn.model_selection import cross_val_predict
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - sklearn optional
        return {"_skipped": f"scikit-learn unavailable: {exc!r}"}

    results: dict[str, Any] = {}
    Xall = StandardScaler().fit_transform(features.astype(np.float64))

    for name, y in labels.items():
        m = valid.get(name)
        m = np.ones(len(y), bool) if m is None else m.astype(bool)
        X, yv = Xall[m], np.asarray(y)[m]
        classes, counts = np.unique(yv, return_counts=True)
        if len(yv) < 30 or len(classes) < 2 or counts.min() < 3:
            results[name] = {"type": "discrete", "n_valid": int(m.sum()), "n_classes": int(len(classes)),
                             "skipped": "too few valid samples/classes for a 3-fold probe"}
            continue
        cv = int(min(5, counts.min()))
        try:
            pred = cross_val_predict(
                LogisticRegression(max_iter=1000, C=1.0, multi_class="auto"), X, yv, cv=cv, n_jobs=1
            )
            acc = float((pred == yv).mean())
            bacc = float(balanced_accuracy_score(yv, pred))
        except Exception as exc:  # pragma: no cover
            results[name] = {"type": "discrete", "n_valid": int(m.sum()), "skipped": f"probe error: {exc!r}"}
            continue
        majority = float(counts.max() / counts.sum())
        results[name] = {
            "type": "discrete",
            "n_valid": int(m.sum()),
            "n_classes": int(len(classes)),
            "probe_accuracy": round(acc, 4),
            "probe_balanced_accuracy": round(bacc, 4),
            "majority_baseline": round(majority, 4),
            "above_baseline": round(acc - majority, 4),
        }

    if continuous is not None:
        m = np.ones(len(continuous), bool) if continuous_valid is None else continuous_valid.astype(bool)
        X, yv = Xall[m], np.asarray(continuous, dtype=np.float64)[m]
        if len(yv) >= 30 and np.std(yv) > 1e-8:
            try:
                pred = cross_val_predict(Ridge(alpha=1.0), X, yv, cv=min(5, max(2, len(yv) // 10)), n_jobs=1)
                results[CONTINUOUS_PROBE] = {
                    "type": "continuous",
                    "n_valid": int(m.sum()),
                    "probe_r2": round(float(r2_score(yv, pred)), 4),
                    "probe_mae": round(float(mean_absolute_error(yv, pred)), 4),
                    "target_std": round(float(np.std(yv)), 4),
                }
            except Exception as exc:  # pragma: no cover
                results[CONTINUOUS_PROBE] = {"type": "continuous", "n_valid": int(m.sum()), "skipped": f"{exc!r}"}
        else:
            results[CONTINUOUS_PROBE] = {"type": "continuous", "n_valid": int(m.sum()),
                                         "skipped": "too few valid samples / no variance"}
    return results

def compute_embeddings(features: np.ndarray, seed: int = 0, do_tsne: bool = True, do_umap: bool = True) -> dict:
    """2-D embeddings of the feature matrix: PCA always; t-SNE and UMAP when available."""
    out: dict[str, np.ndarray] = {}
    try:
        from sklearn.decomposition import PCA

        out["pca"] = PCA(n_components=2, random_state=seed).fit_transform(features)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"PCA unavailable: {exc!r}")
    if do_tsne and len(features) >= 10:
        try:
            from sklearn.manifold import TSNE

            perp = float(min(30, max(5, len(features) // 4)))
            out["tsne"] = TSNE(n_components=2, random_state=seed, perplexity=perp, init="pca").fit_transform(features)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"t-SNE skipped: {exc!r}")
    if do_umap:
        try:
            import umap  # type: ignore

            out["umap"] = umap.UMAP(n_components=2, random_state=seed).fit_transform(features)
        except Exception:
            pass  # UMAP optional
    return out

def _cosine_sims(features: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity. Uses torch's matmul when available and falls back to numpy
    otherwise, so the computation does not depend on the platform's numpy BLAS."""
    f = features.astype(np.float32)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    try:
        import torch

        return (torch.from_numpy(f) @ torch.from_numpy(f).T).numpy()
    except Exception:  # pragma: no cover - numpy fallback
        return f @ f.T

def nearest_neighbor_examples(
    features: np.ndarray, meta_rows: list[dict], k: int = 6, n_queries: int = 12, seed: int = 0
) -> list[dict]:
    """Cosine nearest-neighbour retrieval examples (query + top-k neighbours with metadata)."""
    rng = np.random.default_rng(seed)
    sims = _cosine_sims(features).astype(np.float64)
    np.fill_diagonal(sims, -np.inf)
    n = len(features)
    q_idx = rng.choice(n, size=int(min(n_queries, n)), replace=False)
    out = []
    for qi in q_idx:
        nn = np.argsort(-sims[qi])[:k]
        out.append(
            {
                "query": {"index": int(qi), **_meta_subset(meta_rows[qi])},
                "neighbors": [{"index": int(j), "cos_sim": round(float(sims[qi, j]), 4), **_meta_subset(meta_rows[j])} for j in nn],
            }
        )
    return out

def _meta_subset(row: dict) -> dict:
    return {k: row.get(k) for k in ("tile_id", "modality", "organ", "tissue", "effective_nm_per_px", "source_id", "dataset_id")}

def plot_embeddings_by_metadata(embeddings: dict, meta_rows: list[dict], out_dir: Path) -> list[str]:
    """Scatter each embedding coloured by each metadata factor; one PNG per embedding method."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"matplotlib unavailable; skipping embedding plots: {exc!r}")
        return []
    color_fields = ["effective_nm_per_px", "modality", "organ", "tissue", "source_id", "dataset_id"]
    written = []
    for method, emb in embeddings.items():
        fig, axes = plt.subplots(1, len(color_fields), figsize=(4 * len(color_fields), 4), squeeze=False)
        for ax, field in zip(axes[0], color_fields):
            vals = [r.get(field) for r in meta_rows]
            if field == "effective_nm_per_px":
                c = np.array([float(v) if v not in (None, "") else np.nan for v in vals], dtype=float)
                c = np.log(np.where(c > 0, c, np.nan))
                sc = ax.scatter(emb[:, 0], emb[:, 1], c=c, s=6, cmap="viridis")
                fig.colorbar(sc, ax=ax, fraction=0.046)
                ax.set_title("log(nm/px)")
            else:
                cats = sorted({str(v) for v in vals if v not in (None, "")})
                idx = {c: i for i, c in enumerate(cats)}
                col = np.array([idx.get(str(v), -1) for v in vals])
                ax.scatter(emb[:, 0], emb[:, 1], c=col, s=6, cmap="tab20")
                ax.set_title(f"{field} ({len(cats)})")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{method.upper()} embedding by metadata")
        fig.tight_layout()
        path = out_dir / f"embedding_{method}_by_metadata.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(str(path))
    return written

# --------------------------------------------------------------------------- #
# Feature extraction (torch + DINOv3; gated)
# --------------------------------------------------------------------------- #
def extract_frozen_features(run_dir: Path, data_root: str | None, num_tiles: int, device: str, seed: int):
    """Load the run's teacher checkpoint via checkpoint_index and extract frozen CLS features.

    Returns ``(features [N, D], meta_rows)``. Requires torch + the pinned DINOv3.
    """
    import torch

    from ..config.schema import load_experiment, resolve_data_paths
    from ..data.shard_dataset import EMShardDataset, list_shard_urls
    from ..utils.checkpoint_index import CheckpointIndex

    idx = CheckpointIndex.load(run_dir)
    rec = idx.latest("teacher") or idx.latest("encoder")
    if rec is None:
        raise FileNotFoundError(f"No teacher/encoder checkpoint recorded in {run_dir}/checkpoint_index.json")
    entry = idx.manifest.feature_entry_point
    encoder = _build_dinov3_teacher(entry, rec.path, device)

    # Resolve shards from the run's resolved config / data-root.
    cfg_path = run_dir / "experiment_config.source.yaml"
    spec = load_experiment(cfg_path) if cfg_path.exists() else load_experiment(idx.manifest.config_path)
    spec = resolve_data_paths(spec, data_root=data_root)
    urls = list_shard_urls(spec.data.shard_dir, spec.data.shard_prefix)
    if not urls:
        raise FileNotFoundError(f"No shards found for diagnostics under {spec.data.shard_dir}")

    mean, std = spec.resolved_mean_std()
    crop = spec.max_global_crop
    ds = EMShardDataset(urls=urls, transform=None, min_side=crop, resampled=False, shuffle_buffer=0, seed=seed)

    feats, rows = [], []
    import io as _io

    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import v2 as T

    resize = T.Resize(crop, interpolation=InterpolationMode.BICUBIC, antialias=True)
    crop_t = T.CenterCrop(crop)
    with torch.no_grad():
        for img, meta in ds:
            if not isinstance(img, Image.Image):
                img = Image.open(_io.BytesIO(img)).convert("L")
            x = T.functional.pil_to_tensor(crop_t(resize(img))).float() / 255.0
            x = (x - mean) / std
            x = x.unsqueeze(0).to(device)
            f = encoder(x).squeeze(0).float().cpu().numpy()
            feats.append(f)
            rows.append({k: meta.get(k) for k in ("tile_id", "modality", "organ", "tissue", "effective_nm_per_px", "source_id", "dataset_id")})
            if len(feats) >= num_tiles:
                break
    return np.stack(feats), rows

def _build_dinov3_teacher(entry: dict, ckpt_path: str, device: str):
    """Build a 1-channel ViT from the feature_entry_point and load the teacher backbone."""
    import importlib

    import torch

    import warnings

    from ..utils.checkpoint_index import infer_dinov3_build_kwargs

    build = entry["build"]
    module = importlib.import_module(build["module"])
    factory = getattr(module, build["factory"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get(entry.get("checkpoint_key", "teacher"), ckpt)
    prefix = entry.get("backbone_prefix", "backbone.")
    backbone_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    # Rebuild the block config from the checkpoint's own keys — the entry point's kwargs omit LayerScale
    # (off by default in the bare factory, on in training), so a naive factory(**kwargs) would drop
    # ls1/ls2.gamma and corrupt the teacher features the diagnostics run on.
    model = factory(**infer_dinov3_build_kwargs(backbone_sd, build["kwargs"]))
    model.init_weights() if hasattr(model, "init_weights") else None
    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
    if unexpected:
        warnings.warn(f"[fino_diagnostics] {len(unexpected)} teacher keys did not load into the "
                      f"backbone ({list(unexpected)[:6]}); the diagnostics then describe a "
                      f"representation that differs from the checkpoint's.")
    model.eval().to(device)

    def encode(x):
        out = model(x, is_training=True) if "is_training" in model.forward.__code__.co_varnames else model(x)
        if isinstance(out, dict):
            return out.get("x_norm_clstoken", out.get("x_norm_patchtokens"))
        return out

    return encode

# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output) if args.output else run_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nearest_neighbors").mkdir(exist_ok=True)

    features, meta_rows = extract_frozen_features(run_dir, args.data_root, args.num_tiles, args.device, args.seed)
    result = analyze(features, meta_rows, out_dir, seed=args.seed,
                     do_tsne=not args.no_tsne, do_umap=not args.no_umap)
    print(f"[fino_diagnostics] {len(features)} tiles -> {out_dir}")
    return result

def build_label_arrays(meta_rows: list[dict]):
    """Build discrete label/valid arrays + continuous log-scale array from metadata rows.

    Discrete factors -> dense class indices (per-call vocab) + validity mask; scale ->
    ``log(effective_nm_per_px)`` with a positive-finite validity mask. Pure numpy (no BLAS),
    so it is unit-testable everywhere.
    """
    labels, valid = {}, {}
    for fld in DISCRETE_PROBE_FIELDS:
        raw = [r.get(fld) for r in meta_rows]
        cats = sorted({str(v) for v in raw if v not in (None, "")})
        idx = {c: i for i, c in enumerate(cats)}
        labels[fld] = np.array([idx.get(str(v), -1) for v in raw])
        valid[fld] = np.array([v not in (None, "") for v in raw])
    nm = np.array(
        [float(r["effective_nm_per_px"]) if r.get("effective_nm_per_px") not in (None, "") else np.nan for r in meta_rows],
        dtype=float,
    )
    nm_valid = np.isfinite(nm) & (nm > 0)
    log_nm = np.where(nm_valid, np.log(np.where(nm > 0, nm, 1.0)), 0.0)
    return labels, valid, log_nm, nm_valid

def analyze(features: np.ndarray, meta_rows: list[dict], out_dir: Path, seed: int = 0,
            do_tsne: bool = True, do_umap: bool = True) -> dict:
    """Run probes + embeddings + NN retrieval on a feature matrix; write artifacts."""
    labels, valid, log_nm, nm_valid = build_label_arrays(meta_rows)

    probes = probe_predictability(features, labels, valid, continuous=log_nm, continuous_valid=nm_valid, seed=seed)
    embeddings = compute_embeddings(features, seed=seed, do_tsne=do_tsne, do_umap=do_umap)
    plots = plot_embeddings_by_metadata(embeddings, meta_rows, out_dir)
    nn = nearest_neighbor_examples(features, meta_rows, seed=seed)

    (out_dir / "metadata_probe_results.json").write_text(
        json.dumps({"n_tiles": len(features), "probes": probes, "plots": plots}, indent=2, default=str), encoding="utf-8"
    )
    _write_probe_csv(out_dir / "metadata_probe_results.csv", probes)
    (out_dir / "nearest_neighbors" / "examples.json").write_text(json.dumps(nn, indent=2, default=str), encoding="utf-8")
    return {"probes": probes, "embeddings": list(embeddings.keys()), "plots": plots, "n_neighbors_examples": len(nn)}

def _write_probe_csv(path: Path, probes: dict) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["factor", "type", "n_valid", "score", "score_name", "baseline_or_std", "above_baseline"])
        for name, r in probes.items():
            if name.startswith("_") or "skipped" in r:
                w.writerow([name, r.get("type", ""), r.get("n_valid", ""), "", "", "", r.get("skipped", "")])
                continue
            if r["type"] == "discrete":
                w.writerow([name, "discrete", r["n_valid"], r["probe_balanced_accuracy"], "balanced_accuracy",
                            r["majority_baseline"], r["above_baseline"]])
            else:
                w.writerow([name, "continuous", r["n_valid"], r["probe_r2"], "r2", r["target_std"], r["probe_mae"]])

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="FINO post-checkpoint representation diagnostics.")
    p.add_argument("--run-dir", required=True, help="Run/stage dir containing checkpoint_index.json.")
    p.add_argument("--data-root", default=None, help="Data bundle root (resolves shards).")
    p.add_argument("--output", default=None, help="Output dir (default: <run-dir>/diagnostics).")
    p.add_argument("--num-tiles", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-tsne", action="store_true")
    p.add_argument("--no-umap", action="store_true")
    args = p.parse_args(argv)
    run(args)

if __name__ == "__main__":
    main()
