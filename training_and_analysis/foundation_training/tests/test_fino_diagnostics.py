"""Metadata diagnostics: label assembly, nearest-neighbour examples, and the predictability
probe. Torch and encoder weights are not needed — the probe is exercised on synthetic
features, so it returns a structured result rather than raising."""

from __future__ import annotations

import numpy as np

from em_ssl.tools.fino_diagnostics import build_label_arrays, nearest_neighbor_examples, probe_predictability

def _rows(n=40):
    mods = ["FIB-SEM", "TEM", None]
    return [
        {
            "tile_id": f"t{i}",
            "modality": mods[i % 3],
            "organ": "Brain",
            "tissue": "brain_neuropil",
            "effective_nm_per_px": (5.0 if i % 4 else -1.0),
            "source_id": f"s{i % 2}",
            "dataset_id": "d",
        }
        for i in range(n)
    ]

def test_build_label_arrays_masks_missing_and_invalid():
    rows = _rows(40)
    labels, valid, log_nm, nm_valid = build_label_arrays(rows)
    # modality: every 3rd is None -> masked
    assert valid["modality"].sum() == sum(1 for r in rows if r["modality"] is not None)
    # nm: every 4th row is -1.0 -> invalid; the other three quarters are positive
    assert nm_valid.sum() == sum(1 for r in rows if r["effective_nm_per_px"] > 0)
    assert log_nm.shape == (40,) and np.all(log_nm[~nm_valid] == 0.0)
    assert set(labels.keys()) == {"modality", "organ", "tissue", "source_id", "dataset_id"}

def test_nearest_neighbor_examples_structure():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(40, 16)).astype("float32")
    nn = nearest_neighbor_examples(feats, _rows(40), k=4, n_queries=5, seed=0)
    assert len(nn) == 5
    ex = nn[0]
    assert "query" in ex and len(ex["neighbors"]) == 4
    assert "cos_sim" in ex["neighbors"][0] and "modality" in ex["neighbors"][0]
    # no self-neighbors
    for ex in nn:
        assert ex["query"]["index"] not in {nb["index"] for nb in ex["neighbors"]}

def test_probe_predictability_returns_structured_result():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(60, 16)).astype("float32")
    rows = _rows(60)
    labels, valid, log_nm, nm_valid = build_label_arrays(rows)
    res = probe_predictability(feats, labels, valid, continuous=log_nm, continuous_valid=nm_valid, seed=0)
    assert isinstance(res, dict)
    # Either scikit-learn is unavailable (graceful skip) or per-factor results are present.
    assert "_skipped" in res or any(k in res for k in ("modality", "source_id", "log_effective_nm_per_px"))
