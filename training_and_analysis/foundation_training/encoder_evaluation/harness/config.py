"""Probe configuration: a dataclass over YAML, with a default for every field.

Self-contained — it does not import em_ssl.config — so this subfolder stands alone. Unknown YAML
keys are dropped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

@dataclass
class ProbeConfig:
    # --- feature extraction (identical across every compared encoder) ---
    tile_size: int = 512  # encoder input / context window; /16 gives a 32x32 token grid. For the
                          # context sweep this is the window the RoPE encoders read (512/768/1024),
                          # while the learned-position encoders stay at their native tile.
    # Common comparison region in px. None means it equals tile_size, the standard probe. When set
    # below tile_size, the encoder reads the full tile_size context but its central token grid is
    # cropped to this region before the decoder, so every encoder's decoder does identical work and
    # the metrics score identical central pixels while wider-context encoders still see the surround.
    compare_tile: int | None = None
    feature_layers: Any = "last4"  # "last4" | "last1" | explicit list of block indices
    apply_encoder_norm: bool = True  # apply the encoder's final LayerNorm to each selected layer

    # --- decoder ---
    decoder: str = "linear"  # linear | light_conv | upernet | unet
    num_classes: int = 2  # background + organelle

    # --- optimisation ---
    max_steps: int = 1500
    warmup_steps: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dice_weight: float = 1.0
    ce_weight: float = 1.0
    seed: int = 0

    # --- data / augmentation ---
    num_workers: int = 4
    min_fg_frac_keep: float = 0.0  # if >0, prefer crops with >= this fg fraction; 0.0 = no oversample
    flip: bool = True
    rot90: bool = True
    intensity_jitter: float = 0.1  # +/- brightness/contrast jitter (single-channel-safe), 0 disables

    # --- evaluation (sliding window over the full annotated region) ---
    eval_overlap: float = 0.25
    eval_batch_windows: int = 32  # sliding-window tiles forwarded per encoder call
    boundary_theta_frac: float = 0.0075  # Boundary-F1 tolerance (fraction of image diagonal)
    boundary_dilation_ratio: float = 0.02  # Boundary-IoU band width (fraction of image diagonal)
    fg_threshold: float = 0.5  # foreground probability threshold for the binary mask + instances
    auprc_bins: int = 256  # binning for the per-crop AUPRC sweep
    hd95_pct: float = 95.0  # percentile for the (HD95) Hausdorff distance
    # mito instance metrics (PQ/AP/VI): fixed connected-components post-proc, min object size in px
    instance_min_size: int = 16
    bootstrap_n: int = 1000  # bootstrap resamples for macro CIs (0 disables)
    bootstrap_ci: float = 95.0
    # Per-crop eval metrics across a thread pool; the distance transforms and instance PQ/AP/VI
    # are the main GPU-idle cost. Results are identical to serial. 1 = serial.
    eval_workers: int = 8

    # --- frozen-feature caching (opt-in) ---
    # Forward the frozen encoder over a fixed grid tiling of the train set once per
    # (checkpoint, organelle), cache the features, and train every decoder and label fraction on the
    # cache. Much faster for the small heads, at the cost of per-step image-space augmentation
    # (random crop, flip, jitter), which the cache cannot provide. Eval feature sharing across
    # decoders is always on and unaffected.
    cache_train_features: bool = False
    # Bound cached-feature memory by uniformly subsampling the train tiles to at most this many per
    # (checkpoint, organelle). 0 caches every tile.
    cache_max_tiles: int = 6000
    # Hard memory ceiling per cache, independent of tile count: a wider encoder costs more per tile,
    # so a fixed tile cap alone does not bound the total. Shrinks the effective tile count until no
    # single cache exceeds this many GB. 0 disables.
    cache_max_gb: float = 40.0
    # Keep the feature cache on the GPU so cached training reads features already on-device, avoiding
    # per-step host-to-device streaming. Falls back to host RAM if it would not fit in VRAM.
    cache_on_gpu: bool = True

    # --- label-efficiency sweep ---
    label_fractions: list[float] = field(default_factory=lambda: [1.0])

    # --- encoder adaptation ---
    # frozen | lora | lora_ln | last_n | full; see harness/encoder_adaptation.py. When not "frozen"
    # the encoder forward runs with grad and the trainable adapter or unfrozen-block params join the
    # optimizer at adapter_lr as a second group. Adaptation bypasses the frozen-feature cache, whose
    # features are static and would defeat it.
    adapt: str = "frozen"
    adapt_params: dict = field(default_factory=dict)  # {rank: 8, conv: false} for lora, {n: 4} for last_n
    adapter_lr: float = 1e-3  # LR for encoder-side adapters and unfrozen blocks

    # --- runtime ---
    device: str = "cuda"  # falls back to cpu if cuda unavailable
    amp: bool = True  # bf16 autocast on cuda

    def effective_compare(self) -> int:
        """Common comparison region in px; equals tile_size unless compare_tile is set."""
        return int(self.compare_tile) if self.compare_tile is not None else int(self.tile_size)

    def resolved_layers(self, depth: int) -> list[int]:
        """Concrete 0-based block indices for this arch depth."""
        fl = self.feature_layers
        if isinstance(fl, (list, tuple)):
            return [int(i) for i in fl]
        if fl == "last1":
            return [depth - 1]
        if fl == "last4":
            return [depth - 4, depth - 3, depth - 2, depth - 1]
        raise ValueError(f"feature_layers must be 'last1', 'last4', or a list; got {fl!r}")

    def to_dict(self) -> dict:
        return asdict(self)

def load_probe_config(path: str | Path | None) -> ProbeConfig:
    """Load a ProbeConfig from YAML (or defaults if path is None). Unknown keys are dropped."""
    if path is None:
        return ProbeConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(ProbeConfig)}
    kw = {k: v for k, v in raw.items() if k in known}
    return ProbeConfig(**kw)
