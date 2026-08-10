"""Two-scale co-centered input — a fine branch plus a coarse-context branch.

A fine branch at the model's usual scale, and a coarse branch whose biological field is
``coarse_factor`` times wider in each dimension at the same token budget, obtained by downsampling the
larger region to the same tile size. Both pass through the shared encoder. The coarse tokens supply field-of-view context that helps
separate touching mitochondria and follow ER continuity. The fine tokens are fused with the full set of
coarse tokens, by cross-attention or feature concatenation, and drive the fine-resolution neck and
decoder.

Reuses the encoder feature-tap contract, the neck and decoder factories, ``build_loss`` and the
true-instance evaluation; the two-stream fusion and the co-centered tiling are specific to this arm.
Training updates the fusion, neck, decoder and encoder adapters; the base backbone stays frozen.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Fusion modules (fine tokens ← coarse-context tokens)
# --------------------------------------------------------------------------- #
class CrossScaleFusion(nn.Module):
    """Multi-head cross-attention: each fine token attends to the full set of coarse-context tokens,
    added residually. Fine and coarse grids share the token count (both are tile×tile), but attention is
    global so a fine token can pull context from anywhere in the larger coarse field. Zero-init output
    projection → identity at init (fine features unchanged until the fusion learns to use context)."""

    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.h = heads
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.norm_f = nn.LayerNorm(dim)
        self.norm_c = nn.LayerNorm(dim)

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        # fine/coarse: [B, C, h, w]
        b, c, h, w = fine.shape
        hc, wc = coarse.shape[-2:]
        qf = self.norm_f(fine.flatten(2).transpose(1, 2))          # [B, Nf, C]
        kvc = self.norm_c(coarse.flatten(2).transpose(1, 2))       # [B, Nc, C]
        q = self.q(qf).reshape(b, h * w, self.h, c // self.h).transpose(1, 2)
        kv = self.kv(kvc).reshape(b, hc * wc, 2, self.h, c // self.h).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(b, h * w, c)
        out = self.proj(out).transpose(1, 2).reshape(b, c, h, w)
        return fine + out


class ConcatScaleFusion(nn.Module):
    """Cheaper fusion: concat fine+coarse channels → 1×1 conv → GroupNorm → GELU back to ``dim``.
    Zero-init the coarse contribution path so it starts as identity on the fine features."""

    def __init__(self, dim: int, groups: int = 32):
        super().__init__()
        self.reduce = nn.Conv2d(2 * dim, dim, 1)
        self.gn = nn.GroupNorm(min(groups, dim), dim)
        self.act = nn.GELU()
        nn.init.zeros_(self.reduce.weight)
        nn.init.zeros_(self.reduce.bias)

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        if coarse.shape[-2:] != fine.shape[-2:]:
            coarse = F.interpolate(coarse, fine.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.act(self.gn(self.reduce(torch.cat([fine, coarse], 1))))
        return fine + fused


def build_fusion(kind: str, dim: int, n_taps: int) -> nn.ModuleList:
    cls = {"xattn": CrossScaleFusion, "concat": ConcatScaleFusion}[kind]
    return nn.ModuleList([cls(dim) for _ in range(n_taps)])


# --------------------------------------------------------------------------- #
# Two-scale model
# --------------------------------------------------------------------------- #
class TwoScaleSegModel(nn.Module):
    """Fine + coarse-context two-stream encoder → per-tap fusion → neck → decoder.

    ``forward(fine, coarse)`` returns fine-resolution logits ``[B, num_classes, H, W]``. ``aux_logits``
    proxies the decoder's auxiliary outputs for the true-instance evaluation."""

    def __init__(self, encoder, neck, decoder, layers, fusion: nn.ModuleList, encoder_trainable: bool = True):
        super().__init__()
        self.encoder = encoder
        self.neck = neck
        self.decoder = decoder
        self.layers = list(layers)
        self.fusion = fusion
        self.encoder_trainable = bool(encoder_trainable)

    def features(self, x):
        return self.encoder.features(x, self.layers, grad=self.encoder_trainable)

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        feats_f = self.features(fine)
        feats_c = self.features(coarse)
        fused = [self.fusion[i](ff, fc) for i, (ff, fc) in enumerate(zip(feats_f, feats_c))]
        pyramid = self.neck(fused, fine)
        return self.decoder(pyramid, out_hw=fine.shape[-2:])

    @property
    def aux_logits(self):
        return getattr(self.decoder, "aux_logits", []) or []

    def trainable_parameters(self):
        ps = list(self.neck.parameters()) + list(self.decoder.parameters()) + list(self.fusion.parameters())
        if self.encoder_trainable and getattr(self.encoder, "_conv_lora", None) is not None:
            ps += list(self.encoder._conv_lora.parameters())
        return [p for p in ps if p.requires_grad]


def build_two_scale_model(cfg, encoder, *, fuse: str = "xattn", coarse_factor: int = 2) -> TwoScaleSegModel:
    """Assemble a TwoScaleSegModel from an adapted-baseline-style cfg (neck/decoder/adapt from cfg, like build_segmodel)."""
    from ...hooks.encoder_adaptation import apply_adaptation
    from ...models.decoders import build_decoder
    from ...models.necks import build_neck

    layers = cfg.encoder.resolved_layers(encoder.depth)
    n_taps = len(layers)
    embed_dim = encoder.embedding_dim
    out_channels = int((cfg.neck.params or {}).get("out_channels", 256))
    neck = build_neck(cfg.neck, embed_dim, n_taps, encoder.patch_size, out_channels=out_channels)
    decoder = build_decoder(cfg.decoder, neck.out_channels, neck.strides, cfg.data.num_classes)
    encoder_trainable = apply_adaptation(encoder, getattr(cfg.encoder, "adapt", "frozen") or "frozen",
                                         cfg.encoder.adapt_params or {})
    fusion = build_fusion(fuse, embed_dim, n_taps)
    m = TwoScaleSegModel(encoder, neck, decoder, layers, fusion, encoder_trainable=encoder_trainable)
    m._coarse_factor = int(coarse_factor)
    return m


# --------------------------------------------------------------------------- #
# Two-scale co-centered crops (train dataset + eval tiling)
# --------------------------------------------------------------------------- #
class TwoScaleDataset(torch.utils.data.Dataset):
    """Yields co-centered ``{fine, coarse, target[, inst]}`` from a ``tile*coarse_factor`` crop.

    The fine target/inst is the center-``tile`` region (predicted at canonical resolution); the coarse
    branch supplies the surrounding FOV. Reuses the base dataset's crop-containing / pad logic by wrapping
    ``SegTrainDataset`` at the larger tile, then splitting fine/coarse in ``__getitem__``."""

    def __init__(self, records, data_root, cfg, mean, std, patch_size=16, coarse_factor=2):
        import copy as _copy

        from ...harness.dataset import SegTrainDataset
        self.cf = int(coarse_factor)
        p = int(patch_size)
        # the fine tile is rounded to a whole number of encoder patches; otherwise the encoder rejects
        # the input. Patch size differs across the compared encoders (14 for the DINOv2-derived ViT-L
        # backbones, OmniEM among them; 16 for the rest), so it is taken from the encoder, not assumed.
        self.tile = ((int(cfg.encoder.tile_size) + p - 1) // p) * p
        self.mean, self.std, self.patch = float(mean), float(std), p
        big = _copy.deepcopy(cfg)
        big.encoder.tile_size = self.tile * self.cf          # crop the larger field
        self._base = SegTrainDataset(records, data_root, big, mean, std, patch_size=patch_size)
        self.want_inst = getattr(cfg.data, "task", "semantic") == "instance"

    def __len__(self):
        return len(self._base)

    def reseed(self, salt):
        self._base.reseed(salt)

    def __getitem__(self, idx):
        item = self._base[idx]                               # image [1,L,L], target [L,L], inst?
        L = item["image"].shape[-1]
        tile = self.tile
        off = (L - tile) // 2
        # the base image is already normalized and augmented, so the fine/coarse split is taken directly
        # in normalized space, which preserves the augmentation.
        img = item["image"][0].numpy()                       # normalized float [L,L]
        fine = img[off:off + tile, off:off + tile]
        coarse = F.interpolate(torch.from_numpy(img)[None, None], (tile, tile),
                               mode="bilinear", align_corners=False)[0, 0].numpy()
        out = {"fine": torch.from_numpy(np.ascontiguousarray(fine))[None].float(),
               "coarse": torch.from_numpy(np.ascontiguousarray(coarse))[None].float(),
               "target": item["target"][off:off + tile, off:off + tile].clone()}
        if self.want_inst and "inst" in item:
            out["inst"] = item["inst"][off:off + tile, off:off + tile].clone()
        return out


@torch.no_grad()
def predict_region_two_scale(model, em_uint8, cfg, mean, std, device, coarse_factor=2, collect_aux=False):
    """Sliding-window two-scale eval: for each fine tile, extract the co-centered ``coarse_factor``×
    window from the (reflect-padded) region, downsample to tile, run ``model(fine, coarse)``, Hann-blend.
    Mirrors ``harness.evaluate.predict_region`` with the extra coarse-context extraction."""
    from ...constants import BACKGROUND
    from ...harness.dataset import normalize_em
    from ...harness.evaluate import _hann2d, _round_up, _window_starts

    patch = int(getattr(model.encoder, "patch_size", 16))
    t = _round_up(int(cfg.encoder.tile_size), patch)
    ctx = t * int(coarse_factor)
    H0, W0 = em_uint8.shape
    pad = (ctx - t) // 2                                     # coarse halo each side of a fine tile
    # pad region so every fine tile has a full coarse context + is a whole tile.
    ph = max(t - H0, 0); pw = max(t - W0, 0)
    Ht, Wt = H0 + ph, W0 + pw
    ph += _round_up(Ht, patch) - Ht; pw += _round_up(Wt, patch) - Wt
    pad_mode = "reflect" if (H0 > 1 and W0 > 1) else "constant"
    em_p = np.pad(em_uint8, ((0, ph), (0, pw)), mode=pad_mode)
    H, W = em_p.shape
    em_ctx = np.pad(em_p, ((pad, pad), (pad, pad)), mode=pad_mode)   # extra halo for coarse windows
    xnorm = normalize_em(em_p, mean, std)
    overlap = float(cfg.eval.overlap)
    stride = max(1, int(round(t * (1.0 - overlap))))
    win = _hann2d(t)
    K = int(cfg.data.num_classes)
    acc = np.zeros((K, H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    aux_acc = None
    for y in _window_starts(H, t, stride):
        for x0 in _window_starts(W, t, stride):
            fine = torch.from_numpy(np.ascontiguousarray(xnorm[y:y + t, x0:x0 + t]))[None, None].to(device)
            cy, cx = y, x0                                    # top-left of the coarse ctx window in em_ctx
            coarse_raw = em_ctx[cy:cy + ctx, cx:cx + ctx]
            coarse = F.interpolate(torch.from_numpy(coarse_raw.astype(np.float32))[None, None], (t, t),
                                   mode="bilinear", align_corners=False)[0, 0].numpy()
            coarse = torch.from_numpy(np.ascontiguousarray(normalize_em(coarse.clip(0, 255).astype(np.uint8),
                                                                        mean, std)))[None, None].to(device)
            logits = model(fine, coarse)
            probs = torch.softmax(logits[0].float(), 0).cpu().numpy()
            acc[:, y:y + t, x0:x0 + t] += probs * win[None]
            wsum[y:y + t, x0:x0 + t] += win
            if collect_aux:
                auxs = getattr(model, "aux_logits", None) or []
                if aux_acc is None:
                    aux_acc = [np.zeros((int(a.shape[1]), H, W), np.float32) for a in auxs]
                for k, a in enumerate(auxs):
                    aux_acc[k][:, y:y + t, x0:x0 + t] += a[0].float().cpu().numpy() * win[None]
    probs = (acc / np.maximum(wsum, 1e-6)[None])[:, :H0, :W0]
    fg = (probs[1] if K == 2 else probs[BACKGROUND + 1:].max(0)).astype(np.float32)
    if collect_aux:
        wn = np.maximum(wsum[:H0, :W0], 1e-6)[None]
        aux_full = [(a[:, :H0, :W0] / wn).astype(np.float32) for a in (aux_acc or [])]
        return fg, aux_full
    return fg


# --------------------------------------------------------------------------- #
# Two-scale train + eval (compact; mirrors harness.train / evaluate but two-stream)
# --------------------------------------------------------------------------- #
def _worker_reseed(worker_id: int) -> None:
    """Module-level (picklable) DataLoader worker init — reseeds the per-worker RNG. A closure would fail
    to pickle under the Windows spawn start method; ``get_worker_info().dataset`` reaches the worker's copy."""
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "reseed"):
        info.dataset.reseed(worker_id)


def _lr_scale(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def train_two_scale(cfg, encoder, records, data_root, device, *, fuse="xattn", coarse_factor=2,
                    logger=None) -> "TwoScaleSegModel":
    """Train the two-scale model (fusion + neck + decoder + LoRA adapters). baseline-matched optim/steps."""
    from torch.utils.data import DataLoader

    from ...models.losses import build_loss

    model = build_two_scale_model(cfg, encoder, fuse=fuse, coarse_factor=coarse_factor).to(device)
    ds = TwoScaleDataset(records, data_root, cfg, encoder.image_mean, encoder.image_std,
                         patch_size=encoder.patch_size, coarse_factor=coarse_factor)

    nw = int(cfg.num_workers)
    bs = min(int(cfg.optim.batch_size), max(1, len(ds)))         # never > dataset (else drop_last empties it)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=(len(ds) >= int(cfg.optim.batch_size)),
                        num_workers=nw, worker_init_fn=(_worker_reseed if nw > 0 else None))
    criterion = build_loss(cfg.loss, cfg.data.num_classes)
    # param groups: fusion+neck+decoder @ lr; adapters @ adapter_lr.
    core = list(model.fusion.parameters()) + list(model.neck.parameters()) + list(model.decoder.parameters())
    groups = [{"params": [p for p in core if p.requires_grad], "base_lr": cfg.optim.lr}]
    if model.encoder_trainable and getattr(encoder, "_conv_lora", None) is not None:
        groups.append({"params": list(encoder._conv_lora.parameters()), "base_lr": cfg.optim.adapter_lr})
    opt = torch.optim.AdamW([{**g, "lr": g["base_lr"]} for g in groups], weight_decay=cfg.optim.weight_decay)
    use_amp = cfg.amp and str(device).startswith("cuda")
    total, warmup = int(cfg.optim.max_steps), int(cfg.optim.warmup_steps)
    model.train(); encoder.backbone.eval()
    step = 0
    while step < total:
        n_in_epoch = 0
        for batch in loader:
            n_in_epoch += 1
            if step >= total:
                break
            fine = batch["fine"].to(device); coarse = batch["coarse"].to(device)
            y = batch["target"].to(device); inst = batch.get("inst")
            inst = inst.to(device) if inst is not None else None
            scale = _lr_scale(step, warmup, total)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(fine, coarse)
                loss, report = criterion(logits, y, model.aux_logits, inst=inst)
            loss.backward()
            if cfg.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            opt.step()
            if logger and step % 50 == 0:
                logger(step, float(loss.detach()))
            step += 1
        if n_in_epoch == 0:                                     # empty loader (dataset < batch) — avoid spin
            raise RuntimeError(f"train_two_scale: DataLoader yielded no batches "
                               f"(len(ds)={len(ds)}, batch_size={cfg.optim.batch_size}).")
    return model.eval()


def evaluate_two_scale(model, records, cfg, data_root, device, mean, std, *, coarse_factor=2) -> dict:
    """Two-scale sliding-window eval (same scoring as evaluate_head; both metrics)."""
    from ..common.region_eval import evaluate_with_predictor

    def predict_fn(m, em, c, mn, sd, dev, collect_aux=False):
        return predict_region_two_scale(m, em, c, mn, sd, dev, coarse_factor=coarse_factor,
                                        collect_aux=collect_aux)

    return evaluate_with_predictor(model, records, cfg, data_root, device, mean, std, predict_fn,
                                   extra_summary={"two_scale": {"coarse_factor": coarse_factor,
                                                                "fuse": getattr(model, "_fuse", "xattn")}})
