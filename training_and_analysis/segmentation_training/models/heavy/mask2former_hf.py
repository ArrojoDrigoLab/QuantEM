"""Mask2Former query-decoder bridge via HuggingFace ``transformers`` (no detectron2).

Implementation
--------------
detectron2, the reference Mask2Former implementation, does not build against recent CUDA toolchains
(torch 2.5.1+cu124 and later: its CUDA ops do not compile against the commit it requires), so the
query decoder is built without it. It reuses HuggingFace ``transformers``' pure-PyTorch Mask2Former
internals — the exact same architecture
(MSDeformAttn multi-scale pixel decoder + masked-attention transformer decoder + learnable queries +
Hungarian matcher + set criterion with deep supervision) — fed the frozen-encoder pyramid. Nothing is
reimplemented: every correctness-critical piece is HF's own class.

Reference classes reused (``transformers.models.mask2former.modeling_mask2former``):
  * ``Mask2FormerPixelDecoder(config, feature_channels)`` — MSDeformAttn pixel decoder.
        ``forward(features)`` -> ``.mask_features`` [B,C,H/4,W/4], ``.multi_scale_features`` (3 levels).
  * ``Mask2FormerTransformerModule(in_features, config)`` — masked-attention query decoder.
        ``forward(multi_scale_features, mask_features)`` -> ``.masks_queries_logits`` (per-layer
        [B,Q,H/4,W/4]) + ``.intermediate_hidden_states`` (per-layer [Q,B,C] query embeddings).
  * ``Mask2FormerLoss(config, weight_dict)`` — DETR-style set loss (CE + mask BCE + dice, point-sampled),
        aux-supervised over every decoder layer; embeds ``Mask2FormerHungarianMatcher``.

Decoder contract (see ``segmentation_training.models.base``):
  * ``forward(pyramid, out_hw) -> [B, num_classes, H, W]`` dense semantic logits (softmax over class
    queries, drop no-object, einsum with sigmoid(mask queries)), background channel synthesised as
    ``1 - sum(fg)`` in prob space then ``log()``, so ``evaluate.py`` scores this arm through the same
    dense-logits path as every other decoder.
  * ``uses_query_loss = True`` -> ``segmentation_training.harness.train`` routes to ``compute_loss`` (Hungarian set loss).
  * ``build_targets(batch, device)`` / ``compute_loss(batch, device)`` — HF SetCriterion over instances.
  * ``self.aux_logits = [per_query_masks]`` — for the true-instance eval (``native_instance_labels``).
  * ``native_instance_labels(aux, fg)`` — per-query-mask -> instance-id map (query argmax over fg),
    the query decoder's own post-proc, scored under the ``inst_*`` keys.

All heavy imports (``transformers``) happen inside ``__init__`` so ``import segmentation_training`` and the CPU test
suite stay importable in environments where ``transformers`` is absent.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import STRIDES, resize_to
from ...constants import FOREGROUND, IGNORE_INDEX


class Mask2FormerHFBridge(nn.Module):
    """Adapt the segmentation_training 4-level pyramid into HF Mask2Former's (pixel decoder -> query decoder) stack.

    Args:
        in_channels: channel width of every pyramid level (the neck's ``out_channels``).
        strides:     pyramid strides; must equal ``STRIDES`` (4, 8, 16, 32).
        num_classes: segmentation_training dense-logit class count including background (channel 0). The HF head is built
                     with ``num_classes - 1`` foreground (mask-classification) classes; HF adds its own
                     no-object class internally; background is synthesised on read-back.
        params:      optional overrides — ``num_queries`` (100), ``dec_layers`` (10), ``hidden_dim`` (256),
                     ``enc_layers`` (6), ``no_object_weight`` (0.1), ``class_weight`` (2.0),
                     ``mask_weight`` (5.0), ``dice_weight`` (5.0), ``train_num_points`` (12544),
                     ``oversample_ratio`` (3.0), ``importance_sample_ratio`` (0.75),
                     ``inst_mask_thresh`` (0.5, eval per-pixel mask threshold),
                     ``max_target_instances`` (None, cap on the instances the set loss supervises).
    """

    def __init__(self, in_channels: int, strides: tuple, num_classes: int, *, params: dict | None = None):
        super().__init__()
        # --- lazy heavy import (no GPU needed) ------------------------------------------------------
        from transformers import Mask2FormerConfig
        from transformers.models.mask2former import modeling_mask2former as M

        self.strides = tuple(strides)
        if self.strides != STRIDES:
            raise ValueError(f"Mask2Former(HF) bridge expects strides {STRIDES}, got {self.strides}")
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.ref_num_classes = max(1, self.num_classes - 1)  # HF foreground (mask-classification) classes
        self.uses_query_loss = True     # train.py routes to compute_loss (HF set criterion)
        self.aux_logits: list[torch.Tensor] = []

        p = dict(params or {})
        hidden_dim = int(p.get("hidden_dim", 256))
        num_queries = int(p.get("num_queries", 100))
        dec_layers = int(p.get("dec_layers", 10))
        enc_layers = int(p.get("enc_layers", 6))
        class_w = float(p.get("class_weight", 2.0))
        mask_w = float(p.get("mask_weight", 5.0))
        dice_w = float(p.get("dice_weight", 5.0))
        no_obj_w = float(p.get("no_object_weight", 0.1))
        num_points = int(p.get("train_num_points", 12544))
        self.inst_mask_thresh = float(p.get("inst_mask_thresh", 0.5))
        # Cap the number of target instances the set-loss supervises per image (largest-area kept). Default
        # None = every instance (mito: a handful per tile). ER's dense reticulum can raster to hundreds of
        # tiny CC fragments per 512 tile, whose [N,H,W] target-mask stack OOMs the Hungarian set loss; a
        # cap of ~num_queries keeps the loss bounded (never affects the dense-semantic Dice readback).
        _mti = p.get("max_target_instances", None)
        self.max_target_instances = int(_mti) if _mti is not None else None

        # HF Mask2Former config. ``backbone_config`` is left as the default Swin (never instantiated —
        # the backbone is bypassed entirely); only the pixel decoder + transformer module + class
        # predictor + criterion are built directly, fed the neck pyramid, so the Swin weights never allocate.
        cfg = Mask2FormerConfig(
            num_labels=self.ref_num_classes,
            feature_size=hidden_dim,
            mask_feature_size=hidden_dim,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            decoder_layers=dec_layers,
            encoder_layers=enc_layers,
            no_object_weight=no_obj_w,
            class_weight=class_w,
            mask_weight=mask_w,
            dice_weight=dice_w,
            train_num_points=num_points,
            oversample_ratio=float(p.get("oversample_ratio", 3.0)),
            importance_sample_ratio=float(p.get("importance_sample_ratio", 0.75)),
            use_auxiliary_loss=True,
        )
        self._cfg = cfg

        # The neck pyramid feeds 4 same-channel levels; the pixel decoder's per-level 1x1 input projections
        # adapt ``feature_channels`` -> hidden_dim (it consumes the coarsest ``num_feature_levels`` = 3
        # for the deformable encoder + lifts res2 to the stride-4 mask feature).
        feature_channels = [self.in_channels] * len(self.strides)
        self.pixel_decoder = M.Mask2FormerPixelDecoder(cfg, feature_channels=feature_channels)
        self.transformer_module = M.Mask2FormerTransformerModule(in_features=cfg.feature_size, config=cfg)
        self.class_predictor = nn.Linear(cfg.hidden_dim, cfg.num_labels + 1)  # +1 no-object (HF convention)

        weight_dict = {"loss_cross_entropy": class_w, "loss_mask": mask_w, "loss_dice": dice_w}
        self.criterion = M.Mask2FormerLoss(config=cfg, weight_dict=weight_dict)

        # Raw per-layer outputs from the last forward(); compute_loss reuses these for the same batch.
        self._last_masks: tuple | None = None    # per-layer masks_queries_logits (each [B,Q,h,w])
        self._last_classes: tuple | None = None  # per-layer class_queries_logits (each [B,Q,K+1])

    # --------------------------------------------------------------------------------------------- #
    # Forward: pyramid -> pixel decoder -> query decoder -> dense semantic logits
    # --------------------------------------------------------------------------------------------- #
    def forward(self, pyramid: list[torch.Tensor], out_hw) -> torch.Tensor:
        if len(pyramid) != len(self.strides):
            raise ValueError(f"expected {len(self.strides)} pyramid levels, got {len(pyramid)}")
        # HF pixel decoder reverses `features` internally and takes the coarsest num_feature_levels; the
        # pyramid is passed ascending-resolution (res2..res5 == strides 4,8,16,32) as HF's backbone would.
        pdo = self.pixel_decoder(pyramid, return_dict=True)
        tmo = self.transformer_module(
            multi_scale_features=pdo.multi_scale_features,
            mask_features=pdo.mask_features,
            output_hidden_states=True,
        )
        masks_per_layer = tmo.masks_queries_logits                     # tuple len=dec_layers, [B,Q,h,w]
        classes_per_layer = tuple(
            self.class_predictor(dec.transpose(0, 1)) for dec in tmo.intermediate_hidden_states
        )                                                              # each [B,Q,K+1]
        self._last_masks, self._last_classes = masks_per_layer, classes_per_layer

        mask_logits = masks_per_layer[-1]                              # [B,Q,h,w] (h,w = 1/4 res)
        class_logits = classes_per_layer[-1]                          # [B,Q,K+1]

        # per-query masks upsampled to out_hw for the mito instance metrics / post-proc.
        mask_up = F.interpolate(mask_logits, size=tuple(out_hw), mode="bilinear", align_corners=False)
        self.aux_logits = [mask_up.detach()]

        sem = self._semantic_inference(class_logits, mask_logits)     # [B, ref_K, h, w] probabilities
        logits = self._to_seg_logits(sem)                            # [B, num_classes, h, w] log-space
        return resize_to(logits, out_hw)

    @staticmethod
    def _semantic_inference(class_logits: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
        """HF ``post_process_semantic_segmentation`` core, batched: softmax(class)[..., :-1] einsum
        sigmoid(mask). Returns per-pixel foreground-class probabilities [B, ref_K, h, w] in [0,1]."""
        mask_cls = F.softmax(class_logits, dim=-1)[..., :-1]         # [B, Q, ref_K] (drop no-object)
        mask_pred = mask_logits.sigmoid()                            # [B, Q, h, w]
        return torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)

    def _to_seg_logits(self, sem_prob: torch.Tensor) -> torch.Tensor:
        """Foreground-class probs [B, ref_K, h, w] -> segmentation_training dense logits [B, num_classes, h, w].

        Synthesise background = ``1 - sum(fg)`` (clamped) as channel 0, then ``log()`` so evaluate.py's
        softmax/argmax over num_classes channels behaves (binary case: fg=p, bg=1-p)."""
        eps = 1e-6
        fg = sem_prob.clamp(0.0, 1.0)
        bg = (1.0 - fg.sum(dim=1, keepdim=True)).clamp(min=0.0)
        prob = torch.cat([bg, fg], dim=1)
        if prob.shape[1] != self.num_classes:
            raise RuntimeError(
                f"channel mismatch: built {prob.shape[1]} (=1 bg + {fg.shape[1]} fg) but "
                f"num_classes={self.num_classes}. Set decoder num_classes to fg classes + 1."
            )
        return torch.log(prob.clamp(min=eps))

    # --------------------------------------------------------------------------------------------- #
    # Query-mode training interface (train.py calls forward() then compute_loss() same-batch)
    # --------------------------------------------------------------------------------------------- #
    def build_targets(self, batch: dict, device):
        """HF SetCriterion targets from a segmentation_training batch: per-image ``(mask_labels [N,H,W], class_labels [N])``.

        Instances come from ``batch['inst']`` (int map, 0 = background); pixels where the semantic target
        is IGNORE (255) are zeroed out of the masks so ignore regions never supervise. Every instance gets
        foreground class 0 (HF's 0-based object-class convention; segmentation_training fg class 1 -> HF 0). Falls back to a
        single connected foreground region from ``target == FOREGROUND`` if ``inst`` is absent."""
        target = batch["target"].to(device)     # [B, H, W] in {0,1,255}
        valid = (target != IGNORE_INDEX)
        inst = batch.get("inst")
        if inst is not None:
            inst = inst.to(device)
        b = target.shape[0]
        mask_labels, class_labels = [], []
        for i in range(b):
            v_i = valid[i]
            if inst is not None:
                ids = torch.unique(inst[i])
                ids = ids[ids != 0]
                masks = [(inst[i] == iid) & v_i for iid in ids.tolist()]
                masks = [m for m in masks if m.any()]
                if self.max_target_instances is not None and len(masks) > self.max_target_instances:
                    # Keep the largest-area instances (stable, deterministic) so the set loss stays bounded
                    # on dense-fragment organelles (ER). Tiny fragments dropped from supervision here still
                    # appear in the eval's own instance post-proc; this bounds training memory only.
                    areas = torch.stack([m.sum() for m in masks])
                    keep = torch.argsort(areas, descending=True)[: self.max_target_instances].tolist()
                    masks = [masks[k] for k in keep]
                mask_t = (torch.stack(masks, 0).to(torch.float32) if masks
                          else torch.zeros((0, *v_i.shape), dtype=torch.float32, device=device))
            else:
                fg = (target[i] == FOREGROUND) & v_i
                mask_t = (fg[None].to(torch.float32) if fg.any()
                          else torch.zeros((0, *v_i.shape), dtype=torch.float32, device=device))
            cls_t = torch.zeros((mask_t.shape[0],), dtype=torch.int64, device=device)  # single fg class
            mask_labels.append(mask_t)
            class_labels.append(cls_t)
        return mask_labels, class_labels

    def compute_loss(self, batch: dict, device):
        """HF ``Mask2FormerLoss`` over the stored forward() outputs (same batch).

        Returns ``(loss_tensor, {term: float})``. Reuses ``self._last_masks`` / ``self._last_classes``
        from the most recent forward() (never re-runs the network); aux predictions are every decoder
        layer but the last, exactly as ``Mask2FormerForUniversalSegmentation`` assembles them."""
        if self._last_masks is None:
            raise RuntimeError("compute_loss called before forward(); no stored outputs.")
        mask_labels, class_labels = self.build_targets(batch, device)
        # The Hungarian matcher point-samples the mask logits and calls scipy ``linear_sum_assignment``,
        # which raises "matrix contains invalid numeric entries" on any non-finite cost. Under bf16
        # autocast the point-sampled CE/dice costs can overflow to inf/NaN, and the matcher's ±1e10 clamp
        # does not catch NaN. HF's own training runs this loss in fp32, so the stored logits are cast to
        # fp32 and residual non-finites sanitised before the criterion (the set loss is cheap here).
        def _f32(t: torch.Tensor) -> torch.Tensor:
            return torch.nan_to_num(t.float(), nan=0.0, posinf=30.0, neginf=-30.0)

        masks_f = tuple(_f32(m) for m in self._last_masks)
        classes_f = tuple(_f32(c) for c in self._last_classes)
        aux = [{"masks_queries_logits": m, "class_queries_logits": c}
               for m, c in zip(masks_f[:-1], classes_f[:-1])]
        with torch.autocast(device_type="cuda", enabled=False):
            loss_dict = self.criterion(
                masks_queries_logits=masks_f[-1],
                class_queries_logits=classes_f[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=aux,
            )
        wd = self.criterion.weight_dict
        total = None
        report: dict[str, float] = {}
        for k, v in loss_dict.items():
            w = wd.get(k, 1.0)  # HF pre-weights via weight_dict inside forward; parity is kept by summing all
            term = v if k in wd else v * 0.0
            total = term if total is None else total + term
            if not k.endswith(tuple(f"_{i}" for i in range(self._cfg.decoder_layers))):
                report[k] = float(v.detach())
        if total is None:
            total = sum(m.sum() for m in self._last_masks) * 0.0
        report["loss_total"] = float(total.detach())
        return total, report

    # --------------------------------------------------------------------------------------------- #
    # True-instance eval: per-query masks -> instance-id label map (the query decoder's own post-proc)
    # --------------------------------------------------------------------------------------------- #
    def native_instance_labels(self, aux, fg):
        """Instance-id map [H,W] int32 from the per-query masks (eval-only).

        ``aux = [per_query_mask_logits]`` (the single window's [Q,H,W] mask logits, not sliding-window
        accumulated — this arm is evaluated with the region capped to one tile so a single forward covers
        it; see the query arm config's ``max_region_px``). ``fg`` = semantic foreground [H,W] bool.

        Post-proc (mirrors Mask2Former instance inference): a per-pixel argmax over the sigmoid mask
        probabilities assigns each foreground pixel to its best query; a pixel whose winning query
        scores below ``inst_mask_thresh`` stays background; the surviving query ids are relabelled to
        consecutive ids >=1 with background (outside ``fg``) = 0."""
        masks = np.asarray(aux[0])              # [Q, H, W] mask logits (single window)
        if masks.ndim == 4:
            masks = masks[0]
        prob = 1.0 / (1.0 + np.exp(-masks.astype(np.float32)))  # sigmoid -> [Q,H,W] in [0,1]
        fg_bool = np.asarray(fg).astype(bool)
        Q, H, W = prob.shape
        if fg_bool.shape != (H, W):
            # crop fg to the mask grid; it matches out_hw on the normal path
            fg_bool = fg_bool[:H, :W]
        best_q = prob.argmax(axis=0)            # [H,W] winning query per pixel
        best_p = prob.max(axis=0)               # [H,W] its prob
        lab = np.zeros((H, W), dtype=np.int32)
        keep = fg_bool & (best_p >= self.inst_mask_thresh)
        # relabel winning-query ids to consecutive instance ids (1..)
        assigned = best_q[keep]
        uniq = np.unique(assigned)
        remap = {int(q): i + 1 for i, q in enumerate(uniq.tolist())}
        out = lab.copy()
        # vectorised remap over kept pixels
        flat_q = best_q[keep]
        flat_ids = np.array([remap[int(q)] for q in flat_q], dtype=np.int32) if flat_q.size else flat_q
        out[keep] = flat_ids
        return out


# =================================================================================================
# Builder (dispatched by segmentation_training.models.decoders)
# =================================================================================================
def build_mask2former_query_hf(in_channels: int, strides: tuple, num_classes: int,
                               params: dict | None = None) -> Mask2FormerHFBridge:
    """'query' mode via HF transformers (no detectron2): dense output plus build_targets + compute_loss
    (Hungarian set loss) + native_instance_labels. ``segmentation_training.harness.train`` detects ``uses_query_loss`` and
    trains with the HF SetCriterion; per-query masks ride on ``self.aux_logits`` for the mito instance
    metrics."""
    return Mask2FormerHFBridge(in_channels, strides, num_classes, params=params)
