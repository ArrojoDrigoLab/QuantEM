"""Loss registry for the segmentation harness (ignore-aware, multi-class, no GPU needed).

The loss-function experiment builds the loss up cumulatively: Dice+BCE -> +Skeleton-Recall -> +clDice,
the three arms under ``configs/loss_function/``. Each term is an ``nn.Module`` returning a scalar, and
``CombinedLoss`` sums ``term(...) * weight`` and reports every term's value in a dict so the training
loop can log the progression.

Conventions (shared with harness/metrics.py):
  * ``target`` is ``[B, H, W]`` in ``{0 (background), 1 (foreground), 255 (ignore)}``.
  * Foreground class index == ``FOREGROUND`` (1); multi-class heads treat classes ``1..K-1`` as
    foreground and average Dice/topology over them.
  * All terms operate over valid pixels only (``target != ignore_index``). Invalid pixels contribute
    zero gradient (they are masked, not relabelled).

Portability: torch is the only third-party package imported at module level. numpy, ``scipy.ndimage``
(for the Skeleton-Recall GT skeleton and the orientation/centerline targets) and the optional
``skimage`` skeletonizer are imported lazily inside the functions that need them, so the module loads
in a CPU-only environment. The clDice term is a fully-differentiable torch soft-skeletonize (no scipy) so
it back-props; the numpy ``cldice`` in metrics.py is the (non-differentiable) evaluation metric.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.schema import LossSpec
from ..constants import FOREGROUND, IGNORE_INDEX

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _valid_mask(target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Boolean ``[B, H, W]`` — pixels that are labelled (not ignore)."""
    return target != ignore_index


def _fg_prob(logits: torch.Tensor) -> torch.Tensor:
    """Per-class foreground probability maps ``[B, K, H, W]`` via softmax (K>=2) or sigmoid (K==1)."""
    if logits.shape[1] == 1:
        p = torch.sigmoid(logits)
        return p  # single foreground channel
    return F.softmax(logits, dim=1)


def _onehot_valid(target: torch.Tensor, num_classes: int, valid: torch.Tensor) -> torch.Tensor:
    """One-hot ``[B, K, H, W]`` of the target with ignore pixels zeroed (so they don't count)."""
    safe = torch.where(valid, target, torch.zeros_like(target)).long()
    oh = F.one_hot(safe.clamp(min=0, max=num_classes - 1), num_classes).permute(0, 3, 1, 2).float()
    return oh * valid.unsqueeze(1).float()


def _foreground_classes(num_classes: int) -> list[int]:
    """Class indices treated as foreground (everything but background 0). Binary -> [1]."""
    if num_classes <= 1:
        return [0]
    return list(range(FOREGROUND, num_classes))


# --------------------------------------------------------------------------- #
# Term modules
# --------------------------------------------------------------------------- #
class SoftDiceLoss(nn.Module):
    """Soft Dice over valid pixels, averaged over the foreground classes (1 - Dice)."""

    def __init__(self, ignore_index: int = IGNORE_INDEX, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1] if logits.shape[1] > 1 else 2
        valid = _valid_mask(target, self.ignore_index)
        probs = _fg_prob(logits)
        if logits.shape[1] == 1:
            probs = torch.cat([1.0 - probs, probs], dim=1)  # [bg, fg] for a sigmoid head
        oh = _onehot_valid(target, num_classes, valid)
        vmask = valid.unsqueeze(1).float()
        probs = probs * vmask
        dims = (0, 2, 3)
        inter = (probs * oh).sum(dims)
        denom = probs.sum(dims) + oh.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        fg = _foreground_classes(num_classes)
        return 1.0 - dice[fg].mean()


class CEIgnoreLoss(nn.Module):
    """Ignore-aware cross-entropy (multi-class) / BCE (single-channel head)."""

    def __init__(self, ignore_index: int = IGNORE_INDEX, weight: torch.Tensor | None = None):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.register_buffer("class_weight", weight if weight is not None else None, persistent=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            valid = _valid_mask(target, self.ignore_index)
            tgt = (target == FOREGROUND).float()
            loss = F.binary_cross_entropy_with_logits(
                logits[:, 0], tgt, reduction="none"
            )
            loss = loss * valid.float()
            return loss.sum() / valid.float().sum().clamp(min=1.0)
        return F.cross_entropy(
            logits, target.long(), weight=self.class_weight,
            ignore_index=self.ignore_index,
        )


class FocalLoss(nn.Module):
    """Ignore-aware multi-class focal loss (Lin et al. 2017)."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None,
                 ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = _valid_mask(target, self.ignore_index)
        if logits.shape[1] == 1:
            p = torch.sigmoid(logits[:, 0])
            tgt = (target == FOREGROUND).float()
            pt = torch.where(tgt > 0.5, p, 1.0 - p).clamp(min=_EPS, max=1.0 - _EPS)
            ce = F.binary_cross_entropy_with_logits(logits[:, 0], tgt, reduction="none")
        else:
            logp = F.log_softmax(logits, dim=1)
            safe = torch.where(valid, target, torch.zeros_like(target)).long()
            logpt = logp.gather(1, safe.unsqueeze(1)).squeeze(1)
            pt = logpt.exp().clamp(min=_EPS, max=1.0 - _EPS)
            ce = -logpt
        focal = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            focal = self.alpha * focal
        focal = focal * valid.float()
        return focal.sum() / valid.float().sum().clamp(min=1.0)


class DiceBCELoss(nn.Module):
    """Soft Dice + ignore-aware CE/BCE (the baseline term)."""

    def __init__(self, ignore_index: int = IGNORE_INDEX, dice_weight: float = 1.0,
                 ce_weight: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.dice = SoftDiceLoss(ignore_index=ignore_index, smooth=smooth)
        self.ce = CEIgnoreLoss(ignore_index=ignore_index)
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.covers_ce = True  # tells CombinedLoss the ignore-aware CE is already included

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.ce_weight * self.ce(logits, target)


class DiceFocalLoss(nn.Module):
    """Soft Dice + ignore-aware focal loss."""

    def __init__(self, ignore_index: int = IGNORE_INDEX, gamma: float = 2.0,
                 alpha: float | None = None, dice_weight: float = 1.0, focal_weight: float = 1.0,
                 smooth: float = 1.0):
        super().__init__()
        self.dice = SoftDiceLoss(ignore_index=ignore_index, smooth=smooth)
        self.focal = FocalLoss(gamma=gamma, alpha=alpha, ignore_index=ignore_index)
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.covers_ce = True  # focal is the classification term here

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.focal_weight * self.focal(logits, target)


# --------------------------------------------------------------------------- #
# clDice (differentiable soft-clDice; Shit et al. 2021)
# --------------------------------------------------------------------------- #
def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Soft erosion via cross-shaped min-pooling (= -maxpool(-x) over a 3x1 + 1x3 structuring elem)."""
    p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """Soft dilation via 3x3 max-pooling."""
    return F.max_pool2d(img, (3, 3), stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img: torch.Tensor, n_iter: int = 10) -> torch.Tensor:
    """Differentiable soft-skeleton (Shit et al. 2021) of a soft mask ``[B, 1, H, W]`` in [0, 1]."""
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(n_iter):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftClDiceLoss(nn.Module):
    """Differentiable soft-clDice loss (Shit et al. 2021): 1 - centerline Dice of soft fg prob vs GT.

    Operates on the softmax foreground probability and the GT foreground, over valid pixels only
    (invalid pixels zeroed on both prediction and target before skeletonizing).
    """

    def __init__(self, ignore_index: int = IGNORE_INDEX, iters: int = 10, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.iters = int(iters)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = _valid_mask(target, self.ignore_index).unsqueeze(1).float()
        probs = _fg_prob(logits)
        if logits.shape[1] == 1:
            fg_prob = probs  # already the fg channel
        else:
            # sum over foreground classes -> a single tubular-foreground probability
            fg_prob = probs[:, _foreground_classes(probs.shape[1])].sum(dim=1, keepdim=True)
        fg_prob = (fg_prob * valid).clamp(0.0, 1.0)
        gt = ((target == FOREGROUND).unsqueeze(1).float() * valid)

        skel_pred = soft_skeletonize(fg_prob, self.iters)
        skel_gt = soft_skeletonize(gt, self.iters)

        dims = (1, 2, 3)
        # topology precision: pred skeleton mass covered by GT mask
        tprec = (skel_pred * gt).sum(dims) + self.smooth
        tprec = tprec / (skel_pred.sum(dims) + self.smooth)
        # topology sensitivity: GT skeleton mass covered by pred mask
        tsens = (skel_gt * fg_prob).sum(dims) + self.smooth
        tsens = tsens / (skel_gt.sum(dims) + self.smooth)
        cl = 2.0 * tprec * tsens / (tprec + tsens)
        return (1.0 - cl).mean()


# --------------------------------------------------------------------------- #
# Skeleton-Recall (Kirchhoff et al., ECCV 2024)
# --------------------------------------------------------------------------- #
def _batch_gt_skeleton(target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """GT skeleton of the foreground, per sample, as a float ``[B, 1, H, W]`` tensor in {0,1}.

    ``scipy.ndimage`` and numpy are imported lazily here so torch stays the module's only
    module-level third-party import. Falls back to a morphological skeleton when skimage is absent —
    matching harness/metrics._skeletonize so the loss and metric agree.
    """
    from scipy import ndimage as ndi  # lazy

    def _skel(mask):
        if not mask.any():
            return mask
        try:
            from skimage.morphology import skeletonize  # optional; see the fallback below
            import numpy as np
            return np.asarray(skeletonize(mask), dtype=bool)
        except Exception:
            import numpy as np
            skel = np.zeros_like(mask, dtype=bool)
            eroded = mask.copy()
            while eroded.any():
                opened = ndi.binary_dilation(ndi.binary_erosion(eroded))
                skel |= eroded & ~opened
                eroded = ndi.binary_erosion(eroded)
            return skel

    import numpy as np
    tgt = target.detach().cpu().numpy()
    valid = tgt != ignore_index
    fg = (tgt == FOREGROUND) & valid
    out = np.zeros(tgt.shape, dtype=np.float32)
    for b in range(tgt.shape[0]):
        out[b] = _skel(fg[b]).astype(np.float32)
    return torch.from_numpy(out).unsqueeze(1).to(target.device)


class SkeletonRecallLoss(nn.Module):
    """Skeleton-Recall loss (Kirchhoff et al. 2024): recall of the GT centerline by the soft fg prob.

    The GT skeleton is precomputed on the fly (scipy, lazy). Skeleton pixels the network misses are
    penalised: ``1 - (sum over GT skeleton of pred_prob) / (skeleton size)`` — a soft recall over the
    tubular centerline, over valid pixels only. Emphasises thin structures the region Dice under-weights.
    """

    def __init__(self, ignore_index: int = IGNORE_INDEX, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = _valid_mask(target, self.ignore_index).unsqueeze(1).float()
        probs = _fg_prob(logits)
        if logits.shape[1] == 1:
            fg_prob = probs
        else:
            fg_prob = probs[:, _foreground_classes(probs.shape[1])].sum(dim=1, keepdim=True)
        fg_prob = (fg_prob * valid).clamp(0.0, 1.0)
        skel = _batch_gt_skeleton(target, self.ignore_index) * valid  # no grad through skeleton
        dims = (1, 2, 3)
        recall = (fg_prob * skel).sum(dims) + self.smooth
        recall = recall / (skel.sum(dims) + self.smooth)
        return (1.0 - recall).mean()


# --------------------------------------------------------------------------- #
# Orientation + centerline-distance auxiliary (activates only if aux heads exist)
# --------------------------------------------------------------------------- #
def _orientation_centerline_targets(target: torch.Tensor, ignore_index: int, n_bins: int):
    """Derive (orientation-class, centerline-distance, skeleton-mask) targets from the GT mask.

    ``scipy.ndimage`` is imported lazily. orientation = quantized local tangent angle (mod pi) along
    the skeleton; centerline distance = EDT to the skeleton (0 on the centerline). Returns numpy arrays.
    """
    from scipy import ndimage as ndi  # lazy
    import numpy as np

    tgt = target.detach().cpu().numpy()
    valid = tgt != ignore_index
    fg = (tgt == FOREGROUND) & valid
    B, H, W = tgt.shape
    ori = np.zeros((B, H, W), dtype=np.int64)
    dist = np.zeros((B, H, W), dtype=np.float32)
    skelm = np.zeros((B, H, W), dtype=np.float32)

    # local-tangent Sobel kernels reused across the batch
    for b in range(B):
        m = fg[b]
        if not m.any():
            continue
        try:
            from skimage.morphology import skeletonize
            sk = np.asarray(skeletonize(m), dtype=bool)
        except Exception:
            sk = np.zeros_like(m, dtype=bool)
            eroded = m.copy()
            while eroded.any():
                opened = ndi.binary_dilation(ndi.binary_erosion(eroded))
                sk |= eroded & ~opened
                eroded = ndi.binary_erosion(eroded)
        skelm[b] = sk.astype(np.float32)
        # distance to the centerline (0 on skeleton, growing outward), for the foreground region
        if sk.any():
            dist[b] = ndi.distance_transform_edt(~sk).astype(np.float32)
        # local tangent orientation of the skeleton via smoothed gradient of the skeleton field
        skf = ndi.gaussian_filter(sk.astype(np.float32), sigma=1.0)
        gy = ndi.sobel(skf, axis=0, mode="nearest")
        gx = ndi.sobel(skf, axis=1, mode="nearest")
        angle = np.arctan2(gy, gx)  # gradient direction; tangent is orthogonal but binning is mod pi
        angle = np.mod(angle, np.pi)  # orientation is undirected (mod pi)
        ori[b] = np.clip((angle / np.pi * n_bins).astype(np.int64), 0, n_bins - 1)
    return ori, dist, skelm, valid


class OrientationCenterlineAuxLoss(nn.Module):
    """Auxiliary per-pixel orientation (quantized angle) + centerline-distance regression.

    Consumes prediction channels from ``aux_logits`` emitted by the decoder. The expected aux tensor
    has ``n_orientation_bins + 1`` channels: the first ``n_bins`` are orientation-class logits, the
    last is the (>=0) centerline-distance regression. Both are supervised only on foreground pixels
    (orientation is only defined on the structure) over valid pixels.

    Documented no-op: if ``aux_logits`` is empty / None, or no aux tensor carries at least
    ``n_bins + 1`` channels, this term returns 0 (with a single warning). No decoder in the registry
    emits orientation/centerline aux heads and no released configuration selects this term, so the
    cumulative loss progression (Dice+BCE -> +Skeleton-Recall -> +clDice) is unaffected; the term
    takes effect only for a decoder whose aux tensor supplies those channels.
    """

    def __init__(self, ignore_index: int = IGNORE_INDEX, n_orientation_bins: int = 8,
                 orientation_weight: float = 1.0, distance_weight: float = 1.0,
                 distance_norm: float = 32.0):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.n_bins = int(n_orientation_bins)
        self.orientation_weight = float(orientation_weight)
        self.distance_weight = float(distance_weight)
        self.distance_norm = float(distance_norm)
        self._warned = False

    def _select_aux(self, aux_logits, hw):
        """Pick the aux tensor with >= n_bins+1 channels at the target spatial size (or None)."""
        need = self.n_bins + 1
        if not aux_logits:
            return None
        for a in aux_logits:
            if isinstance(a, torch.Tensor) and a.dim() == 4 and a.shape[1] >= need:
                return a
        return None

    def forward(self, logits: torch.Tensor, target: torch.Tensor, aux_logits=None) -> torch.Tensor:
        hw = target.shape[-2:]
        aux = self._select_aux(aux_logits, hw)
        if aux is None:
            if not self._warned:
                warnings.warn(
                    "orientation_centerline_aux: no aux tensor with >= n_orientation_bins+1 channels "
                    f"({self.n_bins + 1}) found in aux_logits; the term contributes zero unless the "
                    "decoder emits the orientation/centerline aux heads.",
                    RuntimeWarning, stacklevel=2,
                )
                self._warned = True
            return logits.sum() * 0.0  # keeps it in the graph, zero contribution

        if aux.shape[-2:] != tuple(hw):
            aux = F.interpolate(aux, size=tuple(hw), mode="bilinear", align_corners=False)
        ori_logits = aux[:, : self.n_bins]
        dist_pred = aux[:, self.n_bins]

        ori_t, dist_t, skel, valid = _orientation_centerline_targets(
            target, self.ignore_index, self.n_bins
        )
        import numpy as np
        device = aux.device
        ori_t = torch.from_numpy(ori_t).to(device)
        skel_t = torch.from_numpy(skel).to(device)                       # [B,H,W]
        dist_t = torch.from_numpy(dist_t / max(self.distance_norm, _EPS)).to(device)
        valid_t = torch.from_numpy(valid.astype(np.float32)).to(device)

        # orientation supervised on the skeleton (that is where a tangent is defined)
        ori_mask = skel_t * valid_t
        ce = F.cross_entropy(ori_logits, ori_t.clamp(0, self.n_bins - 1), reduction="none")
        ori_loss = (ce * ori_mask).sum() / ori_mask.sum().clamp(min=1.0)

        # centerline-distance regression on all valid pixels (smooth-L1)
        dreg = F.smooth_l1_loss(dist_pred, dist_t, reduction="none")
        dist_loss = (dreg * valid_t).sum() / valid_t.sum().clamp(min=1.0)

        return self.orientation_weight * ori_loss + self.distance_weight * dist_loss


# --------------------------------------------------------------------------- #
# instance-head losses — supervise the decoder instance heads (affinity / center+offset) so the native
# instance post-proc (mutex-watershed / panoptic grouping) works at eval. Targets derived from the
# ``inst`` map (segmentation_training.models.instance_targets). Each takes an extra ``inst`` kwarg (mirroring ``aux_logits``);
# both return 0 when the arm supplies neither (aux_logits/inst absent), so adding them is safe on any arm.
# --------------------------------------------------------------------------- #
class AffinityLoss(nn.Module):
    """BCE on the AffinityMWS head's predicted affinities (aux_logits[0], already sigmoid'd) vs targets from
    ``inst``, masked to in-bounds non-ignore pixels. Offsets must match the decoder's — defaults to
    ``AffinityMWS.DEFAULT_OFFSETS`` so they align by construction."""

    def __init__(self, ignore_index: int = IGNORE_INDEX, offsets=None):
        super().__init__()
        if offsets is None:
            from segmentation_training.models.decoders import AffinityMWS
            offsets = AffinityMWS.DEFAULT_OFFSETS
        self.offsets = tuple(tuple(int(v) for v in o) for o in offsets)
        self.ignore_index = int(ignore_index)

    def forward(self, logits, target, aux_logits=None, inst=None):
        if not aux_logits or inst is None:
            return logits.new_zeros(())          # documented no-op when the arm isn't an affinity head
        from segmentation_training.models.instance_targets import seg_to_affinities
        aff_pred = aux_logits[0].float()          # [B, n_off, H, W] in [0,1]
        if aff_pred.shape[1] != len(self.offsets):
            return logits.new_zeros(())           # channel mismatch -> not this head; no-op
        aff_tgt, valid = seg_to_affinities(inst, self.offsets, self.ignore_index)
        eps = 1e-6
        p = aff_pred.clamp(eps, 1.0 - eps)
        bce = -(aff_tgt * torch.log(p) + (1.0 - aff_tgt) * torch.log(1.0 - p))
        v = valid.float()
        return (bce * v).sum() / v.sum().clamp_min(1.0)


class PanopticInstanceLoss(nn.Module):
    """PanopticDeepLab instance-head loss: MSE on the center heatmap (sigmoid of aux_logits[0]) + fg-masked
    L1 on the offset field (aux_logits[1], pixel units) vs targets from ``inst``. offset_weight keeps the
    (large-magnitude) offset term in scale (Panoptic-DeepLab convention)."""

    def __init__(self, ignore_index: int = IGNORE_INDEX, sigma: float = 8.0, offset_weight: float = 0.01):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.sigma = float(sigma)
        self.offset_weight = float(offset_weight)

    def forward(self, logits, target, aux_logits=None, inst=None):
        if not aux_logits or len(aux_logits) < 2 or inst is None:
            return logits.new_zeros(())          # documented no-op when the arm isn't a panoptic head
        from segmentation_training.models.instance_targets import seg_to_center_offset
        center_pred = torch.sigmoid(aux_logits[0].float())   # [B,1,H,W]
        offset_pred = aux_logits[1].float()                  # [B,2,H,W] pixel units
        if center_pred.shape[1] != 1 or offset_pred.shape[1] != 2:
            return logits.new_zeros(())
        center_tgt, offset_tgt, fg = seg_to_center_offset(inst, self.sigma)
        center_loss = ((center_pred - center_tgt) ** 2).mean()
        off_l1 = (offset_pred - offset_tgt).abs() * fg       # fg [B,1,H,W] broadcasts over the 2 channels
        offset_loss = off_l1.sum() / fg.sum().clamp_min(1.0) / 2.0
        return center_loss + self.offset_weight * offset_loss


# --------------------------------------------------------------------------- #
# Registry + CombinedLoss
# --------------------------------------------------------------------------- #
LOSSES: dict[str, object] = {
    "dice_bce": DiceBCELoss,
    "dice_focal": DiceFocalLoss,
    "cldice": SoftClDiceLoss,
    "skeleton_recall": SkeletonRecallLoss,
    "orientation_centerline_aux": OrientationCenterlineAuxLoss,
    "affinity": AffinityLoss,
    "panoptic_instance": PanopticInstanceLoss,
}

# Terms that already contribute an ignore-aware classification term (so CombinedLoss doesn't add CE).
_COVERS_CE = {"dice_bce", "dice_focal"}


def _term_accepts_aux(term: nn.Module) -> bool:
    """Whether a term's forward takes an ``aux_logits`` argument (the orientation aux does)."""
    import inspect
    try:
        return "aux_logits" in inspect.signature(term.forward).parameters
    except (TypeError, ValueError):
        return False


def _term_accepts_inst(term: nn.Module) -> bool:
    """Whether a term's forward takes an ``inst`` argument (the instance-head losses do)."""
    import inspect
    try:
        return "inst" in inspect.signature(term.forward).parameters
    except (TypeError, ValueError):
        return False


class CombinedLoss(nn.Module):
    """Weighted sum of the loss-function experiment loss terms. Reports every term's raw (pre-weight) value in a dict.

    Always guarantees an ignore-aware classification signal: if none of the configured terms already
    covers CE/BCE (dice_bce / dice_focal), an auto ``CEIgnoreLoss`` is appended so training is stable
    even for a topology-only spec.
    """

    def __init__(self, terms: list[nn.Module], weights: list[float], names: list[str],
                 ignore_index: int = IGNORE_INDEX, add_auto_ce: bool = True):
        super().__init__()
        self.terms = nn.ModuleList(terms)
        self.weights = list(weights)
        self.names = list(names)
        self.ignore_index = int(ignore_index)
        self._accepts_aux = [_term_accepts_aux(t) for t in terms]
        self._accepts_inst = [_term_accepts_inst(t) for t in terms]
        if add_auto_ce:
            self.terms.append(CEIgnoreLoss(ignore_index=ignore_index))
            self.weights.append(1.0)
            self.names.append("ce")
            self._accepts_aux.append(False)
            self._accepts_inst.append(False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, aux_logits=None, inst=None):
        target = target.long()
        total = logits.new_zeros(())
        report: dict[str, float] = {}
        for term, w, name, wants_aux, wants_inst in zip(
                self.terms, self.weights, self.names, self._accepts_aux, self._accepts_inst):
            kw = {}
            if wants_aux:
                kw["aux_logits"] = aux_logits
            if wants_inst:
                kw["inst"] = inst
            val = term(logits, target, **kw)
            report[name] = float(val.detach())
            total = total + w * val
        report["total"] = float(total.detach())
        return total, report


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
# Decoder -> the instance-loss term its native head requires. An instance decoder trained without this
# term leaves the instance head at its random initialisation while the instance evaluation still scores it.
_INSTANCE_LOSS_FOR = {"affinity_mws": "affinity", "panoptic_deeplab": "panoptic_instance"}


def build_loss(spec: LossSpec, num_classes: int, decoder_type: str | None = None) -> CombinedLoss:
    """Build a ``CombinedLoss`` from a ``LossSpec`` (the loss-function experiment). Unknown term type -> ValueError.

    Each term is instantiated as ``LOSSES[term.type](ignore_index=..., **term.params)``; the term's
    configured ``weight`` scales its contribution in the sum. An ignore-aware CE is auto-appended
    unless a configured term already covers classification (dice_bce / dice_focal).

    ``decoder_type`` (optional): when it names an instance decoder (affinity_mws / panoptic_deeplab)
    whose matching instance-loss term is absent, that term is auto-appended (weight 1.0) with a
    warning, so the native instance head is never left unsupervised.
    """
    ignore_index = int(getattr(spec, "ignore_index", IGNORE_INDEX))
    terms: list[nn.Module] = []
    weights: list[float] = []
    names: list[str] = []
    covers_ce = False
    for lt in spec.terms:
        if lt.type not in LOSSES:
            raise ValueError(
                f"Unknown loss term type {lt.type!r}. Valid keys: {sorted(LOSSES)}"
            )
        params = dict(lt.params or {})
        # every term takes an ignore_index; pass it unless the caller overrode it in params
        params.setdefault("ignore_index", ignore_index)
        module = LOSSES[lt.type](**params)
        terms.append(module)
        weights.append(float(lt.weight))
        names.append(lt.type)
        if lt.type in _COVERS_CE or getattr(module, "covers_ce", False):
            covers_ce = True

    want = _INSTANCE_LOSS_FOR.get(str(decoder_type))
    if want is not None and want not in names:
        import warnings
        warnings.warn(
            f"decoder {decoder_type!r} requires the {want!r} instance-loss term, but loss.terms {names} "
            f"omit it -> auto-appending {want!r} (weight 1.0) so the native instance head is supervised. "
            f"Add it explicitly in the config to silence this warning.", RuntimeWarning, stacklevel=2)
        terms.append(LOSSES[want](ignore_index=ignore_index))
        weights.append(1.0)
        names.append(want)

    return CombinedLoss(
        terms, weights, names, ignore_index=ignore_index, add_auto_ce=not covers_ce
    )
