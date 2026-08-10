"""Instance-target derivation.

Turns the dense instance-id map ``inst`` (already emitted by the dataloader, 0=background, >0=instance id,
``ignore_index``=ignore) into the supervision targets the native instance heads need:

* ``seg_to_affinities`` — for the ``affinity_mws`` head (mutex-watershed): per-offset affinities, 1 = the two
  pixels are the SAME foreground instance (attractive within-object), 0 = across a boundary or to background.
* ``seg_to_center_offset`` — for the ``panoptic_deeplab`` head: a center heatmap (Gaussian bumps at instance
  centroids) + a 2-channel offset field pointing each foreground pixel at its instance centroid.

Pure torch tensor ops (no BLAS matmul / sklearn), and every tensor is allocated on ``inst.device``, so
targets are derived inside the training loop on whichever device the batch already sits on.
The math is unit-tested in ``segmentation_training/tests/test_instance_targets.py``.
"""
from __future__ import annotations

import torch


def seg_to_affinities(inst: torch.Tensor, offsets, ignore_index: int = -100):
    """Instance-id map -> per-offset affinity targets + validity mask.

    Args:
        inst:    ``[B,H,W]`` or ``[H,W]`` long. 0=background, >0=instance id, ``ignore_index``=ignore.
        offsets: sequence of ``(dy, dx)`` displacement vectors (matching the decoder's affinity channel order).
        ignore_index: label value marking ignore pixels.

    Returns ``(aff, valid)``, both ``[B, n_offsets, H, W]``:
        aff   float {0,1}: 1 iff ``inst[y,x] == inst[y+dy,x+dx]`` and both are foreground (>0).
        valid bool:        False where the shifted pixel is out of bounds (no wrap-around) or either endpoint is ignore.
    """
    squeezed = inst.dim() == 2
    if squeezed:
        inst = inst[None]
    B, H, W = inst.shape
    n = len(offsets)
    aff = torch.zeros((B, n, H, W), dtype=torch.float32, device=inst.device)
    valid = torch.zeros((B, n, H, W), dtype=torch.bool, device=inst.device)
    fg = inst > 0
    ig = inst == ignore_index
    for o, (dy, dx) in enumerate(offsets):
        dy, dx = int(dy), int(dx)
        # shifted[y,x] = inst[y+dy, x+dx] over the in-bounds region only (no torch.roll wrap-around).
        shifted = torch.full_like(inst, ignore_index)
        ys0, ys1 = max(0, -dy), H - max(0, dy)   # source rows whose target y+dy is in [0,H)
        xs0, xs1 = max(0, -dx), W - max(0, dx)
        if ys0 < ys1 and xs0 < xs1:
            shifted[:, ys0:ys1, xs0:xs1] = inst[:, ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        inb = torch.zeros((B, H, W), dtype=torch.bool, device=inst.device)
        if ys0 < ys1 and xs0 < xs1:
            inb[:, ys0:ys1, xs0:xs1] = True
        same = (inst == shifted) & fg & (shifted > 0)
        aff[:, o] = same.float()
        # valid: in-bounds, neither endpoint ignore. Background endpoints are supervised → affinity 0.
        valid[:, o] = inb & (~ig) & (shifted != ignore_index)
    return aff, valid


def seg_to_center_offset(inst: torch.Tensor, sigma: float = 8.0):
    """Instance-id map -> (center heatmap, offset field, foreground mask) for the panoptic head.

    Args:
        inst:  ``[B,H,W]`` or ``[H,W]`` long. 0=background, >0=instance id.
        sigma: Gaussian std (px) for the center bumps.

    Returns:
        center ``[B,1,H,W]`` float in [0,1] — max of per-instance Gaussians at the (y,x) centroids.
        offset ``[B,2,H,W]`` float — channel 0 = ``centroid_y - y``, channel 1 = ``centroid_x - x`` for
               foreground pixels, 0 elsewhere.
        fg     ``[B,1,H,W]`` float {0,1} — foreground mask (offset loss is applied only here).
    """
    squeezed = inst.dim() == 2
    if squeezed:
        inst = inst[None]
    B, H, W = inst.shape
    dev = inst.device
    center = torch.zeros((B, 1, H, W), dtype=torch.float32, device=dev)
    offset = torch.zeros((B, 2, H, W), dtype=torch.float32, device=dev)
    yy = torch.arange(H, device=dev, dtype=torch.float32).view(H, 1).expand(H, W)
    xx = torch.arange(W, device=dev, dtype=torch.float32).view(1, W).expand(H, W)
    two_s2 = 2.0 * float(sigma) ** 2
    for b in range(B):
        ids = torch.unique(inst[b])
        for i in ids.tolist():
            if i <= 0:
                continue
            m = inst[b] == i
            cy = yy[m].mean()
            cx = xx[m].mean()
            offset[b, 0][m] = cy - yy[m]
            offset[b, 1][m] = cx - xx[m]
            g = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / two_s2)
            center[b, 0] = torch.maximum(center[b, 0], g)
    fg = (inst > 0).unsqueeze(1).to(torch.float32)
    return center, offset, fg


# Default MWS offsets: 4 short-range attractive (the 4-connected nearest neighbours) followed by the
# long-range repulsive edges. The attractive block leads the list, matching the
# ``number_of_attractive_channels = min(n_attractive, n_channels)`` convention in
# ``decoders.mutex_watershed_postproc``, whose ``n_attractive`` defaults to 4.
DEFAULT_MWS_OFFSETS = [
    (0, 1), (1, 0),                 # short-range attractive
    (0, -1), (-1, 0),               # short-range attractive (other direction)
    (0, 9), (9, 0), (0, -9), (-9, 0),   # long-range repulsive
    (0, 27), (27, 0),               # longer-range repulsive
]
