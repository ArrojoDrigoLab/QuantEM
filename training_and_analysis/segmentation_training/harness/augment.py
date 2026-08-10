"""augmentation — one fixed paired image+mask recipe, identical across all arms.

Geometric ops act on image + mask (+ instance map) together; intensity ops act on the image only.
Deliberately excludes arbitrary (non-90 degree) rotation and large scale jitter: those either
interpolate thin-ER labels or confound the scale variable. Elastic uses ``scipy.ndimage`` (BLAS-free,
no GPU needed); warped-in border regions of the mask become IGNORE (255), never spurious background.

Operates on numpy arrays (uint8 EM, uint8 mask in {0,1,255}, optional int32 instance ids) so it slots
into the dataset before normalisation. Labels never interpolate (nearest / order-0 everywhere).
"""

from __future__ import annotations

import numpy as np

from ..constants import IGNORE_INDEX


class PairedAug:
    """Callable paired augmentation configured from a DataSpec. Give it a numpy Generator for
    reproducibility (the dataset seeds one per worker)."""

    def __init__(self, cfg, rng: np.random.Generator | None = None):
        self.flip = cfg.aug_flip
        self.rot90 = cfg.aug_rot90
        self.elastic = cfg.aug_elastic
        self.elastic_alpha = cfg.aug_elastic_alpha
        self.elastic_sigma = cfg.aug_elastic_sigma
        self.brightness = cfg.aug_brightness
        self.contrast = cfg.aug_contrast
        self.gamma = cfg.aug_gamma
        self.noise_std = cfg.aug_noise_std
        self.rng = rng or np.random.default_rng()

    # -- geometric (image + mask + inst) -----------------------------------
    def _geom(self, em, mask, inst):
        r = self.rng
        if self.flip:
            if r.random() < 0.5:
                em, mask = em[:, ::-1], mask[:, ::-1]
                inst = inst[:, ::-1] if inst is not None else None
            if r.random() < 0.5:
                em, mask = em[::-1, :], mask[::-1, :]
                inst = inst[::-1, :] if inst is not None else None
        if self.rot90:
            k = int(r.integers(0, 4))
            if k:
                em, mask = np.rot90(em, k), np.rot90(mask, k)
                inst = np.rot90(inst, k) if inst is not None else None
        em = np.ascontiguousarray(em)
        mask = np.ascontiguousarray(mask)
        inst = np.ascontiguousarray(inst) if inst is not None else None
        if self.elastic and self.elastic_alpha > 0:
            em, mask, inst = self._elastic(em, mask, inst)
        return em, mask, inst

    def _elastic(self, em, mask, inst):
        from scipy import ndimage as ndi

        h, w = em.shape
        r = self.rng
        dx = ndi.gaussian_filter((r.random((h, w)) * 2 - 1), self.elastic_sigma, mode="reflect")
        dy = ndi.gaussian_filter((r.random((h, w)) * 2 - 1), self.elastic_sigma, mode="reflect")
        # normalise the smoothed field to unit std, scale to alpha px of displacement
        dx = self.elastic_alpha * dx / (dx.std() + 1e-6)
        dy = self.elastic_alpha * dy / (dy.std() + 1e-6)
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        coords = [yy + dy, xx + dx]
        em2 = ndi.map_coordinates(em.astype(np.float32), coords, order=1, mode="reflect")
        mask2 = ndi.map_coordinates(mask, coords, order=0, mode="constant", cval=IGNORE_INDEX)
        inst2 = None
        if inst is not None:
            inst2 = ndi.map_coordinates(inst, coords, order=0, mode="constant", cval=0).astype(np.int32)
        return (np.clip(np.round(em2), 0, 255).astype(np.uint8), mask2.astype(np.uint8), inst2)

    # -- intensity (image only) --------------------------------------------
    def _intensity(self, em):
        r = self.rng
        x = em.astype(np.float32) / 255.0
        if self.brightness > 0:
            x = x + float(r.uniform(-self.brightness, self.brightness))
        if self.contrast > 0:
            f = 1.0 + float(r.uniform(-self.contrast, self.contrast))
            x = (x - 0.5) * f + 0.5
        x = np.clip(x, 0.0, 1.0)
        if self.gamma > 0:
            g = float(np.exp(r.uniform(-self.gamma, self.gamma)))  # multiplicative around 1
            x = np.power(x, g)
        if self.noise_std > 0:
            x = x + r.normal(0.0, self.noise_std, x.shape).astype(np.float32)
        return np.clip(x * 255.0, 0, 255).astype(np.uint8)

    def __call__(self, em: np.ndarray, mask: np.ndarray, inst: np.ndarray | None = None):
        em, mask, inst = self._geom(em, mask, inst)
        em = self._intensity(em)
        return em, mask, inst
