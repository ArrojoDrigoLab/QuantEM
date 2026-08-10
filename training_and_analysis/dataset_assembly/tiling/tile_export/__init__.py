"""Deterministic tissue-aware tile export helpers."""

# PNGs that are structurally complete (valid IEND, full pixel data) can still fail
# PIL's strict loader with "image file is truncated". Reading tolerantly loads the
# full pixel data without loss; a genuinely truncated file yields a blank tail that
# the tissue filter rejects. Set at package import: the tiling driver reads PNG
# planes through PIL and draws its parameters and helpers from this package, so the
# setting is in force before it opens its first plane.
from PIL import ImageFile as _PILImageFile

_PILImageFile.LOAD_TRUNCATED_IMAGES = True

from .config import TileExportConfig

__all__ = ["TileExportConfig"]
