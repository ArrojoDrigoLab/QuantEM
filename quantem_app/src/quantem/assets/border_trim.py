"""Detect and remove solid black or white borders from imported 2-D images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def should_trim_initial_import(
    *,
    source_is_canonical_png: bool,
    rendition_metadata: dict | None,
) -> bool:
    """Whether this decode is the one import pass allowed to crop pixels.

    ``upload_state=staged`` is written when the source bytes are first claimed
    and changed to ``canonical`` only after the post-trim PNG and pyramid are
    safely published. Canonical retries and all later reprocessing therefore
    remain byte-for-byte stable.
    """
    metadata = rendition_metadata or {}
    return not source_is_canonical_png and metadata.get("upload_state") == "staged"


@dataclass(frozen=True)
class BorderTrim:
    left: int
    top: int
    right: int
    bottom: int
    original_width: int
    original_height: int

    def as_metadata(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "original_width": self.original_width,
            "original_height": self.original_height,
        }


def _solid_edge_colour(edge: np.ndarray) -> int | None:
    if edge.size == 0:
        return None
    colour = int(edge.flat[0])
    if colour not in (0, 255) or not np.all(edge == colour):
        return None
    return colour


def trim_black_or_white_border(
    plane: np.ndarray,
) -> tuple[np.ndarray, BorderTrim | None]:
    """Crop contiguous pure-black/pure-white rows and columns at each edge.

    Edges are peeled repeatedly because a rectangular frame hides the left and
    right border until its top and bottom rows have gone. Black and white may
    alternate between layers; every removed row or column must itself be
    uniform. A completely black or white image is left unchanged rather than
    collapsed to an empty or one-pixel image.
    """
    image = np.asarray(plane)
    if image.ndim != 2 or image.shape[0] < 2 or image.shape[1] < 2:
        return image, None

    original_height, original_width = image.shape
    outer_colour = _solid_edge_colour(image[0, :])
    if outer_colour is not None and np.all(image == outer_colour):
        # Besides being the only lossless result, this avoids repeatedly
        # scanning a large blank acquisition one shrinking perimeter at a
        # time.
        return image, None

    top, bottom = 0, original_height
    left, right = 0, original_width
    while top < bottom and left < right:
        changed = False
        if _solid_edge_colour(image[top, left:right]) is not None:
            top += 1
            changed = True
        if top < bottom and _solid_edge_colour(image[bottom - 1, left:right]) is not None:
            bottom -= 1
            changed = True
        if top < bottom and _solid_edge_colour(image[top:bottom, left]) is not None:
            left += 1
            changed = True
        if left < right and _solid_edge_colour(image[top:bottom, right - 1]) is not None:
            right -= 1
            changed = True
        if not changed:
            break

    # Blank images, and borders that consume the image, are valid inputs. They
    # are not useful crops and keeping them intact is the only lossless choice.
    if top >= bottom or left >= right:
        return image, None
    if top == 0 and bottom == original_height and left == 0 and right == original_width:
        return image, None

    cropped = np.ascontiguousarray(image[top:bottom, left:right])
    return cropped, BorderTrim(
        left=left,
        top=top,
        right=original_width - right,
        bottom=original_height - bottom,
        original_width=original_width,
        original_height=original_height,
    )
