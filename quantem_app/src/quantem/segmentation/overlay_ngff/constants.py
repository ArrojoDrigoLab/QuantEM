"""Shared constants for segmentation overlay ID-map bundles.

The overlay is an integer **label image** (``labels`` array, ``uint32``) plus a
1-bit **border mask** (``border`` array, ``uint8``). Each pixel of ``labels``
holds a *dense label* (``1..N``, ``0`` = background) that the frontend maps to a
colour at render time via a lookup table (LUT). Object *state* (confirmed /
candidate / excluded / refined / labeled) lives entirely in the LUT, so state
changes never touch the raster -- only geometry edits do.
"""

from __future__ import annotations

import numpy as np

OVERLAY_FORMAT_VERSION = 5
OVERLAY_CHUNK_SIZE = 256

# In-memory rasterisation uses int32 (OpenCV's label-image depth, CV_32SC1);
# the on-disk ``labels`` array is uint32. Dense labels are positive and bounded
# by the live object count (renumbered on every full rebuild), so they fit
# comfortably below 2**31.
LABEL_RASTER_DTYPE = np.int32
LABEL_STORE_DTYPE = np.uint32
BORDER_STORE_DTYPE = np.uint8

LABELS_ARRAY_KEY = "labels"
BORDER_ARRAY_KEY = "border"
OVERLAY_ARRAY_KEYS = (LABELS_ARRAY_KEY, BORDER_ARRAY_KEY)

# Width of the baked border, in level-0 pixels.
OVERLAY_BORDER_WIDTH = 2

# Level-0 rasterisation is parallelised across macro-tiles of this size. The
# value is a multiple of OVERLAY_CHUNK_SIZE so each worker writes whole,
# disjoint zarr chunks (no read-modify-write contention).
MACRO_TILE_SIZE = 2048
RASTER_PROCESS_POOL_MAX = 4
# Pyramid downsampling processes the parent level in large blocks (parent-pixel
# size) rather than per 256px chunk: far fewer zarr read/write calls, and empty
# (all-background) blocks are skipped entirely so no zero-chunks are written.
PYRAMID_BLOCK_SIZE = 2048
# Below this many objects a full build runs in-process (pool spawn overhead is
# not worth it for small maps).
RASTER_POOL_MIN_OBJECTS = 2000

# ---------------------------------------------------------------------------
# State -> colour palette (render-time; resolved per object into the LUT).
#
# Colours mirror the legacy 10-channel palette so the viewer looks unchanged.
# Borders are derived in-shader as a darkened tint of the fill, so only fill
# colours are stored here.
# ---------------------------------------------------------------------------
COLOR_CONFIRMED = "33CC66"
COLOR_CANDIDATE = "FF0000"
COLOR_EXCLUDED = "F59E0B"
COLOR_LABELED = "38BDF8"
COLOR_REFINED = "3B82F6"

STATE_CONFIRMED = "confirmed"
STATE_CANDIDATE = "candidate"
STATE_EXCLUDED = "excluded"
STATE_LABELED = "labeled"
STATE_REFINED = "refined"

# Which states are visible by default (mirrors legacy omero ``active`` flags:
# confirmed + candidate on, excluded/labeled/refined off). User toggles patch
# LUT alpha on the client without a rebuild.
STATE_DEFAULT_VISIBLE = {
    STATE_CONFIRMED: True,
    STATE_CANDIDATE: True,
    STATE_EXCLUDED: False,
    STATE_LABELED: True,
    STATE_REFINED: True,
}

# Pixel-priority ladder: when two objects overlap, the higher-priority object
# wins the contested pixel. Encoded as rasterisation paint order (highest is
# painted last). Within a tier, larger objects paint first so small objects are
# not swallowed.
PRIORITY_CANDIDATE = 1
PRIORITY_EXCLUDED = 2
PRIORITY_LABELED = 3
PRIORITY_CONFIRMED = 4
PRIORITY_REFINED = 5
PRIORITY_MANUAL = 6

# Rebuild policy thresholds (unchanged: geometry edits still drive these).
SYNC_PARTIAL_MAX_LEVEL0_CHUNKS = 32
SYNC_PARTIAL_MAX_CHANGED_PIXELS = 2_000_000
ASYNC_PARTIAL_MAX_LEVEL0_CHUNKS = 512
ASYNC_PARTIAL_MAX_IMAGE_COVERAGE = 0.20

ACTIVE_OVERLAY_JOB_STATUSES = frozenset({"PENDING", "RUNNING", "RETRY"})
OVERLAY_VERSIONED_DIRNAME = "bundles"
OVERLAY_STAGING_DIRNAME = "staging"

# On-disk artifact name (was ``overlay.zarr`` for the legacy 10-channel store).
OVERLAY_STORE_DIRNAME = "labels.zarr"
