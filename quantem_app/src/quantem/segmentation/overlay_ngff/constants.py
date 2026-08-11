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
#: Fallback worker count. Read it through :func:`raster_process_pool_size`,
#: never directly -- the machine profile is allowed to lower it (see below).
RASTER_PROCESS_POOL_MAX = 4

#: How many tasks the level-0 rasteriser keeps outstanding, as a multiple of the
#: worker count. ``ProcessPoolExecutor.map`` submits *every* tile at once: it
#: bounds in-flight calls but places no bound on **completed results**, and the
#: consumer (64 compressed chunk writes per tile) is slower than the producers,
#: so finished 2048^2 crops pile up in the parent. Measured on the reference
#: 419 MP / 20 000-object canvas that backlog is the whole of a 2 051 MB parent
#: peak, and it grows with canvas area rather than with object count.
#:
#: A window of ``2 x workers`` keeps every worker fed (one task running, one
#: queued behind it) while capping the parent at that many crops. Two is the
#: smallest multiple that does both; one would idle a worker for the length of
#: each write.
RASTER_POOL_WINDOW_MULTIPLIER = 2

# Pyramid downsampling processes the parent level in large blocks (parent-pixel
# size) rather than per 256px chunk: far fewer zarr read/write calls, and empty
# (all-background) blocks are skipped entirely so no zero-chunks are written.
PYRAMID_BLOCK_SIZE = 2048
# Below this many objects **level-0 rasterisation** runs in-process (pool spawn
# overhead is not worth it for few draw ops). Gates level 0 only: that is the
# stage whose cost really does scale with the number of draw ops. The pyramid
# pass costs per visited block, not per object, so it has its own gate below --
# an empty canvas has zero objects and could still have thousands of blocks, and
# a handful of objects on a gigapixel canvas has neither.
RASTER_POOL_MIN_OBJECTS = 2000
# Below this many visited pyramid blocks the pyramid runs in-process, reusing a
# single open store handle across every level and array. Pool workers must
# re-open the staged store per block (process isolation leaves them no choice),
# so a pool only pays off once there are enough blocks to amortise both that and
# the spawn: a block costs well under a second, so a hundred-odd of them finish
# sooner in-process than a pool takes to pay for itself.
RASTER_POOL_MIN_PYRAMID_BLOCKS = 128


def raster_process_pool_size() -> int:
    """Worker count for the overlay rasterisation and pyramid pools.

    Comes from ``quantem.core.machine`` -- the single ``MachineProfile``
    computed once at startup (2 raster workers on ``small``, 4 on ``standard``,
    8 on ``workstation``), which BIG_IMAGE_DESIGN section 1.4(a) and owner
    ruling R2 make the *only* place allowed to ask how big this machine is.
    Nothing here may grow a second capability probe: no ``psutil``, no
    ``os.cpu_count()``.

    The import is deferred because ``core.machine`` pins the OpenMP/BLAS thread
    counts as a side effect of being used, and this module is imported early and
    widely; the pool size is only ever needed once a build is already running.

    ``RASTER_PROCESS_POOL_MAX`` remains as the value to use when there is no
    profile to read. That is not the normal path -- a missing
    ``quantem.core.machine`` means a broken install -- but an overlay build
    failing to start because the *machine probe* did not import would be a worse
    failure than building it with four workers.
    """
    try:
        from quantem.core.machine import get_machine_profile
    except ImportError:
        return RASTER_PROCESS_POOL_MAX
    return max(1, int(get_machine_profile().raster_workers))


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
