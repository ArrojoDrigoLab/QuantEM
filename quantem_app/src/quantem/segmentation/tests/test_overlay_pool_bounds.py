"""Stage S2: the overlay rasteriser's parent memory is bounded, and flat.

The defect. ``_rasterize_level0`` used to hand the whole tile list to
``ProcessPoolExecutor.map``. ``map`` bounds how many calls are *in flight*; it
places no bound at all on how many *results* are waiting to be consumed, and
every result is a 2048x2048 ``uint32`` labels crop plus a 2048x2048 ``uint8``
border crop -- 21 MB a tile. Four workers produce those faster than
``_write_tile_result`` can push 64 compressed chunks into zarr, so the finished
tiles pile up in the parent. Measured on a 419 MP canvas with 20 000 objects:
**2 051 MB** of parent, growing with canvas *area* and not at all with object
count. At the 3 224 MP reference image that is about 16 GB, on a machine the
design says has 8 GB in total.

The fix is a submission window: at most ``RASTER_POOL_WINDOW_MULTIPLIER x
raster_workers`` tasks outstanding, consumed in submission order. Bounded, and
constant in canvas size.

What the tests here pin, and what they deliberately do not:

* :class:`_RecordingExecutor` counts *held results* exactly, with no timing in
  it, so the window bound is asserted as an invariant rather than as a
  benchmark. The flat-versus-linear property is asserted on that count at 100
  and at 400 macro tiles. The corresponding measurement in megabytes of real
  process memory, at 419 MP and 1 678 MP with real pixels and a real pool, is
  out of suite -- a 1 678 MP rebuild takes minutes -- and lives in
  ``.scratch/tmp/s2_overlay_report.md``.
* One test does run the real pool twice over the real store and compares
  sha256, because "the output is byte-identical" is the acceptance that matters
  most and a mock cannot answer it.
* One test caps the child's committed memory with a Windows Job Object.
  ``JOB_OBJECT_LIMIT_PROCESS_MEMORY`` makes an over-budget allocation *fail*
  where it happens rather than letting the machine swap, which bounds the same
  quantity a small machine runs out of and reproduces the exact allocation that
  is too big. It does **not** reproduce what an 8 GB laptop actually does first
  -- compress memory, page, thrash, and only then have something killed -- so it
  models the failure, not the slow misery before it. ``JOB_OBJECT_LIMIT_JOB_-
  MEMORY`` is deliberately *not* set: the property under test is the *parent's*
  peak, and a job-wide cap would also be capping the four worker processes,
  which is a different measurement. ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` is
  set so a child that dies under the cap cannot leave its pool workers behind.
  On macOS there is no Job Object at all; ``resource.RLIMIT_AS`` is the POSIX
  analogue and a real MacBook Air has neither.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest
from django.test import TestCase
from shapely.geometry import box

from quantem.assets.models import Asset, Rendition
from quantem.jobs.reporter import JobCancelledError
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.overlay_ngff import mutations
from quantem.segmentation.overlay_ngff.constants import (
    MACRO_TILE_SIZE,
    RASTER_POOL_MIN_OBJECTS,
    RASTER_POOL_WINDOW_MULTIPLIER,
    RASTER_PROCESS_POOL_MAX,
    raster_process_pool_size,
)
from quantem.segmentation.overlay_ngff.paths import get_overlay_active_bundle_path
from quantem.segmentation.type_service import get_or_create_mitochondria_type

#: Bytes one real macro tile costs the parent: 2048^2 uint32 + 2048^2 uint8.
REAL_TILE_BYTES = MACRO_TILE_SIZE * MACRO_TILE_SIZE * 5

#: Stand-in tile for the invariant tests. Same shape of object, 1/64th the
#: bytes, so asserting on 400 tiles costs megabytes instead of gigabytes.
STAND_IN_TILE_SIDE = 256
STAND_IN_TILE_BYTES = STAND_IN_TILE_SIDE * STAND_IN_TILE_SIDE * 5


# ---------------------------------------------------------------------------
# A stand-in executor that counts what the parent is holding
# ---------------------------------------------------------------------------
class _RecordingFuture(Future):
    def __init__(self, recorder: _RecordingExecutor, value):
        super().__init__()
        self._recorder = recorder
        self.set_result(value)

    def result(self, timeout=None):
        value = super().result(timeout)
        self._recorder._release(id(self))
        return value


class _RecordingExecutor:
    """Runs tasks inline and records the peak backlog of *unconsumed results*.

    Deliberately synchronous. The quantity that blew up in production is not a
    race -- it is "how many finished crops has the parent been handed and not
    yet written", which is decided by the submission discipline alone. Running
    the worker at submit time is the worst case of that quantity and makes the
    assertion deterministic; a real pool is exercised by the byte-identity and
    capped-memory tests below.
    """

    instances: list[_RecordingExecutor] = []

    def __init__(self, max_workers: int | None = None, initializer=None, **kwargs):
        self.max_workers = max_workers
        self.initializer = initializer
        self.submitted = 0
        self.peak_outstanding = 0
        self.peak_held_bytes = 0
        self._held: dict[int, int] = {}
        _RecordingExecutor.instances.append(self)

    # -- bookkeeping --------------------------------------------------------
    @staticmethod
    def _result_bytes(value) -> int:
        return sum(int(part.nbytes) for part in value if isinstance(part, np.ndarray))

    def _hold(self, future: Future, value) -> None:
        self._held[id(future)] = self._result_bytes(value)
        self.peak_outstanding = max(self.peak_outstanding, len(self._held))
        self.peak_held_bytes = max(self.peak_held_bytes, sum(self._held.values()))

    def _release(self, key: int) -> None:
        self._held.pop(key, None)

    # -- executor surface ---------------------------------------------------
    def submit(self, fn, *args, **kwargs) -> Future:
        self.submitted += 1
        value = fn(*args, **kwargs)
        future = _RecordingFuture(self, value)
        self._hold(future, value)
        return future

    def map(self, fn, iterable, **kwargs):
        """What ``ProcessPoolExecutor.map`` does: every future up front."""
        futures = [self.submit(fn, item) for item in iterable]
        return (future.result() for future in futures)

    def shutdown(self, wait: bool = True, **kwargs) -> None:
        return None

    def __enter__(self) -> _RecordingExecutor:
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()


class _SinkArray:
    """Accepts the slice writes ``_write_tile_result`` makes, keeps nothing."""

    def __init__(self) -> None:
        self.writes = 0

    def __setitem__(self, key, value) -> None:
        self.writes += 1


def _stand_in_worker(payload: dict) -> tuple[int, int, np.ndarray, np.ndarray]:
    side = STAND_IN_TILE_SIDE
    labels = np.full((side, side), payload["interior"][0] + 1, dtype=np.uint32)
    border = np.zeros((side, side), dtype=np.uint8)
    return payload["interior"][0], payload["interior"][1], labels, border


def _stand_in_payloads(count: int) -> list[dict]:
    return [{"interior": (index, index, index + 1, index + 1)} for index in range(count)]


def _legacy_rasterize_level0(
    arrays,
    payloads,
    *,
    use_pool: bool,
    on_progress=None,
    cancel_check=None,
) -> None:
    """The body that shipped before stage S2, kept as a negative control.

    Copied from ``mutations._rasterize_level0`` as it stood at the start of this
    stage. Its only job here is to prove the recorder above can tell the two
    apart -- a bound that passes for both is not measuring anything.

    Two deliberate deviations from what shipped, both of the same kind: this
    control must differ from the real function in *exactly one* way, the
    submission discipline, because that is the only thing under test.

    * The worker count is read from the same accessor the fixed code reads, not
      from the module constant it used to be.
    * ``on_progress`` and ``cancel_check`` did not exist when this body shipped;
      v0.1.6 added them to ``_rasterize_level0`` so an overlay job can report
      granular progress and answer a cancel between tiles. They are honoured
      here rather than accepted and dropped. Dropping them would still satisfy
      the byte-identity test -- neither one touches a pixel -- but it would make
      this control silently uncancellable and silently mute, so the day someone
      drives the legacy arm to show that cancellation or progress reporting is
      *not* what changed the bytes, they would be reading a difference this file
      invented. The call sites match the real function's: cancel before writing
      each tile, report after writing it, counting tiles written.
    """
    if not payloads:
        return
    if use_pool and len(payloads) > 1:
        with mutations.ProcessPoolExecutor(
            max_workers=mutations.raster_process_pool_size(),
            initializer=mutations.django_pool_initializer,
        ) as executor:
            completed = 0
            for result in executor.map(mutations.render_module.rasterize_tile_worker, payloads):
                if cancel_check is not None:
                    cancel_check()
                mutations._write_tile_result(arrays, result)
                completed += 1
                if on_progress is not None:
                    on_progress("raster", completed, len(payloads))
    else:
        for index, payload in enumerate(payloads, start=1):
            if cancel_check is not None:
                cancel_check()
            mutations._write_tile_result(
                arrays, mutations.render_module.rasterize_tile_worker(payload)
            )
            if on_progress is not None:
                on_progress("raster", index, len(payloads))


def _run_rasterizer(function, *, tiles: int, workers: int):
    """Drive one rasteriser over ``tiles`` stand-in tiles; return the recorder.

    ``create=True`` on the worker-count patch is on purpose: it lets this
    harness be pointed at a build of ``mutations`` that has no such accessor
    (the pre-S2 file sized its pool from a module constant), so that version
    fails on the numbers below rather than on a missing attribute.
    """
    _RecordingExecutor.instances.clear()
    sink = _SinkArray()
    arrays = {"labels": [sink], "border": [sink]}
    with (
        patch.object(mutations, "ProcessPoolExecutor", _RecordingExecutor),
        patch.object(mutations, "raster_process_pool_size", lambda: workers, create=True),
        patch.object(mutations.render_module, "rasterize_tile_worker", _stand_in_worker),
    ):
        function(arrays, _stand_in_payloads(tiles), use_pool=True)
    assert len(_RecordingExecutor.instances) == 1
    recorder = _RecordingExecutor.instances[0]
    assert recorder.submitted == tiles
    assert sink.writes == tiles * 2
    assert recorder.max_workers == workers, (
        f"the pool was built with {recorder.max_workers} workers when the "
        f"machine profile asked for {workers}: the size is not coming from the "
        "profile"
    )
    return recorder


# ---------------------------------------------------------------------------
# 1. The window bound, and that it is flat in canvas size
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tiles", [100, 400])
@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_the_parent_never_holds_more_than_two_tiles_per_worker(tiles, workers):
    recorder = _run_rasterizer(mutations._rasterize_level0, tiles=tiles, workers=workers)

    expected = max(2, workers * RASTER_POOL_WINDOW_MULTIPLIER)
    assert recorder.peak_outstanding == expected, (
        f"{tiles} tiles at {workers} workers left {recorder.peak_outstanding} "
        f"finished tiles in the parent at once, not {expected}. At the real "
        f"tile size of {REAL_TILE_BYTES / 1e6:.0f} MB that is "
        f"{recorder.peak_outstanding * REAL_TILE_BYTES / 1e6:.0f} MB instead of "
        f"{expected * REAL_TILE_BYTES / 1e6:.0f} MB."
    )
    assert recorder.peak_held_bytes == expected * STAND_IN_TILE_BYTES


def test_the_backlog_is_flat_in_canvas_area_not_linear():
    """400 macro tiles must cost the parent exactly what 100 cost it.

    This is the property the design asks for ("parent peak at 400 macro tiles
    equals parent peak at 100 within 10 %"), asserted on the quantity that
    causes the peak rather than on the peak itself -- exactly equal, because
    there is nothing here for a machine to be noisy about. The megabytes
    version, measured end to end on 419 MP and 1 678 MP canvases, is in
    ``.scratch/tmp/s2_overlay_report.md``.
    """
    workers = 4
    small = _run_rasterizer(mutations._rasterize_level0, tiles=100, workers=workers)
    large = _run_rasterizer(mutations._rasterize_level0, tiles=400, workers=workers)

    assert large.peak_held_bytes == small.peak_held_bytes, (
        "the parent's tile backlog grew with the canvas: "
        f"{small.peak_held_bytes} bytes at 100 macro tiles, "
        f"{large.peak_held_bytes} at 400"
    )
    assert large.submitted == 4 * small.submitted


def test_the_recorder_catches_the_unbounded_submission_it_is_guarding_against():
    """The negative control: the pre-S2 body, measured the same way.

    Without this, a recorder that silently stopped counting would let the bound
    tests pass forever.
    """
    small = _run_rasterizer(_legacy_rasterize_level0, tiles=100, workers=4)
    large = _run_rasterizer(_legacy_rasterize_level0, tiles=400, workers=4)

    assert small.peak_outstanding == 100
    assert large.peak_outstanding == 400
    assert large.peak_held_bytes == 4 * small.peak_held_bytes


class _HookProbe:
    """The two v0.1.6 hooks, instrumented, owned by the caller and not the run.

    Everything the run touches lives on the probe rather than being returned,
    because the interesting case here is the run that *raises*: a cancelled
    rasterisation has to be inspected for what it managed to write before it
    stopped, and tallies returned by value die with the frame.
    """

    def __init__(self, *, cancel_after: int | None = None) -> None:
        self.sink = _SinkArray()
        self.progress: list[tuple[str, int, int]] = []
        self.checks = 0
        self._cancel_after = cancel_after

    def on_progress(self, stage, done, total) -> None:
        self.progress.append((stage, done, total))

    def cancel_check(self) -> None:
        self.checks += 1
        if self._cancel_after is not None and self.checks > self._cancel_after:
            raise JobCancelledError("Job cancellation requested.")


def _drive_with_hooks(function, probe: _HookProbe, *, tiles: int, workers: int) -> None:
    """Run one rasteriser over stand-in tiles with the v0.1.6 hooks attached.

    Separate from :func:`_run_rasterizer` because that helper asserts every tile
    was written, which is exactly what a cancelled run must *not* do.
    """
    _RecordingExecutor.instances.clear()
    arrays = {"labels": [probe.sink], "border": [probe.sink]}
    with (
        patch.object(mutations, "ProcessPoolExecutor", _RecordingExecutor),
        patch.object(mutations, "raster_process_pool_size", lambda: workers, create=True),
        patch.object(mutations.render_module, "rasterize_tile_worker", _stand_in_worker),
    ):
        function(
            arrays,
            _stand_in_payloads(tiles),
            use_pool=True,
            on_progress=probe.on_progress,
            cancel_check=probe.cancel_check,
        )


@pytest.mark.parametrize(
    "function",
    [mutations._rasterize_level0, _legacy_rasterize_level0],
    ids=["windowed", "legacy"],
)
def test_both_arms_report_and_check_for_cancellation_once_per_written_tile(function):
    """The control's *other* fidelity: it differs in submission discipline only.

    v0.1.6 gave ``_rasterize_level0`` an ``on_progress`` reporter and a
    ``cancel_check``, and the negative control above had to grow the same two
    parameters to stay callable. Accepting them and quietly dropping them would
    have passed the byte-identity test just as well -- neither hook touches a
    pixel -- and would have left a control that is silently uncancellable and
    silently mute, so anyone later using this file to show that granular
    progress or responsive cancellation is not what moved the bytes would be
    reading a difference the test file invented. Pinned here so the two bodies
    cannot drift apart again.
    """
    probe = _HookProbe()

    _drive_with_hooks(function, probe, tiles=20, workers=4)

    assert probe.progress == [("raster", index, 20) for index in range(1, 21)]
    assert probe.checks == 20, (
        f"cancellation was asked about {probe.checks} times over 20 tiles: the "
        "user's Cancel is answered once a tile, not once a run"
    )
    assert probe.sink.writes == 40  # a labels crop and a border crop per tile


@pytest.mark.parametrize(
    "function",
    [mutations._rasterize_level0, _legacy_rasterize_level0],
    ids=["windowed", "legacy"],
)
def test_both_arms_stop_on_the_cancel_instead_of_finishing_the_canvas(function):
    """Cancel before the third tile: two tiles written, not twenty.

    The windowed loop has already *submitted* a window's worth of tiles by then,
    which is the point -- responsiveness is decided by where the check sits in
    the consume loop, not by how much work is in flight.
    """
    probe = _HookProbe(cancel_after=2)

    with pytest.raises(JobCancelledError):
        _drive_with_hooks(function, probe, tiles=20, workers=4)

    assert probe.checks == 3
    assert probe.progress == [("raster", 1, 20), ("raster", 2, 20)]
    assert probe.sink.writes == 4


# ---------------------------------------------------------------------------
# 2. Pool size comes from the machine profile
# ---------------------------------------------------------------------------
def test_pool_size_is_the_machine_profiles_raster_worker_count():
    machine = pytest.importorskip(
        "quantem.core.machine",
        reason="stage S0 owns core/machine.py; without it the pool falls back",
    )

    assert raster_process_pool_size() == machine.get_machine_profile().raster_workers


def test_the_small_profile_asks_for_two_workers_and_a_window_of_four():
    """The laptop case, stated in numbers rather than left to the box we build on.

    A 4-core / 8 GB Windows laptop and an 8 GB Air are ``small``: two raster
    workers, so four outstanding tiles, so about 84 MB of backlog against the
    2 051 MB that shipped.
    """
    machine = pytest.importorskip("quantem.core.machine")

    small = machine.profile_for(total_ram_bytes=8 * 1024**3, cpu_count=4)

    assert small.name == "small"
    assert small.raster_workers == 2
    window = small.raster_workers * RASTER_POOL_WINDOW_MULTIPLIER
    assert window * REAL_TILE_BYTES < 100 * 1024**2


def test_the_fallback_worker_count_is_still_the_documented_one():
    """If the machine probe cannot be imported the build still runs, at four."""
    with patch.dict(sys.modules, {"quantem.core.machine": None}):
        assert raster_process_pool_size() == RASTER_PROCESS_POOL_MAX


def test_a_dead_pool_names_the_worker_count_it_actually_had():
    """The failure sentence must not go on saying 4 on a two-worker laptop."""
    from concurrent.futures.process import BrokenProcessPool

    with pytest.raises(mutations.OverlayRenderPoolError) as caught:
        mutations._raise_broken_pool(
            BrokenProcessPool("a process died"),
            stage="rasterisation",
            task_count=17,
            worker_count=2,
        )

    message = str(caught.value)
    assert "the 2 background rendering workers" in message
    assert "17 tiles" in message
    assert "a process died" in message


# ---------------------------------------------------------------------------
# 3. The output is byte-identical, through a real pool and a real store
# ---------------------------------------------------------------------------
def _sha256_of_tree(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = 0
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
        digest.update(path.read_bytes())
        files += 1
    return digest.hexdigest(), files


class OverlayBundleIsByteIdenticalTests(TestCase):
    """A windowed rebuild and an unbounded one write the same bundle.

    This is the acceptance that decides whether the change may ship: the overlay
    is a published, versioned, immutable artifact, and a memory fix that moved a
    single pixel would be a different feature. Both arms run the real
    ``ProcessPoolExecutor`` over the real zarr store, so it is a comparison of
    products and not of intentions.
    """

    EXTENT = 6144  # 9 macro tiles: enough for a pool to matter, small enough to run
    OBJECT_COUNT = RASTER_POOL_MIN_OBJECTS + 100

    def setUp(self):
        self.asset = Asset.objects.create(
            display_name=f"S2 byte-identity {uuid4().hex[:8]}",
            original_filename="s2_identity.tif",
            logical_width=self.EXTENT,
            logical_height=self.EXTENT,
            channels=1,
            bit_depth=8,
            pixel_size_nm=5.0,
            preprocess_stage="DONE",
            preprocess_progress=100.0,
        )
        Rendition.objects.create(
            asset=self.asset,
            type=Rendition.TYPE_FULL,
            storage_root="DATA_DIR",
            stored_path=f"images/s2_identity_{self.asset.id}.png",
            path_exists=False,
            is_directory=False,
            stored_width=self.EXTENT,
            stored_height=self.EXTENT,
            stored_channels=1,
            stored_bit_depth=8,
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self._seed_objects()

    def _seed_objects(self) -> None:
        side = 24
        columns = int(self.OBJECT_COUNT**0.5) + 1
        step = max(side + 2, self.EXTENT // columns)
        rows = []
        for index in range(self.OBJECT_COUNT):
            x0 = min(self.EXTENT - side - 1, (index % columns) * step)
            y0 = min(self.EXTENT - side - 1, (index // columns) * step)
            polygon = box(x0, y0, x0 + side, y0 + side)
            rows.append(
                SegmentObject(
                    segmentation=self.segmentation,
                    geometry=polygon,
                    centroid=polygon.centroid,
                    bbox=polygon.envelope,
                    label_state="INFERRED",
                    confidence_score=0.8,
                    features={},
                )
            )
        SegmentObject.objects.bulk_create(rows, batch_size=500)

    def test_the_windowed_rebuild_writes_the_same_bytes_as_the_unbounded_one(self):
        state = mutations.rebuild_overlay_full(self.segmentation)
        windowed_sha, windowed_files = _sha256_of_tree(get_overlay_active_bundle_path(state))
        self.assertGreater(windowed_files, 100, "the store came out suspiciously empty")

        with patch.object(mutations, "_rasterize_level0", _legacy_rasterize_level0):
            state = mutations.rebuild_overlay_full(self.segmentation)
        legacy_sha, legacy_files = _sha256_of_tree(get_overlay_active_bundle_path(state))

        self.assertEqual(legacy_files, windowed_files)
        self.assertEqual(
            legacy_sha,
            windowed_sha,
            "the submission window changed the published overlay bytes",
        )


# ---------------------------------------------------------------------------
# 4. The constrained-memory acceptance
# ---------------------------------------------------------------------------
CAPPED_CHILD = '''
"""Rasterise TILES macro tiles with the parent's committed memory capped.

Run as: child.py <arm> <data_dir> <tiles> <headroom_mb>
Prints one JSON line: {"ok": bool, "arm": ..., "error": ...}
"""
import ctypes
import ctypes.wintypes as wt
import json
import os
import sys

ARM, DATA_DIR, TILES, HEADROOM_MB = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

os.environ["QUANTEM_DATA_DIR"] = DATA_DIR
os.environ["QUANTEM_AUTOSTART_JOBS"] = "0"
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "quantem.core.settings")

import django

django.setup()

import numpy as np
import zarr
from shapely.geometry import box

from quantem.segmentation.overlay_ngff import mutations
from quantem.segmentation.overlay_ngff import render as render_module

TILE = mutations.MACRO_TILE_SIZE
SIDE = int(TILES ** 0.5)
EXTENT = SIDE * TILE


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


class IOC(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class BASIC(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
        ("Affinity", ctypes.POINTER(ctypes.c_ulong)), ("PriorityClass", wt.DWORD),
        ("SchedulingClass", wt.DWORD),
    ]


class EXT(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", BASIC), ("IoInfo", IOC),
        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


psapi = ctypes.WinDLL("psapi", use_last_error=True)
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
hproc = k32.GetCurrentProcess()


def committed():
    counters = PMC()
    counters.cb = ctypes.sizeof(counters)
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
    if not psapi.GetProcessMemoryInfo(hproc, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PrivateUsage


def cap(limit_bytes):
    k32.CreateJobObjectW.restype = wt.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
    handle = k32.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    info = EXT()
    # PROCESS_MEMORY caps this process only (the parent is what is under test);
    # KILL_ON_JOB_CLOSE makes sure a parent that dies takes its pool with it.
    info.BasicLimitInformation.LimitFlags = 0x00000100 | 0x00002000
    info.ProcessMemoryLimit = limit_bytes
    k32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
    if not k32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    k32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
    if not k32.AssignProcessToJobObject(handle, hproc):
        raise ctypes.WinError(ctypes.get_last_error())


def legacy_rasterize_level0(arrays, payloads, *, use_pool, on_progress=None, cancel_check=None):
    """The pre-S2 body again, in the capped child. Kept in step with the
    in-process control above on purpose: the two arms of this file must not
    drift apart, or "unbounded" would mean two different things depending on
    which test you read. ``on_progress``/``cancel_check`` are honoured for the
    same reason they are there -- the only difference from the shipped
    rasteriser must be the submission discipline.
    """
    from concurrent.futures import ProcessPoolExecutor

    from quantem.jobs.pool import django_pool_initializer

    with ProcessPoolExecutor(
        max_workers=mutations.raster_process_pool_size(),
        initializer=django_pool_initializer,
    ) as executor:
        completed = 0
        for result in executor.map(render_module.rasterize_tile_worker, payloads):
            if cancel_check is not None:
                cancel_check()
            mutations._write_tile_result(arrays, result)
            completed += 1
            if on_progress is not None:
                on_progress("raster", completed, len(payloads))


def main():
    store = os.path.join(DATA_DIR, "store.zarr")
    group = zarr.open_group(store, mode="w", zarr_format=2)
    labels = group.create_array(
        "labels", shape=(EXTENT, EXTENT), chunks=(256, 256), dtype="uint32"
    )
    border = group.create_array(
        "border", shape=(EXTENT, EXTENT), chunks=(256, 256), dtype="uint8"
    )
    arrays = {"labels": [labels], "border": [border]}

    draw_ops = []
    label = 0
    for row in range(SIDE * 8):
        for column in range(SIDE * 8):
            label += 1
            x0, y0 = column * (TILE // 8) + 40, row * (TILE // 8) + 40
            polygon = box(x0, y0, x0 + 90, y0 + 90)
            draw_ops.append({
                "label": label,
                "priority": 1,
                "area": 8100.0,
                "rings": render_module.geometry_to_rings(polygon),
                "bbox": (x0, y0, x0 + 90, y0 + 90),
            })
    payloads = mutations._macro_tile_payloads(draw_ops, width=EXTENT, height=EXTENT)
    assert len(payloads) == TILES, (len(payloads), TILES)

    cap(committed() + HEADROOM_MB * 1024 * 1024)

    function = (
        mutations._rasterize_level0 if ARM == "windowed" else legacy_rasterize_level0
    )
    try:
        function(arrays, payloads, use_pool=True)
    except BaseException as exc:
        print(json.dumps({"ok": False, "arm": ARM, "error": f"{type(exc).__name__}: {exc}"[:400]}))
        return
    print(json.dumps({"ok": True, "arm": ARM, "error": "", "written": int(np.asarray(labels[0:256, 0:256]).max())}))


if __name__ == "__main__":
    # Not optional. A spawned pool worker re-imports __main__, so an unguarded
    # main() runs again inside every worker, the second create_array raises
    # ContainsArrayError, and the pool comes back "terminated abruptly" -- a
    # perfect impostor of the memory failure this test is looking for.
    main()
'''


def _run_capped_child(script: Path, work: Path, *, arm: str, headroom_mb: int) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    # Pin the target machine, not this one. On the build box the profile says 8
    # raster workers, which would make the window 16 tiles and quietly change
    # what the cap is testing; ``small`` is the 8 GB laptop the design is for.
    environment["QUANTEM_MACHINE_PROFILE"] = "small"
    completed = subprocess.run(
        [sys.executable, str(script), arm, str(work), "25", str(headroom_mb)],
        capture_output=True,
        text=True,
        timeout=900,
        env=environment,
        check=False,
    )
    reported = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if reported:
        return json.loads(reported[-1])
    return {
        "ok": False,
        "arm": arm,
        "error": f"child exited {completed.returncode}: {completed.stderr.strip()[-400:]}",
    }


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are Windows-only")
def test_the_parent_survives_a_cap_the_unbounded_backlog_does_not(tmp_path):
    """25 macro tiles on the ``small`` profile, under a committed-memory cap.

    The arithmetic the cap is chosen from: a macro tile costs the parent 21 MB;
    ``small`` gives two raster workers, so the window holds four tiles, about
    84 MB, while the pre-S2 backlog can reach all 25, about 525 MB. A headroom
    of 400 MB above the child's own measured steady state sits between those
    with margin both ways, and measuring the baseline rather than assuming it
    keeps this from becoming a test of how heavy Django is this week.

    Three arms, because "the unbounded one failed" is only evidence if it would
    have passed with room: windowed at 400 MB must succeed, unbounded at 400 MB
    must fail, and unbounded at 4 000 MB must succeed. Without the third the
    test could be passing on a plain bug in the legacy arm.

    The failure the middle arm produces is ``BrokenProcessPool``, not
    ``MemoryError``: the allocation that loses is the un-pickling of an incoming
    result inside the executor's own manager thread, and a manager thread that
    dies is reported to the caller as a broken pool. That is what running out of
    room for the backlog looks like from the parent, and it is worth knowing
    that this is the shape the user would see.

    See the module docstring for what a Job Object does and does not model.
    """
    script = tmp_path / "capped_child.py"
    script.write_text(textwrap.dedent(CAPPED_CHILD), encoding="utf-8")

    windowed = _run_capped_child(script, tmp_path / "windowed", arm="windowed", headroom_mb=400)
    unbounded = _run_capped_child(script, tmp_path / "unbounded", arm="unbounded", headroom_mb=400)
    unbounded_uncapped = _run_capped_child(
        script, tmp_path / "unbounded_roomy", arm="unbounded", headroom_mb=4000
    )

    assert windowed["ok"], (
        "a windowed rasterisation did not fit in 400 MB above its own baseline: "
        f"{windowed['error']}"
    )
    assert unbounded_uncapped["ok"], (
        "the unbounded arm failed even with 4 000 MB of headroom, so the 400 MB "
        f"result below is not about memory: {unbounded_uncapped['error']}"
    )
    assert not unbounded["ok"], (
        "the pre-S2 rasteriser fitted inside the cap, so this test is no longer "
        "measuring the backlog it was written for"
    )
