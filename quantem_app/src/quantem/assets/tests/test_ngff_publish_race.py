"""Publishing must be invisible to a reader. Measured across real processes.

Round 3 built the new pyramid in a sibling directory and swapped it in with two
renames, which shrank the window from minutes to milliseconds but did not close
it: **22 all-zero windows in 3 921 reads**, and 6 in 3 636 from a separate
process, every one landing in the swap. They were silent -- zarr substitutes
``fill_value`` for a chunk whose file has just been renamed away and raises
nothing, so the source-file fallback never ran.

Nothing here is renamed. A generation directory is written under its final name
and becomes live by one database ``UPDATE``, so there is no instant at which
the path a reader is inside changes. The strict store is the backstop for
everything else -- a sweep, a disk fault, a half-copied directory -- and the
happy path must never fire it.

These run out of process (``test_ngff_race_driver``) against a real on-disk
SQLite database, because the suite's own database is in memory and two of the
five findings were only visible across a process boundary.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import tifffile
from django.test import SimpleTestCase, TestCase

from quantem.assets.canonical_decode import decode_canonical_plane
from quantem.assets.ngff import regenerate_ngff_from_plane
from quantem.assets.pyramid_authority import (
    NGFF_DRAIN_SECONDS,
    Intent,
    PublishedPyramid,
    PyramidChunkMissing,
    asset_generation_dir,
    resolve_pyramid,
    sweep_asset,
)
from quantem.assets.task_utils import _open_generation_level_cache_clear, load_image_roi_array
from quantem.core.config import DATA_DIR

from . import test_ngff_reference as ref
from .test_ngff_race_driver import CHUNK_STRADDLING_WINDOW
from .test_ngff_source_matrix import stage_asset

SRC = Path(__file__).resolve().parents[3]
DRIVER = "quantem.assets.tests.test_ngff_race_driver"

#: The design's numbers: >= 40 publishes, >= 3 reader processes, >= 3 000 reads.
PUBLISH_ROUNDS = int(os.environ.get("QUANTEM_NGFF_RACE_ROUNDS", "40"))
READER_PROCESSES = int(os.environ.get("QUANTEM_NGFF_RACE_READERS", "3"))
MINIMUM_READS = int(os.environ.get("QUANTEM_NGFF_RACE_MIN_READS", "3000"))


def run_driver(mode: str, *args, timeout: int = 1800) -> dict:
    """Run one driver mode in its own process, with its own data directory."""

    workdir = Path(DATA_DIR) / f"race-{uuid.uuid4().hex[:8]}"
    data_dir = workdir / "data"
    tmp_dir = workdir / "tmp"
    for directory in (data_dir, tmp_dir):
        directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["QUANTEM_DATA_DIR"] = str(data_dir)
    env["PYTHONPATH"] = str(SRC)
    for name in ("TEMP", "TMP", "TMPDIR"):
        env[name] = str(tmp_dir)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            DRIVER,
            mode,
            *[str(value) for value in args],
            str(workdir / "work"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(SRC),
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"driver {mode} exited {completed.returncode}\n"
            f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}"
        )
    line = [row for row in completed.stdout.strip().splitlines() if row.startswith("{")][-1]
    result = json.loads(line)
    result["_workdir"] = str(workdir)
    return result


class PublishRaceTests(SimpleTestCase):
    """Forty publishes under three reading processes."""

    def test_no_reader_ever_sees_a_blank_or_partial_window_during_a_publish(self):
        result = run_driver("publish_race", PUBLISH_ROUNDS, READER_PROCESSES)
        try:
            verdicts = result["verdicts"]
            self.assertGreaterEqual(
                result["reads"],
                MINIMUM_READS,
                f"only {result['reads']} reads landed during {result['publishes']} "
                "publishes; the race did not get a chance to happen",
            )
            self.assertEqual(result["publishes"], PUBLISH_ROUNDS)
            self.assertEqual(
                verdicts.get("BLANK", 0),
                0,
                f"a reader was handed an all-zero window: {result['samples']}",
            )
            self.assertEqual(
                verdicts.get("PARTIAL", 0),
                0,
                f"a reader was handed a partially-zeroed window: {result['samples']}",
            )
            self.assertEqual(
                verdicts.get("WRONG", 0),
                0,
                f"a reader was handed the wrong pixels: {result['samples']}",
            )
            # The strict store is the backstop, and a publish is not an
            # emergency: if this fires on the happy path the mechanism has
            # turned a silent wrong answer into a loud one, which is better but
            # is not the claim being made here.
            self.assertEqual(
                verdicts.get("CHUNK_MISSING", 0),
                0,
                "the strict store fired during an ordinary publish",
            )
            self.assertEqual(
                [key for key in verdicts if key.startswith("ERROR:")],
                [],
                f"unexpected reader errors: {verdicts}",
            )
            self.assertEqual(verdicts.get("OK", 0), result["reads"])
        finally:
            shutil.rmtree(result["_workdir"], ignore_errors=True)


class StrictStoreTests(TestCase):
    """Missing data must be loud. It used to be black."""

    def _asset_with_pyramid(self):
        source = Path(DATA_DIR) / f"strict-{uuid.uuid4().hex[:8]}.tif"
        source.parent.mkdir(parents=True, exist_ok=True)
        yy, xx = np.mgrid[0:1152, 0:1152]
        tifffile.imwrite(str(source), ((yy % 251) + (xx % 199)).astype(np.uint16) * 100)
        asset = stage_asset(source)
        from quantem.assets.asset_openable import get_asset_openable

        openable = get_asset_openable(asset)
        plane = decode_canonical_plane(source)
        regenerate_ngff_from_plane(openable, plane)
        return asset, openable, plane.array

    def test_a_chunk_that_vanishes_mid_read_raises_instead_of_reading_as_black(self):
        asset, openable, expected = self._asset_with_pyramid()
        x, y, width, height = CHUNK_STRADDLING_WINDOW
        np.testing.assert_array_equal(
            load_image_roi_array(openable, x, y, width, height),
            expected[y : y + height, x : x + width],
        )
        resolved = resolve_pyramid(asset, intent=Intent.SERVE)
        self.assertIsInstance(resolved, PublishedPyramid)
        # Exactly the state zarr answers with fill_value and no exception.
        (resolved.root / "0" / "0.0.0").unlink()
        with self.assertRaises(PyramidChunkMissing):
            load_image_roi_array(openable, 0, 0, 64, 64)

    def test_every_level_is_written_dense_so_a_blank_tile_is_still_a_file(self):
        """A genuinely blank EM tile must have a chunk file like any other.

        zarr elides an all-fill chunk by default. That makes the strict store
        raise on correct data, and it makes "count the chunk files" -- the
        publish-time completeness proof -- unsound. Both are only safe because
        the writes are dense, so this is the assertion that holds them up.
        """

        source = Path(DATA_DIR) / f"blank-{uuid.uuid4().hex[:8]}.tif"
        plane = np.full((1152, 1152), 40, dtype=np.uint8)
        plane[:1024, :1024] = 0  # one chunk that is genuinely all fill_value
        tifffile.imwrite(str(source), plane)
        asset = stage_asset(source)
        from quantem.assets.asset_openable import get_asset_openable

        openable = get_asset_openable(asset)
        regenerate_ngff_from_plane(openable, decode_canonical_plane(source))
        resolved = resolve_pyramid(asset, intent=Intent.SERVE)
        self.assertTrue((resolved.root / "0" / "0.0.0").exists(), "the all-fill chunk was elided")
        np.testing.assert_array_equal(
            load_image_roi_array(openable, 0, 0, 64, 64), ref.decode(source)[0:64, 0:64]
        )
        for level in resolved.manifest["levels"]:
            present = [
                child.name
                for child in (resolved.root / level["path"]).iterdir()
                if not child.name.startswith(".")
            ]
            self.assertEqual(len(present), level["chunk_count"])


class SweepUnderOpenHandleTests(TestCase):
    """A superseded generation that a reader still holds is not "removed"."""

    def test_a_held_generation_is_reported_still_held_and_collected_afterwards(self):
        source = Path(DATA_DIR) / f"held-{uuid.uuid4().hex[:8]}.tif"
        tifffile.imwrite(str(source), (np.mgrid[0:1152, 0:1152][1] % 200).astype(np.uint8))
        asset = stage_asset(source)
        from quantem.assets.asset_openable import get_asset_openable

        openable = get_asset_openable(asset)
        plane = decode_canonical_plane(source)
        first = regenerate_ngff_from_plane(openable, plane)
        # A reader inside the old generation, exactly as a streaming tile is.
        handle = open(first / "0" / "0.0.0", "rb")
        try:
            second = regenerate_ngff_from_plane(openable, plane)
            self.assertNotEqual(first, second)
            # Pretend the drain window has passed.
            owner = json.loads((first / "owner.json").read_text(encoding="utf-8"))
            owner["sealed_at"] = owner["sealed_at"] - NGFF_DRAIN_SECONDS - 10
            (first / "owner.json").write_text(json.dumps(owner), encoding="utf-8")
            result = sweep_asset(asset.id, published_generation=second.name)
            self.assertEqual(
                result.still_held,
                1,
                "a generation with a chunk open inside it was reported as removed; "
                "rmtree(ignore_errors=True) silently leaves the root behind, which "
                "is how a 44 MB build root survived a restart",
            )
            self.assertEqual(result.removed, 0)
            self.assertTrue(first.exists())
        finally:
            handle.close()
        _open_generation_level_cache_clear()
        after = sweep_asset(asset.id, published_generation=second.name)
        self.assertEqual(after.removed, 1)
        self.assertFalse(first.exists())
        self.assertEqual(
            sorted(child.name for child in asset_generation_dir(asset.id).iterdir()),
            [second.name],
        )
