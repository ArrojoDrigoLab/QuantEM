"""A kill at any instant must leave nothing permanent.

Round 3 introduced a new debris class and no way to collect it. Killing the
server mid-rebuild stranded a build root that survived a **restart** and
survived the **rebuild of the very same image** the restart resumed twenty
seconds later, because ``_sweep_stale_build_dirs`` only ran at the start of a
build of that image and only deleted directories older than six hours.
Reproduced here before the change: the directory was still there after the
product's own sweeper *and* after a full rebuild.

Three things make it impossible now, and none of them is a clock:

* there is no ``.building``, ``.superseded`` or ``withdrawn/`` any more -- a
  generation is written under its final name, so there is only one kind of
  object to collect;
* every generation writes ``owner.json`` before its first chunk, and a
  generation whose ``boot_id`` is not this boot is debris **with no age
  threshold at all** -- which is what makes the startup pass sufficient and
  removes the dependence on a later build ever happening;
* deletion is honest. ``rmtree(ignore_errors=True)`` silently leaves the root
  in place when a handle is open (MEASURED, WinError 32) and reports nothing,
  which is how the 44 MB survived. The sweeper counts it as ``still_held`` and
  tries again.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import tifffile
from django.test import SimpleTestCase, TestCase

from quantem.assets.canonical_decode import decode_canonical_plane
from quantem.assets.ngff import regenerate_ngff_from_plane
from quantem.assets.pyramid_authority import (
    GENERATION_PREFIX,
    Intent,
    PublishedPyramid,
    Reason,
    Unavailable,
    _tree_bytes,
    asset_generation_dir,
    boot_id,
    request_build,
    resolve_pyramid,
    sweep_asset,
    sweep_ngff_generations,
)
from quantem.core.config import DATA_DIR, NGFF_TMP_DIR

from .test_ngff_publish_race import run_driver
from .test_ngff_source_matrix import stage_asset

#: The design asks for at least ten instants sampled across a build.
KILL_COUNT = int(os.environ.get("QUANTEM_NGFF_KILL_COUNT", "10"))


class KillDebrisTests(SimpleTestCase):
    """Ten real kills of a real builder, then a restart and two sweeps.

    Two sweeps because there are two rules and they must not be allowed to
    cover for each other. The first (the startup pass) has to collect every
    generation a kill interrupted, with **no age threshold** -- that is the
    rule round 3 lacked, and the one that made a 44 MB build root permanent.
    The second runs once the drain window has elapsed and collects the
    superseded-but-sealed generations the first pass deliberately kept, because
    a reader that resolved the old pointer may still be inside one.
    """

    def test_nothing_but_the_published_generation_survives_ten_kills(self):
        result = run_driver("kill_debris", KILL_COUNT)
        try:
            self.assertEqual(len(result["kill_instants"]), KILL_COUNT, result)
            self.assertIsNotNone(
                result["published"],
                "the published pyramid did not survive the kills",
            )
            # The kills have to have left something, or this is asserting about
            # an empty directory.
            self.assertGreater(
                len(result["before"]["children"]),
                1,
                f"no debris was created to collect: {result['before']}",
            )
            # Rule 1/2/3, at the startup pass: a generation a kill interrupted
            # mid-build is unsealed and owned by a dead pid, and goes with no
            # age threshold at all. This is the assertion that fails against
            # round 3's six-hour rule.
            self.assertEqual(
                result["unsealed_left_after_startup_sweep"],
                [],
                "a build that a kill interrupted survived the startup sweep: "
                f"{result['unsealed_left_after_startup_sweep']} "
                f"(before: {result['before']['children']})",
            )
            self.assertEqual(result["swept"]["still_held"], 0, result["swept"])
            # Rule 4, once the drain window has elapsed: a superseded
            # generation is kept only for as long as a reader might be inside
            # it, and then it goes too.
            self.assertEqual(
                result["remaining"],
                [result["published"]],
                "the drained sweep left something behind: "
                f"{result['remaining']} (after the startup pass: "
                f"{result['after_startup_sweep']})",
            )
            self.assertEqual(result["drained"]["still_held"], 0, result["drained"])
            self.assertLessEqual(
                result["total_bytes"],
                int(result["published_bytes"] * 1.05),
                "more than 5 % of the published size is debris: "
                f"{result['total_bytes']} vs {result['published_bytes']}",
            )
        finally:
            shutil.rmtree(result["_workdir"], ignore_errors=True)


class SweepContractTests(TestCase):
    """Each rule of the sweep contract, on its own."""

    @staticmethod
    def _age_out(root: Path) -> None:
        """Pretend this sealed generation's drain window has passed."""

        owner = json.loads((root / "owner.json").read_text(encoding="utf-8"))
        owner["sealed_at"] = (owner.get("sealed_at") or time.time()) - 10_000
        (root / "owner.json").write_text(json.dumps(owner), encoding="utf-8")

    def _asset_with_two_generations(self):
        source = Path(DATA_DIR) / f"sweep-{uuid.uuid4().hex[:8]}.tif"
        source.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(source), (np.mgrid[0:256, 0:256][0] % 200).astype(np.uint8))
        asset = stage_asset(source)
        from quantem.assets.asset_openable import get_asset_openable

        openable = get_asset_openable(asset)
        plane = decode_canonical_plane(source)
        first = regenerate_ngff_from_plane(openable, plane)
        second = regenerate_ngff_from_plane(openable, plane)
        return asset, openable, first, second

    def test_a_generation_from_a_previous_boot_goes_with_no_age_threshold(self):
        asset, _openable, first, second = self._asset_with_two_generations()
        owner = json.loads((first / "owner.json").read_text(encoding="utf-8"))
        owner["boot_id"] = "boot-from-a-previous-run"
        owner["sealed_at"] = time.time()  # young: the old rule would keep it 6 h
        (first / "owner.json").write_text(json.dumps(owner), encoding="utf-8")
        result = sweep_asset(asset.id, published_generation=second.name)
        self.assertEqual(result.removed, 1, result.summary())
        self.assertFalse(first.exists())
        self.assertTrue(second.exists())

    def test_a_generation_whose_process_is_gone_goes_immediately(self):
        asset, _openable, first, second = self._asset_with_two_generations()
        owner = json.loads((first / "owner.json").read_text(encoding="utf-8"))
        owner["pid"] = 999_999_999  # no such process
        owner["sealed"] = False
        (first / "owner.json").write_text(json.dumps(owner), encoding="utf-8")
        result = sweep_asset(asset.id, published_generation=second.name)
        self.assertEqual(result.removed, 1, result.summary())

    def test_a_live_unsealed_build_of_this_boot_is_left_alone(self):
        asset, _openable, _first, second = self._asset_with_two_generations()
        ticket = request_build(asset)
        self.assertFalse(isinstance(ticket, Unavailable))
        owner = json.loads((ticket.root / "owner.json").read_text(encoding="utf-8"))
        self.assertEqual(owner["pid"], os.getpid())
        self.assertEqual(owner["boot_id"], boot_id())
        result = sweep_asset(asset.id, published_generation=second.name)
        self.assertTrue(ticket.root.exists(), "the sweeper deleted a live build")
        self.assertGreaterEqual(result.kept, 1)

    def test_a_legacy_in_place_store_is_rebuilt_rather_than_adopted(self):
        """The compatibility branch that produced two findings is gone.

        The NGFF tree is documented as a rebuildable cache, so a store with no
        state row is not adopted -- it is collected and rebuilt. That deletes
        the "does level 0 hold at least one chunk file" heuristic, which is not
        merely weak but unsound: zarr elides an all-fill chunk, so a blank tile
        looks exactly like a chunk that was never written.
        """

        asset, _openable, _first, second = self._asset_with_two_generations()
        legacy = asset_generation_dir(asset.id) / "legacy-looking-store"
        (legacy / "0").mkdir(parents=True)
        (legacy / ".zattrs").write_text("{}", encoding="utf-8")
        (legacy / "0" / "0.0.0").write_bytes(b"chunkish")
        result = sweep_asset(asset.id, published_generation=second.name)
        self.assertFalse(legacy.exists(), result.summary())
        self.assertTrue(second.exists())

    def test_a_deleted_asset_takes_its_whole_tree_with_it(self):
        """Closes the pre-existing leak: ``tombstone_asset`` left the store."""

        from quantem.assets.asset_mutations import tombstone_asset

        asset, _openable, first, second = self._asset_with_two_generations()
        root = asset_generation_dir(asset.id)
        self.assertTrue(root.exists())
        tombstone_asset(asset)
        sweep_ngff_generations()
        self.assertFalse(root.exists(), f"{sorted(NGFF_TMP_DIR.iterdir())}")
        self.assertFalse(first.exists() or second.exists())

    def test_the_published_generation_is_never_swept(self):
        asset, _openable, _first, second = self._asset_with_two_generations()
        for _ in range(3):
            sweep_ngff_generations()
        resolved = resolve_pyramid(asset, intent=Intent.SERVE)
        self.assertIsInstance(resolved, PublishedPyramid)
        self.assertEqual(resolved.generation_id, second.name)
        self.assertTrue(second.exists())

    def test_a_publish_that_never_happened_leaves_no_pointer_and_no_bytes(self):
        """A kill between "sealed" and "published" is not a partial state."""

        asset, _openable, _first, second = self._asset_with_two_generations()
        ticket = request_build(asset)
        # Sealed, correct, and never published -- the CAS did not commit.
        from quantem.assets.ngff import build_pyramid

        build_pyramid(ticket, _openable, decode_canonical_plane(_openable.path))
        resolved = resolve_pyramid(asset, intent=Intent.SERVE)
        self.assertEqual(resolved.generation_id, second.name)
        self._age_out(ticket.root)
        self._age_out(_first)
        sweep_asset(asset.id, published_generation=second.name)
        self.assertFalse(ticket.root.exists())
        self.assertEqual(
            sorted(
                child.name
                for child in asset_generation_dir(asset.id).iterdir()
                if child.name.startswith(GENERATION_PREFIX)
            ),
            [second.name],
        )

    def test_a_missing_published_generation_reads_as_unbuilt_rather_than_ready(self):
        asset, _openable, _first, second = self._asset_with_two_generations()
        from quantem.assets.task_utils import _open_generation_level_cache_clear

        _open_generation_level_cache_clear()
        self._age_out(_first)
        sweep_asset(asset.id, published_generation=second.name)
        shutil.rmtree(second)
        resolved = resolve_pyramid(asset, intent=Intent.SERVE)
        self.assertIsInstance(resolved, Unavailable)
        self.assertEqual(resolved.reason, Reason.NEVER_BUILT)
        self.assertEqual(_tree_bytes(asset_generation_dir(asset.id)), 0)
