"""Out-of-process driver for the NGFF race harnesses.

Two of the five findings were only visible across a process boundary, and the
publish window is measured in milliseconds, so the races have to be real
processes over a real on-disk SQLite database -- not threads over the suite's
in-memory test database.

This module therefore holds no tests of its own. It is run as
``python -m quantem.assets.tests.test_ngff_race_driver <mode> <args...>``, with
``QUANTEM_DATA_DIR`` pointing at a directory the calling test owns, and it
prints one JSON object on its last line. Every application import is inside a
function so the module can be imported before ``django.setup()``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

CHUNK_STRADDLING_WINDOW = (960, 960, 160, 160)


def _setup_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "quantem.core.settings")
    import django

    django.setup()
    from django.core.management import call_command

    call_command("migrate", run_syncdb=True, verbosity=0)


def _source_plane(side: int = 1152):
    import numpy as np

    yy, xx = np.mgrid[0:side, 0:side].astype(np.float64)
    values = 0.18 + 0.55 * (xx / (side - 1)) + 0.18 * np.sin(yy / 19.0) * np.cos(xx / 29.0)
    return (np.clip(values, 0.02, 0.94) * 65535).astype(np.uint16)


def _write_source(path: Path, side: int) -> None:
    import tifffile

    tifffile.imwrite(str(path), _source_plane(side))


def _make_asset(source: Path):
    """An Asset + FULL rendition pointing at ``source``, as an import leaves it."""

    import shutil

    from quantem.assets.asset_openable import get_asset_openable
    from quantem.assets.models import Asset, Rendition
    from quantem.assets.utils import extract_image_metadata
    from quantem.core.config import DATA_DIR, IMAGES_DIR
    from quantem.core.local_storage import normalize_stored_path_value

    metadata = extract_image_metadata(source)
    asset_id = uuid.uuid4()
    target = IMAGES_DIR / str(asset_id) / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    asset = Asset.objects.create(
        id=asset_id,
        display_name="race",
        original_filename=source.name,
        logical_width=int(metadata["width"]),
        logical_height=int(metadata["height"]),
        channels=int(metadata["channels"]),
        bit_depth=int(metadata["bit_depth"]),
        preprocess_stage="DONE",
    )
    Rendition.objects.create(
        asset=asset,
        type=Rendition.TYPE_FULL,
        storage_root="DATA_DIR",
        stored_path=normalize_stored_path_value(target, relative_to=DATA_DIR),
        path_exists=True,
        is_directory=False,
        stored_width=int(metadata["width"]),
        stored_height=int(metadata["height"]),
        stored_channels=int(metadata["channels"]),
        stored_bit_depth=int(metadata["bit_depth"]),
        metadata={},
    )
    return asset, get_asset_openable(asset)


def _spawn(mode: str, *args) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            __spec__.name if __spec__ else __name__,
            mode,
            *[str(a) for a in args],
        ],
        env=dict(os.environ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# publish_race
# ---------------------------------------------------------------------------


def mode_publish_race(rounds: str, readers: str, minimum_reads: str, workdir: str) -> dict:
    import numpy as np

    from quantem.assets.canonical_decode import decode_canonical_plane
    from quantem.assets.ngff import regenerate_ngff_from_plane

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    source = work / "source.tif"
    _write_source(source, 1152)
    asset, openable = _make_asset(source)
    plane = decode_canonical_plane(source)
    np.save(work / "expected.npy", plane.array)
    regenerate_ngff_from_plane(openable, plane)

    stop = work / "stop"
    children = [
        _spawn("reader", str(asset.id), str(work), str(index)) for index in range(int(readers))
    ]
    # Let every reader get past django.setup() before the publishing starts,
    # so the window under test is actually covered.
    deadline = time.time() + 120
    while time.time() < deadline and len(list(work.glob("reader-*.ready"))) < len(children):
        time.sleep(0.1)

    published = 0
    for _ in range(int(rounds)):
        regenerate_ngff_from_plane(openable, plane)
        published += 1
    # A fixed sleep made the coverage assertion depend on runner speed. Wait
    # for the readers themselves to report the requested sample count instead.
    read_deadline = time.time() + 120
    observed_reads = 0
    while time.time() < read_deadline:
        observed_reads = 0
        for progress in work.glob("reader-*.progress"):
            try:
                observed_reads += int(progress.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        if observed_reads >= int(minimum_reads):
            break
        if any(child.poll() is not None for child in children):
            break
        time.sleep(0.05)
    stop.write_text("stop", encoding="utf-8")
    for child in children:
        child.wait(timeout=120)

    totals: dict[str, int] = {}
    samples: list[dict] = []
    for result in sorted(work.glob("reader-*.json")):
        payload = json.loads(result.read_text(encoding="utf-8"))
        for key, value in payload["verdicts"].items():
            totals[key] = totals.get(key, 0) + value
        samples += payload["samples"][:5]
    return {
        "mode": "publish_race",
        "publishes": published,
        "readers": len(children),
        "reads": sum(totals.values()),
        "verdicts": totals,
        "samples": samples,
        "asset_id": str(asset.id),
    }


def mode_reader(asset_id: str, workdir: str, index: str) -> dict:
    import numpy as np

    from quantem.assets.asset_openable import get_asset_openable
    from quantem.assets.models import Asset
    from quantem.assets.pyramid_authority import PyramidChunkMissing
    from quantem.assets.task_utils import load_image_roi_array

    work = Path(workdir)
    expected_full = np.load(work / "expected.npy")
    x, y, width, height = CHUNK_STRADDLING_WINDOW
    expected = expected_full[y : y + height, x : x + width]
    openable = get_asset_openable(Asset.objects.get(id=asset_id))

    verdicts: dict[str, int] = {}
    samples: list[dict] = []
    reads = 0
    progress = work / f"reader-{index}.progress"
    (work / f"reader-{index}.ready").write_text("ready", encoding="utf-8")
    stop = work / "stop"
    while not stop.exists():
        try:
            window = load_image_roi_array(openable, x, y, width, height)
        except PyramidChunkMissing:
            verdicts["CHUNK_MISSING"] = verdicts.get("CHUNK_MISSING", 0) + 1
            reads += 1
            if reads % 64 == 0:
                progress.write_text(str(reads), encoding="utf-8")
            continue
        except Exception as exc:  # noqa: BLE001
            key = f"ERROR:{type(exc).__name__}"
            verdicts[key] = verdicts.get(key, 0) + 1
            if len(samples) < 5:
                samples.append({"verdict": key, "detail": str(exc)[:200]})
            reads += 1
            if reads % 64 == 0:
                progress.write_text(str(reads), encoding="utf-8")
            continue
        if np.array_equal(window, expected):
            verdict = "OK"
        else:
            zero_fraction = float((window == 0).mean())
            verdict = (
                "BLANK" if zero_fraction == 1.0 else "PARTIAL" if zero_fraction > 0 else "WRONG"
            )
            if len(samples) < 5:
                samples.append(
                    {
                        "verdict": verdict,
                        "mean": round(float(window.mean()), 3),
                        "zero_fraction": round(zero_fraction, 4),
                    }
                )
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        reads += 1
        if reads % 64 == 0:
            progress.write_text(str(reads), encoding="utf-8")
    progress.write_text(str(reads), encoding="utf-8")
    (work / f"reader-{index}.json").write_text(
        json.dumps({"verdicts": verdicts, "samples": samples}), encoding="utf-8"
    )
    return {"mode": "reader", "verdicts": verdicts}


# ---------------------------------------------------------------------------
# kill_debris
# ---------------------------------------------------------------------------


def mode_build_forever(asset_id: str, workdir: str) -> dict:
    """A builder that never stops, so the parent can kill it at any instant."""

    from quantem.assets.asset_openable import get_asset_openable
    from quantem.assets.canonical_decode import decode_canonical_plane
    from quantem.assets.models import Asset
    from quantem.assets.ngff import regenerate_ngff_from_plane

    source = Path(workdir) / "source.tif"
    openable = get_asset_openable(Asset.objects.get(id=asset_id))
    plane = decode_canonical_plane(source)
    (Path(workdir) / "builder.ready").write_text("ready", encoding="utf-8")
    while True:
        regenerate_ngff_from_plane(openable, plane)


#: Mirrors ``pyramid_authority._UNOWNED_GRACE_SECONDS``; a generation whose
#: owner tag never landed is left alone for that long.
_UNOWNED_GRACE = 5.0


def _read_owner_json(root: Path) -> dict | None:
    try:
        return json.loads((root / "owner.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _hard_kill(child: subprocess.Popen) -> None:
    """Stop a builder we started, by pid, and be sure it is gone.

    Windows needs ``taskkill /T /F`` to include grandchildren. POSIX
    ``Popen.kill`` is SIGKILL and this builder has no child processes.
    """

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(child.pid), "/T", "/F"],
            capture_output=True,
            text=True,
        )
    else:
        child.kill()
    try:
        child.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    child.kill()
    child.wait(timeout=60)


def mode_enqueue_once(asset_id: str, workdir: str) -> dict:
    """Wait on the shared start file, then ask for a lazy build. One process."""

    from quantem.assets.models import Asset
    from quantem.assets.pyramid_authority import request_lazy_build

    go = Path(workdir) / "go"
    (Path(workdir) / f"ready-{os.getpid()}").write_text("ready", encoding="utf-8")
    deadline = time.time() + 180
    while not go.exists() and time.time() < deadline:
        time.sleep(0.005)
    job = request_lazy_build(Asset.objects.get(id=asset_id))
    return {"mode": "enqueue_once", "job": str(job.id) if job else None}


def mode_concurrent_enqueue(clients: str, workdir: str) -> dict:
    """N real processes racing one lazy build, over one on-disk SQLite file.

    Threads cannot do this here -- the suite's own database is in memory -- and
    the finding was a cross-process race in the first place: three concurrent
    GETs created three NGFF jobs, they fought for the storage lease, and their
    conflict reconciled over the real error.
    """

    from quantem.assets.pyramid_authority import begin_attempt
    from quantem.jobs.constants import JOB_TYPE_ENSURE_IMAGE_NGFF
    from quantem.jobs.models import Job

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    source = work / "source.tif"
    _write_source(source, 256)
    asset, _openable = _make_asset(source)
    begin_attempt(asset)

    count = int(clients)
    children = [_spawn("enqueue_once", str(asset.id), str(work)) for _ in range(count)]
    deadline = time.time() + 300
    while time.time() < deadline and len(list(work.glob("ready-*"))) < count:
        time.sleep(0.05)
    ready = len(list(work.glob("ready-*")))
    (work / "go").write_text("go", encoding="utf-8")
    exits = [child.wait(timeout=300) for child in children]

    jobs = list(
        Job.objects.filter(type=JOB_TYPE_ENSURE_IMAGE_NGFF, payload_json__asset_id=str(asset.id))
    )
    return {
        "mode": "concurrent_enqueue",
        "clients": count,
        "ready_before_go": ready,
        "child_exit_codes": exits,
        "job_count": len(jobs),
        "job_tokens": sorted({job.payload_json.get("attempt_token") for job in jobs}),
    }


def mode_kill_debris(kills: str, workdir: str) -> dict:
    """Kill a real builder at sampled instants; restart; sweep; measure."""

    import random

    from quantem.assets.canonical_decode import decode_canonical_plane
    from quantem.assets.ngff import regenerate_ngff_from_plane
    from quantem.assets.pyramid_authority import (
        Intent,
        PublishedPyramid,
        _tree_bytes,
        asset_generation_dir,
        resolve_pyramid,
        sweep_ngff_generations,
    )

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    source = work / "source.tif"
    _write_source(source, 1152)
    asset, openable = _make_asset(source)
    plane = decode_canonical_plane(source)
    regenerate_ngff_from_plane(openable, plane)

    rng = random.Random(20260810)
    instants = []
    for _index in range(int(kills)):
        ready = work / "builder.ready"
        ready.unlink(missing_ok=True)
        child = _spawn("build_forever", str(asset.id), str(work))
        deadline = time.time() + 120
        while time.time() < deadline and not ready.exists():
            time.sleep(0.02)
        # Sampled uniformly across a build, including "before it wrote anything".
        delay = rng.uniform(0.0, 1.2)
        time.sleep(delay)
        _hard_kill(child)
        instants.append(round(delay, 3))

    root = asset_generation_dir(asset.id)
    before = {
        "children": sorted(child.name for child in root.iterdir()),
        "bytes": _tree_bytes(root),
    }

    # A restart is what the sweep contract turns on: a generation tagged with a
    # live pid from this boot is left alone, so the parent must forget it has
    # already swept before the "restart" pass is meaningful.
    import quantem.assets.pyramid_authority as authority

    authority._process_sweep_done = False
    swept = sweep_ngff_generations()

    resolved = resolve_pyramid(asset, intent=Intent.SERVE)
    published_ok = isinstance(resolved, PublishedPyramid)
    published = resolved.generation_id if published_ok else None
    after_startup_sweep = sorted(child.name for child in root.iterdir())
    # What is left after the startup pass must be *only* generations that were
    # sealed and published at some point and are still inside their drain
    # window -- readers may be in them. Anything a kill interrupted mid-build is
    # unsealed, and must already be gone.
    unsealed_left = []
    for name in after_startup_sweep:
        if name == published:
            continue
        owner = _read_owner_json(root / name)
        # No owner.json at all means the kill landed between mkdir and the
        # first write. The sweeper leaves those for a five-second grace so it
        # cannot delete a directory another process is about to claim, so they
        # count here too -- the second pass has to have taken them.
        if owner is None or not owner.get("sealed"):
            unsealed_left.append(name)

    # Now let the drain window elapse for every superseded generation and sweep
    # again: the second rule, on its own.
    time.sleep(_UNOWNED_GRACE + 0.5)
    for name in after_startup_sweep:
        if name == published:
            continue
        owner = _read_owner_json(root / name)
        if owner is None:
            continue
        owner["sealed_at"] = (owner.get("sealed_at") or time.time()) - 10_000
        (root / name / "owner.json").write_text(json.dumps(owner), encoding="utf-8")
    drained = sweep_ngff_generations()

    published_bytes = _tree_bytes(resolved.root) if published_ok else 0
    return {
        "mode": "kill_debris",
        "kill_instants": instants,
        "before": before,
        "swept": {
            "removed": swept.removed,
            "bytes_freed": swept.bytes_freed,
            "still_held": swept.still_held,
            "kept": swept.kept,
        },
        "after_startup_sweep": after_startup_sweep,
        "unsealed_left_after_startup_sweep": unsealed_left,
        "drained": {
            "removed": drained.removed,
            "bytes_freed": drained.bytes_freed,
            "still_held": drained.still_held,
        },
        "remaining": sorted(child.name for child in root.iterdir()),
        "published": published,
        "published_bytes": published_bytes,
        "total_bytes": _tree_bytes(root),
    }


_MODES = {
    "publish_race": mode_publish_race,
    "reader": mode_reader,
    "build_forever": mode_build_forever,
    "enqueue_once": mode_enqueue_once,
    "concurrent_enqueue": mode_concurrent_enqueue,
    "kill_debris": mode_kill_debris,
}


def main(argv: list[str]) -> int:
    _setup_django()
    mode = argv[0]
    result = _MODES[mode](*argv[1:])
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
