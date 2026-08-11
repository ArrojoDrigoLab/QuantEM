"""What an export bundle records about how its numbers were produced.

A bundle is re-derivable only if it says which build of a model ran, at which
threshold, on which pixels, under which library versions. This module collects
that record.

Four things it captures, and why each is not obvious:

* **Model bytes.** ``quantem:mito`` is a name, not an identity. The head and the
  encoder are separate downloads and either can be re-released, so the SHA-256
  of each is what goes in the record.
* **The scale the model ran at.** Most packs declare a ``canonical_nm`` and
  resample to it; an uncalibrated image runs at native scale instead, and the
  object count differs.
* **Library versions.** ``scikit-image`` sits under every measurement here, and
  ``torch``, ``numpy`` and ``scipy`` under the probability map.
* **The image.** ``image_key`` is a local database UUID and identifies nothing
  on another machine, so a file is recorded by name and SHA-256 instead.

Three rules govern this module:

1. **Nothing is guessed.** A value that cannot be obtained is written as ``null``
   with a sentence saying why, in the section's ``unavailable`` map. A manifest
   that quietly omits the adapter reads exactly like one from a run that had no
   adapter.
2. **Nothing here can fail a run.** Provenance is metadata: a missing git
   directory, an unreadable file or an absent optional package becomes a
   recorded reason, never an exception out of an analysis job.
3. **Nothing here names this machine.** A full path says which drive and which
   folders one person keeps their data in, and nothing about the image. Files
   are identified by name and SHA-256, which mean the same thing everywhere.
   :func:`scrub_local_paths` is the backstop, using the same detector as the
   model-release build gate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Distributions whose version can move a number in the bundle. ``scikit-image``
#: is here for ``binary_closing`` and ``regionprops``; ``torch`` because a
#: kernel change moves the probability map; ``numpy``/``scipy`` because they are
#: under every statistic; ``shapely`` because it rasterises the masks.
PINNED_DISTRIBUTIONS: tuple[str, ...] = (
    "torch",
    "numpy",
    "scipy",
    "scikit-image",
    "shapely",
    "opencv-python-headless",
    "pandas",
    "Django",
)

#: Files larger than this are recorded without a checksum rather than making
#: every analysis run re-hash them. Stated in the manifest so the reader knows
#: the difference between "not checksummed" and "checksum failed".
MAX_CHECKSUM_BYTES = 4 * 1024**3

_HASH_CHUNK = 4 * 1024 * 1024


def section(values: dict[str, Any], unavailable: dict[str, str]) -> dict[str, Any]:
    """One manifest section: known values, plus ``null`` and a reason for the rest.

    ``unavailable`` maps a field to the sentence explaining why it is null. The
    field is still present with a ``null`` value, so a reader diffing two
    manifests sees "we could not get this" rather than nothing at all.

    A field may be named by **dotted path** when the null it explains is one
    entry of a mapping the section already carries: ``packages.torch``,
    ``canonical_nm_by_pack.somebody:else``. What has to exist beside the reason
    is then the mapping, not a literal key with a dot in it. Creating one put a
    bogus top-level ``"canonical_nm_by_pack.somebody:else": null`` in every such
    manifest, beside the real ``{"canonical_nm_by_pack": {"somebody:else":
    null}}`` -- two keys for one fact, one of which named nothing. The manifest's
    whole contract is that every key means something, so only the head of the
    path is ensured.
    """
    out = dict(values)
    for key in unavailable:
        out.setdefault(key.split(".", 1)[0], None)
    if unavailable:
        out["unavailable"] = dict(sorted(unavailable.items()))
    return out


# ---------------------------------------------------------------------------
# The release
# ---------------------------------------------------------------------------


def release() -> dict[str, Any]:
    """QuantEM's own identity: version string and, if it is a checkout, the commit.

    ``0.1.0`` is the same string for every build made between two releases.
    On its own it cannot distinguish the code that produced a number from the
    code that produced a different one a week later.

    The checkout's *location* is deliberately not recorded. It used to be, as
    ``git_repository``, and it was the largest thing in the bundle naming the
    machine that wrote it; the commit is what identifies the code, and it means
    the same thing on every clone.
    """
    from quantem import __version__

    values: dict[str, Any] = {"quantem_version": __version__}
    unavailable: dict[str, str] = {}

    if getattr(sys, "frozen", False):
        # A frozen (PyInstaller) build ships no checkout at all. Walking up
        # from the install directory can only ever find somebody *else's*
        # repository -- an app unzipped inside any git checkout used to stamp
        # THAT repo's HEAD and dirty state into scientific manifests
        # (git_worktree_clean: false about a repo QuantEM has never read a
        # line of). Not applicable is the honest answer, not a lookup.
        reason = (
            "This is a packaged (frozen) build of QuantEM: it does not run "
            "from a git checkout, so there is no commit or working-tree state "
            "to record. Any git repository near the install directory belongs "
            "to whatever the app was installed inside, not to QuantEM. Only "
            f"the release version ({__version__}) identifies the code."
        )
        unavailable["git_commit"] = reason
        unavailable["git_worktree_clean"] = reason
        return section(values, unavailable)

    repo = _repo_root()
    if repo is None:
        reason = (
            "This copy of QuantEM was installed from a package: there is no .git "
            "directory beside it, so the exact commit cannot be recovered. Only "
            f"the release version ({__version__}) identifies the code."
        )
        unavailable["git_commit"] = reason
        unavailable["git_worktree_clean"] = reason
        return section(values, unavailable)

    commit = _git(repo, "rev-parse", "HEAD") or _read_git_head(repo)
    if commit:
        values["git_commit"] = commit.strip()
    elif (_git(repo, "rev-list", "--count", "--all") or "").strip() == "0":
        unavailable["git_commit"] = (
            "The checkout this ran from has no commits yet, so there is no "
            "commit to name. Only the release version identifies this code."
        )
    else:
        unavailable["git_commit"] = (
            "A git directory exists beside this checkout but HEAD could not be "
            "resolved, either by running git or by reading the refs directly."
        )

    status = _git(repo, "status", "--porcelain")
    if status is None:
        unavailable["git_worktree_clean"] = (
            "Whether the working tree had uncommitted changes could not be "
            "determined: the git executable is not available on this machine. "
            "The commit above is HEAD, not necessarily the code that ran."
        )
    else:
        values["git_worktree_clean"] = status.strip() == ""
        if status.strip():
            values["git_uncommitted_files"] = len(status.strip().splitlines())

    return section(values, unavailable)


def _repo_root() -> Path | None:
    """The checkout containing the ``quantem`` package, or None for an installed copy.

    Never used when running frozen -- :func:`release` returns before calling
    this -- and refuses to answer there anyway: inside a PyInstaller bundle the
    package sits in an install directory, and any ``.git`` above it belongs to
    an unrelated repository that happens to enclose the install.
    """
    if getattr(sys, "frozen", False):
        return None
    try:
        import quantem

        here = Path(quantem.__file__).resolve()
    except Exception:  # pragma: no cover - quantem is always importable here
        return None
    for parent in here.parents:
        # An installed copy has no checkout of its own, and the walk must stop
        # here rather than continue into whatever repository happens to enclose
        # the environment. A venv created inside someone else's checkout is
        # ordinary, and without this guard that stranger's commit and dirty
        # state are stamped into the user's scientific manifests as if they
        # described the code that produced the numbers. The frozen build is
        # covered by the early return above; this is the same defence for pip.
        if parent.name in {"site-packages", "dist-packages"}:
            return None
        if (parent / ".git").exists():
            return parent
    return None


def _git(repo: Path, *args: str) -> str | None:
    """Run a read-only git command, or None when git cannot be run at all."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _read_git_head(repo: Path) -> str | None:
    """Resolve HEAD by reading ``.git`` directly, for a machine with no git binary."""
    git_dir = repo / ".git"
    try:
        if git_dir.is_file():  # a worktree or submodule: "gitdir: <path>"
            pointer = git_dir.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir:"):
                return None
            git_dir = (repo / pointer.split(":", 1)[1].strip()).resolve()
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or None
        ref = head.split(":", 1)[1].strip()
        loose = git_dir / ref
        if loose.is_file():
            return loose.read_text(encoding="utf-8").strip() or None
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.startswith(("#", "^")) or " " not in line:
                    continue
                sha, name = line.split(" ", 1)
                if name.strip() == ref:
                    return sha.strip()
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return None


# ---------------------------------------------------------------------------
# The machine
# ---------------------------------------------------------------------------


#: Importable module behind each pinned distribution, for the frozen-build
#: fallback below. The distribution name is what ``importlib.metadata`` knows;
#: the module name is what actually carries ``__version__`` when the dist-info
#: is gone.
_DISTRIBUTION_MODULES: dict[str, str] = {
    "torch": "torch",
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-image": "skimage",
    "shapely": "shapely",
    "opencv-python-headless": "cv2",
    "pandas": "pandas",
    "Django": "django",
}


def _module_version(dist_name: str) -> str | None:
    """``module.__version__`` for a distribution, or None if truly absent.

    The fallback for a frozen (PyInstaller) build, whose bundler strips the
    dist-info directories ``importlib.metadata`` reads. The library itself is
    right there -- the same manifest names scikit-image as the estimator under
    every measurement -- so recording it as "not installed" was simply false.
    The import is cheap in the only environment where this runs: the analysis
    job has already imported these libraries to compute the numbers.
    """
    module_name = _DISTRIBUTION_MODULES.get(dist_name)
    if module_name is None:
        return None
    try:
        import importlib

        module = sys.modules.get(module_name) or importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        return str(version) if version else None
    except Exception:
        return None


def environment() -> dict[str, Any]:
    """Interpreter, platform, the versions that move numbers, and the device."""
    packages: dict[str, str | None] = {}
    unavailable: dict[str, str] = {}
    for name in PINNED_DISTRIBUTIONS:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            # No distribution metadata does not mean no library: a frozen
            # build strips dist-info while shipping the package itself. Ask
            # the module for its own version before declaring it absent.
            packages[name] = _module_version(name)
            if packages[name] is None:
                unavailable[f"packages.{name}"] = (
                    f"{name} is not installed in the environment that wrote this bundle."
                )

    values: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        # Machine-readable, so a bundle from before the 2026-08-07 estimator
        # ruling (regionprops.perimeter) is distinguishable from one after it
        # (perimeter_crofton) without parsing prose. Perimeter and circularity
        # are not comparable across that boundary; area and everything else are.
        "perimeter_estimator": "skimage.regionprops.perimeter_crofton",
        "skimage_note": (
            "scikit-image is pinned because binary_closing and regionprops are "
            "under every measurement in this bundle. It is NOT what decides the "
            "min-area boundary: 0.26 changed remove_small_objects from 'smaller "
            "than' to 'smaller than or equal to' the minimum size, which is "
            "exactly why QuantEM does not use it — "
            "quantem.inference.postprocess.filter_min_area counts the components "
            "itself, so the object count cannot move with the library version. "
            "The rule is QuantEM's own and is stated there: a component of "
            "exactly min_area pixels meets the minimum and is kept."
        ),
    }

    available, why = _torch_devices()
    values["torch_devices_available"] = available
    if why:
        unavailable["torch_devices_available"] = why
    unavailable["inference_device"] = (
        "This is the machine that wrote the bundle, not the one that ran the "
        "inference: the analysis job routinely runs in a different process, and "
        "on a shared install a different machine. The device each *run* used is "
        "reported per compartment under models.compartments[].run."
        "inference_device, read from the objects themselves, where "
        "quantem.segmentation.run_identity records it; it is null there only "
        "for objects made before that record existed. Only the devices this "
        "machine offers are recorded above."
    )
    return section(values, unavailable)


def _torch_devices() -> tuple[list[str] | None, str | None]:
    try:
        from quantem.inference.device import available_devices

        return list(available_devices()), None
    except Exception as exc:  # pragma: no cover - torch-free install
        return None, (
            f"The available compute devices could not be listed ({exc.__class__.__name__}: {exc}); "
            "torch is an optional dependency of an analysis-only install."
        )


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def file_identity(path: str | os.PathLike[str] | None, *, what: str) -> dict[str, Any]:
    """SHA-256, size and **name** of one file, or nulls with the reason.

    ``what`` names the file in the reason sentence ("the image", "the model
    head"), because a manifest is read by someone who does not have this code
    open.

    The directory is read to find the bytes and is then thrown away. A path is
    an accident of one installation -- ``<data dir>/data/images/<uuid>.png`` is
    where *this* copy of QuantEM filed the import -- while the filename and the
    digest are what a collaborator can check against the file they were sent.
    This used to emit ``path`` and it was the second thing the release-bundle
    scanner caught in an analysis manifest (rule 3 in the module docstring).
    """
    values: dict[str, Any] = {}
    unavailable: dict[str, str] = {}
    if not path:
        unavailable["sha256"] = f"No local file is recorded for {what}."
        unavailable["size_bytes"] = unavailable["sha256"]
        unavailable["filename"] = unavailable["sha256"]
        return section(values, unavailable)

    p = Path(path)
    values["filename"] = p.name
    if p.is_dir():
        unavailable["sha256"] = (
            f"{what} is stored as a directory of chunks named {p.name!r}, not a "
            "single file; a file checksum does not apply to it."
        )
        unavailable["size_bytes"] = unavailable["sha256"]
        return section(values, unavailable)
    if not p.is_file():
        unavailable["sha256"] = (
            f"{what} is recorded as {p.name!r}, which is not present on the "
            "machine that wrote this bundle, so it could not be checksummed."
        )
        unavailable["size_bytes"] = unavailable["sha256"]
        return section(values, unavailable)

    size = p.stat().st_size
    values["size_bytes"] = size
    if size > MAX_CHECKSUM_BYTES:
        unavailable["sha256"] = (
            f"{what} is {size} bytes, above the {MAX_CHECKSUM_BYTES} byte limit "
            "this export checksums, so hashing it was skipped rather than added "
            "to every analysis run."
        )
        return section(values, unavailable)

    digest = sha256_file(p)
    if digest is None:
        unavailable["sha256"] = (
            f"{what} ({p.name!r}) is present but could not be read to checksum it."
        )
    else:
        values["sha256"] = digest
        values["checksum_algorithm"] = "sha256"
    return section(values, unavailable)


#: What the sweep below reports about itself. Named so the reason a bundle is
#: *clean* is in the bundle, not only in this docstring.
LOCAL_PATH_SCANNER = "quantem.registry.release.find_local_paths"


def scrub_local_paths(document: Any) -> tuple[Any, dict[str, Any]]:
    """``(document with no machine paths in it, what the sweep did)``.

    Every field that could carry a path is built without one -- see
    :func:`file_identity` and :func:`release` -- and this is the check that the
    intent held, run over the finished manifest rather than trusted per field.
    It is the *same* detector the model-release build gate uses
    (:data:`LOCAL_PATH_SCANNER`), imported rather than restated: two definitions
    of "this names somebody's disk" is one too many, and the release side is the
    one that has been through a bundle.

    Anything it finds in a string value is replaced with the release module's
    own placeholder, and the count is recorded. The spans themselves are
    **never** recorded -- writing "we removed" followed by the path it removed
    would put back exactly what was taken out.

    The report says ``clean`` only after re-scanning the serialised result, so a
    leak in a dict *key* -- which is not rewritten, because keys are field names
    and pack ids -- is reported rather than passed over.
    """
    try:
        from quantem.registry.release import find_local_paths, redact_local_paths
    except Exception as exc:  # pragma: no cover - registry always ships
        return document, section(
            {"scanner": LOCAL_PATH_SCANNER},
            {
                "spans_removed": (
                    f"The release module could not be imported ({exc.__class__.__name__}: "
                    f"{exc}), so this manifest was not swept for paths naming the "
                    "machine that wrote it. Every field is built without one, but "
                    "that was not verified here."
                ),
                "clean": "The sweep did not run; see spans_removed.",
            },
        )

    removed = 0

    def walk(node: Any) -> Any:
        nonlocal removed
        if isinstance(node, dict):
            return {key: walk(value) for key, value in node.items()}
        if isinstance(node, list):
            return [walk(value) for value in node]
        if isinstance(node, str):
            hits = find_local_paths(node)
            if not hits:
                return node
            removed += len(hits)
            return redact_local_paths(node)
        return node

    cleaned = walk(document)
    try:
        leftover = len(find_local_paths(json.dumps(cleaned, default=str)))
    except (TypeError, ValueError):  # pragma: no cover - the manifest is JSON
        leftover = -1

    report: dict[str, Any] = {
        "scanner": LOCAL_PATH_SCANNER,
        "spans_removed": removed,
        "clean": leftover == 0,
        "note": (
            "Every string in this manifest was scanned for spans naming a "
            "filesystem path, a network host or a drive letter on the machine "
            "that wrote it, with the same detector that gates a model release. "
            "Files are identified by name and sha256 instead: those mean the "
            "same thing to whoever receives this bundle."
        ),
    }
    if leftover > 0:
        report["still_matching"] = leftover
        report["note"] += (
            f" {leftover} span(s) still match after the sweep — they are in "
            "dictionary keys, which are field names and model pack ids and are "
            "not rewritten. Treat this bundle as naming its author's machine."
        )
    elif leftover < 0:  # pragma: no cover - the manifest is JSON by construction
        report["clean"] = None
        report = section(
            report,
            {"clean": "The swept manifest could not be re-serialised to verify it."},
        )
    return cleaned, report


def sha256_file(path: Path) -> str | None:
    """Streaming SHA-256, or None when the file cannot be read."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while chunk := fh.read(_HASH_CHUNK):
                h.update(chunk)
    except OSError:
        logger.warning("Could not checksum %s", path, exc_info=True)
        return None
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Model packs
# ---------------------------------------------------------------------------


def model_pack(pack_id: str) -> dict[str, Any]:
    """Everything identifying one installed model pack.

    The digests come from the install record the registry writes. They attest
    that the bytes have not changed since installation -- upstream ships
    ``"sha256": null`` in every ``checkpoint_index.json``, so there is nothing
    else to check them against, and the manifest says so rather than implying a
    publisher's signature.
    """
    values: dict[str, Any] = {"pack_id": pack_id}
    unavailable: dict[str, str] = {}

    values.update(_pack_spec(pack_id, unavailable))
    values.update(_pack_record(pack_id, unavailable))
    return section(values, unavailable)


def _pack_spec(pack_id: str, unavailable: dict[str, str]) -> dict[str, Any]:
    """Architecture facts QuantEM declares for a pack, independent of the install."""
    try:
        from quantem.inference.specs import get_model_spec, parse_family

        family = parse_family(pack_id)
        organelle = pack_id.split(":", 1)[1] if ":" in pack_id else ""
        spec = get_model_spec(family, organelle)
    except Exception as exc:
        reason = (
            f"{pack_id!r} is not one of the released model packs ({exc}), so "
            "its architecture, canonical pixel size and default threshold are "
            "unknown to this build."
        )
        for key in ("family", "organelle", "canonical_nm", "tile_size", "default_threshold"):
            unavailable[key] = reason
        return {}
    return {
        "family": spec.family,
        "organelle": spec.organelle,
        "neck": spec.neck,
        "decoder": spec.decoder,
        "adapt": spec.adapt,
        "canonical_nm": spec.canonical_nm,
        "tile_size": spec.tile_size,
        "patch_size": spec.patch_size,
        "default_threshold": spec.threshold,
        "default_min_area_px": spec.organelle_spec.default_min_area,
        "close_radius_px": spec.organelle_spec.close_radius,
        "encoder_norm": {"mean": spec.image_mean, "std": spec.image_std},
        "tiling": _tiling(spec),
    }


def _tiling(spec: Any) -> dict[str, Any]:
    """The sliding-window geometry, which the manuscript quotes and pins.

    ``tile_size`` alone does not describe it: the same tile at 25% overlap and
    at 50% overlap is a different number of windows, a different Hann weighting
    of every pixel, and a different probability map. The overlap is the
    manuscript's own number, so it belongs in the bundle beside the tile.
    """
    values: dict[str, Any] = {"tile_size": spec.tile_size, "patch_size": spec.patch_size}
    unavailable: dict[str, str] = {}
    try:
        from quantem.inference import tiling

        overlap = float(tiling.DEFAULT_OVERLAP)
        values["window_overlap"] = overlap
        values["stride_px"] = tiling.stride_for(spec.tile_size, overlap)
        values["blend"] = (
            "2-D separable Hann window with a "
            f"{tiling.HANN_FLOOR} floor, accumulated weighted and normalised by "
            "the summed weight"
        )
    except Exception as exc:  # pragma: no cover - inference always ships
        reason = (
            f"This build could not read its own tiling settings ({exc}), so "
            "the sliding-window overlap it uses could not be recorded."
        )
        for key in ("window_overlap", "stride_px", "blend"):
            unavailable[key] = reason
    values["note"] = (
        "Inference is run as sliding tiles over the model's own grid. Window "
        "starts walk in steps of stride_px with the last window flush to the "
        "edge, so the right and bottom margins are covered rather than dropped."
    )
    return section(values, unavailable)


def _pack_record(pack_id: str, unavailable: dict[str, str]) -> dict[str, Any]:
    """Digests and install facts, from the registry's ``pack.json``."""
    try:
        from quantem.registry.cache import read_record
    except Exception as exc:  # pragma: no cover - registry always ships
        unavailable["weights"] = (
            f"The model registry is not importable ({exc.__class__.__name__}: {exc}), "
            "so the installed weight digests could not be read."
        )
        return {}

    try:
        record = read_record(pack_id)
    except Exception as exc:
        unavailable["weights"] = (
            f"The install record for {pack_id!r} could not be read "
            f"({exc.__class__.__name__}: {exc})."
        )
        return {}

    if record is None:
        unavailable["weights"] = (
            f"{pack_id!r} is not installed in this machine's model cache, so its "
            "head and encoder digests could not be read here. Recover them from "
            "the pack.json the registry writes beside every installed pack (in "
            "the models / packs directory of that machine's QuantEM data "
            "directory) or by re-installing the pack. The architecture and "
            "threshold above come from the release and are correct regardless."
        )
        return {}

    weights: dict[str, Any] = {}
    for key in ("head", "encoder", "config", "index", "export"):
        entry = record.get(key) or {}
        if entry.get("sha256"):
            weights[key] = {
                "filename": entry.get("filename"),
                "sha256": entry.get("sha256"),
                "size_bytes": entry.get("size_bytes"),
            }
    out: dict[str, Any] = {
        "weights": weights,
        "installed_at": record.get("installed_at"),
        "install_source": record.get("source"),
        "digest_origin": record.get("digest_origin"),
    }
    out.update(_encoder_identity(pack_id, record, unavailable))
    if not weights:
        unavailable["weights"] = (
            f"The install record for {pack_id!r} exists but carries no digests."
        )
        out.pop("weights")
    return out


def _encoder_identity(
    pack_id: str, record: dict[str, Any], unavailable: dict[str, str]
) -> dict[str, Any]:
    """Which encoder checkpoint the head was trained against.

    Two install paths write it in two places and neither writes both. A raw
    install (``install.py`` building a pack out of training outputs) records
    ``encoder_run_dir`` and ``checkpoint_step`` at the top of the record; a
    release-bundle install carries the publisher's descriptor instead, which
    names the encoder as ``encoder.run_id`` and ``encoder.checkpoint_step``.
    Reading only the first wrote two bare nulls into every bundle produced from
    a released pack, next to a manifest whose first rule is that nothing is
    guessed and nothing is silently omitted.

    A raw install records the run dir as it stood on the *training* machine, so
    it goes through :func:`quantem.registry.release.encoder_run_dir` -- the same
    reduction the release build applies -- which keeps the directory's name (the
    run id, the part that identifies the encoder) and drops the parents when
    they are a path on somebody's disk.
    """
    descriptor = record.get("release_descriptor")
    encoder = (descriptor or {}).get("encoder") or {}

    raw_run_dir = record.get("encoder_run_dir")
    run_dir = _portable_run_dir(raw_run_dir) or None
    reduced = bool(raw_run_dir) and run_dir != str(raw_run_dir)
    run_id = encoder.get("run_id") or None
    step = record.get("checkpoint_step")
    if step is None:
        step = encoder.get("checkpoint_step")

    out: dict[str, Any] = {
        "encoder_run_dir": run_dir,
        "encoder_run_id": run_id or (Path(str(run_dir)).name if run_dir else None),
        "checkpoint_step": step,
    }
    if reduced:
        out["encoder_run_dir_note"] = (
            "The install record names this run by its absolute path on the "
            "machine that built the pack. Only the directory's name is kept — "
            "it is the run id, which is the part that identifies the encoder, "
            "and the parents identify somebody's disk."
        )
    if encoder:
        out["encoder_export"] = {
            "tier_exported_from": encoder.get("tier_exported_from"),
            "traced_tile": encoder.get("traced_tile"),
            "dynamic_spatial": encoder.get("dynamic_spatial"),
            "max_abs_diff_vs_eager": encoder.get("max_abs_diff_vs_eager"),
        }

    if run_dir is None:
        unavailable["encoder_run_dir"] = (
            f"{pack_id!r} was installed from a {record.get('source') or 'unknown'} "
            "source, which ships the encoder already exported and does not record "
            "the training run directory it came out of. The encoder is identified "
            "by encoder_run_id and by the sha256 of its exported file above."
            if run_id
            else (
                f"The install record for {pack_id!r} carries no encoder run "
                "directory, and no release descriptor that names one."
            )
        )
    if out["encoder_run_id"] is None:
        unavailable["encoder_run_id"] = (
            f"Neither the install record for {pack_id!r} nor its release "
            "descriptor names the encoder run the head was trained against."
        )
    if step is None:
        unavailable["checkpoint_step"] = (
            f"Neither the install record for {pack_id!r} nor its release "
            "descriptor records which encoder checkpoint step was used. The "
            "encoder bytes are still pinned by the sha256 above."
        )
    return out


def _portable_run_dir(raw: Any) -> str:
    """An encoder run dir with any build machine's directories taken off it."""
    try:
        from quantem.registry.release import encoder_run_dir

        return encoder_run_dir(None if raw is None else str(raw))[0]
    except Exception:  # pragma: no cover - registry always ships
        # Never a fabricated value and never a leak: without the reducer the
        # only safe thing to say about a path is its last segment.
        text = str(raw or "").replace("\\", "/")
        return text.rsplit("/", 1)[-1]
