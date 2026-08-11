"""Installing model packs into the local cache.

Three sources:

* :func:`install_pack_from_bundle` -- **the offline path.** Copy a pack out of
  a downloaded, unzipped release bundle (:mod:`quantem.registry.release`),
  check every file against the bundle's own manifest, and record it. A bundle
  carries an exported TorchScript encoder per pack, so a pack installed this
  way runs on any machine with no research dependency present.
* :func:`install_pack_from_path` -- **the maintainer's path.** Assemble a pack
  out of raw training outputs: a head directory, an encoder checkpoint index,
  and the encoder weights, each named explicitly. Useful for testing a head
  before a release exists; from raw outputs it produces an install with no
  exported encoder, so the four QuantEM packs then need Meta's ``dinov3`` at run
  time. If the directory happens to hold ``encoder_ts.pt``, that is used and
  nothing else is asked for.
* :func:`quantem.registry.hf_install.install_pack_from_hf` -- **the default.**
  Download the pack from the QuantEM Hugging Face repository, verify every
  artifact's digest at a pinned revision, convert it to this pack format and
  export its encoder. Lives in its own module so this one stays importable
  without huggingface_hub.

Where digests come from, and what they mean
-------------------------------------------
For a **bundle** install, the expected digest is the publisher's: it is read
from ``MANIFEST.json``, every copied file is re-hashed against it, and a
mismatch aborts the install. The record says ``digest_origin:
verified-against-release-manifest`` and names the release.

For a **local-path** install there is no publisher to trust: every upstream
``checkpoint_index.json`` carries ``"sha256": null``, so the digests are
computed here and attest only that the bytes have not changed since
installation. The record says so.

Either way :func:`quantem.registry.cache.verify_pack` re-checks them later.

CLI::

    python -m quantem.registry.install bundle ./quantem-models-0.1.0 --all
    python -m quantem.registry.install bundle ./quantem-models-0.1.0 quantem:mito
    python -m quantem.registry.install list
    python -m quantem.registry.install verify quantem:mito

    # maintainer only; there are no default roots, on purpose
    python -m quantem.registry.install local --all \\
        --heads-root <dir> --weights-root <dir> --search-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from quantem.registry import cache, release
from quantem.registry.manifest import ARCHITECTURE

logger = logging.getLogger(__name__)

#: Called with a human-readable progress line ("quantem:mito: hashing head.pt").
StatusFn = Callable[[str], None]

#: Maintainer roots for :func:`install_all_from_paths`, read from the
#: environment. There are deliberately **no** built-in defaults: this module
#: used to hard-code one developer's own directory layout, on their own drive,
#: as the default for the heads and weights roots, and because that default was
#: printed in the README, in a 501 body and in every not-installed error, the
#: single documented way to obtain the models was a command that could not work
#: on any other computer. See :mod:`quantem.registry.release` for the roots'
#: meaning.
HEADS_ROOT_ENV_VAR = release.HEADS_ROOT_ENV_VAR
WEIGHTS_ROOT_ENV_VAR = release.WEIGHTS_ROOT_ENV_VAR
SEARCH_DIRS_ENV_VAR = release.SEARCH_DIRS_ENV_VAR


#: How a pack id maps onto the released head directory name.
def head_dirname(pack_id: str) -> str:
    """``"quantem:mito"`` -> ``"mito_quantem"`` (the released directory naming)."""
    family, organelle = pack_id.split(":", 1)
    return f"{organelle}_{family}"


class InstallError(RuntimeError):
    """A pack could not be installed. The message names the missing piece."""


@dataclass(frozen=True)
class InstalledPack:
    pack_id: str
    root: Path
    head_sha256: str
    encoder_sha256: str | None
    bytes_written: int
    reused_blobs: int


# --- Blob store -------------------------------------------------------------


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link ``src`` into ``dst``, copying when the filesystem refuses.

    A hard link is what makes three QuantEM packs share one 525 MB encoder while
    each pack directory still lists a plain ``encoder.pth``. Windows supports
    them on NTFS; a copy is the correct fallback anywhere else, and costs disk
    rather than correctness.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        logger.debug("hard link refused for %s -> %s; copying", src, dst)
        shutil.copy2(src, dst)


def store_blob(src: Path, *, on_progress: cache.HashProgress | None = None) -> tuple[str, int, bool]:
    """Hash ``src`` and put it in the content-addressed store.

    Returns ``(sha256, size_bytes, reused)``. ``reused`` is True when a blob
    with that digest was already present -- the shared-encoder case.

    Either way the blob's mtime is stamped to *now*. That timestamp is the
    blob's GC lease: ``quantem.registry.hf_install._gc_orphan_blobs`` collects
    only blobs no installed pack references whose mtime is older than the
    stale-staging window, so touching on store **and on reuse** is what keeps a
    blob a concurrent install is about to link alive. Without the stamp a
    fresh store would carry the *source* file's mtime (``copy2`` preserves it)
    and could look days old the moment it lands.
    """
    src = Path(src)
    if not src.is_file():
        raise InstallError(f"not a file: {src}")
    digest = cache.sha256_file(src, on_progress=on_progress)
    size = src.stat().st_size
    target = cache.blob_path(digest)
    if target.exists() and target.stat().st_size == size:
        _touch(target)
        return digest, size, True
    target.parent.mkdir(parents=True, exist_ok=True)
    # Unique per process+call: two installs of packs sharing a trunk may hash
    # the same blob concurrently, and a shared ".partial" would have them
    # writing over each other's half-copied bytes. With unique temp names the
    # last atomic replace wins, and both wrote identical content.
    tmp = target.with_suffix(f".partial-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        shutil.copy2(src, tmp)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    _touch(target)
    return digest, size, False


def _touch(path: Path) -> None:
    """Set ``path``'s mtime to now; a failure costs a lease, not an install."""
    try:
        os.utime(path)
    except OSError:
        logger.debug("could not refresh the mtime lease on %s", path)


# --- Encoder resolution -----------------------------------------------------


def resolve_encoder_file(
    index_path: Path,
    checkpoint_step: int | None,
    search_dirs: list[Path],
) -> Path:
    """Find the encoder weight file a pack's config asks for.

    The paths recorded in ``checkpoint_index.json`` are the research machine's
    (UNC shares, WSL mounts) and will not resolve here, so the recorded entry is
    used to choose *which* checkpoint -- by step -- and the file itself is found
    by basename in ``search_dirs``.
    """
    from quantem.inference.encoders import EncoderManifest

    try:
        manifest = EncoderManifest.from_index(index_path)
        candidates = manifest.checkpoint_paths(index_path, checkpoint_step)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # A truncated or partial index is a bad index, not a crash: the caller
        # may have an exported encoder that makes this file decorative.
        raise InstallError(f"{index_path} could not be read as an encoder index: {exc}") from exc
    if not candidates:
        raise InstallError(
            f"{index_path} lists no checkpoint for step={checkpoint_step!r}; "
            "the config and the index disagree."
        )
    tried: list[str] = []
    for recorded in candidates:
        direct = Path(recorded)
        if direct.is_file():
            return direct
        name = Path(recorded.replace("\\", "/")).name
        for d in search_dirs:
            candidate = Path(d) / name
            tried.append(str(candidate))
            if candidate.is_file():
                return candidate
    wanted = [Path(c.replace(chr(92), "/")).name for c in candidates]
    raise InstallError(
        f"no encoder checkpoint for step={checkpoint_step!r} on this machine. "
        f"{index_path.name} names {wanted}; looked for those in "
        f"{[str(d) for d in search_dirs] or '(nowhere -- no directory was given)'}. "
        f"These are research files: a QuantEM release bundle needs none of them, "
        f"because each of its packs carries {cache.EXPORTED_ENCODER_NAME} instead."
    )


# --- Local install ----------------------------------------------------------


def install_pack_from_path(
    pack_id: str,
    head_dir: str | Path,
    *,
    encoder_index: str | Path | None = None,
    encoder_file: str | Path | None = None,
    weights_root: str | Path | None = None,
    search_dirs: list[Path] | None = None,
    force: bool = False,
    on_progress: StatusFn | None = None,
) -> InstalledPack:
    """Install one pack from a directory of files on this machine.

    Mostly a maintainer path -- raw training outputs, a head being tested before
    a release exists. Nothing is guessed from a machine layout: every root this
    reads is either passed in or derived from a file the caller named.

    **An exported encoder is enough.** When ``head_dir`` holds
    ``encoder_ts.pt``, the checkpoint index and the raw foundation weights are
    not looked for at all: the exported artifact is self-describing, it is the
    tier the engine prefers, and demanding a research checkpoint beside it would
    refuse a directory that can already run. That is the shape of a release
    bundle's pack directory, and it used to be refused with a message telling
    the user to go and find a ``.pth`` they have never seen.

    Args:
        pack_id: e.g. ``"quantem:mito"``.
        head_dir: directory holding ``head.pt`` and ``resolved_config.yaml``.
        encoder_index: the family's ``checkpoint_index.json``. When omitted it
            is looked for beside the head, then at
            ``<weights_root>/<run id from the config>/``.
        encoder_file: the encoder weights. Taken from beside the head when it is
            there, else resolved from the index.
        weights_root: directory holding ``<run_id>/checkpoint_index.json``.
        search_dirs: extra directories to find the encoder weights in.
        force: reinstall even if the pack is already present.

    Raises:
        InstallError: naming the missing file.
    """
    if pack_id not in ARCHITECTURE:
        raise InstallError(f"unknown pack id {pack_id!r}; known: {sorted(ARCHITECTURE)}")
    if cache.installed(pack_id) and not force:
        return _already_installed(pack_id)

    head_dir = Path(head_dir)
    head_src = head_dir / cache.HEAD_NAME
    config_src = head_dir / cache.CONFIG_NAME
    for p in (head_src, config_src):
        if not p.is_file():
            raise InstallError(f"{pack_id}: missing {p}")

    from quantem.inference._fig3.schema import load_head_config

    cfg = load_head_config(config_src)
    run_dir, run_id = release.encoder_run_dir(cfg.encoder.run_dir)

    # The exported encoder, when the source has one. This is the *shipping*
    # tier: with it a pack runs on any machine, and without it the four QuantEM
    # packs need Meta's `dinov3` package, which QuantEM does not redistribute --
    # so they install perfectly and then raise at inference. Skipping this file
    # was exactly how a released bundle produced an unrunnable install.
    exported_src = head_dir / cache.EXPORTED_ENCODER_NAME
    has_export = exported_src.is_file()

    # Beside the head first -- that is where a release puts it, and a pack
    # directory should install with nothing else named. Otherwise the config
    # names its encoder run directory, and that is where the index lives.
    if encoder_index is None and (head_dir / cache.INDEX_NAME).is_file():
        encoder_index = head_dir / cache.INDEX_NAME
    if encoder_index is None and weights_root and run_id:
        candidate = Path(weights_root) / run_id / cache.INDEX_NAME
        if candidate.is_file():
            encoder_index = candidate
    if encoder_index is not None and not Path(encoder_index).is_file():
        encoder_index = None
    if encoder_file is None and (head_dir / cache.ENCODER_NAME).is_file():
        encoder_file = head_dir / cache.ENCODER_NAME
    if encoder_index is None and not has_export:
        raise InstallError(
            f"{pack_id}: {head_dir} has {cache.HEAD_NAME} and {cache.CONFIG_NAME}, but "
            f"neither {cache.EXPORTED_ENCODER_NAME} — the exported encoder that "
            f"every pack in a QuantEM release ships beside its head, and all that is "
            f"needed to run one — nor a {cache.INDEX_NAME} for encoder run "
            f"{run_id or '(unnamed)'}.\n"
            f"If you unzipped a QuantEM model release, install from the directory "
            f"holding {release.MANIFEST_NAME}, from its {release.PACKS_DIRNAME}/ "
            f"folder, or from one pack directory inside that "
            f"({release.PACKS_DIRNAME}/{cache.pack_dirname(pack_id)}/).\n"
            f"If these are raw training outputs, the folder holding the encoder runs "
            f"must contain a folder named for the run with {cache.INDEX_NAME} in it."
        )

    dirs = [*(search_dirs or []), head_dir]
    if encoder_index is not None:
        encoder_index = Path(encoder_index)
        dirs.append(encoder_index.parent)
        if encoder_file is None:
            try:
                encoder_file = resolve_encoder_file(
                    encoder_index, cfg.encoder.checkpoint_step, dirs
                )
            except InstallError as exc:
                # Only fatal when there is no exported encoder to fall back on:
                # a pack directory out of a bundle carries an index for
                # provenance and needs nothing it points at.
                if not has_export:
                    raise InstallError(f"{pack_id}: {exc}") from exc
                logger.debug("%s: %s; the exported encoder covers it", pack_id, exc)

    root = cache.pack_dir(pack_id)
    root.mkdir(parents=True, exist_ok=True)
    written = 0
    reused = 0
    record: dict[str, object] = {
        "pack_id": pack_id,
        "schema_version": 1,
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "local-path",
        "digest_origin": (
            "computed-at-install: this pack was copied from a directory of files, not "
            "from a release with a manifest to check them against, so these attest only "
            "that the bytes have not changed since installation."
        ),
        "architecture": dict(ARCHITECTURE[pack_id]),
        "adapt": cfg.encoder.adapt,
        "neck": cfg.neck.type,
        "decoder": cfg.decoder.type,
        # Normalised, not copied: a training config may name its run by an
        # absolute path inside the training container, and this value ends up in
        # an exported analysis manifest.
        "encoder_run_dir": run_dir,
        "checkpoint_step": cfg.encoder.checkpoint_step,
    }

    def _install_file(src: Path, dst_name: str, key: str) -> None:
        nonlocal written, reused
        if on_progress is not None:
            on_progress(f"{pack_id}: hashing {src.name}")
        digest, size, was_reused = store_blob(src)
        _link_or_copy(cache.blob_path(digest), root / dst_name)
        record[key] = {
            "filename": dst_name,
            "sha256": digest,
            "size_bytes": size,
            "source_path": str(src),
        }
        reused += int(was_reused)
        written += 0 if was_reused else size

    _install_file(head_src, cache.HEAD_NAME, "head")
    _install_file(config_src, cache.CONFIG_NAME, "config")
    if encoder_index is not None:
        _install_file(encoder_index, cache.INDEX_NAME, "index")
    if encoder_file is not None:
        _install_file(Path(encoder_file), cache.ENCODER_NAME, "encoder")
    if has_export:
        _install_file(exported_src, cache.EXPORTED_ENCODER_NAME, "export")

    (root / cache.RECORD_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    head_entry = record.get("head") or {}
    encoder_entry = record.get("encoder") or {}
    return InstalledPack(
        pack_id=pack_id,
        root=root,
        head_sha256=str(head_entry.get("sha256", "")),  # type: ignore[union-attr]
        encoder_sha256=encoder_entry.get("sha256"),  # type: ignore[union-attr]
        bytes_written=written,
        reused_blobs=reused,
    )


def install_all_from_paths(
    heads_root: str | Path,
    weights_root: str | Path,
    *,
    pack_ids: list[str] | None = None,
    search_dirs: list[Path] | None = None,
    force: bool = False,
    on_progress: StatusFn | None = None,
) -> list[InstalledPack]:
    """Install every released pack found under ``heads_root``.

    Both roots are required. They used to default to one developer's drive,
    which is why the command that named them was useless to everybody else.
    """
    heads_root = Path(heads_root)
    if not heads_root.is_dir():
        raise InstallError(f"heads root {heads_root} is not a directory.")
    out: list[InstalledPack] = []
    for pack_id in pack_ids or sorted(ARCHITECTURE):
        head_dir = heads_root / head_dirname(pack_id)
        if not head_dir.is_dir():
            raise InstallError(f"{pack_id}: no head directory at {head_dir}")
        out.append(
            install_pack_from_path(
                pack_id,
                head_dir,
                weights_root=weights_root,
                search_dirs=search_dirs,
                force=force,
                on_progress=on_progress,
            )
        )
    return out


# --- Release-bundle install (the documented path) ---------------------------


def install_pack_from_bundle(
    pack_id: str,
    bundle_root: str | Path,
    *,
    force: bool = False,
    on_progress: StatusFn | None = None,
    bundle: release.Bundle | None = None,
) -> InstalledPack:
    """Install one pack from a downloaded, unzipped release bundle.

    This is the path a user takes and the only one any user-facing string names.
    It needs nothing but the directory: no roots, no search paths, no research
    tree, and no network.

    Every file is hashed as it is copied and compared with the digest the
    publisher recorded in ``MANIFEST.json``. A mismatch aborts before the pack's
    install record is written, so a half-verified pack never becomes an
    installed one.

    Args:
        pack_id: e.g. ``"quantem:mito"``.
        bundle_root: the unzipped bundle directory.
        force: reinstall even if the pack is already present.
        bundle: a manifest already read from ``bundle_root``; installing eight
            packs should parse it once.

    Raises:
        InstallError: naming the pack and what was wrong with it.
    """
    if pack_id not in ARCHITECTURE:
        raise InstallError(f"unknown pack id {pack_id!r}; known: {sorted(ARCHITECTURE)}")
    if cache.installed(pack_id) and not force:
        return _already_installed(pack_id)

    try:
        bundle = bundle or release.read_bundle(bundle_root)
        entry = bundle.pack(pack_id)
    except release.BundleError as exc:
        raise InstallError(str(exc)) from exc

    missing_roles = [r for r in release.REQUIRED_ROLES if r not in entry.files]
    if missing_roles:
        raise InstallError(
            f"{pack_id}: the bundle at {bundle.root} lists no {missing_roles} for this "
            "pack, so it could be installed but not run. The bundle is incomplete; "
            "re-download it."
        )

    root = cache.pack_dir(pack_id)
    root.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "pack_id": pack_id,
        "schema_version": 1,
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "release-bundle",
        "release": bundle.release,
        "digest_origin": (
            f"verified-against-release-manifest: every file was re-hashed on install and "
            f"matched the sha256 recorded in {release.MANIFEST_NAME} of QuantEM model "
            f"release {bundle.release}."
        ),
        "architecture": dict(ARCHITECTURE[pack_id]),
    }
    written = 0
    reused = 0

    for role in sorted(entry.files):
        src = entry.file_path(bundle.root, role)
        rel = entry.manifest_path(role)
        if src is None or rel is None:
            continue
        try:
            expected = bundle.file(rel)
        except release.BundleError as exc:
            raise InstallError(f"{pack_id}: {exc}") from exc
        if not src.is_file():
            raise InstallError(
                f"{pack_id}: {rel} is listed in {release.MANIFEST_NAME} but missing from "
                f"{bundle.root}. The download is incomplete."
            )
        if on_progress is not None:
            on_progress(f"{pack_id}: verifying {src.name}")

        digest, size, was_reused = store_blob(src)
        if digest != expected.sha256 or size != expected.size_bytes:
            raise InstallError(
                f"{pack_id}: {rel} does not match the release manifest "
                f"(expected sha256 {expected.sha256[:16]}… / {expected.size_bytes} B, "
                f"got {digest[:16]}… / {size} B). Refusing to install a file the "
                "publisher did not sign off. Re-download the bundle."
            )
        _link_or_copy(cache.blob_path(digest), root / src.name)
        record[role] = {"filename": src.name, "sha256": digest, "size_bytes": size}
        reused += int(was_reused)
        written += 0 if was_reused else size

    # Carry the publisher's own pack descriptor into the record rather than
    # re-deriving licence and provenance here: two places that both claim to say
    # what a pack is licensed under is one place too many.
    descriptor = entry.dir_path(bundle.root) / release.PACK_DESCRIPTOR_NAME
    if descriptor.is_file():
        try:
            record["release_descriptor"] = json.loads(descriptor.read_text(encoding="utf-8"))
        except ValueError:
            logger.warning("Unreadable %s in %s", release.PACK_DESCRIPTOR_NAME, bundle.root)

    record.update(_encoder_provenance(record.get("release_descriptor"), root))

    (root / cache.RECORD_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    head = record.get("head")
    return InstalledPack(
        pack_id=pack_id,
        root=root,
        head_sha256=str(head.get("sha256", "")) if isinstance(head, dict) else "",
        encoder_sha256=None,
        bytes_written=written,
        reused_blobs=reused,
    )


def install_all_from_bundle(
    bundle_root: str | Path,
    *,
    pack_ids: list[str] | None = None,
    force: bool = False,
    on_progress: StatusFn | None = None,
) -> list[InstalledPack]:
    """Install every pack a bundle carries (or the subset named).

    The manifest is parsed once and reused, so eight packs cost one read of it.
    """
    try:
        bundle = release.read_bundle(bundle_root)
    except release.BundleError as exc:
        raise InstallError(str(exc)) from exc

    wanted = list(pack_ids or bundle.pack_ids)
    absent = [p for p in wanted if p not in bundle.pack_ids]
    if absent:
        raise InstallError(
            f"the bundle at {bundle.root} does not contain {absent}. It holds: "
            f"{', '.join(bundle.pack_ids) or '(nothing)'}."
        )
    return [
        install_pack_from_bundle(
            pack_id, bundle.root, force=force, on_progress=on_progress, bundle=bundle
        )
        for pack_id in wanted
    ]


def _encoder_provenance(descriptor: object, root: Path) -> dict[str, object]:
    """``encoder_run_dir`` and ``checkpoint_step`` for a bundle install record.

    Both keys are read straight off the install record by
    :mod:`quantem.analysis.provenance`, so an analysis manifest says which
    encoder run and step produced its masks. A local-path install has the
    training config to hand and writes them there; a bundle install has to get
    them from the publisher, because the shipped ``checkpoint_index.json``
    deliberately no longer names a path (see
    :func:`quantem.registry.release.sanitise_checkpoint_index`).

    The pack descriptor is the source. Its ``encoder`` block is written by
    :func:`quantem.registry.release._write_descriptor` from the very config the
    head was trained with. The installed ``resolved_config.yaml`` is the
    fallback, for a bundle whose descriptor is missing or older than this field.
    """
    if isinstance(descriptor, dict):
        encoder = descriptor.get("encoder")
        if isinstance(encoder, dict) and (encoder.get("run_dir") or encoder.get("run_id")):
            return {
                "encoder_run_dir": str(encoder.get("run_dir") or encoder.get("run_id")),
                "checkpoint_step": encoder.get("checkpoint_step"),
            }

    config_path = root / cache.CONFIG_NAME
    if not config_path.is_file():
        return {}
    try:
        from quantem.inference._fig3.schema import load_head_config

        cfg = load_head_config(config_path)
    except Exception:  # noqa: BLE001 -- provenance is worth less than the install
        logger.debug("Could not read encoder provenance from %s", config_path, exc_info=True)
        return {}
    run_dir, _ = release.encoder_run_dir(cfg.encoder.run_dir)
    if not run_dir:
        return {}
    return {"encoder_run_dir": run_dir, "checkpoint_step": cfg.encoder.checkpoint_step}


def _already_installed(pack_id: str) -> InstalledPack:
    resolved = cache.resolve_pack(pack_id)
    existing = resolved.record or {}
    return InstalledPack(
        pack_id=pack_id,
        root=resolved.root,
        head_sha256=str((existing.get("head") or {}).get("sha256", "")),
        encoder_sha256=(existing.get("encoder") or {}).get("sha256"),
        bytes_written=0,
        reused_blobs=0,
    )


# --- Remote install ---------------------------------------------------------
#
# Downloading from the QuantEM Hugging Face repository lives in
# :mod:`quantem.registry.hf_install` (fetch + verify + convert + export), built
# on :func:`store_blob` and the same content-addressed cache as the two local
# sources above. It is a separate module because it is the only source that
# imports huggingface_hub, and this one must stay importable without it.


# --- CLI --------------------------------------------------------------------


def _report(results: list[InstalledPack]) -> int:
    for r in results:
        print(f"{r.pack_id:18s} head={r.head_sha256[:16]}  encoder={(r.encoder_sha256 or '-')[:16]}")
    total = sum(r.bytes_written for r in results)
    print(f"\n{len(results)} pack(s) installed under {cache.packs_root()}")
    print(f"{total / 1e9:.2f} GB of new blobs ({sum(r.reused_blobs for r in results)} reused)")
    return 0


def _cmd_bundle(args: argparse.Namespace) -> int:
    pack_ids = None if args.all else (args.packs or None)
    if not args.all and not pack_ids:
        print("nothing to do: pass pack ids or --all", file=sys.stderr)
        return 2
    return _report(
        install_all_from_bundle(
            args.bundle,
            pack_ids=pack_ids,
            force=args.force,
            on_progress=(lambda msg: print(msg, file=sys.stderr)) if args.verbose else None,
        )
    )


def _cmd_local(args: argparse.Namespace) -> int:
    pack_ids = None if args.all else (args.packs or None)
    if not args.all and not pack_ids:
        print("nothing to do: pass pack ids or --all", file=sys.stderr)
        return 2
    heads_root = args.heads_root or os.environ.get(HEADS_ROOT_ENV_VAR, "").strip()
    weights_root = args.weights_root or os.environ.get(WEIGHTS_ROOT_ENV_VAR, "").strip()
    missing = []
    if not heads_root:
        missing.append(f"--heads-root (or ${HEADS_ROOT_ENV_VAR})")
    if not weights_root:
        missing.append(f"--weights-root (or ${WEIGHTS_ROOT_ENV_VAR})")
    if missing:
        print(
            "error: `install local` builds a pack out of raw training outputs and needs "
            "to be told where they are: " + ", ".join(missing) + ".\n"
            "There is no default, because that layout differs on every machine.\n"
            "If you are a user rather than the maintainer, you want `install bundle` "
            "instead:\n"
            f"  {cache.INSTALL_COMMAND_MODULE}",
            file=sys.stderr,
        )
        return 2
    search = [Path(d) for d in (args.search_dir or [])] or release.default_search_dirs()
    return _report(
        install_all_from_paths(
            heads_root,
            weights_root,
            pack_ids=pack_ids,
            search_dirs=search,
            force=args.force,
            on_progress=(lambda msg: print(msg, file=sys.stderr)) if args.verbose else None,
        )
    )


def _cmd_list(_args: argparse.Namespace) -> int:
    ids = cache.installed_packs()
    if not ids:
        print(f"no packs installed under {cache.packs_root()}")
        return 0
    for pack_id in ids:
        resolved = cache.resolve_pack(pack_id)
        arch = ARCHITECTURE.get(pack_id, {})
        export = "exported" if resolved.has_export else "eager"
        print(
            f"{pack_id:18s} {arch.get('neck', '?'):16s} {arch.get('decoder', '?'):14s} "
            f"{arch.get('adapt', '?'):8s} tile={arch.get('tile', '?')} [{export}]"
        )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    failed = 0
    for pack_id in args.packs or cache.installed_packs():
        results = cache.verify_pack(pack_id)
        bad = [name for name, ok in results.items() if not ok]
        status = "OK" if not bad else f"MISMATCH {bad}"
        failed += bool(bad)
        print(f"{pack_id:18s} {status}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    # -v is on a shared parent so it works either side of the subcommand; the
    # natural thing to type is `install local --all -v`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true")

    p = argparse.ArgumentParser(
        prog="python -m quantem.registry.install",
        description=(
            "Install QuantEM model packs into the local cache. Packs go under "
            "$QUANTEM_DATA_DIR/models; `quantem models install` is the same thing "
            "with a --data-dir flag."
        ),
        epilog=cache.INSTALL_INSTRUCTIONS,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )
    sub = p.add_subparsers(dest="command", required=True)

    bun = sub.add_parser(
        "bundle",
        parents=[common],
        help="install from a downloaded release bundle (this is the normal way)",
        description=(
            "Install packs from a QuantEM model release you downloaded and unzipped. "
            "Every file is checked against the sha256 the publisher recorded in the "
            "bundle's MANIFEST.json before the pack becomes installed."
        ),
    )
    bun.add_argument("bundle", help="the unzipped bundle directory")
    bun.add_argument("packs", nargs="*", help="pack ids, e.g. quantem:mito")
    bun.add_argument("--all", action="store_true", help="install every pack in the bundle")
    bun.add_argument("--force", action="store_true", help="reinstall packs already present")
    bun.set_defaults(func=_cmd_bundle)

    loc = sub.add_parser(
        "local",
        parents=[common],
        help="maintainer: assemble packs from raw training outputs",
        description=(
            "Assemble packs directly from training outputs. This produces an install "
            "with no exported encoder, so the four QuantEM packs will then need Meta's "
            "`dinov3` at run time. Users want `bundle`; this is for testing a head "
            "before a release exists, and for feeding "
            "`python -m quantem.registry.release build`."
        ),
    )
    loc.add_argument("packs", nargs="*", help="pack ids, e.g. quantem:mito")
    loc.add_argument("--all", action="store_true", help="install all eight released packs")
    loc.add_argument("--heads-root", default=None,
                     help="directory holding <organelle>_<family>/head.pt "
                          f"(or ${HEADS_ROOT_ENV_VAR}); required, no default")
    loc.add_argument("--weights-root", default=None,
                     help="directory holding <run_id>/checkpoint_index.json "
                          f"(or ${WEIGHTS_ROOT_ENV_VAR}); required, no default")
    loc.add_argument("--search-dir", action="append",
                     help="extra directory to find encoder weight files in (repeatable; "
                          f"or ${SEARCH_DIRS_ENV_VAR})")
    loc.add_argument("--force", action="store_true", help="reinstall packs already present")
    loc.set_defaults(func=_cmd_local)

    lst = sub.add_parser("list", parents=[common], help="show installed packs")
    lst.set_defaults(func=_cmd_list)

    ver = sub.add_parser("verify", parents=[common],
                         help="re-hash installed packs against their records")
    ver.add_argument("packs", nargs="*")
    ver.set_defaults(func=_cmd_verify)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    try:
        return int(args.func(args))
    except (InstallError, release.BundleError, cache.PackNotInstalled) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
