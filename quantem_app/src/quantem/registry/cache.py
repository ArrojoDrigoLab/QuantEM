"""The on-disk model cache: pack id -> verified local files.

Layout under ``<QUANTEM_DATA_DIR>/models/``::

    blobs/<aa>/<sha256>              every weight file, content-addressed
    packs/<family>__<organelle>/
        pack.json                    the install record (digests, sizes, source)
        head.pt        -> blob       hard link, or a copy where links are refused
        resolved_config.yaml
        checkpoint_index.json
        encoder.pth    -> blob       the raw foundation encoder, when one was
                                     installed: a local-path install copies it,
                                     a release-bundle install does not need it
        encoder_ts.pt  -> blob       the exported TorchScript encoder. Present in
                                     every release bundle; this is what lets a
                                     pack run with no architecture package

**Content addressing is what makes the encoder shared.** ``quantem:mito``,
``quantem:ld`` and ``quantem:nucleus`` all sit on the same 525 MB ViT-B. Storing
per pack would cost 1.5 GB instead of 525 MB, so the blob is written once under
its digest and each pack directory links to it. The pack directory still shows
plain filenames, which is what makes a pack inspectable with ``ls``.

Digests are computed at install time and recorded. They are not read from the
upstream ``checkpoint_index.json``: every one of those files carries
``"sha256": null``.

Nothing here downloads anything. See :mod:`.install`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Called with the fraction of a file hashed so far.
HashProgress = Callable[[float], None]

#: Read size for hashing. 4 MB keeps a 1.2 GB encoder at a few hundred reads.
_HASH_CHUNK = 4 * 1024 * 1024

#: The things a user is ever told to do to obtain models. **Every user-facing
#: string that names a command must use these.** They are here, in the module
#: with no dependencies, because the alternative -- each error message spelling
#: out its own advice -- is how the application once ended up telling strangers
#: to run a command that only worked on one developer's computer.
#:
#: Two routes: the default downloads the pack from the QuantEM Hugging Face
#: repository (in the app: the Models screen's Install button); the offline
#: route installs from a downloaded, unzipped release bundle.
INSTALL_COMMAND_REMOTE = "quantem models install <pack id, e.g. quantem:mito>"
INSTALL_COMMAND = "quantem models install <the directory you unzipped the release into>"
INSTALL_COMMAND_MODULE = (
    "python -m quantem.registry.install bundle <the directory you unzipped "
    "the release into> --all"
)
#: One line, for embedding in a longer sentence or an API ``reason`` field.
INSTALL_HINT = (
    f"Install it from the Models screen or with `{INSTALL_COMMAND_REMOTE}` -- QuantEM "
    "downloads and verifies it from Hugging Face. Offline, download a QuantEM model "
    f"release instead, unzip it, and install it with `{INSTALL_COMMAND}`."
)
#: Several lines, for a terminal: an error the user has just hit, or CLI help.
INSTALL_INSTRUCTIONS = (
    "Model packs are downloaded on demand. Install one with:\n"
    f"  {INSTALL_COMMAND_REMOTE}\n"
    "which downloads and verifies it from Hugging Face "
    "(https://huggingface.co/ArrojoeDrigoLab/quantem).\n"
    "On a machine with no internet access, download a QuantEM model release "
    "elsewhere,\nunzip it, and install it with:\n"
    f"  {INSTALL_COMMAND}\n"
    f"(or, if the console script is not on your PATH:\n  {INSTALL_COMMAND_MODULE})"
)

#: Filenames inside a pack directory.
HEAD_NAME = "head.pt"
CONFIG_NAME = "resolved_config.yaml"
INDEX_NAME = "checkpoint_index.json"
ENCODER_NAME = "encoder.pth"
RECORD_NAME = "pack.json"
EXPORTED_ENCODER_NAME = "encoder_ts.pt"


class PackNotInstalled(FileNotFoundError):
    """A pack was requested that has not been installed into the cache."""


def models_root() -> Path:
    """``<QUANTEM_DATA_DIR>/models``.

    Read through :mod:`quantem.core.config` when it is importable so a single
    process cannot disagree with itself about where the data directory is; falls
    back to the environment variable so the registry stays usable from a plain
    script with no Django on the path.
    """
    try:
        from quantem.core.config import MODELS_DIR

        return Path(MODELS_DIR)
    except Exception:  # pragma: no cover - only when core.config is unavailable
        raw = os.environ.get("QUANTEM_DATA_DIR", "").strip()
        if not raw:
            raise RuntimeError(
                "QUANTEM_DATA_DIR is not set and quantem.core.config is not importable; "
                "cannot locate the model cache."
            ) from None
        return Path(raw) / "models"


def blobs_root() -> Path:
    return models_root() / "blobs"


def packs_root() -> Path:
    return models_root() / "packs"


def pack_dirname(pack_id: str) -> str:
    """``"quantem:mito"`` -> ``"quantem__mito"`` (a colon is not portable on Windows)."""
    return pack_id.replace(":", "__")


def pack_dir(pack_id: str) -> Path:
    return packs_root() / pack_dirname(pack_id)


def blob_path(sha256: str) -> Path:
    """Content-addressed location of one blob, fanned out by the first two hex digits."""
    digest = sha256.lower()
    return blobs_root() / digest[:2] / digest


def sha256_file(path: str | Path, *, on_progress: HashProgress | None = None) -> str:
    """Streaming SHA-256 of a file. Never loads a 1.2 GB encoder into memory."""
    h = hashlib.sha256()
    size = Path(path).stat().st_size
    done = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            h.update(chunk)
            done += len(chunk)
            if on_progress is not None and size:
                on_progress(done / size)
    return h.hexdigest()


@dataclass(frozen=True)
class ResolvedPack:
    """Verified local artifacts for one installed pack.

    ``encoder_path`` is None when the head embeds a fully fine-tuned encoder
    (``quantem:er``, ``adapt: full``) and no shared blob was installed.
    ``export_path`` is the TorchScript encoder if one has been built; it is the
    tier the engine prefers.
    """

    pack_id: str
    root: Path
    head_path: Path
    config_path: Path
    encoder_path: Path | None = None
    index_path: Path | None = None
    export_path: Path | None = None
    record: dict | None = None

    @property
    def has_export(self) -> bool:
        return self.export_path is not None and self.export_path.exists()


def read_record(pack_id: str) -> dict | None:
    """The install record for a pack, or None if it is not installed."""
    record_path = pack_dir(pack_id) / RECORD_NAME
    if not record_path.exists():
        return None
    try:
        return json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Unreadable install record at %s", record_path)
        return None


def installed(pack_id: str) -> bool:
    """True when the pack's recorded files are all present.

    Existence only -- digests are checked at install time and by
    :func:`verify_pack`, not on the hot path: re-hashing a 1.2 GB encoder on
    every ``load_model`` would add seconds to a cached load.
    """
    record = read_record(pack_id)
    if record is None:
        return False
    root = pack_dir(pack_id)
    required = [HEAD_NAME, CONFIG_NAME]
    if record.get("encoder", {}).get("sha256"):
        required.append(record["encoder"].get("filename", ENCODER_NAME))
    return all((root / name).exists() for name in required)


def installed_packs() -> list[str]:
    """Every pack id currently installed, sorted."""
    root = packs_root()
    if not root.is_dir():
        return []
    found = []
    for child in sorted(root.iterdir()):
        if (child / RECORD_NAME).exists():
            record = read_record(child.name.replace("__", ":", 1))
            if record and record.get("pack_id"):
                found.append(str(record["pack_id"]))
    return sorted(found)


def resolve_pack(pack_id: str) -> ResolvedPack:
    """Locate a pack's installed files.

    This is the function :func:`quantem.inference.engine.resolve_model_files`
    calls. Weight *location* belongs here and not to inference: the cache is the
    only thing that knows where a verified blob landed.

    Raises:
        PackNotInstalled: when the pack has no install record or its files are
            missing.
    """
    root = pack_dir(pack_id)
    record = read_record(pack_id)
    if record is None:
        raise PackNotInstalled(
            f"Model pack {pack_id!r} is not installed (no {RECORD_NAME} under {root}).\n"
            f"{INSTALL_INSTRUCTIONS}"
        )

    head_path = root / HEAD_NAME
    if not head_path.exists():
        raise PackNotInstalled(
            f"Model pack {pack_id!r} has an install record but {head_path} is missing; "
            "reinstall it."
        )

    encoder_entry = record.get("encoder") or {}
    encoder_name = encoder_entry.get("filename")
    encoder_path = root / encoder_name if encoder_name else None
    if encoder_path is not None and not encoder_path.exists():
        raise PackNotInstalled(
            f"Model pack {pack_id!r} records an encoder at {encoder_path}, which is missing; "
            "reinstall it."
        )

    config_path = root / CONFIG_NAME
    index_path = root / INDEX_NAME
    export_path = root / EXPORTED_ENCODER_NAME
    return ResolvedPack(
        pack_id=pack_id,
        root=root,
        head_path=head_path,
        config_path=config_path,
        encoder_path=encoder_path,
        index_path=index_path if index_path.exists() else None,
        export_path=export_path if export_path.exists() else None,
        record=record,
    )


def verify_pack(pack_id: str, *, on_progress: HashProgress | None = None) -> dict[str, bool]:
    """Re-hash a pack's files against its record. ``{filename: matches}``.

    Slow and deliberate: this is the "is my install intact" check, not something
    the inference path runs.
    """
    record = read_record(pack_id)
    if record is None:
        raise PackNotInstalled(f"Model pack {pack_id!r} is not installed.")
    root = pack_dir(pack_id)
    results: dict[str, bool] = {}
    for key in ("head", "encoder", "config", "index", "export"):
        entry = record.get(key) or {}
        name, expected = entry.get("filename"), entry.get("sha256")
        if not name or not expected:
            continue
        path = root / name
        results[name] = path.exists() and sha256_file(path, on_progress=on_progress) == expected
    return results
