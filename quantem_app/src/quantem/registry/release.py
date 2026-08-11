"""Release bundles: the artifact users download, and the command that builds it.

A **release bundle** is a plain directory tree. Unzip it anywhere and point
QuantEM at it; nothing in it refers to the machine that produced it, and every
byte in it is covered by a SHA-256 in one top-level manifest::

    quantem-models-0.1.0/
        MANIFEST.json              every file, its size and its sha256
        MANIFEST.json.sha256       digest of the manifest itself
        README.txt                 the install command, for someone holding a zip
        packs/
            quantem__mito/
                pack.json          this pack's descriptor: architecture, licence, digests
                head.pt            the released segmentation head
                resolved_config.yaml
                checkpoint_index.json
                encoder_ts.pt      the *exported* TorchScript encoder

Why this exists
---------------
The eight released heads are bare ``state_dict``s that need an encoder rebuilt
around them, and four of them (the QuantEM family) sit on a DINOv3 ViT-B whose
architecture code is Meta's and is not redistributed here. A bundle of head
files alone therefore installs perfectly on a stranger's machine and then
refuses to run. The fix is not a better error message: it is to put the
**exported** encoder in the bundle, built once by the maintainer on the machine
that does have the architecture code. That is what :func:`build_bundle` does,
per pack, through :func:`quantem.inference.export.export_encoder_files` -- the
same code path, and the same verification against the eager model, that the
developer-facing export command uses.

What is *not* in a bundle, and why
----------------------------------
The shared foundation encoder blob (``encoder.pth``, 525 MB for QuantEM and
1.2 GB for OmniEM) is deliberately absent. Once ``encoder_ts.pt`` exists nothing
at run time reads it -- see :func:`quantem.inference.encoders.build_encoder`,
which returns at tier (a) before it ever looks -- so shipping it would add
1.7 GB of bytes that only exist to be ignored, and would redistribute a third
party's foundation weights to do it. ``checkpoint_index.json`` *is* kept: it is
small, it is the provenance record for which encoder run and step a pack was
trained against, and it is what a reinstall would need if an export ever had to
be rebuilt.

Local paths are not recorded anywhere in a bundle
-------------------------------------------------
Two of the four files a pack ships are the maintainer's own training outputs,
and they arrive full of the build machine: ``checkpoint_index.json`` records
every encoder checkpoint by absolute path -- a network share, a mounted drive,
whatever the training box had -- and ``resolved_config.yaml`` records the
training ``data_root`` and the config's path inside the training container.
Those are the bytes a user
is told to hash and keep, in an artifact bound for Hugging Face and Zenodo under
a real name.

So they are **sanitised on the way in**, not shipped verbatim
(:func:`sanitise_checkpoint_index`, :func:`sanitise_resolved_config`): what
identifies the encoder is kept -- the run id, the run directory's name, the
checkpoint step, and the checkpoint's *file* name, which carries the step -- and
the directories above it, the host names and the drive letters go. The install
record written by a local-path install carries ``source_path`` values naming the
developer's drive; a bundle's records the encoder run id and checkpoint step
instead.

Because a claim like that is worth exactly what enforces it,
:func:`build_bundle` re-reads every byte it just wrote and refuses to finish a
bundle in which anything still looks like a path or a host
(:func:`scan_bundle_for_local_paths`). ``release scan`` runs the same check on a
downloaded copy.

CLI (maintainer, run once per release, on the build box)::

    python -m quantem.registry.release build \\
        --heads-root  <dir of <organelle>_<family>/head.pt> \\
        --weights-root <dir of <run_id>/checkpoint_index.json> \\
        --search-dir  <dir holding the encoder checkpoint files> \\
        --out .../quantem-models-0.1.0

    python -m quantem.registry.release verify .../quantem-models-0.1.0
    python -m quantem.registry.release scan   .../quantem-models-0.1.0
    python -m quantem.registry.release show   .../quantem-models-0.1.0

Installing one is :mod:`quantem.registry.install` (``quantem models install``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from quantem.registry import cache
from quantem.registry.manifest import ARCHITECTURE

logger = logging.getLogger(__name__)

#: Called with a human-readable progress line.
StatusFn = Callable[[str], None]

BUNDLE_SCHEMA_VERSION = 1

#: What ``MANIFEST.json`` declares itself to be. An installer checks this before
#: anything else, so pointing at the wrong unzipped directory says so plainly
#: instead of failing four files later.
BUNDLE_KIND = "quantem-model-release"

MANIFEST_NAME = "MANIFEST.json"
MANIFEST_DIGEST_NAME = "MANIFEST.json.sha256"
PACKS_DIRNAME = "packs"
README_NAME = "README.txt"

#: The per-pack descriptor inside a bundle. It shares its name with
#: :data:`quantem.registry.cache.RECORD_NAME` because both answer "what is this
#: pack", but they are different documents and neither is copied over the other:
#: this one is written by the maintainer and describes the *release*; the cache's
#: is written by the installer and describes *this machine's copy*.
PACK_DESCRIPTOR_NAME = "pack.json"

#: Files a bundled pack carries, as ``role -> filename``. ``export`` is the one
#: that makes the bundle worth shipping.
PACK_FILE_ROLES: dict[str, str] = {
    "head": cache.HEAD_NAME,
    "config": cache.CONFIG_NAME,
    "index": cache.INDEX_NAME,
    "export": cache.EXPORTED_ENCODER_NAME,
}

#: Roles without which a pack cannot be installed and run.
REQUIRED_ROLES: tuple[str, ...] = ("head", "config", "export")

#: Roles that are the maintainer's own training outputs and are therefore
#: rewritten rather than copied. See the module docstring; the rewriting is
#: :func:`sanitise_pack_file`.
SANITISED_ROLES: tuple[str, ...] = ("config", "index")

#: Environment variables the maintainer can set instead of passing the three
#: build roots every time. There are **no** built-in defaults for these: a path
#: that only exists on one person's computer must never be what a command
#: prints to everybody else.
HEADS_ROOT_ENV_VAR = "QUANTEM_HEADS_ROOT"
WEIGHTS_ROOT_ENV_VAR = "QUANTEM_WEIGHTS_ROOT"
SEARCH_DIRS_ENV_VAR = "QUANTEM_ENCODER_SEARCH_DIRS"


class BundleError(RuntimeError):
    """A bundle could not be built, read, or verified. The message names the file."""


def _write_text(path: Path, text: str) -> int:
    """Write UTF-8 with LF endings, returning bytes written.

    Never :meth:`Path.write_text`: that translates ``\\n`` to ``\\r\\n`` on
    Windows, so the same bundle built on the maintainer's box and on a Linux
    runner would differ in every text file's digest.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(data)
    return len(data)


# --- What may not ship ------------------------------------------------------

#: Stands in for a span that named a filesystem path, a host or a drive on the
#: machine that built the bundle. Deliberately contains no ``/``, ``\`` or ``:``,
#: so redacting an already-redacted document is a no-op and a resumed build
#: produces the same bytes as a fresh one.
REDACTED = "<removed for release>"

#: What counts as naming the build machine, in text. A path runs to the first
#: whitespace or quote, whatever it has in it.
_LOCAL_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    # \\HOST\share\dir\file -- the worst of them: it names a machine on a
    # network, which is the one thing a public artifact can never carry.
    re.compile(r"\\\\[A-Za-z0-9._$-]+(?:\\[^\s\"';,]+)+"),
    # D:\Chris\... or V:/Chris/...
    re.compile(r"(?<![\w:])[A-Za-z]:[\\/][^\s\"';,]*"),
    # /mnt/d/..., /root/..., /home/... -- any absolute POSIX path. The
    # lookbehind keeps the "//" of a URL and the "/" inside doi.org/10.1101/...
    # out of it; a *relative* path such as foundation_weights/m1_dinov3_vitb is
    # not a local path and is kept, because it is how a run identifies itself.
    re.compile(r"(?<![\w:./])/[A-Za-z0-9._-]+/[^\s\"';,]*"),
)

#: Top-level directories an absolute POSIX path really starts with, for the
#: binary rule below. Needed because a TorchScript archive is a zip, its member
#: names are ``encoder_ts/constants/256``, and one stray ``0x2f`` in the central
#: directory in front of one reads as an absolute path in every other respect.
_POSIX_ROOTS = "mnt|home|root|Users|users|media|srv|opt|var|tmp|data|scratch|work|workspace"

#: The same three shapes, for the printable runs inside a weight file. The text
#: patterns cannot be used there: over a 340 MB tensor blob they match float
#: noise dozens of times a file (``V:/aw:!ez``, ``L:\z{=%0``, ``/C/>`BO=``) and
#: they match the archive's own zip members, and a build gate that fails on
#: every clean bundle is a gate someone switches off. So a segment has to look
#: like a directory name, there have to be at least two of them, and an absolute
#: POSIX path has to start somewhere a filesystem actually does.
#:
#: This is deliberately less thorough than the text rules. A weight file is a
#: pickle of tensors and a frozen graph; the paths were in the JSON and the
#: YAML, and those are scanned exactly.
_BINARY_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\\\\[A-Za-z0-9._-]+(?:\\[A-Za-z0-9 ._-]+){2,}"),
    re.compile(r"(?<![\w:])[A-Za-z]:[\\/][A-Za-z0-9 ._-]+(?:[\\/][A-Za-z0-9 ._-]+)+"),
    re.compile(rf"(?<![\w:./])/(?:{_POSIX_ROOTS})(?:/[A-Za-z0-9 ._-]+){{2,}}"),
)

#: A bare drive letter in prose: "synthesised from the V: mirror". Redacted out
#: of string *values*, but deliberately not part of what a file scan looks for:
#: a serialised YAML mapping is full of one-letter keys (``n: 4``) that this
#: cannot be told apart from, so as a file-level rule it would fail every clean
#: bundle. Redaction may be stricter than detection; the reverse would be a
#: build that cries wolf until someone turns the check off.
_DRIVE_MENTION = re.compile(r"(?<![\w:])[A-Za-z]:(?=\s|$)")

#: Files scanned as text, exactly. Everything else goes through the binary
#: rules above: only its long printable runs are read, and only path shapes
#: within them count.
_TEXT_SUFFIXES = frozenset({".json", ".yaml", ".yml", ".txt", ".md", ".csv", ".sha256"})

#: Printable ASCII, long enough that finding one in float noise is not routine:
#: at 0.37 printable bytes per byte, a 24-character run turns up about once per
#: 16 GB of tensor data, while every path worth catching is longer than that and
#: contiguous.
_PRINTABLE_RUN = re.compile(rb"[\x20-\x7e]{24,}")

#: Shortest match reported from a binary.
_MIN_BINARY_MATCH = 12

#: Read size for scanning, and the overlap that keeps a match from being split
#: across two chunks.
_SCAN_CHUNK = 8 * 1024 * 1024
_SCAN_OVERLAP = 4096


def redact_local_paths(text: str) -> str:
    """Replace anything in ``text`` that names this machine with :data:`REDACTED`.

    For a single string *value*, never for a serialised document -- see
    :data:`_DRIVE_MENTION`.
    """
    for pattern in (*_LOCAL_PATH_PATTERNS, _DRIVE_MENTION):
        text = pattern.sub(REDACTED, text)
    return text


def find_local_paths(
    text: str,
    *,
    patterns: tuple[re.Pattern[str], ...] = _LOCAL_PATH_PATTERNS,
    min_len: int = 0,
) -> list[str]:
    """Every span of ``text`` that names a path or a host.

    The check behind :func:`redact_local_paths`, and the reason a bundle can be
    verified rather than trusted: ``find_local_paths(redact_local_paths(s))`` is
    empty for every ``s``.
    """
    found: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            hit = match.group(0)
            if len(hit) >= min_len and hit not in found:
                found.append(hit)
    return found


def encoder_run_dir(raw: str | None) -> tuple[str, str]:
    """``(run dir as it may ship, run id)`` for a config's ``encoder.run_dir``.

    Half the released configs name the encoder run relatively
    (``foundation_weights/m1_dinov3_vitb``) and half name it by its absolute
    path inside the training container (``/root/dino/foundation_weights/...``).
    The directory's *name* is the run id either way and is the thing that
    identifies the encoder; its parents, when they are a path on some machine,
    are not, and an absolute run dir is therefore reduced to the name alone
    rather than redacted -- a bundle that said ``run_dir: <removed>`` would have
    thrown away the one field it was supposed to keep.
    """
    run_dir = str(raw or "").replace("\\", "/")
    run_id = PurePosixPath(run_dir).name if run_dir else ""
    if run_dir and redact_local_paths(run_dir) != run_dir:
        run_dir = run_id
    return run_dir, run_id


def _scrub(value: Any) -> Any:
    """Recursively redact every string *value* in a parsed document."""
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, str):
        return redact_local_paths(value)
    return value


def sanitise_checkpoint_index(text: str) -> str:
    """A pack's ``checkpoint_index.json`` as it should ship.

    The index is the provenance record for *which encoder run and step* a head
    was trained against, and that is all of it a stranger can use. The recorded
    ``path`` of each checkpoint is the research machine's -- installation
    resolves those by basename anyway
    (:func:`quantem.registry.install.resolve_encoder_file`) -- so the path is
    reduced to that basename, which still carries the step, and every remaining
    string is redacted.
    """
    raw = json.loads(text)
    for record in raw.get("checkpoints") or []:
        if not isinstance(record, dict):
            continue
        recorded = str(record.get("path") or "")
        if recorded:
            record["path"] = PurePosixPath(recorded.replace("\\", "/")).name
    return json.dumps(_scrub(raw), indent=2, sort_keys=True) + "\n"


def sanitise_resolved_config(text: str) -> str:
    """A pack's ``resolved_config.yaml`` as it should ship.

    Two keys are dropped outright rather than redacted, because a redaction
    marker where a path used to be is an invitation to wonder what it was:
    ``data.data_root`` (where the training corpus sat) and the top-level
    ``config_path`` (where the experiment YAML sat inside the training
    container). Neither is read by inference --
    :class:`quantem.inference._fig3.schema.HeadConfig` takes ``encoder``,
    ``neck``, ``decoder`` and ``data.num_classes``, and sets ``config_path`` to
    the file it just loaded -- and the rest of the config is kept so the shipped
    head still documents how it was trained.

    ``encoder.run_dir`` is reduced rather than redacted: see
    :func:`encoder_run_dir`.
    """
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise BundleError("resolved_config.yaml is not a YAML mapping")
    raw.pop("config_path", None)
    data = raw.get("data")
    if isinstance(data, dict):
        data.pop("data_root", None)
    enc = raw.get("encoder")
    if isinstance(enc, dict) and enc.get("run_dir"):
        enc["run_dir"] = encoder_run_dir(enc["run_dir"])[0]
    return yaml.safe_dump(
        _scrub(raw), sort_keys=False, default_flow_style=False, allow_unicode=True
    )


#: role -> the function that turns the maintainer's copy into the shipped one.
_SANITISERS: dict[str, Callable[[str], str]] = {
    "config": sanitise_resolved_config,
    "index": sanitise_checkpoint_index,
}


def sanitise_pack_file(role: str, text: str) -> str:
    """Dispatch to the sanitiser for ``role``; identity for roles that have none."""
    fn = _SANITISERS.get(role)
    return fn(text) if fn is not None else text


def _sanitise_in_place(path: Path, role: str) -> None:
    """Rewrite a file already in a bundle, if sanitising changes it.

    Needed by ``--skip-existing``: a pack directory left by an earlier run holds
    verbatim copies, and a resumed build must produce the same bundle a fresh one
    would. Sanitising is idempotent, so a directory this has already touched is
    left alone.
    """
    if not path.is_file():
        return
    original = path.read_text(encoding="utf-8")
    cleaned = sanitise_pack_file(role, original)
    if cleaned != original:
        _write_text(path, cleaned)


def scan_file_for_local_paths(path: str | Path) -> list[str]:
    """Every build-machine path, host or drive still readable in one file.

    Text files are checked exactly, and that is where the leaks actually were.
    Anything else is checked through the long printable runs in it under the
    stricter :data:`_BINARY_PATH_PATTERNS` -- which is what makes the answer
    meaningful for a 341 MB TorchScript archive, where the interesting strings
    (module names, the embedded metadata JSON) are printable, long and
    path-shaped only if something went wrong, and everything else is float
    noise.
    """
    path = Path(path)
    if path.suffix.lower() in _TEXT_SUFFIXES:
        return find_local_paths(path.read_text(encoding="utf-8", errors="replace"))

    found: list[str] = []
    for run in _printable_runs(path):
        hits = find_local_paths(
            run, patterns=_BINARY_PATH_PATTERNS, min_len=_MIN_BINARY_MATCH
        )
        for hit in hits:
            if hit not in found:
                found.append(hit)
    return found


def _printable_runs(path: Path) -> Iterator[str]:
    """Printable-ASCII runs in a file, read in chunks with an overlap.

    The overlap is what keeps a run that straddles a chunk boundary from being
    seen as two short ones. It also means a run inside the overlap is yielded
    twice; the caller deduplicates, and reporting a leak twice would be the
    harmless direction anyway.
    """
    tail = b""
    with open(path, "rb") as fh:
        while chunk := fh.read(_SCAN_CHUNK):
            buf = tail + chunk
            for match in _PRINTABLE_RUN.finditer(buf):
                yield match.group(0).decode("ascii", "replace")
            tail = buf[-_SCAN_OVERLAP:]


def scan_bundle_for_local_paths(
    root: str | Path, *, on_progress: StatusFn | None = None
) -> dict[str, list[str]]:
    """Check a whole bundle tree. ``{relative path: offending snippets}``.

    Empty means the README's claim -- that the bundle does not refer to the
    machine that built it -- is true of these bytes. This is run at the end of
    every :func:`build_bundle` and is available to a downloader as
    ``release scan``.
    """
    root = Path(root).expanduser()
    offenders: dict[str, list[str]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if on_progress is not None:
            on_progress(f"scanning {rel}")
        hits = scan_file_for_local_paths(path)
        if hits:
            offenders[rel] = hits
    return offenders


def _describe_offenders(offenders: dict[str, list[str]], *, limit: int = 6) -> str:
    lines = []
    for rel, hits in offenders.items():
        shown = ", ".join(hits[:limit])
        more = f" (+{len(hits) - limit} more)" if len(hits) > limit else ""
        lines.append(f"  {rel}: {shown}{more}")
    return "\n".join(lines)


# --- The format -------------------------------------------------------------


@dataclass(frozen=True)
class BundleFile:
    """One file in a bundle, addressed by digest.

    ``path`` is relative to the bundle root and always uses forward slashes, so
    a manifest written on Windows verifies unchanged on Linux.
    """

    path: str
    sha256: str
    size_bytes: int

    def to_json(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> BundleFile:
        return cls(
            path=str(raw["path"]).replace("\\", "/"),
            sha256=str(raw["sha256"]).lower(),
            size_bytes=int(raw["size_bytes"]),
        )


@dataclass(frozen=True)
class BundlePack:
    """One pack's entry in a bundle manifest."""

    pack_id: str
    dirname: str
    #: role -> filename inside the pack directory.
    files: dict[str, str]
    architecture: dict[str, Any] = field(default_factory=dict)

    def dir_path(self, root: Path) -> Path:
        return Path(root) / PACKS_DIRNAME / self.dirname

    def file_path(self, root: Path, role: str) -> Path | None:
        name = self.files.get(role)
        return self.dir_path(root) / name if name else None

    def manifest_path(self, role: str) -> str | None:
        name = self.files.get(role)
        return f"{PACKS_DIRNAME}/{self.dirname}/{name}" if name else None

    def to_json(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "dir": f"{PACKS_DIRNAME}/{self.dirname}",
            "files": dict(self.files),
            "architecture": dict(self.architecture),
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> BundlePack:
        dirname = str(raw.get("dir") or "").replace("\\", "/").rstrip("/")
        dirname = dirname.rsplit("/", 1)[-1] or cache.pack_dirname(str(raw["pack_id"]))
        return cls(
            pack_id=str(raw["pack_id"]),
            dirname=dirname,
            files={str(k): str(v) for k, v in (raw.get("files") or {}).items()},
            architecture=dict(raw.get("architecture") or {}),
        )


@dataclass(frozen=True)
class Bundle:
    """A parsed ``MANIFEST.json`` plus the directory it was read from."""

    root: Path
    release: str
    generated_at: str
    packs: list[BundlePack]
    files: list[BundleFile]
    schema_version: int = BUNDLE_SCHEMA_VERSION
    generated_by: dict[str, Any] = field(default_factory=dict)

    @property
    def pack_ids(self) -> list[str]:
        return [p.pack_id for p in self.packs]

    def pack(self, pack_id: str) -> BundlePack:
        for p in self.packs:
            if p.pack_id == pack_id:
                return p
        raise BundleError(
            f"{pack_id} is not in this bundle ({self.root}). It holds: "
            f"{', '.join(self.pack_ids) or '(nothing)'}."
        )

    def file(self, rel_path: str) -> BundleFile:
        wanted = rel_path.replace("\\", "/")
        for f in self.files:
            if f.path == wanted:
                return f
        raise BundleError(f"{rel_path} is not listed in {MANIFEST_NAME} under {self.root}.")

    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)


def _wrong_directory_hint(root: Path) -> str:
    """Say where the bundle probably is, for the three ways users miss it.

    People point an installer at the zip, at their Downloads folder, and at the
    ``packs/`` subdirectory. Each of those is one sentence away from working, and
    the sentence is worth more than a correct-but-silent refusal.
    """
    if root.is_file():
        return " Unzip it first and point at the unzipped directory."
    if root.name == PACKS_DIRNAME and (root.parent / MANIFEST_NAME).is_file():
        return f" Point it one level up, at {root.parent}."
    try:
        nested = sorted(c for c in root.glob(f"*/{MANIFEST_NAME}") if c.is_file())
    except OSError:
        nested = []
    if nested:
        return f" Did you mean {nested[0].parent}?"
    return ""


def read_bundle(root: str | Path) -> Bundle:
    """Parse the manifest of a downloaded, unzipped bundle.

    Raises:
        BundleError: when the directory is not a bundle, naming what was
            expected. Users point installers at their Downloads folder, at the
            zip, and at the ``packs/`` subdirectory; each of those must produce
            a sentence that says what to do instead.
    """
    root = Path(root).expanduser()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise BundleError(
            f"No {MANIFEST_NAME} in {root}; that is not a QuantEM model bundle."
            + _wrong_directory_hint(root)
        )

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise BundleError(f"{manifest_path} is not valid JSON: {exc}") from exc

    if str(raw.get("kind")) != BUNDLE_KIND:
        raise BundleError(
            f"{manifest_path} declares kind={raw.get('kind')!r}, not {BUNDLE_KIND!r}."
        )
    version = int(raw.get("schema_version") or 0)
    if version > BUNDLE_SCHEMA_VERSION:
        raise BundleError(
            f"{manifest_path} is schema version {version}; this QuantEM understands "
            f"up to {BUNDLE_SCHEMA_VERSION}. Upgrade QuantEM to install this release."
        )

    return Bundle(
        root=root,
        release=str(raw.get("release") or "unknown"),
        generated_at=str(raw.get("generated_at") or ""),
        schema_version=version,
        generated_by=dict(raw.get("generated_by") or {}),
        packs=[BundlePack.from_json(p) for p in (raw.get("packs") or [])],
        files=[BundleFile.from_json(f) for f in (raw.get("files") or [])],
    )


def verify_bundle(
    root: str | Path,
    *,
    pack_ids: list[str] | None = None,
    on_progress: StatusFn | None = None,
) -> dict[str, bool]:
    """Re-hash a bundle against its own manifest. ``{relative path: matches}``.

    Run before publishing and again after downloading. A missing file is a
    False, not an exception, so one call reports every problem rather than the
    first one.
    """
    bundle = read_bundle(root)
    root = bundle.root
    wanted: set[str] | None = None
    if pack_ids is not None:
        wanted = set()
        for pack_id in pack_ids:
            pack = bundle.pack(pack_id)
            wanted.update(
                p for p in (pack.manifest_path(r) for r in pack.files) if p is not None
            )

    results: dict[str, bool] = {}
    for entry in bundle.files:
        if wanted is not None and entry.path not in wanted:
            continue
        path = root / entry.path
        if on_progress is not None:
            on_progress(f"verifying {entry.path}")
        if not path.is_file() or path.stat().st_size != entry.size_bytes:
            results[entry.path] = False
            continue
        results[entry.path] = cache.sha256_file(path) == entry.sha256
    return results


# --- Building ---------------------------------------------------------------


@dataclass(frozen=True)
class BuiltPack:
    pack_id: str
    dirname: str
    bytes_written: int
    export_max_abs_diff: float
    export_dynamic_spatial: bool
    export_source_tier: str


@dataclass(frozen=True)
class BuildReport:
    root: Path
    release: str
    packs: list[BuiltPack]
    failures: dict[str, str]

    @property
    def total_bytes(self) -> int:
        return sum(p.bytes_written for p in self.packs)


def _env_paths(name: str) -> list[Path]:
    raw = os.environ.get(name, "").strip()
    return [Path(p) for p in raw.split(os.pathsep) if p.strip()] if raw else []


def default_heads_root() -> Path | None:
    """Maintainer's heads root from the environment, or None. No built-in path."""
    raw = os.environ.get(HEADS_ROOT_ENV_VAR, "").strip()
    return Path(raw) if raw else None


def default_weights_root() -> Path | None:
    raw = os.environ.get(WEIGHTS_ROOT_ENV_VAR, "").strip()
    return Path(raw) if raw else None


def default_search_dirs() -> list[Path]:
    return _env_paths(SEARCH_DIRS_ENV_VAR)


def _copy_into(src: Path, dst: Path) -> int:
    """Copy ``src`` to ``dst`` through a ``.partial``, returning bytes written.

    Via a temporary name so an interrupted build leaves nothing that looks like
    a finished file -- the manifest is written last, but a half-copied
    ``head.pt`` beside a stale manifest would verify as a mismatch rather than
    as the absence it is.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copy2(src, tmp)
    tmp.replace(dst)
    return dst.stat().st_size


def _copy_for_release(src: Path, dst: Path, role: str) -> int:
    """Put one source file into the bundle, sanitised if its role calls for it.

    The two text files a pack ships are the maintainer's training outputs and
    are rewritten, not copied: see the module docstring. Weights are copied
    byte for byte -- an artifact that does not reproduce the published model is
    the one thing worse than a leaked path.
    """
    if role in SANITISED_ROLES:
        return _write_text(
            dst, sanitise_pack_file(role, Path(src).read_text(encoding="utf-8"))
        )
    return _copy_into(src, dst)


def build_bundle(
    dest: str | Path,
    *,
    heads_root: str | Path,
    weights_root: str | Path,
    search_dirs: list[Path] | None = None,
    pack_ids: list[str] | None = None,
    release: str | None = None,
    device: str = "cpu",
    tolerance: float | None = None,
    overwrite: bool = False,
    skip_existing: bool = False,
    keep_going: bool = False,
    on_progress: StatusFn | None = None,
) -> BuildReport:
    """Turn the maintainer's training outputs into a self-contained bundle.

    One command, run once per release on a machine that has whatever each pack's
    encoder needs to be built eagerly (``timm`` for OmniEM, Meta's ``dinov3`` for
    QuantEM). The result is the directory that goes to Hugging Face and Zenodo.

    Args:
        dest: bundle root to create.
        heads_root: directory holding ``<organelle>_<family>/head.pt``.
        weights_root: directory holding ``<run_id>/checkpoint_index.json``.
        search_dirs: extra directories to find encoder checkpoint files in. The
            paths recorded inside a ``checkpoint_index.json`` are the research
            machine's and are resolved by basename; see
            :func:`quantem.registry.install.resolve_encoder_file`.
        pack_ids: subset to build. Default: all eight.
        release: version string recorded in the manifest. Default: the app's.
        device: where to build and verify each export.
        tolerance: max |diff| allowed between exported and eager encoder.
        overwrite: replace files already in ``dest``.
        skip_existing: reuse a pack directory already complete in ``dest``,
            re-hashing rather than re-exporting it. For resuming a build that
            died four packs in -- not for assembling a bundle by hand.
        keep_going: record a pack's failure and continue instead of raising.

    Raises:
        BundleError: naming the pack and the missing piece.
    """
    from quantem.inference.export import (
        DEFAULT_TOLERANCE,
        ExportError,
        export_encoder_files,
    )
    from quantem.registry.install import head_dirname, resolve_encoder_file

    tolerance = DEFAULT_TOLERANCE if tolerance is None else float(tolerance)
    dest = Path(dest).expanduser()
    heads_root = Path(heads_root).expanduser()
    weights_root = Path(weights_root).expanduser()
    extra_dirs = [Path(d) for d in (search_dirs or [])]
    ids = list(pack_ids or sorted(ARCHITECTURE))
    unknown = [p for p in ids if p not in ARCHITECTURE]
    if unknown:
        raise BundleError(f"unknown pack id(s) {unknown}; known: {sorted(ARCHITECTURE)}")
    if not heads_root.is_dir():
        raise BundleError(f"--heads-root {heads_root} is not a directory.")
    if not weights_root.is_dir():
        raise BundleError(f"--weights-root {weights_root} is not a directory.")

    from quantem import __version__ as app_version

    release = str(release or app_version)

    def say(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    dest.mkdir(parents=True, exist_ok=True)
    built: list[BuiltPack] = []
    failures: dict[str, str] = {}
    manifest_files: list[BundleFile] = []
    manifest_packs: list[BundlePack] = []

    for pack_id in ids:
        dirname = cache.pack_dirname(pack_id)
        pack_out = dest / PACKS_DIRNAME / dirname
        try:
            if skip_existing and _pack_dir_complete(pack_out):
                say(f"{pack_id}: reusing {pack_out} (--skip-existing)")
                reused_files, reused_pack = _hash_existing_pack(pack_id, pack_out, dirname)
                manifest_files.extend(reused_files)
                manifest_packs.append(reused_pack)
                built.append(
                    BuiltPack(
                        pack_id=pack_id,
                        dirname=dirname,
                        bytes_written=0,
                        export_max_abs_diff=float("nan"),
                        export_dynamic_spatial=False,
                        export_source_tier="reused",
                    )
                )
                continue

            sources, run_dir, run_id, step = _locate_sources(
                pack_id,
                heads_root=heads_root,
                weights_root=weights_root,
                search_dirs=extra_dirs,
                head_dirname=head_dirname,
                resolve_encoder_file=resolve_encoder_file,
            )

            pack_out.mkdir(parents=True, exist_ok=True)
            written = 0
            copied: dict[str, Path] = {}
            for role, src in (
                ("head", sources.head_path),
                ("config", sources.config_path),
                ("index", sources.index_path),
            ):
                if src is None:
                    continue
                target = pack_out / PACK_FILE_ROLES[role]
                if target.exists() and not overwrite:
                    raise BundleError(
                        f"{target} already exists; pass --overwrite to rebuild this bundle "
                        "or --skip-existing to resume one."
                    )
                say(f"{pack_id}: copying {PACK_FILE_ROLES[role]}")
                written += _copy_for_release(Path(src), target, role)
                copied[role] = target

            export_target = pack_out / PACK_FILE_ROLES["export"]
            say(f"{pack_id}: exporting encoder (this is the slow part)")
            result = export_encoder_files(
                pack_id,
                sources,
                output=export_target,
                tolerance=tolerance,
                device=device,
                overwrite=overwrite,
            )
            written += result.size_bytes
            copied["export"] = export_target

            entries: list[BundleFile] = []
            digests: dict[str, dict[str, Any]] = {}
            for role, path in copied.items():
                rel = f"{PACKS_DIRNAME}/{dirname}/{path.name}"
                say(f"{pack_id}: hashing {path.name}")
                digest = cache.sha256_file(path)
                size = path.stat().st_size
                entries.append(BundleFile(path=rel, sha256=digest, size_bytes=size))
                digests[role] = {
                    "filename": path.name,
                    "sha256": digest,
                    "size_bytes": size,
                }

            descriptor_path = _write_descriptor(
                pack_id,
                pack_out,
                release=release,
                digests=digests,
                encoder_run_dir=run_dir,
                encoder_run_id=run_id,
                checkpoint_step=step,
                export=result,
            )
            entries.append(
                BundleFile(
                    path=f"{PACKS_DIRNAME}/{dirname}/{PACK_DESCRIPTOR_NAME}",
                    sha256=cache.sha256_file(descriptor_path),
                    size_bytes=descriptor_path.stat().st_size,
                )
            )
            written += descriptor_path.stat().st_size

            manifest_files.extend(entries)
            manifest_packs.append(
                BundlePack(
                    pack_id=pack_id,
                    dirname=dirname,
                    files={role: PACK_FILE_ROLES[role] for role in copied},
                    architecture=dict(ARCHITECTURE[pack_id]),
                )
            )
            built.append(
                BuiltPack(
                    pack_id=pack_id,
                    dirname=dirname,
                    bytes_written=written,
                    export_max_abs_diff=result.max_abs_diff,
                    export_dynamic_spatial=result.dynamic_spatial,
                    export_source_tier=result.source_tier,
                )
            )
            say(
                f"{pack_id}: done ({written / 1e6:.0f} MB, "
                f"max|diff|={result.max_abs_diff:.2e} from tier {result.source_tier})"
            )
        except (BundleError, ExportError, FileNotFoundError, RuntimeError) as exc:
            if not keep_going:
                raise BundleError(f"{pack_id}: {exc}") from exc
            logger.warning("bundle build failed for %s: %s", pack_id, exc)
            failures[pack_id] = str(exc)

    readme = _write_readme(dest, release=release, packs=manifest_packs)
    manifest_files.append(
        BundleFile(
            path=README_NAME,
            sha256=cache.sha256_file(readme),
            size_bytes=readme.stat().st_size,
        )
    )

    manifest_path = _write_manifest(
        dest,
        release=release,
        packs=manifest_packs,
        files=sorted(manifest_files, key=lambda f: f.path),
    )

    # The last gate, and the reason the README may say what it says. Every byte
    # that was just written is read back and checked; on a hit the manifest is
    # removed again, so what is left on disk cannot be read, installed, or
    # mistaken for something publishable.
    say("checking the bundle names no path on this machine")
    offenders = scan_bundle_for_local_paths(dest, on_progress=on_progress)
    if offenders:
        manifest_path.unlink(missing_ok=True)
        (dest / MANIFEST_DIGEST_NAME).unlink(missing_ok=True)
        raise BundleError(
            f"{len(offenders)} file(s) in {dest} still name this machine, so the bundle "
            f"was not finished and its {MANIFEST_NAME} has been removed. A release is "
            "published under a real name and must not carry the build box's paths or "
            "host names:\n"
            + _describe_offenders(offenders)
            + f"\nRewriting happens in sanitise_pack_file(); a role that needs it and "
            f"does not have it belongs in {SANITISED_ROLES!r}."
        )

    return BuildReport(root=dest, release=release, packs=built, failures=failures)


def _pack_dir_complete(pack_out: Path) -> bool:
    names = [PACK_FILE_ROLES[r] for r in REQUIRED_ROLES] + [PACK_DESCRIPTOR_NAME]
    return all((pack_out / name).is_file() for name in names)


def _hash_existing_pack(
    pack_id: str, pack_out: Path, dirname: str
) -> tuple[list[BundleFile], BundlePack]:
    """Manifest entries for a pack directory that a previous run finished.

    The two sanitised files are rewritten first. A directory left by an earlier
    build may hold verbatim training outputs, and "resume" must mean the same
    bundle a fresh build would produce, not a bundle with a hole in it.
    """
    for role in SANITISED_ROLES:
        _sanitise_in_place(pack_out / PACK_FILE_ROLES[role], role)

    entries: list[BundleFile] = []
    roles: dict[str, str] = {}
    for role, name in PACK_FILE_ROLES.items():
        path = pack_out / name
        if not path.is_file():
            continue
        roles[role] = name
        entries.append(
            BundleFile(
                path=f"{PACKS_DIRNAME}/{dirname}/{name}",
                sha256=cache.sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    descriptor = pack_out / PACK_DESCRIPTOR_NAME
    entries.append(
        BundleFile(
            path=f"{PACKS_DIRNAME}/{dirname}/{PACK_DESCRIPTOR_NAME}",
            sha256=cache.sha256_file(descriptor),
            size_bytes=descriptor.stat().st_size,
        )
    )
    return entries, BundlePack(
        pack_id=pack_id,
        dirname=dirname,
        files=roles,
        architecture=dict(ARCHITECTURE[pack_id]),
    )


def _locate_sources(
    pack_id: str,
    *,
    heads_root: Path,
    weights_root: Path,
    search_dirs: list[Path],
    head_dirname: Callable[[str], str],
    resolve_encoder_file: Callable[..., Path],
) -> tuple[Any, str, str, int | None]:
    """Find one pack's four source files under the maintainer's roots.

    Returns ``(EncoderSources, encoder run dir, encoder run id, checkpoint
    step)``. Those three values are what goes into the bundle in place of the
    absolute paths, which are the build box's and have no meaning to anyone
    else; the run dir is kept only while it stays relative, because a config
    that named an absolute one would be naming this machine.
    """
    from quantem.inference._fig3.schema import load_head_config
    from quantem.inference.export import EncoderSources
    from quantem.inference.specs import MODEL_SPECS

    head_dir = heads_root / head_dirname(pack_id)
    head_path = head_dir / cache.HEAD_NAME
    config_path = head_dir / cache.CONFIG_NAME
    for path in (head_path, config_path):
        if not path.is_file():
            raise BundleError(f"missing {path}. Is --heads-root right?")

    cfg = load_head_config(config_path)
    run_dir, run_id = encoder_run_dir(cfg.encoder.run_dir)
    if not run_id:
        raise BundleError(
            f"{config_path} names no encoder run_dir, so there is no "
            f"{cache.INDEX_NAME} to take the architecture from."
        )
    index_path = weights_root / run_id / cache.INDEX_NAME
    if not index_path.is_file():
        raise BundleError(
            f"no {index_path}. The pack was trained against encoder run {run_id!r}; "
            "point --weights-root at the directory holding that run."
        )

    step = cfg.encoder.checkpoint_step
    encoder_path: Path | None = None
    dirs = [*search_dirs, index_path.parent]
    if MODEL_SPECS[pack_id].embeds_encoder:
        # `adapt: full` -- the head is a whole fine-tuned ViT-B. The shared blob
        # is not needed and asking for it would make the build fail on a machine
        # that legitimately does not have it.
        try:
            encoder_path = resolve_encoder_file(index_path, step, dirs)
        except Exception:  # noqa: BLE001 -- optional for this pack by construction
            logger.debug("%s: no shared encoder blob; using the head's own", pack_id)
    else:
        encoder_path = resolve_encoder_file(index_path, step, dirs)

    return (
        EncoderSources(
            head_path=head_path,
            config_path=config_path,
            index_path=index_path,
            encoder_path=encoder_path,
        ),
        run_dir,
        run_id,
        step,
    )


def _write_descriptor(
    pack_id: str,
    pack_out: Path,
    *,
    release: str,
    digests: dict[str, dict[str, Any]],
    encoder_run_dir: str,
    encoder_run_id: str,
    checkpoint_step: int | None,
    export: Any,
) -> Path:
    """Write ``pack.json`` -- what this pack is, for a human and for the installer.

    The ``encoder`` block is the whole provenance a bundle install has: it is
    where ``encoder_run_dir`` and ``checkpoint_step`` come from in the install
    record, and from there in the analysis manifest. It has to be here because
    the alternative -- reading them back out of the shipped
    ``checkpoint_index.json`` -- is reading them out of a file whose paths have
    just been removed.
    """
    from quantem.inference.specs import MODEL_SPECS
    from quantem.registry.catalogue import pack_licence, pack_notes, pack_title

    spec = MODEL_SPECS[pack_id]
    body = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "pack_id": pack_id,
        "release": release,
        "title": pack_title(spec),
        "family": spec.family,
        "organelle": spec.organelle,
        "architecture": dict(ARCHITECTURE[pack_id]),
        "canonical_nm": spec.canonical_nm,
        "tile_size": spec.tile_size,
        "default_threshold": spec.threshold,
        "licence": pack_licence(spec.family),
        "notes": pack_notes(spec.family),
        "encoder": {
            "run_dir": encoder_run_dir,
            "run_id": encoder_run_id,
            "checkpoint_step": checkpoint_step,
            "tier_exported_from": export.source_tier,
            "traced_tile": export.traced_tile,
            "dynamic_spatial": export.dynamic_spatial,
            "max_abs_diff_vs_eager": export.max_abs_diff,
        },
        "provenance_note": (
            f"{cache.CONFIG_NAME} and {cache.INDEX_NAME} in this pack are the training "
            "outputs with the build machine's absolute paths, host names and "
            f"{cache.CONFIG_NAME}'s data_root removed, so their digests differ from the "
            "maintainer's copies. Which encoder run and checkpoint step the head was "
            "trained against -- the part that identifies the model -- is in `encoder` "
            "above, and the weights themselves are byte-for-byte."
        ),
        "files": digests,
    }
    path = pack_out / PACK_DESCRIPTOR_NAME
    _write_text(path, json.dumps(body, indent=2, sort_keys=True) + "\n")
    return path


def _write_manifest(
    dest: Path, *, release: str, packs: list[BundlePack], files: list[BundleFile]
) -> Path:
    body = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "release": release,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generated_by": _build_environment(),
        "packs": [p.to_json() for p in packs],
        "files": [f.to_json() for f in files],
        "total_bytes": sum(f.size_bytes for f in files),
    }
    path = dest / MANIFEST_NAME
    _write_text(path, json.dumps(body, indent=2, sort_keys=True) + "\n")
    # The manifest cannot contain its own digest, so it gets a sidecar. This is
    # the one value a publisher should quote on the download page: check it and
    # every other digest in the tree follows.
    _write_text(dest / MANIFEST_DIGEST_NAME, f"{cache.sha256_file(path)}  {MANIFEST_NAME}\n")
    return path


def _build_environment() -> dict[str, Any]:
    """What produced this bundle. Names versions, never paths or the user."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.system(),
    }
    try:
        import torch

        info["torch"] = str(torch.__version__)
    except Exception:  # pragma: no cover - torch is a hard dependency of a build
        pass
    try:
        from quantem import __version__ as app_version

        info["quantem"] = app_version
    except Exception:  # pragma: no cover
        pass
    return info


def _write_readme(dest: Path, *, release: str, packs: list[BundlePack]) -> Path:
    listing = "\n".join(f"  {p.pack_id}" for p in packs) or "  (none)"
    text = f"""\
QuantEM model release {release}
{"=" * (24 + len(release))}

This directory holds the pretrained segmentation models for QuantEM. It is
self-contained: it needs no network access and no research code to install or to
run, and it does not refer to the machine that built it.

Packs included:
{listing}

To install, from the directory you unzipped this into:

    quantem models install .

or, if the `quantem` command is not on your PATH:

    python -m quantem.registry.install bundle . --all

Add `--data-dir <path>` to install somewhere other than the default QuantEM data
directory. Check the download first with:

    python -m quantem.registry.release verify .

Every file here is listed in {MANIFEST_NAME} with its SHA-256. The digest of the
manifest itself is in {MANIFEST_DIGEST_NAME}; that is the single value worth
comparing against the one on the download page.

The claim above that this release names no machine is not one you have to take
on trust. Each pack's {cache.CONFIG_NAME} and {cache.INDEX_NAME} are the
maintainer's training outputs with absolute paths, host names and the training
data root removed -- which encoder run and checkpoint step a head was trained
against is kept, in the `encoder` block of its {PACK_DESCRIPTOR_NAME} -- and the
build re-reads its own output to confirm nothing was missed. To repeat that
check here:

    python -m quantem.registry.release scan .

The model weights carry their own licences -- see the `licence` and `notes`
fields in each pack's {PACK_DESCRIPTOR_NAME}, and NOTICE in the QuantEM
repository. They are not covered by QuantEM's MIT licence.
"""
    path = dest / README_NAME
    _write_text(path, text)
    return path


# --- CLI --------------------------------------------------------------------


def _cmd_build(args: argparse.Namespace) -> int:
    heads_root = Path(args.heads_root) if args.heads_root else default_heads_root()
    weights_root = Path(args.weights_root) if args.weights_root else default_weights_root()
    missing = []
    if heads_root is None:
        missing.append(f"--heads-root (or ${HEADS_ROOT_ENV_VAR})")
    if weights_root is None:
        missing.append(f"--weights-root (or ${WEIGHTS_ROOT_ENV_VAR})")
    if missing or heads_root is None or weights_root is None:
        print(
            "error: this command needs the release inputs it is building from: "
            + ", ".join(missing)
            + ".\nThere is no default: the training outputs live in a different place "
            "on every build machine.",
            file=sys.stderr,
        )
        return 2

    search = [Path(d) for d in (args.search_dir or [])] or default_search_dirs()
    report = build_bundle(
        args.out,
        heads_root=heads_root,
        weights_root=weights_root,
        search_dirs=search,
        pack_ids=args.packs or None,
        release=args.release,
        device=args.device,
        tolerance=args.tolerance,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        keep_going=args.keep_going,
        on_progress=(lambda msg: print(msg, file=sys.stderr)) if args.verbose else None,
    )
    for p in report.packs:
        print(
            f"{p.pack_id:18s} {p.bytes_written / 1e6:9.1f} MB  "
            f"max|diff|={p.export_max_abs_diff:.2e}  from={p.export_source_tier}"
        )
    for pack_id, reason in report.failures.items():
        print(f"{pack_id:18s} FAILED: {reason}", file=sys.stderr)
    print(
        f"\n{len(report.packs)} pack(s), {report.total_bytes / 1e9:.2f} GB in {report.root}"
    )
    print(
        f"release {report.release}; now run: "
        f"python -m quantem.registry.release verify {report.root}"
    )
    return 1 if report.failures else 0


def _cmd_verify(args: argparse.Namespace) -> int:
    results = verify_bundle(
        args.bundle,
        pack_ids=args.packs or None,
        on_progress=(lambda msg: print(msg, file=sys.stderr)) if args.verbose else None,
    )
    bad = [path for path, ok in results.items() if not ok]
    for path in sorted(results):
        print(f"{'OK  ' if results[path] else 'BAD '} {path}")
    print(f"\n{len(results) - len(bad)}/{len(results)} file(s) verified")
    if bad:
        print(f"{len(bad)} file(s) do not match the manifest", file=sys.stderr)

    # Hashes prove the bundle is intact; they say nothing about what is *in* it.
    # A bundle built before the sanitiser existed passes 41/41 while all eight
    # packs still name the lab's file server. ``verify`` is the one command the
    # README tells a downloader -- and the maintainer -- to run, so a leak has
    # to fail here rather than in a separate command nobody is required to
    # invoke. The alternative is publishing it once, permanently.
    offenders = scan_bundle_for_local_paths(
        args.bundle,
        on_progress=(lambda msg: print(msg, file=sys.stderr)) if args.verbose else None,
    )
    if offenders:
        print(
            f"\n{len(offenders)} file(s) name the machine that built this bundle:\n"
            f"{_describe_offenders(offenders)}\n\n"
            "This bundle must not be published. Rebuild it with "
            "`python -m quantem.registry.release build`, which sanitises these "
            "files as it copies them; their digests change with them.",
            file=sys.stderr,
        )
    return 1 if (bad or offenders) else 0


def _cmd_scan(args: argparse.Namespace) -> int:
    offenders = scan_bundle_for_local_paths(
        args.bundle,
        on_progress=(lambda msg: print(msg, file=sys.stderr)) if args.verbose else None,
    )
    if not offenders:
        print(f"clean: nothing in {args.bundle} names a path, host or drive.")
        return 0
    print(f"{len(offenders)} file(s) name the machine that built this bundle:")
    print(_describe_offenders(offenders))
    return 1


def _cmd_show(args: argparse.Namespace) -> int:
    bundle = read_bundle(args.bundle)
    print(f"release      {bundle.release}")
    print(f"generated    {bundle.generated_at}")
    print(f"built with   {bundle.generated_by}")
    print(f"total        {bundle.total_bytes / 1e9:.2f} GB in {len(bundle.files)} files")
    print()
    for pack in bundle.packs:
        arch = pack.architecture
        print(
            f"{pack.pack_id:18s} {arch.get('neck', '?'):16s} {arch.get('decoder', '?'):14s} "
            f"{arch.get('adapt', '?'):8s} tile={arch.get('tile', '?')} "
            f"files={sorted(pack.files)}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true")

    p = argparse.ArgumentParser(
        prog="python -m quantem.registry.release",
        description=(
            "Build and check QuantEM model release bundles. `build` is a maintainer "
            "command run once per release; `verify` is for anyone who downloaded one."
        ),
        parents=[common],
    )
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser(
        "build",
        parents=[common],
        help="assemble a release bundle from the training outputs",
        description=(
            "Assemble a self-contained bundle: per pack the head, its config, the "
            "encoder checkpoint index and an exported TorchScript encoder, plus a "
            "manifest with a sha256 for every file. The export is the slow part and "
            "is verified against the eager model before it is written. The config and "
            "the index are rewritten on the way in to drop this machine's paths and "
            "host names, and the finished tree is re-read to prove none are left."
        ),
    )
    b.add_argument("--out", required=True, help="bundle root to create")
    b.add_argument("packs", nargs="*", help="pack ids to build (default: all eight)")
    b.add_argument(
        "--heads-root",
        default=None,
        help=f"directory holding <organelle>_<family>/head.pt (or ${HEADS_ROOT_ENV_VAR})",
    )
    b.add_argument(
        "--weights-root",
        default=None,
        help=f"directory holding <run_id>/checkpoint_index.json (or ${WEIGHTS_ROOT_ENV_VAR})",
    )
    b.add_argument(
        "--search-dir",
        action="append",
        help=f"extra directory holding encoder checkpoint files (repeatable; or ${SEARCH_DIRS_ENV_VAR})",
    )
    b.add_argument("--release", default=None, help="version recorded in the manifest")
    b.add_argument("--device", default="cpu", help="build/verify device for the export")
    b.add_argument("--tolerance", type=float, default=None)
    b.add_argument("--overwrite", action="store_true", help="replace files already in --out")
    b.add_argument(
        "--skip-existing",
        action="store_true",
        help="reuse pack directories a previous run finished (resume a build)",
    )
    b.add_argument(
        "--keep-going",
        action="store_true",
        help="record a pack's failure and continue instead of stopping",
    )
    b.set_defaults(func=_cmd_build)

    v = sub.add_parser("verify", parents=[common], help="re-hash a bundle against its manifest")
    v.add_argument("bundle", help="the unzipped bundle directory")
    v.add_argument("packs", nargs="*", help="check only these pack ids")
    v.set_defaults(func=_cmd_verify)

    sc = sub.add_parser(
        "scan",
        parents=[common],
        help="check a bundle names no path, host or drive on the build machine",
        description=(
            "Re-read every byte of a bundle and report anything that still looks like "
            "a filesystem path, a UNC host or a drive letter. `build` runs this on its "
            "own output and refuses to finish a bundle that fails it; run it here to "
            "check a copy you downloaded, or one assembled by hand."
        ),
    )
    sc.add_argument("bundle", help="the unzipped bundle directory")
    sc.set_defaults(func=_cmd_scan)

    s = sub.add_parser("show", parents=[common], help="what is in a bundle")
    s.add_argument("bundle")
    s.set_defaults(func=_cmd_show)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    try:
        return int(args.func(args))
    except BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
