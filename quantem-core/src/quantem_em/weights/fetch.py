"""Weight download, verification and caching.

Bundling weights in the wheel is not an option: PyPI limits a distribution file to 100 MiB and a
project to 10 GiB cumulative, and its limit-increase policy names "large pre-trained machine
learning models" as an explicit denial reason. Every comparable plugin downloads at first use.

We use ``huggingface_hub`` rather than ``pooch`` for one decisive reason: **it is the only fetcher
in this ecosystem that resumes**. ``pooch/downloaders.py`` has no Range/206 handling at all, so a
dropped connection at 1.2 GB restarts from zero. It also gives SHA-256 ETag verification, atomic
``.incomplete`` staging, mid-download retry, ``HF_HUB_OFFLINE`` and a size probe before committing.

Cache resolution order
----------------------
1. ``$QUANTEM_MODEL_DIR`` — a flat directory of plainly-named files. This is the air-gap and
   shared-lab-drive path: an admin drops the files in and sets one variable.
2. ``$HF_HUB_CACHE`` / ``$HF_HOME`` — shared with every other HF-using tool on the machine.
3. the huggingface_hub default.

Deliberately **not** napari's ``user_cache_dir()``: it is scoped by a hash of ``sys.prefix``, so
every virtualenv would re-download everything. And never inside ``site-packages``, which upgrades
destroy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_REGISTRY_PATH = Path(__file__).with_name("registry.json")


class WeightsError(RuntimeError):
    pass


class WeightsUnavailableError(WeightsError):
    """Needed artifacts are absent and the network is unavailable or disallowed."""

    def __init__(self, message: str, missing: list[dict]):
        super().__init__(message)
        self.missing = missing


class WeightsCorruptError(WeightsError):
    """A cached file's digest does not match the registry."""


def load_registry() -> dict:
    path = os.environ.get("QUANTEM_MODEL_REGISTRY")
    p = Path(path) if path else _REGISTRY_PATH
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def offline() -> bool:
    return os.environ.get("QUANTEM_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    } or os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def local_dir() -> Path | None:
    """The flat side-load directory, if configured."""
    d = os.environ.get("QUANTEM_MODEL_DIR")
    return Path(d) if d else None


def revision(reg: dict | None = None) -> str:
    """Which commit of the weights repo this build resolves against.

    Not cosmetic. The digests in ``registry.json`` are pinned and verified on every load, so if a
    later release overwrites a filename on ``main``, every already-installed copy would start
    raising :class:`WeightsCorruptError` on a file it had been reading happily for months. Pinning
    the revision makes a new release additive: old wheels keep resolving the commit they shipped
    against, new wheels point at the new one.
    """
    return (reg or load_registry()).get("hf_revision") or "main"


def artifacts_for(spec) -> list[str]:
    """Artifact names a model needs, trunk first. ``quantem/er`` needs only its own."""
    names = []
    if spec.trunk_artifact:
        names.append(spec.trunk_artifact)
    names.append(spec.model_artifact)
    return names


def _entry(name: str, reg: dict | None = None) -> dict:
    reg = reg or load_registry()
    try:
        return reg["artifacts"][name]
    except KeyError:
        raise WeightsError(f"unknown artifact {name!r}") from None


def cached_path(name: str) -> Path | None:
    """Return a local path for ``name`` if present, without touching the network."""
    reg = load_registry()
    entry = _entry(name, reg)
    d = local_dir()
    if d is not None:
        p = d / entry["filename"]
        if p.is_file():
            return p
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    hit = try_to_load_from_cache(
        repo_id=reg["hf_repo"], filename=entry["filename"], revision=revision(reg)
    )
    return Path(hit) if isinstance(hit, str) else None


def is_cached(name: str) -> bool:
    return cached_path(name) is not None


def artifact_info(name: str) -> dict:
    """Everything the UI needs to describe an artifact *before* downloading it."""
    reg = load_registry()
    e = _entry(name, reg)
    return {
        "name": name,
        "filename": e["filename"],
        "bytes": e.get("bytes"),
        "sha256": e.get("sha256"),
        "repo": reg["hf_repo"],
        "revision": revision(reg),
        "url": f"https://huggingface.co/{reg['hf_repo']}/blob/{revision(reg)}/{e['filename']}",
        "license": e.get("license", "see the model's Hugging Face repository"),
        "description": e.get("description", ""),
        "cached": is_cached(name),
    }


def download_plan(specs) -> dict:
    """What a set of models would cost to make runnable. Never touches the network.

    Trunks are counted once even when several models share them, so the UI can say "this model adds
    26 MB because you already have the encoder".
    """
    needed: list[str] = []
    for spec in specs:
        for a in artifacts_for(spec):
            if a not in needed:
                needed.append(a)
    items = [artifact_info(a) for a in needed]
    missing = [i for i in items if not i["cached"]]
    total = sum(i["bytes"] or 0 for i in missing)
    return {
        "artifacts": items,
        "missing": missing,
        "download_bytes": total,
        "all_present": not missing,
    }


def ensure(names, *, progress=None, allow_network: bool = True) -> dict[str, Path]:
    """Make every named artifact available locally. Returns ``{name: path}``.

    ``progress`` is called as ``progress(done_bytes, total_bytes, label)`` with byte counts
    accumulated across **the whole set**, not per file: a user downloading a 1.24 GB model wants
    one bar that fills once, not two that each fill from zero.

    Raises :class:`WeightsUnavailableError` when something is missing and cannot be fetched, with
    enough detail for the caller to render a side-load instruction.
    """
    reg = load_registry()
    out: dict[str, Path] = {}
    missing: list[dict] = []

    to_fetch = [n for n in names if cached_path(n) is None]
    grand_total = sum(_entry(n, reg).get("bytes") or 0 for n in to_fetch)
    done_before = 0

    for name in names:
        p = cached_path(name)
        if p is not None:
            out[name] = _verify(name, p, reg)
            continue
        if offline() or not allow_network:
            missing.append(artifact_info(name))
            continue

        e = _entry(name, reg)
        label = e["filename"]
        base = done_before
        size = e.get("bytes") or 0

        def relay(done, total, _base=base, _size=size, _label=label):
            if progress is not None:
                # Clamp: hub's byte counter is advisory. Depending on whether the xet backend is
                # active it can report coarsely, restart at zero on a retry, or overshoot a
                # resumed file -- none of which should make the bar jump backwards or past 100 %.
                d = min(int(done), _size) if _size else int(done)
                progress(_base + d, grand_total or int(total or 0), _label)

        if progress is not None:
            progress(base, grand_total, label)  # show the file and 0 % before any bytes move
        out[name] = _download(name, reg, progress=relay)
        done_before += size
        if progress is not None:
            # The authoritative point. tqdm granularity varies by backend and the final update is
            # not guaranteed to arrive, so anchor on completion: this is the only thing that makes
            # the bar actually reach 100 %.
            progress(done_before, grand_total, label)

    if missing:
        raise WeightsUnavailableError(
            "Required model files are not available locally and downloading is disabled.\n"
            + "\n".join(f"  {m['filename']}  ({_fmt(m['bytes'])})  {m['url']}" for m in missing)
            + "\n\nPlace these files in a directory and set QUANTEM_MODEL_DIR to it.",
            missing,
        )
    return out


def _reporting_tqdm(on_update):
    """A tqdm subclass that forwards byte counts instead of drawing to a terminal.

    ``tqdm_class`` is the hub's supported hook for this and is present in every 1.x we target, so
    this needs no access to hub internals. It matters because the alternative -- hub's own bar --
    writes to stderr, which a GUI napari launch discards, leaving a multi-gigabyte download
    indistinguishable from a hang.
    """
    from huggingface_hub.utils import tqdm as hf_tqdm

    class _Reporter(hf_tqdm):
        def update(self, n=1):
            done = super().update(n)
            try:
                on_update(self.n or 0, self.total or 0)
            except Exception:  # a reporting failure must never break the download
                pass
            return done

    return _Reporter


def _download(name: str, reg: dict, *, progress=None) -> Path:
    from huggingface_hub import hf_hub_download

    e = _entry(name, reg)
    if not e.get("sha256"):
        raise WeightsError(
            f"artifact {name!r} has no recorded sha256. Refusing to download an unverifiable file; "
            "run weights/convert.py to populate the registry before publishing."
        )
    kw = {}
    if progress is not None:
        kw["tqdm_class"] = _reporting_tqdm(progress)
    path = hf_hub_download(
        repo_id=reg["hf_repo"], filename=e["filename"], revision=revision(reg), **kw
    )
    return _verify(name, Path(path), reg)


def _digest_of(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(name: str, path: Path, reg: dict) -> Path:
    """Check the digest on every use, not only after download.

    Verifying on a cache hit makes the cache self-healing: a truncated file is detected and can be
    re-fetched, instead of being trusted forever.
    """
    e = _entry(name, reg)
    want = e.get("sha256")
    if not want:
        return path
    got = _digest_of(path)
    if got != want:
        raise WeightsCorruptError(
            f"{path} failed verification.\n  expected sha256 {want}\n  actual   sha256 {got}\n"
            "Delete the file and re-download."
        )
    return path


def export_flat(names, dest, *, progress=None) -> dict[str, Path]:
    """Materialise artifacts as plainly-named files in one directory.

    This is what ``QUANTEM_MODEL_DIR`` reads, and the hub cache is not it: a normal download lands
    in ``<cache>/models--org--repo/blobs/<etag>`` with a symlinked ``snapshots`` tree, so copying
    "the downloaded files" to an offline machine produces a directory in which
    :func:`cached_path` finds nothing. Without this, the documented air-gap recipe cannot work.
    """
    import shutil

    reg = load_registry()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name in names:
        e = _entry(name, reg)
        target = dest / e["filename"]
        if not (target.is_file() and _digest_of(target) == e.get("sha256")):
            src = ensure([name], progress=progress)[name]
            shutil.copyfile(src, target)
        out[name] = _verify(name, target, reg)
    return out


def load_tensors(path) -> dict:
    """Read a safetensors artifact into a plain ``{name: tensor}`` dict."""
    from safetensors.torch import load_file

    return load_file(str(path))


def _fmt(n: int | None) -> str:
    """Format a byte count for a user.

    Decimal units (MB = 10^6), matching how browsers, Hugging Face and download managers report
    sizes -- so the number a user sees here is the number they see while it downloads.
    """
    if n is None:
        return "unknown size"
    if n == 0:
        return "0 B"
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1000 or unit == "GB":
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1000.0
    return f"{x:.1f} GB"


format_bytes = _fmt
