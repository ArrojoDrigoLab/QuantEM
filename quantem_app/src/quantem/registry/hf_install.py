"""Installing a pack from the Hugging Face repository: download, verify, convert, export.

This is the third install source beside the release bundle and the local path
(:mod:`quantem.registry.install`), and the only one that touches the network.
The published artifacts are **safetensors** (the release deliberately ships no
pickles), laid out per :mod:`quantem.registry.hf`; the app's runtime pack
format is the one the release bundles use -- ``head.pt`` +
``resolved_config.yaml`` + ``checkpoint_index.json`` + an exported
``encoder_ts.pt``. The gap between the two is closed here, at install time,
in five steps:

1. **Fetch the model card** and refuse anything unverifiable before the first
   weight byte moves.
2. **Download and verify**: the head against the card's sha256, the shared
   trunk against the repository's LFS object id at the pinned revision. A
   mismatch aborts naming both digests; nothing is installed.
3. **Convert** the flat safetensors into the pack shape. The head's
   ``neck.* / decoder.* / adapters.* / encoder.*`` keys become the
   ``head.pt`` dict the loader expects; a ``resolved_config.yaml`` and a
   ``checkpoint_index.json`` are synthesised from the card plus the pinned
   facts in :mod:`quantem.registry.manifest` (the published artifacts carry
   timm-named tensors, so the synthesised index declares the timm entry
   point -- for the QuantEM family that is the ``quantem_dinov3`` variant in
   :mod:`quantem.inference.encoders`).
4. **Export** the encoder to TorchScript (:mod:`quantem.inference.export`)
   inside the staging directory, so the installed pack lands at tier
   ``exported`` exactly like a bundle install. If the export fails the pack is
   still installed -- it runs through the eager timm tier -- and the failure is
   reported, not swallowed.
5. **Promote atomically.** Everything is assembled in a staging directory
   under the models root; the pack directory appears only when complete, so a
   download killed mid-file leaves no half-installed pack. Blobs go through
   the same content-addressed store as every other source, which is what makes
   a trunk shared by three packs cost one download and one copy.
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantem.registry import cache, hf
from quantem.registry.install import InstallError, StatusFn, _link_or_copy, store_blob
from quantem.registry.manifest import ARCHITECTURE, ENCODER_NORM

logger = logging.getLogger(__name__)

#: Filename the shared encoder trunk takes inside an HF-installed pack
#: directory. Deliberately not ``encoder.pth``: the bytes are a safetensors
#: file and the suffix is what tells the loader how to read it.
HF_ENCODER_NAME = "encoder.safetensors"

#: The synthesised ``checkpoint_index.json`` marks the QuantEM family's entry
#: point with this variant; :func:`quantem.inference.encoders.build_encoder`
#: dispatches on it.
QUANTEM_TIMM_VARIANT = "quantem_dinov3"

#: ``adapt`` value and params for each published adaptation name, exactly as
#: the released ``resolved_config.yaml`` files spell them.
_ADAPT_CONFIG: dict[str, tuple[str, dict[str, Any]]] = {
    "last_n": ("last_n", {"n": 4}),
    "full": ("full", {}),
    "lora": ("lora", {"rank": 8, "conv": False}),
}

#: Blocks each family's trunk leaves for the head to overlay. QuantEM's
#: ``last_n`` heads carry their own fine-tuned blocks 8-11; everything else
#: ships complete trunks.
_QUANTEM_OVERLAY_BLOCKS = (8, 9, 10, 11)

#: Anything install-transient older than this is debris: staging directories
#: from killed installs, and blobs no installed pack references. One window for
#: both, because both rest on the same assumption -- no install takes a day.
_STALE_SECONDS = 24 * 3600

#: Backoff before each promote retry, in seconds (bounded: ~30 s in total).
#: On Windows an antivirus or search-indexer scan routinely holds a transient
#: handle on a file that just finished writing -- a fresh 1.2 GB TorchScript
#: export is exactly what they open -- and renaming the staging directory then
#: fails with ``PermissionError: [WinError 5]`` until the handle is released.
#: Observed 4/4 on real ``omniem:ld`` installs; seconds-scale bounded retries
#: absorb it, and a file held forever still fails, loudly, after the schedule.
_PROMOTE_RETRY_DELAYS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 15.0)


@dataclass(frozen=True)
class HfInstalledPack:
    """What one HF install produced. Mirrors ``install.InstalledPack`` + provenance."""

    pack_id: str
    root: Path
    head_sha256: str
    encoder_sha256: str | None
    bytes_written: int
    reused_blobs: int
    downloaded_bytes: int
    revision: str
    exported: bool
    export_error: str | None = None


def install_pack_from_hf(
    pack_id: str,
    *,
    force: bool = False,
    revision: str | None = None,
    export: bool = True,
    on_status: StatusFn | None = None,
    on_bytes: hf.BytesProgress | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> HfInstalledPack:
    """Download ``pack_id`` from the QuantEM Hugging Face repository and install it.

    Args:
        pack_id: e.g. ``"quantem:mito"``.
        force: reinstall even if the pack is already present.
        revision: repository revision; defaults to the pinned
            :data:`quantem.registry.hf.QUANTEM_HF_REVISION` (or its env override).
        export: trace the encoder to ``encoder_ts.pt`` after conversion. Tests
            with stand-in tensors turn this off; the real path leaves it on.
        on_status: human-readable progress lines.
        on_bytes: ``(bytes_done, bytes_total)`` across the whole download.
        cancel_check: called at safe points; may raise to abort. Aborting never
            leaves a half-installed pack -- staging is discarded.

    Raises:
        InstallError: anything that stops the install, with the reason and,
            for a digest mismatch, both digests.
    """
    if pack_id not in ARCHITECTURE:
        raise InstallError(f"unknown pack id {pack_id!r}; known: {sorted(ARCHITECTURE)}")
    if cache.installed(pack_id) and not force:
        from quantem.registry.install import _already_installed

        existing = _already_installed(pack_id)
        return HfInstalledPack(
            pack_id=pack_id,
            root=existing.root,
            head_sha256=existing.head_sha256,
            encoder_sha256=existing.encoder_sha256,
            bytes_written=0,
            reused_blobs=0,
            downloaded_bytes=0,
            revision=revision or hf.hf_revision(),
            exported=(existing.root / cache.EXPORTED_ENCODER_NAME).exists(),
        )

    rev = revision or hf.hf_revision()

    def say(message: str) -> None:
        if on_status is not None:
            on_status(message)

    def check() -> None:
        if cancel_check is not None:
            cancel_check()

    # -- 1. the model card ---------------------------------------------------
    check()
    say(f"{pack_id}: fetching the model card from {hf.HF_REPO_ID}@{rev[:12]}")
    try:
        card = hf.fetch_sidecar(pack_id, revision=rev)
    except hf.HfError as exc:
        raise InstallError(str(exc)) from exc

    trunk_name = card.trunk_basename
    trunk_info: hf.RemoteFile | None = None
    total_bytes = card.head_bytes
    if trunk_name is not None:
        trunk_file = f"{trunk_name}.safetensors"
        try:
            trunk_info = hf.remote_file_info(trunk_file, revision=rev)
        except hf.HfError as exc:
            raise InstallError(str(exc)) from exc
        total_bytes += trunk_info.size_bytes

    downloaded_so_far = 0

    def file_progress(done: int, total: int) -> None:
        if on_bytes is not None and total_bytes:
            on_bytes(min(downloaded_so_far + done, total_bytes), total_bytes)

    # -- 2. download and verify ---------------------------------------------
    check()
    say(f"{pack_id}: downloading {card.head_file} ({card.head_bytes / 1e6:.1f} MB)")
    try:
        head_path = hf.download_file(
            card.head_file,
            revision=rev,
            expected_bytes=card.head_bytes,
            on_bytes=file_progress,
            cancel_check=cancel_check,
        )
        say(f"{pack_id}: verifying {card.head_file}")
        hf.verify_sha256(
            head_path,
            card.head_sha256,
            what=f"{pack_id}: {card.head_file}",
            source=f"the model card published in {hf.HF_REPO_ID}@{rev}",
        )
    except hf.HfError as exc:
        raise InstallError(str(exc)) from exc
    downloaded_so_far += card.head_bytes

    trunk_path: Path | None = None
    if trunk_info is not None:
        check()
        say(
            f"{pack_id}: downloading {trunk_info.filename} "
            f"({trunk_info.size_bytes / 1e6:.1f} MB, shared by the family)"
        )
        try:
            trunk_path = hf.download_file(
                trunk_info.filename,
                revision=rev,
                expected_bytes=trunk_info.size_bytes,
                on_bytes=file_progress,
                cancel_check=cancel_check,
            )
            say(f"{pack_id}: verifying {trunk_info.filename}")
            hf.verify_sha256(
                trunk_path,
                trunk_info.sha256,
                what=f"{pack_id}: {trunk_info.filename}",
                source=f"the LFS object id in {hf.HF_REPO_ID}@{rev}",
            )
        except hf.HfError as exc:
            raise InstallError(str(exc)) from exc
        downloaded_so_far += trunk_info.size_bytes
    if on_bytes is not None and total_bytes:
        on_bytes(total_bytes, total_bytes)

    # -- 3-5. convert, export, promote --------------------------------------
    staging_root = cache.models_root() / "staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    _sweep_stale_staging(staging_root)
    _gc_orphan_blobs(staging_root)
    staging = staging_root / f"{cache.pack_dirname(pack_id)}-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)

    try:
        check()
        say(f"{pack_id}: converting the published artifacts into a pack")
        convert_artifacts_to_pack(
            pack_id, card, head_path, trunk_path, staging, revision=rev
        )

        exported = False
        export_error: str | None = None
        if export:
            check()
            say(f"{pack_id}: exporting the encoder to TorchScript (one-time, minutes)")
            exported, export_error = _export_in_staging(pack_id, staging)
            if export_error:
                say(f"{pack_id}: export failed; the pack will run through timm instead")

        check()
        say(f"{pack_id}: recording digests")
        record, written, reused = _record_and_store(
            pack_id, card, staging, rev,
            trunk_info=trunk_info,
            trunk_path=trunk_path,
            exported=exported,
            export_error=export_error,
        )

        final_root = _promote(pack_id, staging, force=force)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # The install's own blobs are minutes old and every installed pack's are
    # referenced, so this only collects debris earlier failures left behind.
    _gc_orphan_blobs(staging_root)

    head_entry = record.get("head") or {}
    encoder_entry = record.get("encoder") or {}
    say(f"{pack_id}: installed under {final_root}")
    return HfInstalledPack(
        pack_id=pack_id,
        root=final_root,
        head_sha256=str(head_entry.get("sha256", "")),
        encoder_sha256=encoder_entry.get("sha256"),
        bytes_written=written,
        reused_blobs=reused,
        downloaded_bytes=downloaded_so_far,
        revision=rev,
        exported=exported,
        export_error=export_error,
    )


# --- Conversion -------------------------------------------------------------


def convert_artifacts_to_pack(
    pack_id: str,
    card: hf.Sidecar,
    head_path: Path,
    trunk_path: Path | None,
    staging: Path,
    *,
    revision: str,
) -> None:
    """Write ``head.pt``, ``resolved_config.yaml`` and ``checkpoint_index.json``.

    The tensor split is the exact inverse of the packaging conversion that
    produced the artifacts (quantem-core ``weights/convert.py``): ``neck.*`` and
    ``decoder.*`` return to their state dicts unchanged; ``adapters.*`` becomes
    the LoRA ``encoder_trainable`` (``_conv_lora.<i>...``, the naming the heads
    were trained under); ``encoder.*`` -- timm-named, because that is how the
    release publishes them -- becomes ``backbone.*`` tensors for the timm-built
    encoder. A ``last_n`` head must carry *all four* of its fine-tuned blocks:
    anything less would leave trunk-absent blocks at random init, and that is
    checked here rather than discovered as a plausible-looking wrong
    segmentation.
    """
    import torch
    from safetensors.torch import load_file

    family, organelle = pack_id.split(":", 1)
    tensors = load_file(str(head_path))

    neck: dict[str, Any] = {}
    decoder: dict[str, Any] = {}
    encoder: dict[str, Any] = {}
    adapters: dict[str, Any] = {}
    for key, value in tensors.items():
        if key.startswith("neck."):
            neck[key[len("neck."):]] = value
        elif key.startswith("decoder."):
            decoder[key[len("decoder."):]] = value
        elif key.startswith("encoder."):
            encoder[key[len("encoder."):]] = value
        elif key.startswith("adapters."):
            adapters[key[len("adapters."):]] = value
        else:
            raise InstallError(
                f"{pack_id}: unexpected key {key!r} in {card.head_file}; the published "
                "artifact does not match the layout this app understands."
            )
    if not neck or not decoder:
        raise InstallError(
            f"{pack_id}: {card.head_file} carries no neck/decoder tensors "
            f"(neck={len(neck)}, decoder={len(decoder)}); it is not a QuantEM head artifact."
        )

    adaptation = str(card.architecture.get("adaptation") or "")
    if adaptation not in _ADAPT_CONFIG:
        raise InstallError(
            f"{pack_id}: the model card declares adaptation {adaptation!r}, which this "
            f"app cannot rebuild (knows {sorted(_ADAPT_CONFIG)})."
        )
    adapt, adapt_params = _ADAPT_CONFIG[adaptation]

    encoder_trainable: dict[str, Any] | None = None
    adapters_out: dict[str, Any] | None = None
    if family == "quantem":
        if adapters:
            raise InstallError(
                f"{pack_id}: unexpected adapters in a {adaptation} QuantEM head."
            )
        if adaptation == "last_n":
            got_blocks = {
                int(k.split(".")[1]) for k in encoder if k.startswith("blocks.")
            }
            missing = set(_QUANTEM_OVERLAY_BLOCKS) - got_blocks
            if missing:
                raise InstallError(
                    f"{pack_id}: {card.head_file} should carry fine-tuned blocks "
                    f"{sorted(_QUANTEM_OVERLAY_BLOCKS)} but blocks {sorted(missing)} are "
                    "absent. Installing it would run those blocks at random weights."
                )
        if not encoder:
            raise InstallError(
                f"{pack_id}: {card.head_file} carries no encoder tensors for a "
                f"{adaptation} pack."
            )
        encoder_trainable = {f"backbone.{k}": v for k, v in encoder.items()}
    else:
        if encoder:
            raise InstallError(
                f"{pack_id}: unexpected encoder tensors in an OmniEM head "
                "(LoRA never touches the trunk)."
            )
        if not adapters:
            raise InstallError(f"{pack_id}: {card.head_file} carries no LoRA adapters.")
        # Both spellings the released head.pt files use: the loader reads
        # encoder_trainable; adapters is the fallback and inspection surface.
        encoder_trainable = {f"_conv_lora.{k}": v for k, v in adapters.items()}
        adapters_out = dict(adapters)

    head = {
        "neck": neck,
        "decoder": decoder,
        "encoder_trainable": encoder_trainable,
        "adapters": adapters_out,
        "conditioner": None,
        "meta_vocab": None,
    }
    torch.save(head, str(staging / cache.HEAD_NAME))

    (staging / cache.CONFIG_NAME).write_text(
        _resolved_config_yaml(pack_id, card, adapt, adapt_params, revision),
        encoding="utf-8",
    )
    (staging / cache.INDEX_NAME).write_text(
        json.dumps(_checkpoint_index(pack_id, card, revision), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if trunk_path is not None:
        _link_or_copy(trunk_path, staging / HF_ENCODER_NAME)


def _resolved_config_yaml(
    pack_id: str,
    card: hf.Sidecar,
    adapt: str,
    adapt_params: dict[str, Any],
    revision: str,
) -> str:
    """The subset of a ``resolved_config.yaml`` that shapes the module graph.

    Synthesised, and says so. Field-for-field it matches what the released
    YAMLs carry for the same pack (verified against all eight: every released
    neck/decoder ``params`` is ``{}``, ``num_classes`` is 2,
    ``apply_encoder_norm`` is true, ``feature_layers`` is ``last4``).
    """
    import yaml

    _family, organelle = pack_id.split(":", 1)
    doc = {
        "name": str(card.raw.get("arm_name") or pack_id),
        "notes": (
            f"Synthesised at install from {hf.sidecar_filename(pack_id)} published in "
            f"{hf.HF_REPO_ID}@{revision}; carries only the fields inference reads."
        ),
        "encoder": {
            "run_dir": None,
            "checkpoint_step": None,
            "tile_size": 512,
            "feature_layers": str(card.architecture.get("feature_layers") or "last4"),
            "apply_encoder_norm": True,
            "adapt": adapt,
            "adapt_params": adapt_params,
        },
        "neck": {"type": str(card.architecture.get("neck")), "params": {}},
        "decoder": {"type": str(card.architecture.get("decoder")), "params": {}},
        "data": {
            "organelle": organelle,
            "canonical_nm": card.inference.get("canonical_nm"),
            "num_classes": 2,
            "task": str(card.inference.get("task") or "instance"),
        },
    }
    return yaml.safe_dump(doc, sort_keys=False)


def _checkpoint_index(pack_id: str, card: hf.Sidecar, revision: str) -> dict[str, Any]:
    """A ``checkpoint_index.json`` describing the published (timm-named) encoder.

    Framework is ``timm_vit`` for both families -- the artifacts carry timm
    naming and timm builds both architectures. The QuantEM entry point differs
    from OmniEM's in every way that matters (1-channel, caller-normalised
    input, LayerNorm eps 1e-6, live k-bias, bfloat16 rotary periods), so it is
    marked with ``variant: quantem_dinov3`` and built by its own path in
    :mod:`quantem.inference.encoders`.
    """
    family, _organelle = pack_id.split(":", 1)
    arch = card.architecture
    mean, std = ENCODER_NORM["quantem" if family == "quantem" else "omniem"]
    common = {
        "config_path": None,
        "checkpoints": [],
        "notes": (
            f"Synthesised at install from the artifacts published in "
            f"{hf.HF_REPO_ID}@{revision}. Tensors are timm-named; nothing here "
            "points at a research machine."
        ),
    }
    if family == "quantem":
        adaptation = str(arch.get("adaptation") or "")
        entry_point: dict[str, Any] = {
            "loader": "timm_hf",
            "variant": QUANTEM_TIMM_VARIANT,
            "forward": "forward_intermediates",
            "timm_model": str(arch.get("encoder") or "vit_base_patch16_dinov3_qkvb"),
            "in_chans": 1,
            "img_size_build": 512,
            "strip_prefix": "encoder.",
            "norm_eps": 1e-06,
            "rope_periods_bf16": True,
            "n_prefix_tokens": 5,
        }
        if adaptation == "last_n":
            entry_point["overlay_blocks"] = list(_QUANTEM_OVERLAY_BLOCKS)
        encoder = {
            "arch": "vit_base",
            "depth": int(arch.get("depth") or 12),
            "embedding_dim": int(arch.get("embed_dim") or 768),
            "patch_size": int(arch.get("patch_size") or 16),
            "framework": "timm_vit",
            "image_mean": [mean],
            "image_std": [std],
            "input_channels": 1,
            "run_id": "m1_dinov3_vitb",
            "feature_entry_point": entry_point,
        }
    else:
        encoder = {
            "arch": "vit_large",
            "depth": int(arch.get("depth") or 24),
            "embedding_dim": int(arch.get("embed_dim") or 1024),
            "patch_size": int(arch.get("patch_size") or 14),
            "framework": "timm_vit",
            "image_mean": [mean],
            "image_std": [std],
            "input_channels": 3,
            "run_id": "omniem_emdino_vitl",
            "feature_entry_point": {
                "loader": "timm_hf",
                "forward": "forward_intermediates",
                "timm_model": str(arch.get("encoder") or "vit_large_patch14_dinov2.lvd142m"),
                "in_chans": 3,
                "img_size_build": 518,
                "dynamic_img_size": True,
                "strip_prefix": "encoder.",
                "drop_key_prefixes": ["head."],
                "allow_unexpected": ["mask_token"],
            },
        }
    return {"schema_version": 1, "encoder": encoder, **common}


# --- Export, record, promote ------------------------------------------------


def _export_in_staging(pack_id: str, staging: Path) -> tuple[bool, str | None]:
    """Trace the staged pack's encoder to ``encoder_ts.pt`` beside its head.

    Returns ``(exported, error)``. Failure keeps the pack installable at the
    eager timm tier; refusing the whole install over the export would leave the
    user with nothing when they already hold verified weights that run.
    """
    try:
        from quantem.inference.export import EncoderSources, export_encoder_files

        encoder_file = staging / HF_ENCODER_NAME
        result = export_encoder_files(
            pack_id,
            EncoderSources(
                head_path=staging / cache.HEAD_NAME,
                config_path=staging / cache.CONFIG_NAME,
                index_path=staging / cache.INDEX_NAME,
                encoder_path=encoder_file if encoder_file.exists() else None,
            ),
            output=staging / cache.EXPORTED_ENCODER_NAME,
        )
        logger.info(
            "Exported %s from HF install: max|diff|=%.3e dynamic=%s",
            pack_id, result.max_abs_diff, result.dynamic_spatial,
        )
        return True, None
    except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
        logger.warning("TorchScript export failed for %s", pack_id, exc_info=True)
        return False, f"{exc.__class__.__name__}: {exc}"


def _record_and_store(
    pack_id: str,
    card: hf.Sidecar,
    staging: Path,
    revision: str,
    *,
    trunk_info: hf.RemoteFile | None,
    trunk_path: Path | None,
    exported: bool,
    export_error: str | None,
) -> tuple[dict[str, Any], int, int]:
    """Hash every staged file into the blob store and write ``pack.json``.

    Returns ``(record, bytes_written, reused_blobs)``. Linking staged files to
    their blobs is what dedupes the shared trunk: the second pack of a family
    finds the blob already present and writes nothing new.
    """
    roles = [
        ("head", cache.HEAD_NAME),
        ("config", cache.CONFIG_NAME),
        ("index", cache.INDEX_NAME),
    ]
    if (staging / HF_ENCODER_NAME).exists():
        roles.append(("encoder", HF_ENCODER_NAME))
    if exported and (staging / cache.EXPORTED_ENCODER_NAME).exists():
        roles.append(("export", cache.EXPORTED_ENCODER_NAME))

    trunk_provenance = None
    if trunk_info is not None:
        trunk_provenance = {
            "filename": trunk_info.filename,
            "sha256": trunk_info.sha256,
            "size_bytes": trunk_info.size_bytes,
            "metadata": hf.safetensors_metadata(trunk_path) if trunk_path else {},
        }

    record: dict[str, Any] = {
        "pack_id": pack_id,
        "schema_version": 1,
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "huggingface",
        "digest_origin": (
            f"verified-against-hugging-face: {card.head_file} was re-hashed after download "
            f"and matched the sha256 in its published model card at {hf.HF_REPO_ID}@{revision}"
            + (
                f"; {trunk_info.filename} matched the repository's LFS object id at that "
                "revision"
                if trunk_info is not None
                else ""
            )
            + ". head.pt, resolved_config.yaml and checkpoint_index.json were converted "
            "locally from those verified artifacts at install, so their digests attest "
            "that the converted bytes have not changed since."
        ),
        "hf": {
            "repo_id": hf.HF_REPO_ID,
            "revision": revision,
            "head_artifact": {
                "filename": card.head_file,
                "sha256": card.head_sha256,
                "size_bytes": card.head_bytes,
            },
            **({"trunk_artifact": trunk_provenance} if trunk_provenance else {}),
        },
        "architecture": dict(ARCHITECTURE[pack_id]),
        "adapt": ARCHITECTURE[pack_id]["adapt"],
        "neck": ARCHITECTURE[pack_id]["neck"],
        "decoder": ARCHITECTURE[pack_id]["decoder"],
        "encoder_run_dir": (
            "m1_dinov3_vitb" if pack_id.startswith("quantem") else "omniem_emdino_vitl"
        ),
        "checkpoint_step": None,
    }
    if export_error:
        record["export_error"] = export_error

    written = 0
    reused = 0
    for role, name in roles:
        digest, size, was_reused = store_blob(staging / name)
        _link_or_copy(cache.blob_path(digest), staging / name)
        record[role] = {"filename": name, "sha256": digest, "size_bytes": size}
        reused += int(was_reused)
        written += 0 if was_reused else size

    (staging / cache.RECORD_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    return record, written, reused


def _promote(pack_id: str, staging: Path, *, force: bool) -> Path:
    """Move the completed staging directory into place as the installed pack.

    The rename is the commit point: before it there is no pack directory (or
    the previous one, untouched); after it the whole verified pack exists.
    Without ``force``, a pack directory appearing since the install began means
    a concurrent install won the race; this staging is discarded in favour of
    what is already there -- same verified content either way. With ``force``
    the user asked for a reinstall, and the fresh copy always replaces.
    """
    final_root = cache.pack_dir(pack_id)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    if final_root.exists():
        if not force and cache.installed(pack_id):
            logger.info("%s: another install finished first; keeping it.", pack_id)
            shutil.rmtree(staging, ignore_errors=True)
            return final_root
        shutil.rmtree(final_root)
    _rename_with_retry(pack_id, staging, final_root)
    return final_root


def _rename_with_retry(pack_id: str, staging: Path, final_root: Path) -> None:
    """The promote rename, retried on ``PermissionError`` with bounded backoff.

    Windows refuses to rename a directory while anything inside it is open
    without delete-sharing -- and a virus scanner or the search indexer opens
    freshly written gigabyte files exactly when an install finishes. The handle
    is transient; the rename is not wrong, just early. Each failure is logged
    with the attempt count so a genuinely stuck install says why, and the final
    failure becomes an :class:`InstallError` that names the likely culprit and
    the fact that nothing needs re-downloading.
    """
    attempts = len(_PROMOTE_RETRY_DELAYS) + 1
    for attempt, delay in enumerate(_PROMOTE_RETRY_DELAYS, start=1):
        try:
            staging.rename(final_root)
            if attempt > 1:
                logger.info("%s: promote succeeded on attempt %d.", pack_id, attempt)
            return
        except PermissionError as exc:
            logger.warning(
                "%s: promoting the staged pack failed (%s); something -- typically "
                "an antivirus or indexer scanning the fresh files -- still holds a "
                "handle. Attempt %d/%d, retrying in %.1f s.",
                pack_id, exc, attempt, attempts, delay,
            )
            time.sleep(delay)
    try:
        staging.rename(final_root)
        logger.info("%s: promote succeeded on attempt %d.", pack_id, attempts)
    except PermissionError as exc:
        raise InstallError(
            f"{pack_id}: could not move the completed pack into place "
            f"({staging} -> {final_root}): {exc}. Tried {attempts} times over "
            f"{sum(_PROMOTE_RETRY_DELAYS):.0f} s, so something on this machine "
            "(antivirus, search indexer, a backup tool) is holding the staged "
            "files open for longer than a scan should take. The downloaded "
            "artifacts are verified and cached; retrying the install will reuse "
            "them without re-downloading."
        ) from exc


def _sweep_stale_staging(staging_root: Path) -> None:
    """Remove staging directories older than a day -- debris of killed installs."""
    cutoff = time.time() - _STALE_SECONDS
    try:
        for child in staging_root.iterdir():
            try:
                if child.is_dir() and child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
            except OSError:
                continue
    except OSError:
        pass


def _gc_orphan_blobs(staging_root: Path) -> int:
    """Delete blobs no installed pack references, older than the stale window.

    A failed install correctly discards its staging directory, but the
    content-addressed blobs it stored stay behind with nothing to ever delete
    them: four failed attempts at one 2.4 GB pack left over 5 GB of
    unreferenced blobs (each attempt's TorchScript export serialises to
    different bytes, so each got its own blob). This pass runs beside the
    stale-staging sweep and again after a successful promote. Returns the
    bytes freed.

    What makes a blob safe to delete while another install may be running:

    * no installed pack's ``pack.json`` names its digest (an unreadable record
      aborts the whole pass -- the GC must never guess);
    * its mtime is older than :data:`_STALE_SECONDS`. :func:`~quantem.registry
      .install.store_blob` stamps a blob's mtime to *now* both when it writes
      one and when it reuses one, so any blob a live install has touched holds
      a fresh lease -- the same no-install-takes-a-day assumption the staging
      sweep makes;
    * it has a single hard link and that link is not a file in a live staging
      directory. Belt and braces: deleting a multiply-linked blob would not
      corrupt the other links (they keep the content), but a linked blob is in
      use by definition and removing its store entry would only force a
      re-copy.

    ``*.partial-*`` temp files from killed copies age out the same way.
    """
    blobs_root = cache.blobs_root()
    if not blobs_root.is_dir():
        return 0
    cutoff = time.time() - _STALE_SECONDS

    referenced: set[str] = set()
    packs_root = cache.packs_root()
    if packs_root.is_dir():
        try:
            pack_dirs = list(packs_root.iterdir())
        except OSError:
            return 0  # cannot enumerate installs: assume everything referenced
        for child in pack_dirs:
            record_path = child / cache.RECORD_NAME
            if not record_path.is_file():
                continue
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                logger.warning(
                    "Skipping the blob GC pass: unreadable install record at %s",
                    record_path,
                )
                return 0
            for entry in record.values():
                if isinstance(entry, dict) and entry.get("sha256"):
                    referenced.add(str(entry["sha256"]).lower())

    # Files a live (or freshly dead) staging still links to, by identity: hard
    # links share (device, inode), and Windows Python fills both in.
    staged_ids: set[tuple[int, int]] = set()
    try:
        for path in staging_root.rglob("*"):
            try:
                if path.is_file():
                    st = path.stat()
                    staged_ids.add((st.st_dev, st.st_ino))
            except OSError:
                continue
    except OSError:
        pass

    freed = 0
    try:
        shards = [d for d in blobs_root.iterdir() if d.is_dir()]
    except OSError:
        return 0
    for shard in shards:
        try:
            entries = list(shard.iterdir())
        except OSError:
            continue
        for blob in entries:
            try:
                if not blob.is_file():
                    continue
                st = blob.stat()
                if st.st_mtime >= cutoff:
                    continue
                if ".partial-" not in blob.name:
                    if blob.name.lower() in referenced:
                        continue
                    if st.st_nlink > 1 or (st.st_dev, st.st_ino) in staged_ids:
                        continue
                blob.unlink()
                freed += st.st_size
            except OSError:
                continue
        with contextlib.suppress(OSError):
            shard.rmdir()  # only succeeds when the fan-out dir emptied
    if freed:
        logger.info(
            "Blob GC: removed %.2f GB of blobs no installed pack references.",
            freed / 1e9,
        )
    return freed
