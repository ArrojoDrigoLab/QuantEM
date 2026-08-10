"""The Hugging Face install path, with the network mocked at ``hf_hub_download``.

What is under test is everything *around* the byte transfer: the model card is
the gate (no digest, no download), verification aborts on a mismatch naming
both digests, conversion produces a pack the loaders understand, the shared
trunk is downloaded once and reused through the blob store, an offline machine
gets the repo URL and the bundle route rather than a traceback, and a cancel
mid-install leaves nothing half-installed.

The artifacts here are tiny real safetensors -- the conversion code reads them
for real -- with the TorchScript export turned off, because tracing a ViT is
the one step that needs real weights. The export and the numbers it produces
are covered by the maintainer-run parity verification, not by this lane; the
one live-network test lives in ``test_hf_live.py``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from quantem.registry import cache, hf, hf_install
from quantem.registry.hf_install import (
    HF_ENCODER_NAME,
    QUANTEM_TIMM_VARIANT,
    install_pack_from_hf,
)
from quantem.registry.install import InstallError, store_blob

# --- A fake repository ------------------------------------------------------


def _write_head_safetensors(path: Path, family: str) -> None:
    """A minimal but structurally honest head artifact."""
    import torch
    from safetensors.torch import save_file

    tensors = {
        "neck.fuse.0.weight": torch.randn(4, 8),
        "neck.fuse.1.weight": torch.randn(4),
        "decoder.aff_head.weight": torch.randn(2, 4),
    }
    if family == "quantem":
        for i in (8, 9, 10, 11):
            tensors[f"encoder.blocks.{i}.attn.qkv.weight"] = torch.randn(6, 6)
    else:
        for i in (0, 1):
            tensors[f"adapters.{i}.down.weight"] = torch.randn(2, 4)
            tensors[f"adapters.{i}.up.weight"] = torch.randn(4, 2)
    save_file(tensors, str(path))


class FakeRepo:
    """Files served by the patched ``hf_hub_download`` / ``get_paths_info``."""

    def __init__(self, root: Path):
        self.root = root
        self.files: dict[str, Path] = {}
        self.downloads: list[str] = []

    def add(self, filename: str, content: bytes | Path) -> Path:
        path = self.root / filename
        if isinstance(content, Path):
            path = content
        else:
            path.write_bytes(content)
        self.files[filename] = path
        return path

    def add_pack(self, pack_id: str, *, tamper_digest: bool = False) -> dict:
        family, organelle = pack_id.split(":", 1)
        head_name = f"{family}-{organelle}.safetensors"
        head_path = self.root / head_name
        _write_head_safetensors(head_path, family)
        self.files[head_name] = head_path
        digest = cache.sha256_file(head_path)
        if tamper_digest:
            digest = "0" * 64
        trunk = "quantem-vitb-trunk" if family == "quantem" else "omniem-vitl"
        card = {
            "model_id": f"{family}/{organelle}",
            "arm_name": "T_test",
            "family": family,
            "organelle": organelle,
            "architecture": {
                "encoder": "vit_base_patch16_dinov3_qkvb" if family == "quantem"
                else "vit_large_patch14_dinov2.lvd142m",
                "patch_size": 16 if family == "quantem" else 14,
                "embed_dim": 768 if family == "quantem" else 1024,
                "depth": 12 if family == "quantem" else 24,
                "neck": "naive_1x1",
                "decoder": "affinity_mws",
                "adaptation": "last_n" if family == "quantem" else "lora",
                "feature_layers": "last4",
            },
            "inference": {"tile_size": 512, "canonical_nm": 8.0, "task": "instance"},
            "artifact": {
                "filename": head_name,
                "bytes": head_path.stat().st_size,
                "sha256": digest,
            },
            "requires": [trunk, f"{family}-{organelle}"],
        }
        sidecar = self.root / f"{family}-{organelle}.json"
        sidecar.write_text(json.dumps(card), encoding="utf-8")
        self.files[sidecar.name] = sidecar
        if f"{trunk}.safetensors" not in self.files:
            self.add(f"{trunk}.safetensors", b"not-a-real-trunk-but-hashable")
        return card

    # the two patched boundaries ---------------------------------------------

    def hf_hub_download(self, repo_id: str, filename: str, *, revision=None, cache_dir=None, **_):
        assert repo_id == hf.HF_REPO_ID
        assert revision == hf.hf_revision()
        # The explicit cache dir is part of the contract: never HF's home.
        assert cache_dir is not None and str(hf.hf_cache_dir()) in str(cache_dir)
        self.downloads.append(filename)
        try:
            return str(self.files[filename])
        except KeyError:
            raise FileNotFoundError(filename) from None

    def get_paths_info(self, repo_id: str, paths, revision=None, **_):
        assert repo_id == hf.HF_REPO_ID

        class _Lfs:
            def __init__(self, sha256):
                self.sha256 = sha256

        class _Entry:
            def __init__(self, path, size, sha256):
                self.path = path
                self.size = size
                self.lfs = _Lfs(sha256)

        out = []
        for p in paths:
            f = self.files.get(p)
            if f is not None:
                out.append(_Entry(p, f.stat().st_size, cache.sha256_file(f)))
        return out


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fake HF repo, an isolated models root, and the patched boundaries."""
    (tmp_path / "repo").mkdir()
    fake = FakeRepo(tmp_path / "repo")

    models_root = tmp_path / "models"
    monkeypatch.setattr(cache, "models_root", lambda: models_root)
    monkeypatch.setattr(hf, "hf_cache_dir", lambda: tmp_path / "hfcache")

    class _Api:
        def get_paths_info(self, *args, **kwargs):
            return fake.get_paths_info(*args, **kwargs)

    with (
        patch("huggingface_hub.hf_hub_download", fake.hf_hub_download),
        patch("huggingface_hub.HfApi", _Api),
    ):
        yield fake


# --- Install ----------------------------------------------------------------


def test_install_converts_verifies_and_records_provenance(repo):
    card = repo.add_pack("quantem:mito")
    result = install_pack_from_hf("quantem:mito", export=False)

    assert cache.installed("quantem:mito")
    root = cache.pack_dir("quantem:mito")
    for name in (cache.HEAD_NAME, cache.CONFIG_NAME, cache.INDEX_NAME, HF_ENCODER_NAME,
                 cache.RECORD_NAME):
        assert (root / name).exists(), name

    record = cache.read_record("quantem:mito")
    assert record["source"] == "huggingface"
    assert hf.HF_REPO_ID in record["digest_origin"]
    assert hf.hf_revision() in record["digest_origin"]
    assert record["hf"]["head_artifact"]["sha256"] == card["artifact"]["sha256"]
    assert record["hf"]["revision"] == hf.hf_revision()
    assert result.downloaded_bytes > 0
    # No export was requested, so the record must not claim one.
    assert "export" not in record

    # The converted head is the dict shape the loader reads.
    import torch

    head = torch.load(str(root / cache.HEAD_NAME), map_location="cpu", weights_only=False)
    assert set(head) == {"neck", "decoder", "encoder_trainable", "adapters",
                         "conditioner", "meta_vocab"}
    assert all(k.startswith("backbone.blocks.") for k in head["encoder_trainable"])

    # The synthesised config parses through the real loader.
    from quantem.inference._fig3.schema import load_head_config

    cfg = load_head_config(root / cache.CONFIG_NAME)
    assert cfg.neck.type == "naive_1x1"
    assert cfg.decoder.type == "affinity_mws"
    assert cfg.encoder.adapt == "last_n"
    assert cfg.encoder.adapt_params == {"n": 4}

    # The synthesised index routes to the QuantEM timm tier...
    index = json.loads((root / cache.INDEX_NAME).read_text(encoding="utf-8"))
    assert index["encoder"]["framework"] == "timm_vit"
    assert index["encoder"]["feature_entry_point"]["variant"] == QUANTEM_TIMM_VARIANT
    # ...and the variant string is the one the encoder builder dispatches on.
    from quantem.inference.encoders import QUANTEM_TIMM_VARIANT as ENCODERS_VARIANT

    assert QUANTEM_TIMM_VARIANT == ENCODERS_VARIANT

    # The cheap catalogue probe agrees this is runnable through timm (the
    # models root is already redirected by the fixture).
    from quantem.registry.catalogue import probe_runnable

    probe = probe_runnable("quantem:mito", installed=True)
    assert probe.ok, probe.reason
    assert probe.tier == "timm"


def test_omniem_adapters_become_conv_lora_state(repo):
    repo.add_pack("omniem:ld")
    install_pack_from_hf("omniem:ld", export=False)

    import torch

    head = torch.load(
        str(cache.pack_dir("omniem:ld") / cache.HEAD_NAME),
        map_location="cpu", weights_only=False,
    )
    assert all(k.startswith("_conv_lora.") for k in head["encoder_trainable"])
    assert head["adapters"] and all(not k.startswith("_conv_lora") for k in head["adapters"])


def test_the_shared_trunk_is_downloaded_once_and_reused(repo):
    repo.add_pack("quantem:mito")
    repo.add_pack("quantem:ld")
    first = install_pack_from_hf("quantem:mito", export=False)
    second = install_pack_from_hf("quantem:ld", export=False)

    assert repo.downloads.count("quantem-vitb-trunk.safetensors") >= 1
    # The second pack's trunk came out of the content-addressed store: at
    # least one blob reused, and both packs record the same encoder digest.
    assert first.reused_blobs == 0
    assert second.reused_blobs >= 1
    rec1 = cache.read_record("quantem:mito")
    rec2 = cache.read_record("quantem:ld")
    assert rec1["encoder"]["sha256"] == rec2["encoder"]["sha256"]
    assert cache.blob_path(rec1["encoder"]["sha256"]).exists()


def test_a_digest_mismatch_aborts_naming_both_digests(repo):
    repo.add_pack("quantem:mito", tamper_digest=True)
    real_digest = cache.sha256_file(repo.files["quantem-mito.safetensors"])

    with pytest.raises(InstallError) as excinfo:
        install_pack_from_hf("quantem:mito", export=False)

    message = str(excinfo.value)
    assert "0" * 64 in message          # the expected digest
    assert real_digest in message       # the actual digest
    assert not cache.installed("quantem:mito")
    assert not cache.pack_dir("quantem:mito").exists()


def test_a_card_with_no_digest_is_refused_before_any_weight_byte(repo):
    card = repo.add_pack("quantem:mito")
    card["artifact"]["sha256"] = None
    (repo.root / "quantem-mito.json").write_text(json.dumps(card), encoding="utf-8")

    with pytest.raises(InstallError) as excinfo:
        install_pack_from_hf("quantem:mito", export=False)
    assert "no sha256" in str(excinfo.value)
    assert "quantem-mito.safetensors" not in repo.downloads


def test_offline_error_names_the_repo_and_the_bundle_route(repo, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise ConnectionError("simulated: no route to host")

    with patch("huggingface_hub.hf_hub_download", refuse):
        with pytest.raises(InstallError) as excinfo:
            install_pack_from_hf("quantem:mito", export=False)

    message = str(excinfo.value)
    assert hf.HF_REPO_URL in message
    assert cache.INSTALL_COMMAND in message
    assert "Traceback" not in message


def test_cancel_mid_install_leaves_no_staging_and_no_pack(repo):
    repo.add_pack("quantem:mito")

    class Cancelled(Exception):
        pass

    def cancel_check():
        # Fire at the first safe point after staging exists -- the download is
        # done, conversion has not landed -- which is exactly the window a
        # half-installed pack would come from.
        staging_root = cache.models_root() / "staging"
        if staging_root.exists() and any(staging_root.iterdir()):
            raise Cancelled()

    with pytest.raises(Cancelled):
        install_pack_from_hf("quantem:mito", export=False, cancel_check=cancel_check)

    assert not cache.installed("quantem:mito")
    staging_root = cache.models_root() / "staging"
    assert not staging_root.exists() or not any(staging_root.iterdir())


def test_an_installed_pack_short_circuits(repo):
    repo.add_pack("quantem:mito")
    install_pack_from_hf("quantem:mito", export=False)
    repo.downloads.clear()

    again = install_pack_from_hf("quantem:mito", export=False)
    assert again.downloaded_bytes == 0
    assert repo.downloads == []


# --- Promote vs Windows transient handles ------------------------------------


@pytest.mark.skipif(os.name != "nt", reason="POSIX renames a directory with open files")
class TestPromoteRetry:
    """The commit-point rename against an AV/indexer-style transient handle.

    Python's ``open()`` on Windows takes no delete-sharing, so a reader thread
    holding a staged file open makes ``staging.rename()`` raise the exact
    ``PermissionError: [WinError 5]`` the real installs hit 4/4 times -- the
    faithful stand-in for a virus scanner reading a fresh 1.2 GB export.
    """

    def _staged(self, tmp_path, monkeypatch) -> Path:
        monkeypatch.setattr(cache, "models_root", lambda: tmp_path / "models")
        staging = tmp_path / "models" / "staging" / "quantem__mito-deadbeef"
        staging.mkdir(parents=True)
        (staging / cache.HEAD_NAME).write_bytes(b"h" * 4096)
        return staging

    def _hold_open(self, path: Path, seconds: float) -> threading.Thread:
        """Hold a read handle on ``path`` for ``seconds``; returns once held."""
        held = threading.Event()

        def hold() -> None:
            with open(path, "rb") as fh:
                fh.read(1)
                held.set()
                time.sleep(seconds)

        thread = threading.Thread(target=hold, daemon=True)
        thread.start()
        assert held.wait(timeout=5), "the holder thread never opened the file"
        return thread

    def test_a_transient_handle_is_absorbed_by_the_retry(self, tmp_path, monkeypatch):
        staging = self._staged(tmp_path, monkeypatch)
        # A fast schedule with the same total patience shape: the handle is
        # released after ~1 s, well inside the budget.
        monkeypatch.setattr(hf_install, "_PROMOTE_RETRY_DELAYS", (0.2,) * 20)
        holder = self._hold_open(staging / cache.HEAD_NAME, 1.0)
        try:
            final = hf_install._promote("quantem:mito", staging, force=False)
        finally:
            holder.join(timeout=10)
        assert final == cache.pack_dir("quantem:mito")
        assert (final / cache.HEAD_NAME).exists()
        assert not staging.exists()

    def test_without_the_retry_the_same_handle_fails(self, tmp_path, monkeypatch):
        """The 4/4 reproduction: no retries is the pre-fix behaviour, and it
        must fail under the very condition the retry test succeeds in."""
        staging = self._staged(tmp_path, monkeypatch)
        monkeypatch.setattr(hf_install, "_PROMOTE_RETRY_DELAYS", ())
        holder = self._hold_open(staging / cache.HEAD_NAME, 2.0)
        try:
            with pytest.raises(InstallError) as excinfo:
                hf_install._promote("quantem:mito", staging, force=False)
        finally:
            holder.join(timeout=10)
        message = str(excinfo.value)
        assert "antivirus" in message
        assert "re-download" in message  # the cached artifacts survive
        assert not cache.pack_dir("quantem:mito").exists()

    def test_exhausted_retries_name_the_schedule(self, tmp_path, monkeypatch):
        """A handle held past the whole budget still fails -- bounded, not
        forever -- and the error says how long it tried."""
        staging = self._staged(tmp_path, monkeypatch)
        monkeypatch.setattr(hf_install, "_PROMOTE_RETRY_DELAYS", (0.05, 0.05))
        holder = self._hold_open(staging / cache.HEAD_NAME, 2.0)
        try:
            with pytest.raises(InstallError) as excinfo:
                hf_install._promote("quantem:mito", staging, force=False)
        finally:
            holder.join(timeout=10)
        assert "Tried 3 times" in str(excinfo.value)


def test_a_failed_promote_retries_from_the_cache_not_the_network(repo, monkeypatch):
    """After a promote failure, a retry reuses stored blobs (and, live, the
    verified hub cache): the failure costs time, never another download."""
    repo.add_pack("quantem:mito")

    real = hf_install._rename_with_retry
    calls = {"n": 0}

    def flaky(pack_id, staging, final_root):
        calls["n"] += 1
        if calls["n"] == 1:
            raise InstallError(f"{pack_id}: simulated: staged files held open")
        return real(pack_id, staging, final_root)

    monkeypatch.setattr(hf_install, "_rename_with_retry", flaky)
    with pytest.raises(InstallError, match="held open"):
        install_pack_from_hf("quantem:mito", export=False)
    assert not cache.installed("quantem:mito")
    # The discarded staging's blobs survive the failure...
    blob_count = sum(1 for p in cache.blobs_root().rglob("*") if p.is_file())
    assert blob_count > 0

    retry = install_pack_from_hf("quantem:mito", export=False)
    assert cache.installed("quantem:mito")
    # ...and the retry links against them instead of copying fresh bytes: the
    # trunk, config and index are content-identical across attempts.
    assert retry.reused_blobs >= 3


# --- The orphaned-blob GC -----------------------------------------------------


class TestOrphanBlobGc:
    """Failed installs must not leak gigabytes into ``models/blobs`` forever."""

    @pytest.fixture
    def roots(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache, "models_root", lambda: tmp_path / "models")
        blobs = cache.blobs_root()
        blobs.mkdir(parents=True)
        staging_root = cache.models_root() / "staging"
        staging_root.mkdir(parents=True)
        return blobs, staging_root

    def _blob(self, content: bytes, *, age_seconds: float = 0.0) -> Path:
        digest, _size, _reused = self._store(content)
        path = cache.blob_path(digest)
        if age_seconds:
            old = time.time() - age_seconds
            os.utime(path, (old, old))
        return path

    def _store(self, content: bytes):
        src = cache.models_root() / f"src-{abs(hash(content))}.bin"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(content)
        return store_blob(src)

    def test_only_old_unreferenced_unlinked_blobs_are_collected(self, roots):
        _blobs_root, staging_root = roots
        two_days = 2 * hf_install._STALE_SECONDS

        orphan_old = self._blob(b"orphan, old", age_seconds=two_days)
        orphan_fresh = self._blob(b"orphan, fresh")

        referenced = self._blob(b"referenced by an installed pack", age_seconds=two_days)
        pack_root = cache.pack_dir("quantem:mito")
        pack_root.mkdir(parents=True)
        (pack_root / cache.RECORD_NAME).write_text(
            json.dumps({
                "pack_id": "quantem:mito",
                "head": {"filename": cache.HEAD_NAME, "sha256": referenced.name},
            }),
            encoding="utf-8",
        )

        staged = self._blob(b"linked from a live staging", age_seconds=two_days)
        live = staging_root / "quantem__ld-12345678"
        live.mkdir()
        os.link(staged, live / HF_ENCODER_NAME)
        # The hard link backdates both names' shared mtime; that is exactly
        # the nlink/inode guard's job.
        old = time.time() - two_days
        os.utime(staged, (old, old))

        partial_old = orphan_old.parent / f"{'0' * 64}.partial-999-abcdef01"
        partial_old.write_bytes(b"half a copy")
        os.utime(partial_old, (old, old))
        partial_fresh = orphan_old.parent / f"{'1' * 64}.partial-999-abcdef02"
        partial_fresh.write_bytes(b"half a copy, in flight")

        freed = hf_install._gc_orphan_blobs(staging_root)

        assert not orphan_old.exists(), "old orphan must be collected"
        assert not partial_old.exists(), "stale partial must be collected"
        assert orphan_fresh.exists(), "fresh blobs hold a lease"
        assert referenced.exists(), "recorded digests are never collected"
        assert staged.exists(), "a blob linked from live staging is in use"
        assert partial_fresh.exists(), "an in-flight partial is not debris"
        assert freed >= 11  # at least the old orphan's bytes

    def test_an_unreadable_record_aborts_the_pass(self, roots):
        _blobs_root, staging_root = roots
        orphan = self._blob(b"orphan", age_seconds=2 * hf_install._STALE_SECONDS)
        pack_root = cache.pack_dir("quantem:mito")
        pack_root.mkdir(parents=True)
        (pack_root / cache.RECORD_NAME).write_text("{not json", encoding="utf-8")

        assert hf_install._gc_orphan_blobs(staging_root) == 0
        assert orphan.exists(), "GC must never guess what a broken record references"

    def test_store_blob_stamps_and_renews_the_mtime_lease(self, roots):
        """A stored blob is 'now' even when its source is old, and a reuse
        refreshes it -- that timestamp is what keeps a blob a concurrent
        install is using out of the GC's reach."""
        _blobs_root, _staging_root = roots
        src = cache.models_root() / "trunk.bin"
        src.write_bytes(b"shared trunk bytes")
        old = time.time() - 3 * hf_install._STALE_SECONDS
        os.utime(src, (old, old))  # copy2 would preserve this

        digest, _size, reused = store_blob(src)
        assert not reused
        blob = cache.blob_path(digest)
        assert blob.stat().st_mtime > time.time() - 60

        os.utime(blob, (old, old))
        _digest, _size, reused = store_blob(src)
        assert reused
        assert blob.stat().st_mtime > time.time() - 60

    def test_the_install_flow_collects_an_earlier_failure_leak(self, repo):
        """End to end: debris from a failed attempt is gone after the next
        successful install of anything."""
        repo.add_pack("quantem:mito")
        staging_root = cache.models_root() / "staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        blobs_dir = cache.blobs_root() / "ab"
        blobs_dir.mkdir(parents=True, exist_ok=True)
        leak = blobs_dir / ("ab" + "0" * 62)
        leak.write_bytes(b"x" * 128)
        old = time.time() - 2 * hf_install._STALE_SECONDS
        os.utime(leak, (old, old))

        install_pack_from_hf("quantem:mito", export=False)

        assert cache.installed("quantem:mito")
        assert not leak.exists()
        record = cache.read_record("quantem:mito")
        for key in ("head", "config", "index", "encoder"):
            assert cache.blob_path(record[key]["sha256"]).exists()


# --- The hub environment ------------------------------------------------------


class TestPrepareHubEnv:
    """hf-xet must never write on the OS drive: its telemetry log ignores
    HF_XET_CACHE/HF_HOME and lands in ``~/.cache/huggingface/xet/logs`` unless
    HF_XET_LOG_DEST names somewhere else."""

    def _clear(self, monkeypatch, *names):
        for name in names:
            # setenv-then-delenv so the original state is recorded and restored
            # even when the variable did not exist before the test.
            monkeypatch.setenv(name, "sentinel")
            monkeypatch.delenv(name)

    def test_the_xet_log_lands_beside_the_xet_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hf, "hf_cache_dir", lambda: tmp_path / "hfcache")
        self._clear(monkeypatch, "HF_XET_CACHE", "HF_XET_LOG_DEST")

        hf._prepare_hub_env()

        assert os.environ["HF_XET_CACHE"] == str(tmp_path / "hfcache" / "xet")
        dest = os.environ["HF_XET_LOG_DEST"]
        # Trailing separator: hf-xet's dest parser reads that as directory
        # mode (rolling files + its own cleanup) even before the dir exists.
        assert dest.endswith(os.sep)
        assert Path(dest) == tmp_path / "hfcache" / "xet" / "logs"
        assert Path(dest).is_dir()

    def test_an_existing_environment_wins(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hf, "hf_cache_dir", lambda: tmp_path / "hfcache")
        self._clear(monkeypatch, "HF_XET_LOG_DEST")
        monkeypatch.setenv("HF_XET_CACHE", str(tmp_path / "elsewhere"))

        hf._prepare_hub_env()

        # The log destination follows the *effective* xet cache, not ours.
        assert Path(os.environ["HF_XET_LOG_DEST"]) == tmp_path / "elsewhere" / "logs"

        monkeypatch.setenv("HF_XET_LOG_DEST", "already-routed")
        hf._prepare_hub_env()
        assert os.environ["HF_XET_LOG_DEST"] == "already-routed"


# --- The job handler ---------------------------------------------------------


class TestInstallJobHandler:
    """The registered handler drives the install and reports honestly."""

    @pytest.fixture
    def job(self, db):
        from quantem.jobs.models import Job

        return Job.objects.create(
            type="install_model_pack",
            status="RUNNING",
            payload_json={"pack_id": "quantem:mito", "source": "huggingface"},
        )

    def test_handler_installs_and_returns_the_pack_state(self, job, tmp_path):
        from quantem.jobs.handlers import handle_install_model_pack
        from quantem.jobs.reporter import CancelToken, JobReporter
        from quantem.registry.hf_install import HfInstalledPack

        fake = HfInstalledPack(
            pack_id="quantem:mito",
            root=tmp_path,
            head_sha256="a" * 64,
            encoder_sha256="b" * 64,
            bytes_written=123,
            reused_blobs=1,
            downloaded_bytes=456,
            revision="deadbeef",
            exported=True,
        )
        entry = {"runnable": True, "reason": None, "encoder_tier": "exported"}
        with (
            patch("quantem.registry.hf_install.install_pack_from_hf", return_value=fake) as install,
            patch("quantem.registry.catalogue.pack_entry", return_value=entry),
        ):
            result = handle_install_model_pack(
                job.payload_json, JobReporter(job.id), CancelToken(job.id)
            )

        assert install.call_args.args == ("quantem:mito",)
        assert result["source"] == "huggingface"
        assert result["revision"] == "deadbeef"
        assert result["downloaded_bytes"] == 456
        assert result["exported"] is True
        assert result["runnable"] is True

    def test_handler_requires_a_pack_id(self, job):
        from quantem.jobs.handlers import handle_install_model_pack
        from quantem.jobs.reporter import CancelToken, JobReporter

        with pytest.raises(ValueError):
            handle_install_model_pack({}, JobReporter(job.id), CancelToken(job.id))


# --- CLI routing -------------------------------------------------------------


class TestCliRouting:
    """`quantem models install` picks HF for pack ids, the bundle for a directory."""

    def _args(self, *sources, **flags):
        import argparse

        return argparse.Namespace(
            sources=list(sources),
            all=flags.get("all", False),
            hf=flags.get("hf", False),
            force=flags.get("force", False),
            verbose=False,
            data_dir=None,
        )

    def test_pack_ids_route_to_hf(self, monkeypatch, tmp_path):
        from quantem import cli

        monkeypatch.setenv("QUANTEM_DATA_DIR", str(tmp_path))
        with patch("quantem.cli._install_from_hf", return_value=0) as install:
            rc = cli.cmd_models_install(self._args("quantem:mito", "omniem:ld"))
        assert rc == 0
        assert install.call_args.args[0] == ["quantem:mito", "omniem:ld"]

    def test_all_routes_to_hf_with_every_pack(self, monkeypatch, tmp_path):
        from quantem import cli
        from quantem.inference.specs import MODEL_SPECS

        monkeypatch.setenv("QUANTEM_DATA_DIR", str(tmp_path))
        with patch("quantem.cli._install_from_hf", return_value=0) as install:
            rc = cli.cmd_models_install(self._args(all=True))
        assert rc == 0
        assert install.call_args.args[0] == sorted(MODEL_SPECS)

    def test_a_directory_routes_to_the_bundle_installer(self, monkeypatch, tmp_path):
        from quantem import cli

        monkeypatch.setenv("QUANTEM_DATA_DIR", str(tmp_path))
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        with patch(
            "quantem.registry.install.install_all_from_bundle", return_value=[]
        ) as install:
            rc = cli.cmd_models_install(self._args(str(bundle)))
        assert rc == 0
        assert install.call_args.args[0] == bundle

    def test_a_typo_names_the_known_pack_ids(self, monkeypatch, tmp_path, capsys):
        from quantem import cli

        monkeypatch.setenv("QUANTEM_DATA_DIR", str(tmp_path))
        rc = cli.cmd_models_install(self._args("quantem:mitochondria"))
        assert rc == 2
        err = capsys.readouterr().err
        assert "quantem:mito" in err

    def test_nothing_to_do_names_both_routes(self, monkeypatch, tmp_path, capsys):
        from quantem import cli

        monkeypatch.setenv("QUANTEM_DATA_DIR", str(tmp_path))
        rc = cli.cmd_models_install(self._args())
        assert rc == 2
        err = capsys.readouterr().err
        assert "Hugging Face" in err
        assert "bundle" in err
