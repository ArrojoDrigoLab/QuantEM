"""The model catalogue and its usability probe.

Every test here runs against a **fake pack directory** rather than the real
cache, because the point of the probe is what it says about packs that are
missing pieces, and the developer box has all eight installed and exported.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from django.test import SimpleTestCase

from quantem.inference.specs import MODEL_SPECS
from quantem.registry import cache, catalogue

QUANTEM_INDEX = {"encoder": {"framework": "dinov3"}}
OMNIEM_INDEX = {"encoder": {"framework": "timm_vit"}}


class _FakeCache:
    """A models root under a tmp dir, with packs built file by file."""

    def __init__(self, root: Path):
        self.root = root

    def install(
        self,
        pack_id: str,
        *,
        index: dict | None = None,
        exported: bool = False,
    ) -> Path:
        pack = self.root / "packs" / pack_id.replace(":", "__")
        pack.mkdir(parents=True, exist_ok=True)
        (pack / cache.HEAD_NAME).write_bytes(b"head")
        (pack / cache.CONFIG_NAME).write_text("{}", encoding="utf-8")
        (pack / cache.RECORD_NAME).write_text(
            json.dumps({"pack_id": pack_id, "head": {"sha256": "x"}}), encoding="utf-8"
        )
        if index is not None:
            (pack / cache.INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")
        if exported:
            (pack / cache.EXPORTED_ENCODER_NAME).write_bytes(b"ts")
        return pack


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    fake = _FakeCache(tmp_path / "models")
    monkeypatch.setattr(cache, "models_root", lambda: fake.root)
    return fake


# --- The shape ---------------------------------------------------------------


class CatalogueShapeTests(SimpleTestCase):
    def test_every_released_pack_is_listed_once(self):
        ids = [entry["id"] for entry in catalogue.packs()]
        assert ids == sorted(MODEL_SPECS)
        assert len(ids) == 8

    def test_an_entry_carries_every_contract_field(self):
        entry = catalogue.pack_entry("quantem:mito")
        assert set(entry) >= {
            "id", "family", "organelle", "title", "installed", "download_bytes",
            "canonical_nm", "tile_size", "default_threshold", "decoder", "neck",
            "adapt", "licence", "notes",
        }

    def test_titles_read_the_way_the_contract_shows_them(self):
        assert catalogue.pack_entry("quantem:mito")["title"] == "QuantEM — Mitochondria"
        assert catalogue.pack_entry("omniem:ld")["title"] == "OmniEM — Lipid Droplets"

    def test_download_bytes_is_head_plus_the_shared_encoder(self):
        # The number in the API contract, and the sum of two measured sizes.
        # Published HF sizes at the pinned revision: 136,541,856 (head) +
        # 227,685,512 (quantem-vitb trunk). The old pin, 662,337,373, counted
        # the local fp32 research artifacts and overstated the download by 74%.
        assert (
            catalogue.pack_entry("quantem:mito")["download_bytes"]
            == 136_541_856 + 227_685_512
        )

    def test_a_full_finetune_pack_needs_no_separate_encoder(self):
        # quantem:er was adapted with `adapt: full`, so its 465 MB head file is
        # a whole fine-tuned ViT-B. Charging it for the shared encoder as well
        # would overstate the download by 525 MB.
        assert MODEL_SPECS["quantem:er"].embeds_encoder
        assert catalogue.pack_entry("quantem:er")["download_bytes"] == 465_028_184

    def test_the_default_threshold_is_the_published_one_everywhere(self):
        assert {e["default_threshold"] for e in catalogue.packs()} == {0.5}

    def test_the_device_block_names_a_real_device(self):
        device = catalogue.device_block()
        assert device["kind"] in {"cpu", "cuda", "mps"}
        assert set(device) == {"kind", "name", "cuda", "mps"}

    def test_omniem_notes_declare_the_third_party_encoder(self):
        # The base encoder is not ours and its licence is not the repository's.
        notes = catalogue.pack_entry("omniem:mito")["notes"]
        assert "EM-DINO" in notes


# --- The usability probe -----------------------------------------------------


@pytest.mark.usefixtures("fake_cache")
class TestRunnableProbe:
    def test_an_uninstalled_pack_is_not_runnable(self, fake_cache):
        probe = catalogue.probe_runnable("quantem:mito")
        assert probe.ok is False
        assert "Not installed" in probe.reason

    def test_an_exported_encoder_makes_a_pack_runnable(self, fake_cache):
        fake_cache.install("quantem:mito", index=QUANTEM_INDEX, exported=True)
        probe = catalogue.probe_runnable("quantem:mito")
        assert probe.ok is True
        assert probe.reason is None
        assert probe.tier == "exported"

    def test_a_quantem_pack_without_an_export_builds_through_timm(self, fake_cache):
        # A DINOv3-shaped index no longer means Meta's package. The engine
        # renames the tensors and builds the same encoder through timm (see
        # quantem.inference.encoders.build_encoder), so the probe must not grey
        # out a pack that runs -- which is what it did while _EAGER_REQUIREMENT
        # still said this family needed "dinov3".
        fake_cache.install("quantem:mito", index=QUANTEM_INDEX)
        with patch.object(catalogue, "_dinov3_available", return_value=False):
            probe = catalogue.probe_runnable("quantem:mito")
        assert probe.ok is True
        assert probe.tier == "timm"

    def test_a_quantem_pack_needs_one_of_the_two(self, fake_cache):
        # Neither timm nor dinov3: now there really is nothing to build with.
        fake_cache.install("quantem:mito", index=QUANTEM_INDEX)
        with (
            patch.object(catalogue, "_module_available", lambda name: name == "torch"),
            patch.object(catalogue, "_dinov3_available", return_value=False),
        ):
            probe = catalogue.probe_runnable("quantem:mito")
        assert probe.ok is False
        # The reason names the way out, not just the problem -- and the way out
        # has to be one the person reading it can take. This used to point at
        # `python -m quantem.inference.export`, which needs both the dinov3
        # package the message has just said is missing and an already-installed
        # pack, so it was advice only the developer could act on. It then
        # pointed at `quantem models install`, which is advice only a terminal
        # user can act on (I-12, F2). The way out for everyone else is to
        # reinstall, which is a button.
        assert cache.INSTALL_HINT in probe.reason
        assert "reinstalling fixes this" in probe.reason
        assert "quantem.inference.export" not in probe.reason
        # And it says what is missing in words, not in package names: "dinov3"
        # is not a thing the reader can install, look up, or usefully know.
        assert "dinov3" not in probe.reason
        assert cache.EXPORTED_ENCODER_NAME not in probe.reason

    def test_a_quantem_pack_falls_back_to_dinov3_without_timm(self, fake_cache):
        # The developer escape hatch, and the engine's last rung. Reported only
        # when timm is genuinely absent, because timm is what would be built.
        fake_cache.install("quantem:mito", index=QUANTEM_INDEX)
        with (
            patch.object(catalogue, "_module_available", lambda name: name == "torch"),
            patch.object(catalogue, "_dinov3_available", return_value=True),
        ):
            probe = catalogue.probe_runnable("quantem:mito")
        assert probe.ok is True
        assert probe.tier == "dinov3"

    def test_an_omniem_pack_needs_timm(self, fake_cache):
        fake_cache.install("omniem:mito", index=OMNIEM_INDEX)
        with patch.object(catalogue, "_module_available", lambda name: name == "torch"):
            probe = catalogue.probe_runnable("omniem:mito")
        assert probe.ok is False
        # "timm" is a Python package name, and naming it in the app told the
        # reader nothing they could act on (I-12). What they can act on is the
        # reinstall, so that is what the sentence is about.
        assert "timm" not in probe.reason
        assert cache.INSTALL_HINT in probe.reason
        assert "reinstalling fixes this" in probe.reason

    def test_an_omniem_pack_is_runnable_with_timm(self, fake_cache):
        fake_cache.install("omniem:mito", index=OMNIEM_INDEX)
        probe = catalogue.probe_runnable("omniem:mito")
        assert probe.ok is True
        assert probe.tier == "timm"

    def test_a_pack_with_no_index_and_no_export_cannot_be_rebuilt(self, fake_cache):
        fake_cache.install("omniem:mito")
        probe = catalogue.probe_runnable("omniem:mito")
        assert probe.ok is False
        # Not by filename: ``checkpoint_index.json`` is an implementation
        # detail of the pack directory and means nothing on screen.
        assert cache.INDEX_NAME not in probe.reason
        assert "incomplete" in probe.reason
        assert cache.INSTALL_HINT in probe.reason

    def test_an_unknown_encoder_framework_is_refused(self, fake_cache):
        fake_cache.install("omniem:mito", index={"encoder": {"framework": "jax"}})
        probe = catalogue.probe_runnable("omniem:mito")
        assert probe.ok is False
        assert "jax" in probe.reason

    def test_no_torch_means_nothing_runs(self, fake_cache):
        fake_cache.install("quantem:mito", index=QUANTEM_INDEX, exported=True)
        with patch.object(catalogue, "_module_available", return_value=False):
            probe = catalogue.probe_runnable("quantem:mito")
        assert probe.ok is False
        assert "PyTorch" in probe.reason

    def test_the_probe_never_opens_a_weight_file(self, fake_cache):
        # A list request must not pay for a 1.2 GB encoder, eight times over.
        fake_cache.install("quantem:mito", index=QUANTEM_INDEX, exported=True)
        real_open = Path.open
        opened: list[str] = []

        def _record(self, *args, **kwargs):
            opened.append(self.name)
            return real_open(self, *args, **kwargs)

        with patch.object(Path, "open", _record):
            catalogue.probe_runnable("quantem:mito")
        assert cache.EXPORTED_ENCODER_NAME not in opened
        assert cache.HEAD_NAME not in opened

    def test_the_catalogue_reports_the_probe_per_pack(self, fake_cache):
        fake_cache.install("quantem:mito", index=QUANTEM_INDEX, exported=True)
        entries = {e["id"]: e for e in catalogue.packs()}
        assert entries["quantem:mito"]["installed"] is True
        assert entries["quantem:mito"]["runnable"] is True
        assert entries["quantem:mito"]["reason"] is None
        assert entries["omniem:er"]["installed"] is False
        assert entries["omniem:er"]["runnable"] is False
        assert entries["omniem:er"]["reason"]
