"""The eager-encoder fallback must announce itself and repair the export.

UAT round 13, paper-cut 5: with ``encoder_ts.pt`` deleted, a run silently fell
back to rebuilding the encoder from ``encoder.safetensors`` -- ~4.5 minutes
instead of ~30 seconds -- said nothing about why, and left the export missing
so every later start paid it again.

The heavy halves (a real eager build, a real trace) are covered by the
weights-gated suites; what these pin is the *policy* around them, with the
export step stubbed:

* the repair targets ``encoder_ts.pt`` beside the pack's head,
* it never runs when the artifact already exists, and
* a failing rewrite is swallowed and logged -- the run must proceed on the
  eager module it already has.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from quantem.inference import engine
from quantem.inference.engine import ModelFiles, _repair_export
from quantem.inference.specs import MODEL_SPECS


@pytest.fixture
def pack(tmp_path):
    head = tmp_path / "pack" / "head.pt"
    head.parent.mkdir(parents=True)
    head.write_bytes(b"h")
    files = ModelFiles(pack_id="omniem:mito", head_path=head)
    spec = MODEL_SPECS["omniem:mito"]
    cfg = SimpleNamespace(encoder=SimpleNamespace(adapt="lora", apply_encoder_norm=True))
    bundle = SimpleNamespace(
        module=object(),
        contract=SimpleNamespace(
            tier="timm",
            depth=24,
            embedding_dim=1024,
            patch_size=14,
            input_mean=0.0,
            input_std=1.0,
        ),
    )
    model = SimpleNamespace(layers=[5, 11, 17, 23])
    return files, spec, cfg, bundle, model


def test_the_repair_writes_the_export_beside_the_head(pack):
    files, spec, cfg, bundle, model = pack

    with patch("quantem.inference.export.export_built_encoder") as export:
        export.return_value = SimpleNamespace(
            path=files.head_path.parent / "encoder_ts.pt", max_abs_diff=1e-6
        )
        _repair_export(files, spec, cfg, bundle, model, "cpu")

    assert export.call_count == 1
    kwargs = export.call_args.kwargs
    assert export.call_args.args[0] == "omniem:mito"
    assert Path(kwargs["output"]) == files.head_path.parent / "encoder_ts.pt"
    assert kwargs["adapt"] == "lora"
    assert export.call_args.args[2] == [5, 11, 17, 23]


def test_an_existing_export_is_never_rewritten(pack):
    files, spec, cfg, bundle, model = pack
    (files.head_path.parent / "encoder_ts.pt").write_bytes(b"already here")

    with patch("quantem.inference.export.export_built_encoder") as export:
        _repair_export(files, spec, cfg, bundle, model, "cpu")

    export.assert_not_called()


def test_a_failing_rewrite_never_fails_the_run_and_says_why(pack, caplog):
    files, spec, cfg, bundle, model = pack

    with patch(
        "quantem.inference.export.export_built_encoder",
        side_effect=RuntimeError("disk full"),
    ):
        with caplog.at_level(logging.WARNING, logger="quantem.inference.engine"):
            _repair_export(files, spec, cfg, bundle, model, "cpu")  # must not raise

    text = caplog.text
    assert "could not rewrite" in text
    assert "omniem:mito" in text
    assert "quantem.inference.export" in text  # names the manual route


def test_a_successful_rewrite_is_announced(pack, caplog):
    files, spec, cfg, bundle, model = pack

    with patch("quantem.inference.export.export_built_encoder") as export:
        export.return_value = SimpleNamespace(
            path=files.head_path.parent / "encoder_ts.pt", max_abs_diff=2.5e-6
        )
        with caplog.at_level(logging.WARNING, logger="quantem.inference.engine"):
            _repair_export(files, spec, cfg, bundle, model, "cpu")

    assert "rewrote the missing TorchScript encoder" in caplog.text


def test_build_module_and_repair_share_the_export_filename():
    from quantem.inference.encoders import EXPORTED_ENCODER_NAME

    assert engine._exported_encoder_name() == EXPORTED_ENCODER_NAME == "encoder_ts.pt"
