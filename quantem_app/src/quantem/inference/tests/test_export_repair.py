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


@pytest.fixture(autouse=True)
def _forget_refusals():
    """``_EXPORT_REFUSED`` is process state; no test may inherit another's."""
    engine._EXPORT_REFUSED.clear()
    yield
    engine._EXPORT_REFUSED.clear()


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
    assert "could not write" in text
    assert "omniem:mito" in text
    assert "quantem.inference.export" in text  # names the manual route


def test_a_refusal_is_not_paid_for_twice_in_one_process(pack):
    """A trace that cannot be written is expensive to be told about again.

    The QuantEM ViT-B on CUDA is the real case: its trace does not survive the
    export's own verification, so the artifact is never written and every cold
    start would otherwise pay the trace to learn that -- MEASURED 51 s against
    25 s for the eager build alone.
    """
    files, spec, cfg, bundle, model = pack

    with patch(
        "quantem.inference.export.export_built_encoder",
        side_effect=RuntimeError("does not reproduce the published model"),
    ) as export:
        _repair_export(files, spec, cfg, bundle, model, "cpu")
        _repair_export(files, spec, cfg, bundle, model, "cpu")

    assert export.call_count == 1


def test_a_successful_rewrite_is_announced(pack, caplog):
    files, spec, cfg, bundle, model = pack

    with patch("quantem.inference.export.export_built_encoder") as export:
        export.return_value = SimpleNamespace(
            path=files.head_path.parent / "encoder_ts.pt", max_abs_diff=2.5e-6
        )
        with caplog.at_level(logging.WARNING, logger="quantem.inference.engine"):
            _repair_export(files, spec, cfg, bundle, model, "cpu")

    assert "wrote the missing TorchScript encoder for cpu" in caplog.text


def test_build_module_and_repair_share_the_export_filename():
    from quantem.inference.encoders import EXPORTED_ENCODER_NAME

    assert engine._exported_encoder_name() == EXPORTED_ENCODER_NAME == "encoder_ts.pt"


# --- The repair must not poison the pack for another device -----------------
#
# MEASURED (gpu_measure): a CUDA run of a pack with no exported encoder wrote a
# 341 173 139-byte CUDA-traced ``encoder_ts.pt`` into a shared pack directory,
# after which every CPU run of that pack failed with the mirror device error --
# permanently, silently, with nothing on screen. These pin the two properties
# that make it unreachable rather than unlikely.


def test_a_cuda_repair_cannot_write_the_cpu_artifact(pack):
    files, spec, cfg, bundle, model = pack

    with patch("quantem.inference.export.export_built_encoder") as export:
        export.return_value = SimpleNamespace(
            path=files.head_path.parent / "encoder_ts.cuda.pt", max_abs_diff=1e-6
        )
        _repair_export(files, spec, cfg, bundle, model, "cuda")

    written = Path(export.call_args.kwargs["output"])
    assert written.name == "encoder_ts.cuda.pt"
    assert written != files.head_path.parent / "encoder_ts.pt"
    assert export.call_args.kwargs["device"] == "cuda"


def test_a_cpu_artifact_does_not_stop_a_cuda_repair(pack):
    """The two artifacts are independent: having one is not having the other."""
    files, spec, cfg, bundle, model = pack
    (files.head_path.parent / "encoder_ts.pt").write_bytes(b"the cpu one")

    with patch("quantem.inference.export.export_built_encoder") as export:
        export.return_value = SimpleNamespace(
            path=files.head_path.parent / "encoder_ts.cuda.pt", max_abs_diff=1e-6
        )
        _repair_export(files, spec, cfg, bundle, model, "cuda")

    assert export.call_count == 1
    assert (files.head_path.parent / "encoder_ts.pt").read_bytes() == b"the cpu one"


def test_an_existing_device_artifact_is_never_rewritten(pack):
    files, spec, cfg, bundle, model = pack
    (files.head_path.parent / "encoder_ts.cuda.pt").write_bytes(b"already here")

    with patch("quantem.inference.export.export_built_encoder") as export:
        _repair_export(files, spec, cfg, bundle, model, "cuda")

    export.assert_not_called()
