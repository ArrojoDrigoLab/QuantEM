"""Shared fixtures for the segmentation_training CPU tests — synthetic corpus + mock checkpoints only.

Nothing heavy is imported at collection time: every fixture defers its imports to first use, and the
corpus builder itself needs only numpy and tifffile. The mock checkpoints do need DINOv3, because
``_synthetic.write_mock_checkpoint`` builds a randomly-initialised ``dinov3`` backbone; the test
declared optional against that dependency gates on ``pytest.importorskip("dinov3")``.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def synthetic_corpus(tmp_path_factory) -> dict:
    from segmentation_training._synthetic import build_synthetic_corpus

    root = tmp_path_factory.mktemp("segmentations")
    return build_synthetic_corpus(root)


@pytest.fixture()
def mock_checkpoint(tmp_path):
    from segmentation_training._synthetic import write_mock_checkpoint

    def _make(framework: str = "dinov3", arch: str = "vit_small", **kw) -> Path:
        return write_mock_checkpoint(tmp_path / f"mock_{framework}", framework, arch=arch, **kw)

    return _make


# fixtures for the experiment lines
@pytest.fixture(scope="module")
def er_setup(tmp_path_factory):
    from segmentation_training.experiments.common._smoke import synthetic_setup
    return synthetic_setup(tmp_path_factory.mktemp("er"), organelle="er", tile_size=32)


@pytest.fixture(scope="module")
def mito_setup(tmp_path_factory):
    from segmentation_training.experiments.common._smoke import synthetic_setup
    return synthetic_setup(tmp_path_factory.mktemp("mito"), organelle="mito", tile_size=32)
