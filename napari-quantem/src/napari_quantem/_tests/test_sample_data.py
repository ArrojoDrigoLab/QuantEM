"""The sample image must ship, load, be attributed, and actually work with the default model."""

from __future__ import annotations

import numpy as np
import pytest

from napari_quantem._sample_data import SAMPLE, load_sample_em, sample_path


def test_sample_file_ships_and_is_small():
    p = sample_path()
    assert p.is_file(), "the sample image is missing from the package"
    kb = p.stat().st_size / 1024
    assert kb < 1024, f"sample is {kb:.0f} KB; keep the wheel small"


def test_sample_is_declared_as_package_data():
    """A file present in the source tree but absent from package-data ships in git and not in the
    wheel -- which fails only for users who pip install.

    Two ways to check, and which one applies depends on where the suite is running. From a source
    checkout, read the declaration. From an installed copy there is no pyproject to read -- but
    then the far better evidence is already available, because the file either arrived in
    site-packages or it did not, and ``test_sample_file_ships_and_is_small`` has just proven it
    did.
    """
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).parents[3] / "pyproject.toml"
    if not pyproject.is_file():
        assert sample_path().is_file(), "installed copy is missing the sample image"
        return
    cfg = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    patterns = cfg["tool"]["setuptools"]["package-data"]["napari_quantem"]
    assert any("resources" in p and p.endswith(".png") for p in patterns)


def test_loads_as_a_2d_uint8_image():
    layers = load_sample_em()
    assert len(layers) == 1
    data, meta, kind = layers[0]
    assert kind == "image"
    assert data.ndim == 2 and data.dtype == np.uint8
    assert data.shape == (1024, 1024)


def test_pixel_size_is_discoverable_without_hunting():
    """The plugin never infers pixel size, so the sample has to say what its own is."""
    _data, meta, _ = load_sample_em()[0]
    assert "5 nm/px" in meta["name"]
    assert meta["metadata"]["quantem_sample"]["pixel_size_nm"] == 5.0


def test_scale_is_not_set_from_the_pixel_size():
    """Setting `scale` would silently change displayed units and contradict the no-inference rule."""
    _data, meta, _ = load_sample_em()[0]
    assert "scale" not in meta


def test_attribution_travels_with_the_layer():
    _data, meta, _ = load_sample_em()[0]
    s = meta["metadata"]["quantem_sample"]
    assert "Arrojo e Drigo Lab" in s["credit"]
    assert s["license"] == "CC BY 4.0"
    assert "CC BY 4.0" in s["citation"]


def test_attribution_is_embedded_in_the_file_itself():
    """So it survives someone copying the PNG out of site-packages."""
    from PIL import Image

    info = Image.open(sample_path()).info
    blob = " ".join(str(v) for v in info.values())
    assert "Arrojo e Drigo Lab" in blob
    assert "CC BY 4.0" in blob


def test_modality_is_sem_everywhere():
    """This image is SEM. It was briefly mislabelled TEM, so pin it in all four places it appears."""
    from pathlib import Path

    from PIL import Image

    assert SAMPLE["modality"] == "SEM"
    assert "(SEM)" in SAMPLE["title"]
    assert " SEM," in SAMPLE["citation"]

    png = " ".join(str(v) for v in Image.open(sample_path()).info.values())
    assert "SEM" in png and "TEM" not in png

    for f in (
        sample_path().parent / "SAMPLE_DATA.md",
        Path(__file__).parents[1] / "napari.yaml",
    ):
        text = f.read_text(encoding="utf-8")
        assert "TEM" not in text.replace("QUANTEM", ""), f"stale TEM in {f.name}"


def test_sample_docs_exist_and_carry_the_licence():
    doc = sample_path().parent / "SAMPLE_DATA.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    assert "CC BY 4.0" in text
    assert "Arrojo e Drigo Lab" in text


def test_sample_carries_a_real_citation_and_no_dangling_accession():
    """The crop ships with the plugin rather than being deposited separately, so the metadata must
    point at the paper -- and must not leave a half-filled deposition field behind."""
    assert SAMPLE["citation_url"].startswith("https://www.biorxiv.org/")
    assert SAMPLE["citation_url"] in SAMPLE["citation"]
    assert "accession" not in SAMPLE and "doi" not in SAMPLE
    doc = (sample_path().parent / "SAMPLE_DATA.md").read_text(encoding="utf-8")
    assert SAMPLE["citation_url"] in doc


def test_manifest_registers_the_sample():
    from pathlib import Path

    import yaml

    manifest = yaml.safe_load(
        (Path(__file__).parents[1] / "napari.yaml").read_text(encoding="utf-8")
    )
    samples = manifest["contributions"]["sample_data"]
    assert len(samples) == 1
    assert samples[0]["key"] == "quantem_islet_em"
    ids = {c["id"] for c in manifest["contributions"]["commands"]}
    assert samples[0]["command"] in ids


@pytest.mark.skipif(
    not __import__("os").environ.get("QUANTEM_MODEL_DIR"), reason="needs the published artifacts"
)
def test_default_model_finds_mitochondria_in_the_sample():
    """The sample exists to make a good first impression -- so assert it actually does."""
    pytest.importorskip("torch")
    from quantem_em.api import load_model
    from quantem_em.registry import DEFAULT_MODEL_FOR_ORGANELLE

    data, meta, _ = load_sample_em()[0]
    nm = meta["metadata"]["quantem_sample"]["pixel_size_nm"]
    res = load_model(DEFAULT_MODEL_FOR_ORGANELLE["mito"]).segment(data, pixel_size_nm=nm)
    assert res.n_objects >= 8, f"only {res.n_objects} mitochondria found in the sample"
    frac = res.summary()["area_fraction"]
    assert 0.03 < frac < 0.35, f"implausible mitochondrial area fraction {frac:.3f}"


def test_there_is_one_install_command_that_actually_works():
    """`pip install napari-quantem` pulls napari but no Qt binding, so napari cannot start. That
    is the right default -- a plugin must not override a user's existing binding -- but it means an
    `all` extra has to exist, and the README has to lead with it."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).parents[3]
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        import importlib.metadata as md

        extras = md.metadata("napari-quantem").get_all("Provides-Extra") or []
        assert "all" in extras, f"no `all` extra in the built metadata: {extras}"
        return

    cfg = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = cfg["project"]["optional-dependencies"]
    assert "all" in extras and any("napari" in d for d in extras["all"]), extras
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert 'pip install "napari-quantem[all]"' in readme
