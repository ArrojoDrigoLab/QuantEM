"""Sample data: a small EM crop so the plugin can be tried before downloading a model.

Deliberately import-light — napari calls sample-data providers from the Plugins menu, and this must
not drag in torch. Only Pillow and numpy are touched, and only when the sample is actually opened.
"""

from __future__ import annotations

from pathlib import Path

RESOURCES = Path(__file__).parent / "resources"

#: Everything a user needs to cite or re-find this image. Kept next to the loader so the layer
#: metadata, the docs and the citation cannot drift apart.
SAMPLE = {
    "filename": "islet_mito_5nm.png",
    "title": "Mouse pancreatic islet (SEM)",
    "pixel_size_nm": 5.0,
    "modality": "SEM",
    "tissue": "mouse pancreatic islet",
    "credit": "Arrojo e Drigo Lab, Vanderbilt University",
    "license": "CC BY 4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "citation": (
        "Mouse pancreatic islet SEM, Arrojo e Drigo Lab, Vanderbilt University. CC BY 4.0. "
        "Acree et al., bioRxiv 2026, https://www.biorxiv.org/content/10.64898/2026.08.06.743293v1"
    ),
    # Distributed with the plugin rather than deposited separately: it is a single 1024x1024
    # crop whose purpose is to let someone try the plugin before downloading a model.
    "repository": "distributed with napari-quantem",
    "citation_url": "https://www.biorxiv.org/content/10.64898/2026.08.06.743293v1",
    "source_crop_id": "C57_August_CD2_Islet1_5nm_00000",
}


def sample_path() -> Path:
    return RESOURCES / SAMPLE["filename"]


def load_sample_em():
    """napari sample-data provider. Returns one LayerData tuple.

    The pixel size is attached to the layer's metadata rather than to ``scale``: this plugin never
    infers a pixel size, and setting ``scale`` would silently change the displayed units. It is in
    the layer name too, so the value to type into the widget is visible without hunting.
    """
    import numpy as np
    from PIL import Image

    data = np.asarray(Image.open(sample_path()))
    nm = SAMPLE["pixel_size_nm"]
    return [
        (
            data,
            {
                "name": f"{SAMPLE['title']} — {nm:g} nm/px",
                "colormap": "gray",
                "metadata": {"quantem_sample": dict(SAMPLE)},
            },
            "image",
        )
    ]
