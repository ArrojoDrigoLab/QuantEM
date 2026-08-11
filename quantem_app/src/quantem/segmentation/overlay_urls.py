"""Overlay NGFF (OME-Zarr) serving routes.

These sit under ``segmentation-overlays/``, not ``api/``, which is why they are
a second module rather than more entries in :mod:`quantem.segmentation.urls`:
one mount point per URL module. The project URLconf mounts this at the root
with ``path("", include("quantem.segmentation.overlay_urls"))``.
"""

from django.urls import path

from quantem.segmentation.views import (
    segmentation_overlay_ngff_file,
    segmentation_overlay_ngff_root,
)

urlpatterns = [
    path(
        "segmentation-overlays/<uuid:seg_id>.zarr",
        segmentation_overlay_ngff_root,
        name="segmentation-overlay-ngff-root",
    ),
    path(
        "segmentation-overlays/<uuid:seg_id>.zarr/<path:ngff_path>",
        segmentation_overlay_ngff_file,
        name="segmentation-overlay-ngff-file",
    ),
]
