"""NGFF (OME-Zarr) serving routes for assets.

These sit under ``ngff/``, not ``api/``, which is why they are a second module
rather than more entries in :mod:`quantem.assets.urls`: one mount point per
URL module. The project URLconf mounts this at the root with
``path("", include("quantem.assets.ngff_urls"))``.
"""

from django.urls import path

from quantem.assets.views import asset_ngff_file, asset_ngff_root

urlpatterns = [
    path(
        "ngff/assets/<uuid:asset_id>.zarr",
        asset_ngff_root,
        name="asset-ngff-root",
    ),
    path(
        "ngff/assets/<uuid:asset_id>.zarr/<path:ngff_path>",
        asset_ngff_file,
        name="asset-ngff-file",
    ),
]
