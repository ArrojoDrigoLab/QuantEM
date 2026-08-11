"""
URL configuration for the QuantEM local server.

The surface here is exactly what a segmentation + proofreading + analysis
desktop app needs: import an image, run one of the released organelle models
over it (whole image or an ROI), proofread the resulting instances, and read the
overlay back as OME-Zarr. Nothing broader is exposed.

Every route lives in the app that owns it. What is left here is the one thing
that is genuinely global: which apps are mounted, and **in what order** -- the
SPA catch-all is served last so every API route above it wins. Adding a route
means editing your own app's URL module, never this file.
"""

from django.urls import include, path

from quantem.core.spa import serve_frontend

urlpatterns = [
    path("api/", include("quantem.assets.urls")),
    path("api/", include("quantem.jobs.urls")),
    # The DRF router is at the top of quantem.segmentation.urls, which is
    # where it resolved from when it was declared here.
    path("api/", include("quantem.segmentation.urls")),
    # NGFF (OME-Zarr) serving endpoints. Not under api/, so a separate mount.
    path("", include("quantem.assets.ngff_urls")),
    path("", include("quantem.segmentation.overlay_urls")),
    # Feature apps own their own routes; included before the SPA catch-all.
    path("api/", include("quantem.registry.urls")),
    path("api/", include("quantem.analysis.urls")),
    path("api/", include("quantem.finetune.urls")),
    path("api/", include("quantem.sam.urls")),
    # The built frontend, served last so every API route above wins. This is what
    # makes `pip install quantem-app && quantem-app` a complete application rather than
    # just an API server.
    path("", serve_frontend, name="frontend-index"),
    path("<path:path>", serve_frontend, name="frontend-asset"),
]
