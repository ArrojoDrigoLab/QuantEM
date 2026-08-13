"""Download endpoints for user-selected image and segmentation rasters."""

from __future__ import annotations

import re
from io import BytesIO

from django.http import FileResponse
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.assets.asset_resolver import get_active_asset
from quantem.assets.raster_exports import original_image_export, png_bytes, segmentation_export
from quantem.segmentation.models import ImageSegmentation

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def _filename_part(value: object, fallback: str) -> str:
    cleaned = _UNSAFE_FILENAME.sub("_", str(value or "").strip()).strip("._")
    return cleaned[:120] or fallback


class AssetRasterExportView(APIView):
    """Download one selected source on an image as an 8-bit grayscale PNG."""

    def get(self, request, asset_id):
        asset = get_active_asset(asset_id)
        source = str(request.query_params.get("source") or "original").strip().lower()
        image_name = _filename_part(asset.display_name, "image")

        try:
            if source == "original":
                plane = original_image_export(asset)
                filename = f"{image_name}_EM_8bit.png"
            elif source == "segmentation":
                segmentation_id = request.query_params.get("segmentation_id")
                if not segmentation_id:
                    return Response(
                        {"error": "segmentation_id is required for a segmentation export."},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
                segmentation = get_object_or_404(
                    ImageSegmentation.objects.select_related("asset", "segmentation_type"),
                    id=segmentation_id,
                    asset=asset,
                )
                plane = segmentation_export(segmentation)
                segmentation_name = _filename_part(
                    segmentation.display_name or segmentation.segmentation_type.long_name,
                    "segmentation",
                )
                filename = f"{image_name}_{segmentation_name}_8bit.png"
            else:
                return Response(
                    {"error": "source must be original or segmentation."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        except (FileNotFoundError, ValueError) as exc:
            return Response(
                {"error": str(exc) or "The selected raster could not be exported."},
                status=status.HTTP_409_CONFLICT,
            )

        return FileResponse(
            BytesIO(png_bytes(plane)),
            as_attachment=True,
            filename=filename,
            content_type="image/png",
        )
