"""
DRF views for the local image library.
"""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache

from django.db.models import Q
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.core.config import DATA_DIR
from quantem.core.local_storage import resolve_stored_path

from .asset_mutations import (
    create_uploaded_asset,
    enqueue_ngff_for_asset,
    tombstone_asset,
    update_asset,
)
from .asset_openable import (
    asset_ngff_ready,
    get_asset_ngff_path,
    get_asset_openable,
)
from .asset_resolver import active_asset_queryset, get_active_asset
from .ngff import render_lowest_resolution_ngff_png_from_root
from .serializers import serialize_asset_detail, serialize_asset_entry
from .utils import UPLOAD_SUFFIXES

logger = logging.getLogger(__name__)

ASSET_ENTRY_PAGE_DEFAULT_LIMIT = 60
ASSET_ENTRY_PAGE_MAX_LIMIT = 200

ASSET_ORDERINGS = {
    "display_name": ("display_name", "id"),
    "-display_name": ("-display_name", "id"),
    "created_at": ("created_at", "id"),
    "-created_at": ("-created_at", "id"),
    "updated_at": ("updated_at", "id"),
    "-updated_at": ("-updated_at", "id"),
}


@lru_cache(maxsize=1)
def _get_torch_module():
    try:
        return importlib.import_module("torch")
    except Exception:
        return None


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _filtered_asset_queryset(request):
    assets = active_asset_queryset().prefetch_related("renditions")
    search = str(request.query_params.get("search") or "").strip()
    if search:
        assets = assets.filter(
            Q(display_name__icontains=search)
            | Q(original_filename__icontains=search)
            | Q(notes__icontains=search)
        )
    ordering = str(request.query_params.get("ordering") or "display_name")
    return assets.order_by(*ASSET_ORDERINGS.get(ordering, ASSET_ORDERINGS["display_name"]))


def _parse_pagination(request) -> tuple[int, int]:
    try:
        limit = int(request.query_params.get("limit", ASSET_ENTRY_PAGE_DEFAULT_LIMIT))
    except (TypeError, ValueError):
        limit = ASSET_ENTRY_PAGE_DEFAULT_LIMIT
    try:
        offset = int(request.query_params.get("offset", 0))
    except (TypeError, ValueError):
        offset = 0
    limit = min(max(limit, 1), ASSET_ENTRY_PAGE_MAX_LIMIT)
    return limit, max(offset, 0)


class AssetUploadView(APIView):
    """Upload an image and return the canonical Asset detail.

    Accepts TIFF and PNG only (:data:`quantem.assets.utils.UPLOAD_SUFFIXES`) -
    the same set ``assets/volume_readers.py`` can read. A file of any other type
    is rejected with 400 and a message naming the accepted extensions, so the
    client never has to guess; ``supported_upload_formats`` on
    ``/api/system/status/`` carries the same list for the file picker.

    The fields it reads are exactly ``file``, ``display_name``,
    ``pixel_size_nm``, ``notes`` and the four ``segment_*`` flags. That list is
    written down because the client was posting a ``tag_names`` field this view
    never looked at, behind an import-form box labelled "Tags": there is no tag
    field on :class:`~quantem.assets.models.Asset` and no tag model anywhere in
    this tree, so the text was accepted and discarded, and the library went on
    showing no tags. ``notes`` is the field that does exist and is already
    searched by ``AssetListView``; upload was the one door that could not set
    it.
    """

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        del args, kwargs
        if "file" not in request.FILES:
            return Response(
                {
                    "error": (
                        'No file provided. Use multipart/form-data with "file" field.'
                    ),
                    "supported_upload_formats": list(UPLOAD_SUFFIXES),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            payload = create_uploaded_asset(
                uploaded_file=request.FILES["file"],
                display_name=request.data.get("display_name", None),
                pixel_size_nm=request.data.get("pixel_size_nm", None),
                notes=request.data.get("notes", None),
                segment_mito=_truthy(request.data.get("segment_mito")),
                segment_er=_truthy(request.data.get("segment_er")),
                segment_nucleus=_truthy(request.data.get("segment_nucleus")),
                segment_ld=_truthy(request.data.get("segment_ld")),
                swallow_enqueue_errors=True,
            )
            return Response(payload, status=status.HTTP_201_CREATED)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:
            logger.error("Asset upload failed: %s", str(exc), exc_info=True)
            return Response(
                {"error": f"Error processing image: {str(exc)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class AssetListView(APIView):
    """List the local image library.

    Returns a bare list, or a ``{results, total, limit, offset, has_more}`` page
    when ``limit`` is supplied.
    """

    def get(self, request):
        assets = _filtered_asset_queryset(request)
        if request.query_params.get("limit") is None:
            return Response([serialize_asset_entry(asset) for asset in assets])

        limit, offset = _parse_pagination(request)
        total = assets.count()
        page = list(assets[offset : offset + limit + 1])
        has_more = len(page) > limit
        return Response(
            {
                "results": [serialize_asset_entry(asset) for asset in page[:limit]],
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": has_more,
            }
        )


class AssetDetailView(APIView):
    """Retrieve, patch, or tombstone one canonical Asset."""

    def get(self, request, asset_id):
        del request
        return Response(serialize_asset_detail(get_active_asset(asset_id)))

    def patch(self, request, asset_id):
        try:
            return Response(
                update_asset(get_active_asset(asset_id), request.data or {})
            )
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, asset_id):
        del request
        tombstone_asset(get_active_asset(asset_id))
        return Response(status=status.HTTP_204_NO_CONTENT)


class AssetProcessedPngView(APIView):
    def get(self, request, asset_id):
        del request
        asset = get_active_asset(asset_id)
        openable = get_asset_openable(asset)
        if not openable.file_path:
            return Response(
                {"error": "No preview available"}, status=status.HTTP_404_NOT_FOUND
            )
        file_path = getattr(openable, "path", None) or resolve_stored_path(
            openable.file_path,
            relative_to=DATA_DIR,
        )
        if not file_path.exists():
            return Response(
                {"error": "Preview file not found"}, status=status.HTTP_404_NOT_FOUND
            )
        return FileResponse(open(file_path, "rb"), content_type="image/png")


class AssetNgffThumbnailView(APIView):
    def get(self, request, asset_id):
        del request
        asset = get_active_asset(asset_id)
        ngff_path = get_asset_ngff_path(asset)
        if ngff_path is None:
            return Response(
                {"error": "NGFF thumbnail not available"},
                status=status.HTTP_404_NOT_FOUND,
            )
        try:
            png_bytes = render_lowest_resolution_ngff_png_from_root(ngff_path)
        except Exception:
            return Response(
                {"error": "NGFF thumbnail not available"},
                status=status.HTTP_404_NOT_FOUND,
            )
        return HttpResponse(png_bytes, content_type="image/png")


class SystemStatusView(APIView):
    """
    API view for system status information.

    Returns CUDA availability and the image formats this build can import, so
    the upload UI's file filter is driven by the backend rather than a
    hard-coded list that can drift from what the readers accept.
    """

    def get(self, request):
        """
        Return system status including CUDA availability.
        """
        cuda_available = False
        torch = _get_torch_module()
        if torch is not None:
            try:
                cuda_available = torch.cuda.is_available()
            except Exception as e:
                logger.warning(f"Error checking CUDA availability: {str(e)}")

        return Response(
            {
                "cuda_available": cuda_available,
                "supported_upload_formats": list(UPLOAD_SUFFIXES),
            }
        )


class SystemHandshakeView(APIView):
    """
    Lightweight handshake endpoint for desktop-local startup checks.
    """

    def get(self, request):
        return Response(
            {
                "status": "ok",
                "message": "handshake-ok",
            }
        )


def asset_ngff_root(request, asset_id):
    """Ensure and serve NGFF root metadata for a canonical asset."""

    asset = get_active_asset(asset_id)
    get_asset_openable(asset)
    try:
        ngff_root_path = get_asset_ngff_path(asset)
        if ngff_root_path is None or not asset_ngff_ready(asset):
            job = enqueue_ngff_for_asset(asset)
            return JsonResponse(
                {
                    "asset_id": str(asset.id),
                    "ngff_status": "queued",
                    "job_id": str(job.id),
                },
                status=202,
            )
        attrs_path = ngff_root_path / ".zattrs"
        if not attrs_path.exists():
            raise Http404(f"NGFF metadata not found for asset {asset_id}")
        return FileResponse(open(attrs_path, "rb"), content_type="application/json")
    except Http404:
        raise
    except Exception as e:
        logger.error(f"Error serving NGFF root metadata for asset {asset_id}: {str(e)}")
        raise Http404(f"Error serving NGFF metadata: {str(e)}") from e


def asset_ngff_file(request, asset_id, ngff_path):
    """Serve files from a canonical asset NGFF zarr store."""

    asset = get_active_asset(asset_id)
    get_asset_openable(asset)
    try:
        ngff_root_path = get_asset_ngff_path(asset)
        if ngff_root_path is None or not asset_ngff_ready(asset):
            job = enqueue_ngff_for_asset(asset)
            return JsonResponse(
                {
                    "asset_id": str(asset.id),
                    "ngff_status": "queued",
                    "job_id": str(job.id),
                },
                status=202,
            )
        relative_path = str(ngff_path).lstrip("/")
        if not relative_path:
            raise Http404("Invalid NGFF path")

        file_path = (ngff_root_path / relative_path).resolve()
        expected_root = ngff_root_path.resolve()

        try:
            file_path.relative_to(expected_root)
        except ValueError:
            raise Http404("Invalid NGFF path") from None

        if not file_path.exists() or file_path.is_dir():
            if (
                file_path.name == ".zattrs"
                and file_path.parent.exists()
                and file_path.parent.is_dir()
            ):
                return HttpResponse("{}", content_type="application/json")
            raise Http404(f"NGFF file not found: {relative_path}")

        content_type = (
            "application/json"
            if file_path.name in {".zattrs", ".zarray", ".zgroup", ".zmetadata"}
            else "application/octet-stream"
        )
        return FileResponse(open(file_path, "rb"), content_type=content_type)
    except Http404:
        raise
    except Exception as e:
        logger.error(
            "Error serving NGFF file for asset %s path %s: %s",
            asset_id,
            ngff_path,
            str(e),
        )
        raise Http404(f"Error serving NGFF file: {str(e)}") from e
