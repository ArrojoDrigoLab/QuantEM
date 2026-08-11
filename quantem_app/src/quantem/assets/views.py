"""
DRF views for the local image library.
"""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache

from django.conf import settings
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
    normalise_import_grouping,
    tombstone_asset,
    update_asset,
)
from .asset_openable import get_asset_openable
from .asset_resolver import active_asset_queryset, get_active_asset
from .ngff import render_lowest_resolution_ngff_png_from_root
from .pyramid_authority import (
    Intent,
    Reason,
    Unavailable,
    failure_detail,
    request_lazy_build,
    resolve_pyramid,
)
from .serializers import serialize_asset_detail, serialize_asset_entry
from .upload_staging import staged_upload_handlers
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


#: What a caller sends to mean "the images that are in none of them".
#:
#: Unassigned is a bucket, not a gap. It is the state every existing library is
#: in and the state an image returns to when its experiment is deleted, so the
#: filter has to be able to name it -- and it cannot be named by an id, because
#: there is no row to have one. A literal is the only thing left, and it is
#: spelled out here rather than at three call sites.
UNASSIGNED = "none"


def _grouping_filter(assets, request):
    """Narrow to one experiment or dataset, or to the images in neither.

    Repeated parameters are a union (``?experiment=a&experiment=b``), which is
    what a multi-select sends. ``none`` may be mixed in with real ids, so
    "these two experiments, plus everything not yet filed" is one request.
    """
    experiments = [value for value in request.query_params.getlist("experiment") if value]
    if experiments:
        named = [value for value in experiments if value != UNASSIGNED]
        matcher = Q(experiment_id__in=named) if named else Q()
        if UNASSIGNED in experiments:
            matcher = matcher | Q(experiment__isnull=True)
        assets = assets.filter(matcher)

    datasets = [value for value in request.query_params.getlist("dataset") if value]
    if datasets:
        named = [value for value in datasets if value != UNASSIGNED]
        matcher = Q(datasets__id__in=named) if named else Q()
        if UNASSIGNED in datasets:
            matcher = matcher | Q(datasets__isnull=True)
        # ``distinct`` because a many-to-many join returns one row per matching
        # membership, so an image in two of the chosen datasets would otherwise
        # be counted and rendered twice.
        assets = assets.filter(matcher).distinct()
    return assets


def _filtered_asset_queryset(request):
    assets = (
        active_asset_queryset()
        .select_related("experiment")
        .prefetch_related("renditions", "datasets")
    )
    search = str(request.query_params.get("search") or "").strip()
    if search:
        assets = assets.filter(
            Q(display_name__icontains=search)
            | Q(original_filename__icontains=search)
            | Q(notes__icontains=search)
        )
    assets = _grouping_filter(assets, request)
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
    ``pixel_size_nm``, ``notes``, the four ``segment_*`` flags, and the four
    optional grouping fields (``experiment_id``/``experiment_name`` and
    ``dataset_id``/``dataset_name``, each an id to use or a name to create).
    That list is
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
        # Before anything touches request.FILES, which is what parses the body.
        # Django's default handler would write the image to the system
        # temporary directory for save_uploaded_file_to_path to copy into the
        # staging directory; this one writes it into the staging directory
        # once. See quantem.assets.upload_staging for the measurements and for
        # why the copy cannot just become a rename on Windows.
        request.upload_handlers = staged_upload_handlers(request)
        if "file" not in request.FILES:
            return Response(
                {
                    "error": ('No file provided. Use multipart/form-data with "file" field.'),
                    "supported_upload_formats": list(UPLOAD_SUFFIXES),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            # Where in the library this image goes, if anywhere. All four
            # fields may be absent, and absent is the ordinary case -- see
            # `normalise_import_grouping`, which refuses only the one
            # combination that cannot mean anything (a dataset with no
            # experiment) and does so before any bytes are claimed.
            grouping = normalise_import_grouping(
                experiment_id=request.data.get("experiment_id"),
                experiment_name=request.data.get("experiment_name"),
                dataset_id=request.data.get("dataset_id"),
                dataset_name=request.data.get("dataset_name"),
            )
            payload = create_uploaded_asset(
                uploaded_file=request.FILES["file"],
                display_name=request.data.get("display_name", None),
                pixel_size_nm=request.data.get("pixel_size_nm", None),
                notes=request.data.get("notes", None),
                grouping=grouping,
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
            return Response(update_asset(get_active_asset(asset_id), request.data or {}))
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
            return Response({"error": "No preview available"}, status=status.HTTP_404_NOT_FOUND)
        file_path = getattr(openable, "path", None) or resolve_stored_path(
            openable.file_path,
            relative_to=DATA_DIR,
        )
        if not file_path.exists():
            return Response({"error": "Preview file not found"}, status=status.HTTP_404_NOT_FOUND)
        return FileResponse(open(file_path, "rb"), content_type="image/png")


class AssetNgffThumbnailView(APIView):
    """The dashboard preview, rendered from the published generation.

    Routed through the authority like everything else, which also closes the
    verifier's FINDING 6 for free: this view used to answer 200 for a FAILED,
    unopenable asset because it resolved a path on disk rather than asking
    whether the asset was openable.
    """

    def get(self, request, asset_id):
        del request
        asset = get_active_asset(asset_id)
        resolved = resolve_pyramid(asset, intent=Intent.SERVE)
        if isinstance(resolved, Unavailable):
            return Response(
                {"error": "NGFF thumbnail not available", "reason": resolved.reason.value},
                status=status.HTTP_404_NOT_FOUND,
            )
        try:
            png_bytes = render_lowest_resolution_ngff_png_from_root(resolved.root)
        except Exception:
            return Response(
                {"error": "NGFF thumbnail not available"},
                status=status.HTTP_404_NOT_FOUND,
            )
        return HttpResponse(png_bytes, content_type="image/png")


class SystemStatusView(APIView):
    """
    API view for system status information.

    Returns CUDA availability, the image formats this build can import, and the
    largest upload it will accept -- so the upload UI's file filter and its size
    check are both driven by the backend rather than by hard-coded numbers that
    can drift from what the server actually does.
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
                # The size half of the same contract. ``quantem serve`` hands
                # this number to waitress as ``max_request_body_size``, and a
                # request over it is refused from the headers alone -- which
                # arrives at the browser as an aborted connection partway
                # through the upload, with no way to tell it apart from the
                # network dropping. Publishing the limit is what lets the client
                # refuse an impossible file in the file picker, instantly and by
                # name, instead of after a long upload that was never going to
                # be accepted. Bytes, and an integer: a float would arrive in
                # JavaScript as a value that cannot be compared exactly.
                "max_upload_bytes": int(settings.QUANTEM_MAX_UPLOAD_BYTES),
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


#: What the client is told for each reason the authority can give. The two
#: NGFF routes share this table so they cannot drift apart again -- guarding one
#: of them only ever changed which URL the viewer used to re-open a failed
#: asset, because the viewer polls both.
_UNAVAILABLE_MESSAGES = {
    Reason.TERMINAL_FAILURE: (
        "This image's import failed, so it has no pyramid to view. Re-import the file to try again."
    ),
    Reason.CANCELLED: (
        "This image's import was cancelled, so it has no pyramid to view. "
        "Re-import the file to try again."
    ),
    Reason.NO_ASSET: "This image is no longer in the library.",
}


def _ngff_unavailable_response(asset, unavailable: Unavailable) -> JsonResponse:
    """One answer per reason, for both NGFF routes.

    * ``NEVER_BUILT`` -> 202 and *one* lazy build, collapsed by
      :func:`~quantem.assets.pyramid_authority.request_lazy_build`.
    * ``BUILDING`` -> 202 and **no enqueue at all**. Something that can publish
      is already running; adding a second job only creates the lease fight that
      made the import report the wrong error.
    * ``GEOMETRY_MISMATCH`` / ``STALE_DECODER`` -> 202 and a rebuild: the store
      that exists is not this picture, or not this decoder's pixels.
    * ``TERMINAL_FAILURE`` / ``CANCELLED`` -> 409 naming the state and carrying
      the import's own error sentence, so the client can say why instead of
      spinning on a "queued" that will never arrive. 409 rather than 404
      because the store is not missing in the "wrong URL" sense; it is
      unavailable because of the state this asset is in.
    """

    reason = unavailable.reason
    if reason in {Reason.TERMINAL_FAILURE, Reason.CANCELLED, Reason.NO_ASSET}:
        return JsonResponse(
            {
                "asset_id": str(asset.id),
                "ngff_status": "unavailable",
                "reason": reason.value,
                "preprocess_stage": asset.preprocess_stage,
                "detail": _UNAVAILABLE_MESSAGES[reason],
                "preprocess_error": failure_detail(asset),
            },
            status=409,
        )

    job = None
    if reason is not Reason.BUILDING:
        job = request_lazy_build(asset)
    return JsonResponse(
        {
            "asset_id": str(asset.id),
            "ngff_status": "building" if reason is Reason.BUILDING else "queued",
            "reason": reason.value,
            "job_id": str(job.id) if job is not None else None,
        },
        status=202,
    )


def asset_ngff_root(request, asset_id):
    """Serve the published generation's NGFF root metadata, or say why not."""

    asset = get_active_asset(asset_id)
    get_asset_openable(asset)
    try:
        resolved = resolve_pyramid(asset, intent=Intent.SERVE)
        if isinstance(resolved, Unavailable):
            return _ngff_unavailable_response(asset, resolved)
        attrs_path = resolved.file_path(".zattrs")
        if not attrs_path.exists():
            raise Http404(f"NGFF metadata not found for asset {asset_id}")
        # The ETag names the generation a viewer is reading, so it can notice a
        # rebuild and reload rather than mixing chunks from two pyramids.
        return FileResponse(
            open(attrs_path, "rb"),
            content_type="application/json",
            headers={"ETag": f'"{resolved.generation_id}"'},
        )
    except Http404:
        raise
    except Exception as e:
        logger.error(f"Error serving NGFF root metadata for asset {asset_id}: {str(e)}")
        raise Http404(f"Error serving NGFF metadata: {str(e)}") from e


def asset_ngff_file(request, asset_id, ngff_path):
    """Serve one file out of the published generation of an asset's pyramid."""

    asset = get_active_asset(asset_id)
    get_asset_openable(asset)
    try:
        resolved = resolve_pyramid(asset, intent=Intent.SERVE)
        if isinstance(resolved, Unavailable):
            return _ngff_unavailable_response(asset, resolved)

        relative_path = str(ngff_path).lstrip("/")
        if not relative_path:
            raise Http404("Invalid NGFF path")

        file_path = resolved.file_path(relative_path).resolve()
        expected_root = resolved.root.resolve()

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
        return FileResponse(
            open(file_path, "rb"),
            content_type=content_type,
            headers={"ETag": f'"{resolved.generation_id}"'},
        )
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
