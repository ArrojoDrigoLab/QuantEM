"""
DRF views for the local image library.
"""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from uuid import UUID

from django.conf import settings
from django.db import transaction
from django.db.models import Q
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem import __version__
from quantem.core.config import DATA_DIR
from quantem.core.local_storage import resolve_stored_path

from .asset_mutations import (
    create_uploaded_asset,
    enqueue_upload_pipeline,
    normalise_import_grouping,
    tombstone_asset,
    update_asset,
)
from .asset_openable import get_asset_openable
from .asset_resolver import active_asset_queryset, get_active_asset
from .models import Rendition
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


#: What a caller sends to mean "images in no dataset". Active images always
#: have an experiment, so there is intentionally no equivalent experiment
#: bucket.
NO_DATASET = "none"


def _grouping_filter(assets, request):
    """Narrow to one experiment or dataset.

    Repeated parameters are a union (``?experiment=a&experiment=b``), which is
    what a multi-select sends. ``none`` remains meaningful only for datasets.
    """
    experiments = [value for value in request.query_params.getlist("experiment") if value]
    if experiments:
        # Ignore the retired unassigned sentinel from a stale client. If it is
        # the only value this deliberately returns no active images.
        named = [value for value in experiments if value != NO_DATASET]
        assets = assets.filter(experiment_id__in=named)

    datasets = [value for value in request.query_params.getlist("dataset") if value]
    if datasets:
        named = [value for value in datasets if value != NO_DATASET]
        matcher = Q(datasets__id__in=named) if named else Q()
        if NO_DATASET in datasets:
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
                defer_processing=_truthy(request.data.get("defer_processing")),
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


class AssetUploadPipelineBatchStartView(APIView):
    """Start processing only after every file in a multi-image import exists.

    The client uploads files sequentially with ``defer_processing=true``, then
    submits the successful asset ids here. All job rows are committed in one
    transaction, so a worker cannot claim image one while the rest of the batch
    is still being queued. Repeating the request is safe.
    """

    def post(self, request):
        uploads = request.data.get("uploads")
        if not isinstance(uploads, list) or not uploads:
            return Response(
                {"error": '"uploads" must be a non-empty list.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        parsed: list[tuple[str, dict[str, bool]]] = []
        seen: set[str] = set()
        for item in uploads:
            if not isinstance(item, dict):
                return Response(
                    {"error": "Every upload entry must be an object."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            raw_asset_id = str(item.get("asset_id") or "").strip()
            try:
                asset_id = str(UUID(raw_asset_id))
            except ValueError:
                return Response(
                    {"error": "Every upload entry must have a valid asset_id."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if asset_id in seen:
                return Response(
                    {"error": "Every upload entry must have a unique asset_id."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            seen.add(asset_id)
            parsed.append(
                (
                    asset_id,
                    {
                        "segment_mito": _truthy(item.get("segment_mito")),
                        "segment_er": _truthy(item.get("segment_er")),
                        "segment_nucleus": _truthy(item.get("segment_nucleus")),
                        "segment_ld": _truthy(item.get("segment_ld")),
                    },
                )
            )

        job_ids: list[str] = []
        with transaction.atomic():
            assets = {
                str(asset.id): asset
                for asset in active_asset_queryset().select_for_update().filter(id__in=seen)
            }
            missing = seen - assets.keys()
            if missing:
                return Response(
                    {"error": "One or more imported images were not found."},
                    status=status.HTTP_404_NOT_FOUND,
                )
            for asset_id, options in parsed:
                asset = assets[asset_id]
                job = enqueue_upload_pipeline(asset, **options)
                job_ids.append(str(job.id))
        return Response({"job_ids": job_ids}, status=status.HTTP_202_ACCEPTED)


class AssetUploadPipelineRecoveryView(APIView):
    """Resume deferred imports left behind by an interrupted browser session."""

    def post(self, request):
        del request
        job_ids: list[str] = []
        with transaction.atomic():
            assets = list(
                active_asset_queryset()
                .select_for_update()
                .filter(
                    preprocess_stage="ENCODING",
                    renditions__type=Rendition.TYPE_FULL,
                    renditions__metadata__processing_deferred=True,
                )
                .distinct()
            )
            for asset in assets:
                job_ids.append(str(enqueue_upload_pipeline(asset).id))
        return Response({"job_ids": job_ids}, status=status.HTTP_202_ACCEPTED)


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

    Returns the installed app version, CUDA availability, the image formats
    this build can import, and the largest upload it will accept. Settings and
    upload validation therefore stay aligned with the backend automatically.
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
                "app_version": __version__,
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
