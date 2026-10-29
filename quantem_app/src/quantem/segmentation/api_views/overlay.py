"""Segmentation overlay manifest, LUT, and NGFF serving views."""

from __future__ import annotations

import logging
from functools import lru_cache

from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.core.local_storage import ensure_cached_storage_path, storage_relpath_for_path
from quantem.segmentation.models import ImageSegmentation
from quantem.segmentation.overlay_ngff import (
    build_label_lut_json,
    encode_zero_chunk,
    ensure_overlay_manifest,
    full_image_dirty_bbox,
    get_or_create_overlay_state,
    get_overlay_active_bundle_path,
    parse_overlay_chunk_path,
    queue_full_overlay_rebuild,
    register_overlay_mutation,
)
from quantem.segmentation.overlay_ngff.paths import OverlayStoreError
from quantem.segmentation.overlay_ngff.snapshots import read_snapshot
from quantem.segmentation.overlay_ngff.store import stored_chunk_shape
from quantem.segmentation.serializers import SegmentationOverlayRebuildSerializer
from quantem.segmentation.source_models import normalize_source_model

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _sparse_chunk_bytes(array_key: str, chunk_shape: tuple[int, int]) -> bytes:
    return encode_zero_chunk(array_key, chunk_shape)


def _set_revisioned_cache_headers(response, request) -> None:
    if request.GET.get("rev"):
        response["Cache-Control"] = "public, max-age=31536000, immutable"
    else:
        response["Cache-Control"] = "no-cache"


def _requested_store(request, state):
    revision = request.GET.get("rev")
    if revision:
        try:
            root, _ = read_snapshot(
                str(state.segmentation_id),
                state.candidate_source_model,
                revision,
                include_labels=False,
            )
            return root
        except OverlayStoreError as exc:
            raise Http404(str(exc)) from exc
    return get_overlay_active_bundle_path(state)


class SegmentationOverlayManifestView(APIView):
    """Return the current overlay bundle manifest for the viewer."""

    def get(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )
        source_model = normalize_source_model(request.query_params.get("source_model"))
        payload = ensure_overlay_manifest(segmentation, source_model=source_model)
        from quantem.segmentation.overlay_ngff.snapshots import schedule_snapshot_prune

        state = get_or_create_overlay_state(segmentation, source_model)
        schedule_snapshot_prune(str(segmentation.id), source_model, state.bundle_version)
        return Response(payload, status=status.HTTP_200_OK)


class SegmentationOverlayLutView(APIView):
    """Return the render-time colour/state LUT for the active overlay bundle.

    Default response is a compact binary ``(max_label + 1, 4)`` uint8 RGBA
    buffer indexed by dense label. ``?format=json`` returns the label -> object
    map used for picking and client-side toggles.
    """

    def get(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )
        source_model = normalize_source_model(request.query_params.get("source_model"))
        state = get_or_create_overlay_state(segmentation, source_model)
        from quantem.segmentation.overlay_ngff.cache import palette_cache

        assignments = None
        snapshot_root = None
        revision = request.query_params.get("rev")
        snapshot_revision = revision
        if snapshot_revision is None and state.bundle_version > 0:
            # The published store is immutable, so its label map is a cheaper
            # consistency boundary than holding a database row lock while an
            # O(N) palette is built.  A writer publishes the files before it
            # commits this state row; therefore any visible state revision has
            # a matching snapshot.
            snapshot_revision = f"{state.bundle_version}-{state.applied_revision}"
        if snapshot_revision:
            try:
                snapshot_root, snapshot = read_snapshot(
                    str(segmentation.id),
                    source_model,
                    snapshot_revision,
                    include_labels=False,
                )
            except OverlayStoreError as exc:
                if revision:
                    return Response({"detail": str(exc)}, status=status.HTTP_409_CONFLICT)
                # A legacy store can predate snapshots. Keep it readable until
                # its next rebuild, using the database mapping as before.
                snapshot_root = None
            else:
                state.bundle_version = snapshot["bundle_version"]
        # Include the database identity, palette revision and publication time.
        cache_key = (
            state._state.db,
            str(state.pk),
            str(snapshot_root),
            revision,
            state.bundle_version,
            state.lut_revision,
            state.updated_at,
            segmentation.segmentation_type.default_color,
        )
        hidden_states = frozenset(
            token.strip()
            for token in (request.query_params.get("hide") or "").split(",")
            if token.strip()
        )
        binary_key = (*cache_key, tuple(sorted(hidden_states)))
        # A binary palette can fit even when the JSON map exceeds its budget.
        # Check it before rebuilding the much larger visibility/picking map.
        if request.query_params.get("format") != "json":
            binary = palette_cache.get(binary_key)
            if binary is not None:
                return _binary_palette_response(binary, state)
        lut_json = palette_cache.get(cache_key)
        if lut_json is None:
            if snapshot_root is not None:
                try:
                    _, snapshot = read_snapshot(
                        str(segmentation.id), source_model, snapshot_revision
                    )
                except OverlayStoreError as exc:
                    return Response({"detail": str(exc)}, status=status.HTTP_409_CONFLICT)
                assignments = snapshot["labels"]
            lut_json = build_label_lut_json(state, assignments=assignments)
            # Do not hold a write lock across an O(N) read. A concurrent state
            # edit invalidates this attempt rather than caching mixed visibility
            # under the revision read before it started.
            unchanged = (
                type(state)
                .objects.filter(
                    pk=state.pk, lut_revision=state.lut_revision, updated_at=state.updated_at
                )
                .exists()
            )
            if not unchanged:
                return Response(
                    {"detail": "Overlay visibility changed; retry this palette."},
                    status=status.HTTP_409_CONFLICT,
                )
            # Conservative accounting includes Python object overhead, not just
            # compact JSON bytes. Large palettes bypass the cache altogether.
            size = 1024 + len(lut_json["objects"]) * 768
            palette_cache.put(cache_key, lut_json, size)
        if request.query_params.get("format") == "json":
            response = Response(lut_json, status=status.HTTP_200_OK)
            response["Cache-Control"] = "no-store"
            return response
        from quantem.segmentation.overlay_ngff.labels_lut import binary_from_json

        binary = palette_cache.get(binary_key)
        if binary is None:
            binary = binary_from_json(lut_json, hidden_states=hidden_states)
            palette_cache.put(binary_key, binary, len(binary[0]))
        return _binary_palette_response(binary, state)


def _binary_palette_response(binary, state):
    payload, max_label = binary
    response = HttpResponse(payload, content_type="application/octet-stream")
    response["X-Overlay-Lut-Revision"] = str(state.lut_revision)
    response["X-Overlay-Bundle-Version"] = str(state.bundle_version)
    response["X-Overlay-Max-Label"] = str(max_label)
    response["Cache-Control"] = "no-store"
    return response


class SegmentationOverlayRebuildView(APIView):
    """Queue a manual overlay rebuild."""

    def post(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )
        serializer = SegmentationOverlayRebuildSerializer(data=request.data or {})
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        mode = serializer.validated_data["mode"]
        source_model = normalize_source_model(request.data.get("source_model"))
        if mode == "full":
            overlay = queue_full_overlay_rebuild(segmentation, source_model=source_model)
        else:
            overlay = register_overlay_mutation(
                segmentation,
                dirty_bbox=full_image_dirty_bbox(segmentation),
                source_model=source_model,
            )

        state = get_or_create_overlay_state(segmentation, source_model)
        payload = ensure_overlay_manifest(segmentation, source_model=source_model)
        payload["overlay"] = overlay
        payload["status"] = state.status
        return Response(payload, status=status.HTTP_202_ACCEPTED)


def segmentation_overlay_ngff_root(request, seg_id):
    """Serve NGFF root metadata for the active segmentation overlay bundle."""

    segmentation = get_object_or_404(ImageSegmentation.objects.select_related("asset"), id=seg_id)
    source_model = normalize_source_model(request.GET.get("source_model"))
    state = get_or_create_overlay_state(segmentation, source_model)
    store_root = _requested_store(request, state)
    ensure_cached_storage_path(storage_relpath_for_path(store_root), path_type="dir")
    attrs_path = store_root / ".zattrs"
    if not attrs_path.exists():
        raise Http404("Overlay NGFF metadata not found")
    response = FileResponse(open(attrs_path, "rb"), content_type="application/json")  # noqa: SIM115 -- FileResponse closes it after streaming
    _set_revisioned_cache_headers(response, request)
    return response


def segmentation_overlay_ngff_file(request, seg_id, ngff_path):
    """Serve files from the active segmentation overlay NGFF bundle.

    Chunk paths are ``<array>/<level>/<cy>.<cx>`` (array = ``labels`` | ``border``).
    Missing chunks are background and served as compressed zeros of the array's
    dtype so the store stays sparse on disk.
    """

    segmentation = get_object_or_404(ImageSegmentation.objects.select_related("asset"), id=seg_id)
    source_model = normalize_source_model(request.GET.get("source_model"))
    state = get_or_create_overlay_state(segmentation, source_model)
    store_root = _requested_store(request, state)
    ensure_cached_storage_path(storage_relpath_for_path(store_root), path_type="dir")
    if not (store_root / ".zgroup").is_file():
        raise Http404("Overlay NGFF store not found")
    relative_path = str(ngff_path).lstrip("/")
    if not relative_path:
        raise Http404("Invalid overlay NGFF path")

    file_path = (store_root / relative_path).resolve()
    expected_root = store_root.resolve()
    try:
        file_path.relative_to(expected_root)
    except ValueError as exc:
        raise Http404("Invalid overlay NGFF path") from exc

    if not file_path.exists() or file_path.is_dir():
        if file_path.name == ".zattrs" and file_path.parent.exists() and file_path.parent.is_dir():
            response = HttpResponse("{}", content_type="application/json")
            _set_revisioned_cache_headers(response, request)
            return response
        chunk_request = parse_overlay_chunk_path(relative_path)
        if chunk_request is None:
            raise Http404(f"Overlay NGFF file not found: {relative_path}")
        array_key, level, chunk_coords = chunk_request
        chunk_shape = stored_chunk_shape(store_root, array_key, level, chunk_coords)
        if chunk_shape is None:
            raise Http404(f"Overlay NGFF file not found: {relative_path}")
        response = HttpResponse(
            _sparse_chunk_bytes(array_key, chunk_shape),
            content_type="application/octet-stream",
        )
        _set_revisioned_cache_headers(response, request)
        return response

    if file_path.name in {".zattrs", ".zarray", ".zgroup", ".zmetadata"}:
        content_type = "application/json"
    else:
        content_type = "application/octet-stream"

    response = FileResponse(open(file_path, "rb"), content_type=content_type)  # noqa: SIM115 -- FileResponse closes it after streaming
    _set_revisioned_cache_headers(response, request)
    return response
