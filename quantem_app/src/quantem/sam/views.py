"""HTTP for box-prompted object adding.

Three routes: what the weights' situation is, fetch them, and prompt a box.

The prompt is **synchronous**. The implementation this was ported from had two
flows -- a sync preview and an async create-poll-acknowledge with its own table,
batching and TTLs -- which is the right shape for a multi-tenant server and the
wrong one here. This is a single-user loopback desktop app; the first box in a
neighbourhood pays the encoder (about half a second) and every later box in it
pays the decoder (tens of milliseconds), which is inside the budget for a
request a person is waiting on.
"""

from __future__ import annotations

import logging

from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.assets.asset_openable import get_asset_openable
from quantem.sam import checkpoint
from quantem.sam.backends import stub_mode
from quantem.sam.checkpoint import CheckpointMissing
from quantem.sam.config import CHECKPOINT
from quantem.sam.geometry import Box, polygon_coords
from quantem.sam.prompt import (
    PromptRefused,
    candidates_payload,
    run_prompt,
    store_top_candidate,
)

logger = logging.getLogger(__name__)


def _error(message: str, code: int) -> Response:
    return Response({"detail": message}, status=code)


class SamModelView(APIView):
    """``GET /api/sam/model/`` -- are the weights here, and if not, how far along.

    Polled by the client while a download runs. Cheap: a ``stat`` and a
    dictionary copy.
    """

    def get(self, request: Request) -> Response:
        payload = checkpoint.status()
        payload["stub_mode"] = stub_mode()
        return Response(payload, status=status.HTTP_200_OK)


class SamModelDownloadView(APIView):
    """``POST /api/sam/model/download/`` -- fetch the weights.

    Returns immediately with the status the client should start polling. Asking
    twice is harmless: an in-flight transfer is reported, not restarted.
    """

    def post(self, request: Request) -> Response:
        payload = checkpoint.start_download()
        code = status.HTTP_200_OK if payload["installed"] else status.HTTP_202_ACCEPTED
        return Response(payload, status=code)


class SamBoxPromptView(APIView):
    """``POST /api/sam/segmentations/<seg_id>/box/``

    Body::

        {"box": {"x0": 100, "y0": 120, "x1": 260, "y1": 240}}

    On success, ``201`` with the object that was created, the masks that were
    not, and a small ``timing`` block so a slow first prompt is explainable
    rather than merely slow.
    """

    def post(self, request: Request, seg_id) -> Response:
        from django.shortcuts import get_object_or_404

        from quantem.segmentation.api_views.shared import completion_lock_response
        from quantem.segmentation.models import ImageSegmentation

        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked

        box, message = _parse_box(request.data)
        if box is None:
            return _error(message, status.HTTP_400_BAD_REQUEST)

        if not stub_mode() and not checkpoint.installed():
            return _error(
                f"{CHECKPOINT.display_name} has not been downloaded yet. "
                "Download it, then draw the box again.",
                status.HTTP_409_CONFLICT,
            )

        asset = getattr(segmentation, "asset", None)
        if asset is None:
            return _error(
                "This segmentation has no image, so there is nothing to segment.",
                status.HTTP_400_BAD_REQUEST,
            )
        openable = get_asset_openable(asset, require=False)
        if openable is None:
            return _error(
                "QuantEM cannot open this image's pixels on this computer.",
                status.HTTP_409_CONFLICT,
            )

        width, height = int(openable.width), int(openable.height)
        if width <= 0 or height <= 0:
            return _error(
                "This image's size is not recorded, so a box cannot be placed on it.",
                status.HTTP_409_CONFLICT,
            )

        try:
            result = run_prompt(
                segmentation_id=str(segmentation.id),
                openable=openable,
                box=box,
                image_width=width,
                image_height=height,
            )
        except PromptRefused as exc:
            return _error(str(exc), status.HTTP_400_BAD_REQUEST)
        except CheckpointMissing as exc:
            return _error(str(exc), status.HTTP_409_CONFLICT)
        except MemoryError:
            return _error(
                "This computer ran out of memory while segmenting that box. Try a smaller box.",
                status.HTTP_507_INSUFFICIENT_STORAGE,
            )
        except Exception as exc:  # pragma: no cover - last-resort sentence
            if type(exc).__name__ == "OutOfMemoryError":
                return _error(
                    "The graphics card ran out of memory while segmenting that "
                    "box. Try a smaller box.",
                    status.HTTP_507_INSUFFICIENT_STORAGE,
                )
            logger.exception("Box prompt failed for segmentation %s", segmentation.id)
            return _error(
                "Segmenting that box did not work. Try again, or draw the box "
                "a little differently.",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        outcome = store_top_candidate(segmentation=segmentation, result=result)
        measurement = outcome.pop("measurement", None)
        outcome.pop("dirty_bbox", None)

        from quantem.segmentation.services.confirm_batch import (
            register_confirmation_overlay_mutation,
        )

        overlay = register_confirmation_overlay_mutation(
            segmentation=segmentation,
            result={**outcome, "dirty_bbox": None},
            fallback_geometries=[result.top.polygon],
        )

        body = {
            **outcome,
            "overlay": overlay,
            "measurement": measurement.as_payload() if measurement is not None else None,
            "object": {
                "geometry_coords": polygon_coords(result.top.polygon),
                "score": round(result.top.score, 4),
                "area": round(result.top.area, 1),
            },
            "other_candidates": candidates_payload(result),
            "timing": {
                "cache_hit": result.cache_hit,
                "encode_ms": result.encode_ms,
                "decode_ms": result.decode_ms,
                "device": result.device,
            },
        }
        return Response(body, status=status.HTTP_201_CREATED)


def _parse_box(data) -> tuple[Box | None, str]:
    """A validated box from the request body, or the sentence to send back."""
    raw = data.get("box") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return None, 'Send the drawn box as "box", with x0, y0, x1 and y1.'
    try:
        corners = [float(raw[name]) for name in ("x0", "y0", "x1", "y1")]
    except (KeyError, TypeError, ValueError):
        return None, "The box needs four numbers: x0, y0, x1 and y1."
    if any(value != value or value in (float("inf"), float("-inf")) for value in corners):
        return None, "The box needs four ordinary numbers."
    box = Box.normalized(*corners)
    if box.width < 1.0 or box.height < 1.0:
        return None, "That box is too small. Drag out a box around one object."
    return box, ""
