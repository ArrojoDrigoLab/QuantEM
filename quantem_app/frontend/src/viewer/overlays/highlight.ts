import type { SegmentOverlay } from "@/viewer/types";

export function updateOverlayHighlight(
  overlayElement: SVGSVGElement,
  isHighlighted: boolean,
  baseColor: string,
  expectedSegmentId?: string
): void {
  let shape: SVGPolygonElement | SVGPolylineElement | null = null;

  if (expectedSegmentId) {
    shape = overlayElement.querySelector(
      `[data-segment-id="${expectedSegmentId}"]`
    ) as SVGPolygonElement | SVGPolylineElement | null;

    if (!shape) {
      const candidates = overlayElement.querySelectorAll("polygon, polyline");
      if (candidates.length === 1) {
        shape = candidates[0] as SVGPolygonElement | SVGPolylineElement;
      } else {
        console.warn("[updateOverlayHighlight] Could not find shape with expected segment ID", {
          expectedSegmentId,
          availableIds: Array.from(
            overlayElement.querySelectorAll("[data-segment-id]")
          ).map((element) => element.getAttribute("data-segment-id")),
          children: Array.from(overlayElement.children).map((child) => ({
            tag: child.tagName,
            id: child.getAttribute("data-segment-id"),
          })),
        });
        return;
      }
    }
  } else {
    shape = overlayElement.firstElementChild as SVGPolygonElement | SVGPolylineElement | null;
    if (!shape || (shape.tagName !== "polygon" && shape.tagName !== "polyline")) {
      console.warn("[updateOverlayHighlight] No valid shape found and no expectedSegmentId provided");
      return;
    }
  }

  const actualSegmentId = shape.getAttribute("data-segment-id");
  if (expectedSegmentId && actualSegmentId && actualSegmentId !== expectedSegmentId) {
    console.warn("[updateOverlayHighlight] Segment ID mismatch - applying highlight anyway", {
      expectedSegmentId,
      actualSegmentId,
      overlayElementId: overlayElement.id,
      svgChildren: Array.from(overlayElement.children).map((child) => child.tagName),
    });
  }

  applyShapeHighlight(shape, isHighlighted, baseColor);
}

export function applyOverlayHighlight(
  shape: SVGPolygonElement | SVGPolylineElement | SVGPathElement,
  isHighlighted: boolean,
  overlay: Pick<
    SegmentOverlay,
    "fillColor" | "fillOpacity" | "strokeColor" | "strokeOpacity"
  >
): void {
  if (isHighlighted) {
    shape.setAttribute("fill", "#00ffff");
    shape.setAttribute("fill-opacity", "0.15");
    shape.setAttribute("stroke", "#00ffff");
    shape.setAttribute("stroke-opacity", "0.15");
    return;
  }

  shape.setAttribute("fill", overlay.fillColor);
  shape.setAttribute("fill-opacity", overlay.fillOpacity.toString());
  shape.setAttribute("stroke", overlay.strokeColor);
  shape.setAttribute("stroke-opacity", overlay.strokeOpacity.toString());
}

export function applyShapeHighlight(
  shape: SVGPolygonElement | SVGPolylineElement | SVGPathElement,
  isHighlighted: boolean,
  baseColor: string
): void {
  applyOverlayHighlight(shape, isHighlighted, {
    fillColor: baseColor,
    fillOpacity: 0,
    strokeColor: baseColor,
    strokeOpacity: 0.3,
  });
}
