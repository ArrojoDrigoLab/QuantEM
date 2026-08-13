import { memo } from "react";
import type { SegmentOverlay } from "@/viewer/types";
import type { ViewMetrics } from "@/viewer/components/internal/viewerMath";

function toClosedGeometry(points: import("@/utils/geometry").Point[]) {
  if (points.length < 2) return points;
  const first = points[0];
  const last = points[points.length - 1];
  if (!first || !last) return points;
  if (first.x === last.x && first.y === last.y) return points;
  return [...points, first];
}

function normalizeOverlayStyle(overlay: SegmentOverlay, highlighted: boolean) {
  if (highlighted) {
    return {
      fillColor: "#00ffff",
      fillOpacity: 0.15,
      strokeColor: "#00ffff",
      strokeOpacity: 0.15,
      strokeWidth: overlay.strokeWidth ?? 2,
      strokeDasharray: overlay.strokeDasharray,
    };
  }
  return {
    fillColor: overlay.fillColor,
    fillOpacity: overlay.fillOpacity,
    strokeColor: overlay.strokeColor,
    strokeOpacity: overlay.strokeOpacity,
    strokeWidth: overlay.strokeWidth ?? 2,
    strokeDasharray: overlay.strokeDasharray,
  };
}

const OverlaySvgLayer = memo(function OverlaySvgLayer(config: {
  overlays: SegmentOverlay[];
  highlightedSegmentId?: string | null;
}) {
  const { overlays, highlightedSegmentId } = config;
  return (
    <>
      {overlays.map((overlay) => {
        const highlighted = highlightedSegmentId === overlay.id;
        const style = normalizeOverlayStyle(overlay, highlighted);
        const isPolyline = overlay.shape === "polyline";
        const minPoints = isPolyline ? 2 : 3;
        const geometry = isPolyline ? overlay.geometry : toClosedGeometry(overlay.geometry);
        if (geometry.length < minPoints) return null;
        const points = geometry.map((point) => `${point.x},${point.y}`).join(" ");
        const fill = isPolyline || style.fillOpacity <= 0 ? "none" : style.fillColor;
        if (isPolyline) {
          return (
            <polyline
              key={overlay.id}
              points={points}
              fill={fill}
              stroke={style.strokeColor}
              strokeOpacity={style.strokeOpacity}
              strokeWidth={style.strokeWidth}
              strokeDasharray={style.strokeDasharray}
              strokeLinecap="round"
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
            />
          );
        }
        const holeRings = (overlay.holes ?? []).filter((ring) => ring.length >= 3);
        if (holeRings.length > 0) {
          // Render exterior + interior rings as a single even-odd path so the
          // excluded holes are visibly cut out of the fill.
          const ringToSubpath = (ring: import("@/utils/geometry").Point[]) => {
            const closed = toClosedGeometry(ring);
            return (
              "M" +
              closed.map((point) => `${point.x},${point.y}`).join("L") +
              "Z"
            );
          };
          const pathData = [geometry, ...holeRings].map(ringToSubpath).join(" ");
          return (
            <path
              key={overlay.id}
              d={pathData}
              fillRule="evenodd"
              fill={fill}
              fillOpacity={style.fillOpacity}
              stroke={style.strokeColor}
              strokeOpacity={style.strokeOpacity}
              strokeWidth={style.strokeWidth}
              strokeDasharray={style.strokeDasharray}
              vectorEffect="non-scaling-stroke"
            />
          );
        }
        return (
          <polygon
            key={overlay.id}
            points={points}
            fill={fill}
            fillOpacity={isPolyline ? 0 : style.fillOpacity}
            stroke={style.strokeColor}
            strokeOpacity={style.strokeOpacity}
            strokeWidth={style.strokeWidth}
            strokeDasharray={style.strokeDasharray}
            vectorEffect="non-scaling-stroke"
          />
        );
      })}
    </>
  );
});

export function ViewerSvgOverlay(config: {
  metrics: ViewMetrics | null;
  persistentOverlays: SegmentOverlay[];
  transientOverlays: SegmentOverlay[];
  highlightedSegmentId?: string | null;
}) {
  const { metrics, persistentOverlays, transientOverlays, highlightedSegmentId } = config;
  const viewBox = metrics
    ? `${metrics.minX} ${metrics.minY} ${metrics.visibleWidth} ${metrics.visibleHeight}`
    : "0 0 1 1";
  return (
    <svg
      width="100%"
      height="100%"
      viewBox={viewBox}
      preserveAspectRatio="none"
      style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
    >
      <OverlaySvgLayer
        overlays={persistentOverlays}
        highlightedSegmentId={highlightedSegmentId}
      />
      <OverlaySvgLayer
        overlays={transientOverlays}
        highlightedSegmentId={highlightedSegmentId}
      />
    </svg>
  );
}
