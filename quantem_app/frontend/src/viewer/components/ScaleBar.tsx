/**
 * How big is that?
 *
 * Every measurement this app produces is in physical units, and until now the
 * canvas showed none: an object on screen was a number of screen pixels at an
 * unknown zoom, so the one question a microscopist asks first had no answer
 * without leaving the image.
 *
 * The bar is drawn only when the image has a real pixel size. An image with no
 * calibration gets nothing rather than a bar in pixels dressed up as a
 * distance, because a scale bar that does not mean nanometres is worse than no
 * scale bar at all.
 */

import type { ViewMetrics } from "@/viewer/components/internal/viewerMath";
import { scaleBarPlan } from "@/viewer/components/internal/viewerMath";

/** The bar never grows past this, so it stays out of the way of the image. */
export const SCALE_BAR_MAX_PX = 160;

interface ScaleBarProps {
  metrics: ViewMetrics | null;
  pixelSizeNm: number | null | undefined;
}

export function ScaleBar({ metrics, pixelSizeNm }: ScaleBarProps) {
  if (!metrics) return null;
  const plan = scaleBarPlan(metrics, pixelSizeNm, SCALE_BAR_MAX_PX);
  if (!plan) return null;

  return (
    <div
      className="viewer-scale-bar"
      data-testid="viewer-scale-bar"
      role="img"
      aria-label={`Scale bar: ${plan.label}`}
    >
      <span
        className="viewer-scale-bar-rule"
        style={{ width: `${Math.round(plan.lengthPx)}px` }}
      />
      <span className="viewer-scale-bar-label">{plan.label}</span>
    </div>
  );
}
