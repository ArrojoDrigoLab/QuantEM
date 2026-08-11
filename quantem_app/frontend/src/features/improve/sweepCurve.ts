/**
 * Geometry for the threshold-sweep chart.
 *
 * Kept apart from the component so the scaling can be tested without a DOM.
 * The y axis is pinned to [0, 1] rather than fitted to the data: Dice is
 * bounded, and an auto-scaled axis makes a 0.02 improvement look like a
 * transformation.
 */

export interface CurvePoint {
  threshold: number;
  dice: number;
}

export interface PlotBox {
  left: number;
  right: number;
  top: number;
  bottom: number;
  width: number;
  height: number;
}

export const VIEW_WIDTH = 640;
export const VIEW_HEIGHT = 300;

export const PLOT: PlotBox = {
  left: 48,
  right: 16,
  top: 18,
  bottom: 44,
  width: VIEW_WIDTH - 48 - 16,
  height: VIEW_HEIGHT - 18 - 44,
};

/** Drop thresholds whose Dice could not be computed rather than plotting zero. */
export function toCurvePoints(
  thresholds: number[],
  dice: Array<number | null>
): CurvePoint[] {
  const points: CurvePoint[] = [];
  for (let index = 0; index < thresholds.length; index += 1) {
    const value = dice[index];
    if (value === null || value === undefined || !Number.isFinite(value)) continue;
    points.push({ threshold: thresholds[index], dice: value });
  }
  return points;
}

export interface Domain {
  min: number;
  max: number;
}

export function thresholdDomain(thresholds: number[]): Domain {
  if (thresholds.length === 0) return { min: 0, max: 1 };
  const min = Math.min(...thresholds);
  const max = Math.max(...thresholds);
  if (max === min) return { min: min - 0.5, max: max + 0.5 };
  return { min, max };
}

export function scaleX(threshold: number, domain: Domain): number {
  const span = domain.max - domain.min || 1;
  const clamped = Math.min(Math.max(threshold, domain.min), domain.max);
  return PLOT.left + ((clamped - domain.min) / span) * PLOT.width;
}

/** Dice is in [0, 1]; the axis is too. */
export function scaleY(dice: number): number {
  const clamped = Math.min(Math.max(dice, 0), 1);
  return PLOT.top + (1 - clamped) * PLOT.height;
}

export function polylinePoints(points: CurvePoint[], domain: Domain): string {
  return points
    .map((point) => `${scaleX(point.threshold, domain)},${scaleY(point.dice)}`)
    .join(" ");
}
