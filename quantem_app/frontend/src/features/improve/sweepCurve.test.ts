import { describe, expect, it } from "vitest";
import {
  PLOT,
  polylinePoints,
  scaleX,
  scaleY,
  thresholdDomain,
  toCurvePoints,
} from "@/features/improve/sweepCurve";

describe("sweepCurve", () => {
  it("drops thresholds whose Dice could not be computed", () => {
    const points = toCurvePoints([0.1, 0.2, 0.3], [0.5, null, Number.NaN]);
    expect(points).toEqual([{ threshold: 0.1, dice: 0.5 }]);
  });

  it("pins the y axis to [0, 1] so a small gain is not magnified", () => {
    expect(scaleY(1)).toBe(PLOT.top);
    expect(scaleY(0)).toBe(PLOT.top + PLOT.height);
    // Out-of-range values clamp rather than escaping the plot box.
    expect(scaleY(1.4)).toBe(PLOT.top);
    expect(scaleY(-0.2)).toBe(PLOT.top + PLOT.height);
  });

  it("maps the threshold domain across the plot width", () => {
    const domain = thresholdDomain([0.05, 0.5, 0.95]);
    expect(domain).toEqual({ min: 0.05, max: 0.95 });
    expect(scaleX(0.05, domain)).toBeCloseTo(PLOT.left, 6);
    expect(scaleX(0.95, domain)).toBeCloseTo(PLOT.left + PLOT.width, 6);
    expect(scaleX(0.5, domain)).toBeCloseTo(PLOT.left + PLOT.width / 2, 6);
  });

  it("gives a single-threshold sweep a domain it can be drawn in", () => {
    const domain = thresholdDomain([0.5]);
    expect(domain.min).toBeLessThan(domain.max);
    expect(Number.isFinite(scaleX(0.5, domain))).toBe(true);
  });

  it("renders a polyline the browser can parse", () => {
    const domain = thresholdDomain([0, 1]);
    const points = polylinePoints(
      [
        { threshold: 0, dice: 0 },
        { threshold: 1, dice: 1 },
      ],
      domain
    );
    expect(points).toBe(
      `${PLOT.left},${PLOT.top + PLOT.height} ${PLOT.left + PLOT.width},${PLOT.top}`
    );
  });
});
