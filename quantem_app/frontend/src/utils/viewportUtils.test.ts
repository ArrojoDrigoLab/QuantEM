import { describe, expect, it } from "vitest";
import { calculateViewportBbox } from "@/utils/viewportUtils";
import type { ViewportState } from "@/viewer/types";

describe("calculateViewportBbox", () => {
  it("returns undefined for invalid inputs", () => {
    expect(calculateViewportBbox(null, 1000, 500)).toBeUndefined();
    expect(
      calculateViewportBbox(
        { centerX: 0.5, centerY: 0.5, zoom: 1, containerWidth: 100, containerHeight: 100 },
        0,
        500
      )
    ).toBeUndefined();
  });

  it("calculates bbox with padding and container aspect ratio", () => {
    const viewport: ViewportState = {
      centerX: 0.5,
      centerY: 0.25,
      zoom: 2,
      containerWidth: 1000,
      containerHeight: 500,
    };

    expect(calculateViewportBbox(viewport, 1000, 500)).toEqual({
      x_min: 150,
      y_min: 75,
      x_max: 850,
      y_max: 425,
    });
  });

  it("uses image width scale for centerY conversion", () => {
    const viewport: ViewportState = {
      centerX: 0.5,
      centerY: 1,
      zoom: 10,
      containerWidth: 1000,
      containerHeight: 1000,
    };

    const bbox = calculateViewportBbox(viewport, 1000, 2000);
    expect(bbox).toBeDefined();
    expect(bbox?.y_min).toBeGreaterThan(900);
    expect(bbox?.y_min).toBeLessThan(1000);
  });
});
