import { describe, expect, it } from "vitest";
import { polygonIntersectsRect, simplifyPolygon } from "@/utils/geometry";

describe("polygonIntersectsRect", () => {
  const square = [
    { x: 10, y: 10 },
    { x: 20, y: 10 },
    { x: 20, y: 20 },
    { x: 10, y: 20 },
  ];

  it("returns true when rectangle overlaps polygon interior", () => {
    expect(
      polygonIntersectsRect(square, { x0: 15, y0: 15, x1: 25, y1: 25 })
    ).toBe(true);
  });

  it("returns true when rectangle only touches polygon edge", () => {
    expect(
      polygonIntersectsRect(square, { x0: 20, y0: 12, x1: 25, y1: 18 })
    ).toBe(true);
  });

  it("returns false when rectangle is disjoint", () => {
    expect(
      polygonIntersectsRect(square, { x0: 30, y0: 30, x1: 40, y1: 40 })
    ).toBe(false);
  });
});

describe("simplifyPolygon", () => {
  it("returns a closed polygon ring", () => {
    const polygon = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
      { x: 5, y: 11 },
      { x: 0, y: 10 },
      { x: 0, y: 0 },
    ];

    const simplified = simplifyPolygon(polygon, 1.0);
    expect(simplified.length).toBeGreaterThanOrEqual(4);
    expect(simplified[0]).toEqual(simplified[simplified.length - 1]);
  });

  it("closes open polygon input before returning", () => {
    const openTriangle = [
      { x: 0, y: 0 },
      { x: 12, y: 0 },
      { x: 6, y: 8 },
    ];

    const simplified = simplifyPolygon(openTriangle, 1.0);
    expect(simplified[0]).toEqual(simplified[simplified.length - 1]);
    expect(simplified.length).toBe(4);
  });
});
