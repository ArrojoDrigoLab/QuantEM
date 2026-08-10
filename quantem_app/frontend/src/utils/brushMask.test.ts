import { describe, expect, it } from "vitest";
import type { Point } from "@/utils/geometry";
import { brushStrokesToConnectedPolygons } from "@/utils/brushMask";

function area(points: Point[]): number {
  if (points.length < 4) {
    return 0;
  }
  let total = 0;
  for (let index = 0; index < points.length - 1; index += 1) {
    const current = points[index];
    const next = points[index + 1];
    total += current.x * next.y - next.x * current.y;
  }
  return Math.abs(total) / 2;
}

describe("brushStrokesToConnectedPolygons", () => {
  it("merges overlapping brush strokes into one connected polygon", () => {
    const polygons = brushStrokesToConnectedPolygons([
      {
        size: 10,
        points: [
          { x: 10, y: 20 },
          { x: 40, y: 20 },
        ],
      },
      {
        size: 10,
        points: [
          { x: 35, y: 20 },
          { x: 65, y: 20 },
        ],
      },
    ]);

    expect(polygons).toHaveLength(1);
    expect(area(polygons[0])).toBeGreaterThan(0);
  });

  it("keeps unconnected brush regions as separate polygons", () => {
    const polygons = brushStrokesToConnectedPolygons([
      {
        size: 8,
        points: [
          { x: 10, y: 10 },
          { x: 25, y: 10 },
        ],
      },
      {
        size: 8,
        points: [
          { x: 120, y: 120 },
          { x: 140, y: 120 },
        ],
      },
    ]);

    expect(polygons).toHaveLength(2);
  });

  it("produces non-bbox geometry for bent brush paths", () => {
    const [polygon] = brushStrokesToConnectedPolygons([
      {
        size: 6,
        points: [
          { x: 10, y: 10 },
          { x: 30, y: 10 },
          { x: 30, y: 30 },
        ],
      },
    ]);

    const polygonArea = area(polygon);
    expect(polygonArea).toBeGreaterThan(120);
    expect(polygonArea).toBeLessThan(500);
  });

  it("treats repeated painting over the same region as binary", () => {
    const stroke = {
      size: 12,
      points: [
        { x: 20, y: 20 },
        { x: 50, y: 20 },
      ],
    };

    const once = brushStrokesToConnectedPolygons([stroke]);
    const twice = brushStrokesToConnectedPolygons([stroke, stroke]);

    const onceArea = once.reduce((sum, polygon) => sum + area(polygon), 0);
    const twiceArea = twice.reduce((sum, polygon) => sum + area(polygon), 0);

    expect(once).toHaveLength(1);
    expect(twice).toHaveLength(1);
    expect(Math.abs(onceArea - twiceArea)).toBeLessThan(1e-6);
  });
});
