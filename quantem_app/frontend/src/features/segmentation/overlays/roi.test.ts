import { describe, expect, it } from "vitest";
import {
  generateDrawStrokeOverlays,
  generateRoiOverlays,
  generateRoiStrokeOverlays,
} from "@/features/segmentation/overlays/roi";

describe("segmentation overlay roi", () => {
  it("generates ROI frame overlays", () => {
    const overlays = generateRoiOverlays({
      id: "roi-1",
      segmentation: "seg-1",
      x: 20,
      y: 30,
      width: 40,
      height: 50,
      source: "AUTO",
      seed: null,
      is_active: true,
      is_complete: false,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    });

    expect(overlays).toHaveLength(1);
    expect(overlays[0].id).toBe("roi-frame");
    expect(overlays[0].geometry).toHaveLength(5);
  });

  it("keeps the active frame solid and renders inactive frames dashed at 40%", () => {
    const active = {
      id: "roi-active",
      segmentation: "seg-1",
      x: 20,
      y: 30,
      width: 40,
      height: 50,
      source: "AUTO" as const,
      seed: null,
      is_active: true,
      is_complete: false,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    };
    const inactive = {
      ...active,
      id: "roi-inactive",
      x: 100,
      is_active: false,
    };

    const overlays = generateRoiOverlays(active, [active, inactive]);

    expect(overlays).toHaveLength(2);
    expect(overlays.at(-1)).toMatchObject({
      id: "roi-frame",
      strokeColor: "#ffd166",
      strokeOpacity: 0.9,
      strokeDasharray: undefined,
    });
    expect(overlays[0]).toMatchObject({
      id: "roi-frame-roi-inactive",
      strokeColor: "#ffd166",
      strokeOpacity: 0.4,
      strokeDasharray: "8 6",
    });
  });

  it("builds roi stroke overlays for point and brush strokes", () => {
    const overlays = generateRoiStrokeOverlays([
      { id: "point", label: 1, size: 8, points: [{ x: 5, y: 5 }] },
      {
        id: "brush",
        label: 0,
        size: 12,
        points: [
          { x: 1, y: 1 },
          { x: 8, y: 8 },
        ],
      },
    ]);

    expect(overlays).toHaveLength(2);
    expect(overlays[0]?.id).toBe("roi-stroke-point");
    expect(overlays[1]?.shape).toBe("polyline");
  });

  it("builds filled draw overlays from brush strokes", () => {
    const overlays = generateDrawStrokeOverlays([
      {
        id: "draw-1",
        label: 1,
        size: 10,
        points: [
          { x: 0, y: 0 },
          { x: 10, y: 0 },
          { x: 10, y: 10 },
        ],
      },
    ]);

    expect(overlays.length).toBeGreaterThan(0);
    expect(overlays[0]).toMatchObject({
      fillColor: "#33cc66",
      strokeColor: "#2aa957",
    });
  });
});
