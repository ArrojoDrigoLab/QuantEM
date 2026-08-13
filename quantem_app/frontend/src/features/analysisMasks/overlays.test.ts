import { describe, expect, it } from "vitest";
import { analysisMaskObjectOverlays } from "./overlays";
import type { AnalysisMaskObject } from "./types";

describe("analysisMaskObjectOverlays", () => {
  it("renders every polygon in one object with its random color and 10% fill", () => {
    const object: AnalysisMaskObject = {
      id: "object-1",
      segmentation: "seg-1",
      name: "Object 1",
      color: "#f97316",
      sort_order: 1,
      geometry: {
        type: "MultiPolygon",
        coordinates: [
          [
            [
              [0, 0],
              [20, 0],
              [20, 20],
              [0, 20],
              [0, 0],
            ],
            [
              [5, 5],
              [10, 5],
              [10, 10],
              [5, 10],
              [5, 5],
            ],
          ],
          [
            [
              [30, 30],
              [40, 30],
              [40, 40],
              [30, 40],
              [30, 30],
            ],
          ],
        ],
      },
      created_at: "2026-08-13T00:00:00Z",
      updated_at: "2026-08-13T00:00:00Z",
    };

    const overlays = analysisMaskObjectOverlays(object, true);

    expect(overlays).toHaveLength(2);
    expect(overlays.every((overlay) => overlay.fillColor === "#f97316")).toBe(true);
    expect(overlays.every((overlay) => overlay.fillOpacity === 0.1)).toBe(true);
    expect(overlays.every((overlay) => overlay.strokeWidth === 3)).toBe(true);
    expect(overlays[0]?.holes).toHaveLength(1);
  });
});
