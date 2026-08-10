import { describe, expect, it } from "vitest";
import {
  generatePendingPolygonOverlay,
  generateSelectionBBoxOverlay,
} from "@/features/segmentation/overlays/draft";

describe("segmentation draft overlays", () => {
  it("renders the group selection bbox overlay", () => {
    expect(generateSelectionBBoxOverlay({ x0: 1, y0: 2, x1: 5, y1: 6 })?.id).toBe(
      "right-selection-bbox"
    );
    expect(generateSelectionBBoxOverlay(null)).toBeNull();
  });

  it("builds the pending polygon overlay", () => {
    expect(
      generatePendingPolygonOverlay([
        { x: 1, y: 1 },
        { x: 5, y: 1 },
        { x: 5, y: 5 },
      ])
    ).toMatchObject({ id: "pending-polygon" });
    expect(generatePendingPolygonOverlay(null)).toBeNull();
  });
});
