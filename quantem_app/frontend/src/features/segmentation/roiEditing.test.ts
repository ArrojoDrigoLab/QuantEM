import { describe, expect, it } from "vitest";

import {
  resolveRoiEditHandle,
  updateRoiForDrag,
} from "@/features/segmentation/roiEditing";

const bounds = { x: 100, y: 200, width: 512, height: 512 };

describe("ROI area editing", () => {
  it("makes the full edges and corners draggable", () => {
    expect(resolveRoiEditHandle(bounds, { x: 100, y: 200 })).toBe("north-west");
    expect(resolveRoiEditHandle(bounds, { x: 250, y: 200 })).toBe("north");
    expect(resolveRoiEditHandle(bounds, { x: 612, y: 450 })).toBe("east");
    expect(resolveRoiEditHandle(bounds, { x: 300, y: 400 })).toBe("move");
    expect(resolveRoiEditHandle(bounds, { x: 20, y: 20 })).toBeNull();
  });

  it("keeps the opposite edge fixed while resizing", () => {
    expect(
      updateRoiForDrag(
        bounds,
        "north",
        { x: 250, y: 200 },
        { x: 250, y: 150 },
        { width: 2048, height: 1536 }
      )
    ).toEqual({ x: 100, y: 150, width: 512, height: 562 });
  });
});
