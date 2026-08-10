import { describe, expect, it } from "vitest";
import {
  buildVivSelection,
  buildVivSelections,
  getDepthFromSource,
} from "@/viewer/components/internal/vivUtils";
import type { VivLoaderSource } from "@/viewer/components/internal/vivUtils";

function source(labels: string[], shape: number[]): VivLoaderSource {
  return { labels, shape, dtype: "uint8" } as unknown as VivLoaderSource;
}

describe("buildVivSelection", () => {
  it("returns an empty selection for a 2D [c, y, x] source", () => {
    expect(buildVivSelection(source(["c", "y", "x"], [1, 256, 256]))).toEqual({});
  });

  it("pins the z axis to the requested index for a 3D [c, z, y, x] source", () => {
    expect(
      buildVivSelection(source(["c", "z", "y", "x"], [1, 20, 256, 256]), { zIndex: 7 })
    ).toEqual({ z: 7 });
  });

  it("defaults z to 0 and floors/clamps negative indices", () => {
    const labels = ["c", "z", "y", "x"];
    expect(buildVivSelection(source(labels, [1, 20, 256, 256]))).toEqual({ z: 0 });
    expect(
      buildVivSelection(source(labels, [1, 20, 256, 256]), { zIndex: -3 })
    ).toEqual({ z: 0 });
    expect(
      buildVivSelection(source(labels, [1, 20, 256, 256]), { zIndex: 4.9 })
    ).toEqual({ z: 4 });
  });

  it("threads zIndex through buildVivSelections", () => {
    expect(
      buildVivSelections(source(["c", "z", "y", "x"], [1, 20, 256, 256]), { zIndex: 3 })
    ).toEqual([{ z: 3 }]);
  });
});

describe("getDepthFromSource", () => {
  it("reads the z-axis length for 3D sources", () => {
    expect(getDepthFromSource(source(["c", "z", "y", "x"], [1, 20, 256, 256]))).toBe(20);
  });

  it("returns 1 when there is no z axis", () => {
    expect(getDepthFromSource(source(["c", "y", "x"], [1, 256, 256]))).toBe(1);
  });
});
