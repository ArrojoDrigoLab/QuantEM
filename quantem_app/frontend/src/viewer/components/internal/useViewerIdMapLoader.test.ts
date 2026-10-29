import { describe, expect, it, vi } from "vitest";
import { combineOverlaySources } from "./useViewerIdMapLoader";
import type { VivLoaderData } from "./vivUtils";

describe("combineOverlaySources", () => {
  it("uses a stable loader and fetches the matching border selection", async () => {
    const getLabels = vi.fn().mockResolvedValue({ data: new Uint32Array([7]), width: 1, height: 1 });
    const getBorder = vi.fn().mockResolvedValue({ data: new Uint8Array([1]), width: 1, height: 1 });
    const labels = [{ getTile: getLabels }] as unknown as VivLoaderData;
    const borders = [{ getTile: getBorder }] as unknown as VivLoaderData;
    const combined = combineOverlaySources(labels, borders);
    expect(combineOverlaySources(labels, borders)).toBe(combined);
    const signal = new AbortController().signal;
    await combined[0].getTile({ x: 2, y: 3, selection: { overlayArray: 0 }, signal });
    await combined[0].getTile({ x: 2, y: 3, selection: { overlayArray: 1 }, signal });
    expect(getLabels).toHaveBeenCalledExactlyOnceWith({ x: 2, y: 3, selection: {}, signal });
    expect(getBorder).toHaveBeenCalledExactlyOnceWith({ x: 2, y: 3, selection: {}, signal });
  });
});
