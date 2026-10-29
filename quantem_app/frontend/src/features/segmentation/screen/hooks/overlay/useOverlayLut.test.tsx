import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useOverlayLut } from "./useOverlayLut";
import { getSegmentationOverlayLut } from "@/shared/api/segmentations/overlays";
import type { OverlayLutBinary } from "@/shared/types/segmentation";

vi.mock("@/shared/api/segmentations/overlays", () => ({
  getSegmentationOverlayLut: vi.fn(), getSegmentationOverlayLutJson: vi.fn(),
}));

describe("palette generations", () => {
  it("never applies a late palette to a replacement raster", async () => {
    let finishOld!: (value: OverlayLutBinary) => void;
    const old = new Promise<OverlayLutBinary>((resolve) => { finishOld = resolve; });
    const fresh = { rgba: new Uint8Array([0, 0, 0, 0, 0, 255, 0, 255]), maxLabel: 1, bundleVersion: 2, lutRevision: 2 };
    vi.mocked(getSegmentationOverlayLut).mockReturnValueOnce(old).mockResolvedValueOnce(fresh);
    const { result, rerender } = renderHook(({ raster }) => useOverlayLut({
      segmentationId: "seg", lutRevision: 2, rasterRevision: raster,
    }), { initialProps: { raster: "1-1" } });
    rerender({ raster: "2-2" });
    expect(result.current).toBeNull();
    await waitFor(() => expect(result.current?.bundleVersion).toBe(2));
    await act(async () => { finishOld({ ...fresh, bundleVersion: 1 }); await old; });
    expect(result.current?.bundleVersion).toBe(2);
    expect(getSegmentationOverlayLut).toHaveBeenLastCalledWith("seg", null, undefined, "2-2");
  });
});
