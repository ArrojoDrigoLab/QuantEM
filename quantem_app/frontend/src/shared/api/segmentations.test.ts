import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getSegmentsAtPoint,
} from "@/shared/api/segmentations/annotations";
import {
  getSegmentationOverlayManifest,
  getSegmentationOverlayNgffUrl,
} from "@/shared/api/segmentations/overlays";
import { setApiConfig } from "@/shared/api/core/http";

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("shared/api/segmentations", () => {
  beforeEach(() => {
    setApiConfig({ apiBaseUrl: "http://127.0.0.1:9000" });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    setApiConfig({ apiBaseUrl: "http://127.0.0.1:8000" });
  });

  it("resolves relative overlay-manifest NGFF URLs", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          status: "READY",
          ngff_url: "/segmentation-overlays/seg-1.zarr",
          lut_url: "/api/segmentations/seg-1/overlay-lut/",
          arrays: ["labels", "border"],
          label_dtype: "uint32",
          bundle_version: 1,
          applied_revision: 2,
          desired_revision: 2,
          lut_revision: 2,
          chunk_size: [256, 256],
          level_count: 1,
          width: 256,
          height: 256,
        })
      )
    );

    const manifest = await getSegmentationOverlayManifest("seg-1");
    expect(manifest.ngff_url).toBe("http://127.0.0.1:9000/segmentation-overlays/seg-1.zarr");
  });

  it("builds overlay NGFF URLs against the configured API base", () => {
    expect(getSegmentationOverlayNgffUrl("seg-1", "rev-4")).toBe(
      "http://127.0.0.1:9000/segmentation-overlays/seg-1.zarr?v=rev-4"
    );
  });

  it("sends point-query params to the at-point endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse([]));
    vi.stubGlobal("fetch", fetchMock);

    await getSegmentsAtPoint("seg-1", {
      x: 10,
      y: 12,
      states: ["CANDIDATE", "INFERRED"],
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:9000/api/segmentations/seg-1/segments/at-point?x=10&y=12&states=CANDIDATE%2CINFERRED",
      expect.any(Object)
    );
  });
});
