import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  deleteAsset,
  getAssetNgffThumbnailUrl,
  getAssetPreviewPngUrl,
  getAssetPreviewThumbnailUrl,
  getHomeEntries,
  getHomeEntryPage,
  getHomeImages,
  getAssetNgffUrl,
  updateAsset,
  updateExperiment,
  uploadAsset,
} from "@/shared/api/assets";
import { setApiConfig } from "@/shared/api/core/http";

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("shared/api/assets", () => {
  beforeEach(() => {
    setApiConfig({ apiBaseUrl: "http://127.0.0.1:9000" });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    setApiConfig({ apiBaseUrl: "http://127.0.0.1:8000" });
  });

  it("builds preview and NGFF URLs against the configured API base", () => {
    expect(getAssetPreviewPngUrl("asset-1", "cache-1")).toBe(
      "http://127.0.0.1:9000/api/assets/asset-1/preview-png?v=cache-1"
    );
    expect(getAssetNgffThumbnailUrl("asset-1", "cache-1")).toBe(
      "http://127.0.0.1:9000/api/assets/asset-1/ngff-thumbnail/?v=cache-1"
    );
    expect(getAssetPreviewThumbnailUrl("asset-1", "cache-1")).toBe(
      "http://127.0.0.1:9000/api/assets/asset-1/preview-thumbnail/?v=cache-1"
    );
    expect(getAssetNgffUrl("asset-1", "rev-2")).toBe(
      "http://127.0.0.1:9000/ngff/assets/asset-1.zarr?v=rev-2"
    );
  });

  it("sends home-image query params and auth headers", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse([]));
    vi.stubGlobal("fetch", fetchMock);

    await getHomeImages({ search: "mito", ordering: "-created_at" });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:9000/api/assets/?search=mito&ordering=-created_at",
      expect.objectContaining({
        headers: expect.objectContaining({
          "Content-Type": "application/json",
        }),
      })
    );
  });

  it("sends unified home-entry availability params", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse([]));
    vi.stubGlobal("fetch", fetchMock);

    await getHomeEntries({
      search: "mito",
      availability: "catalog",
      ordering: "-created_at",
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:9000/api/assets/?search=mito&ordering=-created_at&availability=catalog",
      expect.objectContaining({
        headers: expect.objectContaining({
          "Content-Type": "application/json",
        }),
      })
    );
  });

  it("sends paged home-entry params", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({ results: [], total: 0, limit: 60, offset: 0, has_more: false })
      );
    vi.stubGlobal("fetch", fetchMock);

    await getHomeEntryPage({
      search: "islet",
      availability: "all" as const,
      species: ["Mus musculus", "__none__"],
      organ: ["Pancreas"],
      confirmed: ["confirmed"],
      tile_status: ["partial"],
      limit: 60,
      offset: 120,
    });

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://127.0.0.1:9000/api/assets/?search=islet&species=Mus+musculus&species=__none__&organ=Pancreas&confirmed=confirmed&tile_status=partial&availability=all&limit=60&offset=120",
      expect.any(Object)
    );
  });

  it("builds asset delete requests", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);

    await deleteAsset("asset-1");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://127.0.0.1:9000/api/assets/asset-1/",
      expect.objectContaining({
        method: "DELETE",
      })
    );
  });

  it("patches the pixel size, including clearing it back to uncalibrated", async () => {
    // A fresh Response per call: a body can only be read once.
    const fetchMock = vi
      .fn()
      .mockImplementation(async () => jsonResponse({ id: "asset-1", pixel_size_nm: 4.2 }));
    vi.stubGlobal("fetch", fetchMock);

    // The one call that lets an EM export with no resolution tag ever produce
    // a µm² number.
    await updateAsset("asset-1", { pixel_size_nm: 4.2 });
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://127.0.0.1:9000/api/assets/asset-1/",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ pixel_size_nm: 4.2 }),
      })
    );

    // null is meaningful: the backend reads it as "unknown".
    await updateAsset("asset-1", { pixel_size_nm: null });
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://127.0.0.1:9000/api/assets/asset-1/",
      expect.objectContaining({ body: JSON.stringify({ pixel_size_nm: null }) })
    );
  });

  it("sends the upload fields the backend actually reads", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ id: "asset-1", pixel_size_nm: 4.2 }));
    vi.stubGlobal("fetch", fetchMock);

    await uploadAsset(new File(["x"], "scan.png", { type: "image/png" }), {
      pixelSizeNm: 4.2,
      segmentMito: true,
    });

    const body = fetchMock.mock.calls[0][1].body as FormData;
    expect(body.get("pixel_size_nm")).toBe("4.2");
    expect(body.get("segment_mito")).toBe("true");
    // `use_mitonet` went out with MitoNet; no view reads it.
    expect(body.get("use_mitonet")).toBeNull();
  });

  it("omits the pixel size when the user did not type one", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ id: "asset-1" }));
    vi.stubGlobal("fetch", fetchMock);

    await uploadAsset(new File(["x"], "scan.tif"), { pixelSizeNm: null });

    // Sending an empty value would not hurt, but omitting it keeps the file's
    // own declared value as the only source.
    expect((fetchMock.mock.calls[0][1].body as FormData).get("pixel_size_nm")).toBeNull();
  });

  it("patches experiment metadata", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ id: "exp-1", confirmed_assets: true }));
    vi.stubGlobal("fetch", fetchMock);

    await updateExperiment("exp-1", { confirmed_assets: true });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:9000/api/experiments/exp-1/",
      expect.objectContaining({
        body: JSON.stringify({ confirmed_assets: true }),
        method: "PATCH",
      })
    );
  });
});
