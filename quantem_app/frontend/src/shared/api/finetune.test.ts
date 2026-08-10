import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  applyAdapter,
  getAdaptCrops,
  getAdapter,
  getModelCatalogue,
  installModelPack,
  startAdaptation,
} from "@/shared/api/finetune";
import { setApiConfig } from "@/shared/api/core/http";

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("shared/api/finetune", () => {
  beforeEach(() => {
    setApiConfig({ apiBaseUrl: "http://127.0.0.1:9000" });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    setApiConfig({ apiBaseUrl: "http://127.0.0.1:8000" });
  });

  it("reads the catalogue and the crops", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ packs: [], adapted: [], device: null }))
      .mockResolvedValueOnce(jsonResponse({ crops: [] }));
    vi.stubGlobal("fetch", fetchMock);

    await getModelCatalogue();
    await getAdaptCrops("seg-1");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://127.0.0.1:9000/api/models/",
      expect.any(Object)
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://127.0.0.1:9000/api/segmentations/seg-1/adapt/crops/",
      expect.any(Object)
    );
  });

  it("escapes the pack id in the install route", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ job_id: "job-1" }));
    vi.stubGlobal("fetch", fetchMock);

    await installModelPack("quantem:mito", "D:/models/quantem-mito");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:9000/api/models/quantem%3Amito/install/",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ source_path: "D:/models/quantem-mito" }),
      })
    );
  });

  it("starts an adaptation with the full payload", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ job_id: "job-1", adapter_id: "ad-1" }));
    vi.stubGlobal("fetch", fetchMock);

    const response = await startAdaptation("seg-1", {
      base_model: "quantem:mito",
      mode: "head",
      steps: 300,
      lr: 0.0001,
      seed: 0,
      name: "mito @ liver",
    });

    expect(response.adapter_id).toBe("ad-1");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:9000/api/segmentations/seg-1/adapt/",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          base_model: "quantem:mito",
          mode: "head",
          steps: 300,
          lr: 0.0001,
          seed: 0,
          name: "mito @ liver",
        }),
      })
    );
  });

  it("reads and applies an adapter", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ id: "ad-1" }))
      .mockResolvedValueOnce(jsonResponse({ id: "ad-1", applied_at: "now" }));
    vi.stubGlobal("fetch", fetchMock);

    await getAdapter("ad-1");
    await applyAdapter("ad-1");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://127.0.0.1:9000/api/adapters/ad-1/",
      expect.any(Object)
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://127.0.0.1:9000/api/adapters/ad-1/apply/",
      expect.objectContaining({ method: "POST" })
    );
  });
});
