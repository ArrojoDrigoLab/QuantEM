import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getAnalysisExportUrl,
  getAnalysisRun,
  getAnalysisRuns,
  startAnalysisRun,
} from "@/shared/api/analysis";
import { setApiConfig } from "@/shared/api/core/http";

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("shared/api/analysis", () => {
  beforeEach(() => {
    setApiConfig({ apiBaseUrl: "http://127.0.0.1:9000" });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    setApiConfig({ apiBaseUrl: "http://127.0.0.1:8000" });
  });

  it("posts a run against the segmentation", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        jsonResponse({ job_id: "job-1", analysis_run_id: "run-1" })
      );
    vi.stubGlobal("fetch", fetchMock);

    const response = await startAnalysisRun("seg-1", {
      compartments: { mito: "seg-1" },
      tissue_segmentation_id: null,
      points_source: "centroids",
      band_edges_nm: [0, 50, 100],
      replicates: 20,
      seed: 12345,
      group: "fasted",
    });

    expect(response.analysis_run_id).toBe("run-1");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:9000/api/segmentations/seg-1/analysis/",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          compartments: { mito: "seg-1" },
          tissue_segmentation_id: null,
          points_source: "centroids",
          band_edges_nm: [0, 50, 100],
          replicates: 20,
          seed: 12345,
          group: "fasted",
        }),
      })
    );
  });

  it("lists runs and reads one run", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(jsonResponse({ id: "run-1" }));
    vi.stubGlobal("fetch", fetchMock);

    await getAnalysisRuns("seg-1");
    await getAnalysisRun("run-1");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://127.0.0.1:9000/api/segmentations/seg-1/analysis/",
      expect.any(Object)
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://127.0.0.1:9000/api/analysis/run-1/",
      expect.any(Object)
    );
  });

  it("builds export URLs against the configured base and escapes the name", () => {
    expect(getAnalysisExportUrl("run-1", "objects.csv")).toBe(
      "http://127.0.0.1:9000/api/analysis/run-1/export/objects.csv"
    );
    // The backend refuses anything that escapes the run directory; the client
    // must not hand it a raw traversal either.
    expect(getAnalysisExportUrl("run-1", "../secret.csv")).toBe(
      "http://127.0.0.1:9000/api/analysis/run-1/export/..%2Fsecret.csv"
    );
  });
});
