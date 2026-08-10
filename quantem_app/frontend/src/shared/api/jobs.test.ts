import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as jobsApi from "@/shared/api/jobs";
import { getJobQueueStatus, retryJob } from "@/shared/api/jobs";
import { setApiConfig } from "@/shared/api/core/http";

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("shared/api/jobs", () => {
  beforeEach(() => {
    setApiConfig({ apiBaseUrl: "http://127.0.0.1:9000" });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    setApiConfig({ apiBaseUrl: "http://127.0.0.1:8000" });
  });

  it("requests queue status with auth headers", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        running: [],
        queues: [],
        failed: [],
        completed: [],
        worker: { scheduler_in_process: true },
        generated_at: "2026-03-27T12:00:00Z",
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    await getJobQueueStatus();

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:9000/api/jobs/queue-status/",
      expect.objectContaining({
        headers: expect.objectContaining({
          "Content-Type": "application/json",
        }),
      })
    );
  });

  it("exposes no worker-restart call", () => {
    // The route does not exist server-side (JobWorkerRestartEndpointTests);
    // shipping a client for it only produces a silent 404.
    expect(Object.keys(jobsApi)).not.toContain("restartJobWorker");
  });

  it("posts to the retry endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ status: "queued", job_id: "job-1" }));
    vi.stubGlobal("fetch", fetchMock);

    await retryJob("job-1");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:9000/api/jobs/job-1/retry/",
      expect.objectContaining({
        method: "POST",
      })
    );
  });
});
