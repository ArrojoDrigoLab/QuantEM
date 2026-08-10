import { http, HttpResponse } from "msw";
import type { ModelCatalogue } from "@/shared/types/finetune";
import type { JobQueueStatus } from "@/shared/types/jobs";

/**
 * An empty-but-well-formed catalogue.
 *
 * Several screens now ask `GET /api/models/` so they can say whether a model
 * can run before offering to run it. `onUnhandledRequest: "error"` means every
 * one of their tests would otherwise fail on the request rather than on its
 * own assertion. An empty `packs` list resolves every pack to "unknown"
 * runnability, which is the neutral state — no test inherits a claim it did not
 * ask for. A test that cares overrides this with `server.use(...)`.
 */
export const EMPTY_MODEL_CATALOGUE: ModelCatalogue = {
  packs: [],
  adapted: [],
  device: { kind: "cpu", name: "CPU", cuda: false, mps: false },
};

/**
 * An idle queue.
 *
 * The Models screen reads `GET /api/jobs/queue-status/` to surface the last
 * failed install per pack. Nothing failed is the neutral state; a test about
 * failure history overrides this with `server.use(...)`.
 */
export const EMPTY_JOB_QUEUE_STATUS: JobQueueStatus = {
  running: [],
  queues: [],
  failed: [],
  completed: [],
  worker: { scheduler_in_process: true },
  generated_at: "2026-01-01T00:00:00Z",
};

// Default no-op handler to keep MSW initialized in tests that do not define
// request handlers explicitly.
export const handlers = [
  http.get("http://127.0.0.1:8000/__msw_health__", () =>
    HttpResponse.json({ ok: true })
  ),
  http.get("http://127.0.0.1:8000/api/models/", () =>
    HttpResponse.json(EMPTY_MODEL_CATALOGUE)
  ),
  http.get("http://127.0.0.1:8000/api/jobs/queue-status/", () =>
    HttpResponse.json(EMPTY_JOB_QUEUE_STATUS)
  ),
];
