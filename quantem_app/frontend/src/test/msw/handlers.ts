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

/**
 * Nothing to fine-tune on.
 *
 * The labeling header asks `POST /api/finetune/preview/` over the image it is
 * open on, because R13 enables its Fine-Tune button only once that organelle
 * has an annotation on that image. Every test that renders the header therefore
 * fires it, and without a default it fires into `onUnhandledRequest: "error"` --
 * sixty-odd error lines in a full run, which is how a suite teaches people to
 * ignore MSW errors. Zero annotations is the neutral answer: the button renders
 * disabled, exactly as it does on an image nobody has annotated.
 */
export const EMPTY_FINETUNE_PREVIEW = {
  experiment: null,
  asset_count: 0,
  annotation_count: 0,
  confirmed_areas: 0,
  done_rois: 0,
  tile_count: 0,
  per_image: [],
  default_mode: "use_all",
  eligible: false,
  blockers: [],
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
  http.post("http://127.0.0.1:8000/api/finetune/preview/", () =>
    HttpResponse.json(EMPTY_FINETUNE_PREVIEW)
  ),
  // An unorganised library. The import form and the library page both read
  // `GET /api/experiments/` so they can offer the experiments that exist; with
  // `onUnhandledRequest: "error"` every one of their tests would otherwise fail
  // on the request rather than on its own assertion. Empty is the neutral state
  // and the one every library that exists today is in -- the grouping controls
  // render as "No experiment" plus "New experiment…", and the library's filter
  // does not appear at all, which is exactly what those tests already assert
  // about the screen. A test that cares overrides this with `server.use(...)`.
  http.get("http://127.0.0.1:8000/api/experiments/", () => HttpResponse.json([])),
];
