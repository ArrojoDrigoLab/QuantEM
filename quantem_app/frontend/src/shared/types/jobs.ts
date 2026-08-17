import type { JobPriority, JobResourceClass, JobStatus } from "@/shared/types/common";
import type { RunLeg } from "@/shared/types/runs";

export interface SystemStatus {
  /** The installed QuantEM package version reported by the backend. */
  app_version?: string;
  cuda_available: boolean;
  /**
   * Extensions this build can import, e.g. `[".tif", ".tiff", ".png"]`.
   *
   * `SystemStatusView` emits the same `UPLOAD_SUFFIXES` the upload endpoint
   * validates against, precisely so the file picker never has to guess. Treat
   * it as optional only for the window between the page loading and the first
   * status response landing.
   */
  supported_upload_formats?: string[];
}

/**
 * Which phase of a job is running. A machine key, never rendered verbatim:
 * `runProgress.ts` is the only place that turns one into English.
 */
export type JobProgressStage =
  | ""
  | "queued"
  | "loading_model"
  | "inference"
  | "extracting"
  | "saving"
  | "downloading_model";

/**
 * Countable work — tiles — or absent when this job counts none.
 *
 * `total` is the sliding-window plan, laid out from the region shape and the
 * pack's canonical scale before the first forward pass, so it is exact and is
 * on the row while the model is still loading. `percent` divides by that same
 * total, which is the whole point: the run's coarse `progress` also carries the
 * phases either side of the tiles and therefore divides by more, and rendering
 * both for the same run is how the drawer came to show a 56% bar beside the
 * words "57% (Tile 32/56)".
 */
export interface JobUnitProgress {
  done: number;
  total: number;
  /** Singular noun for one unit, e.g. `"tile"`. */
  label: string;
  percent: number | null;
  stage: string;
  eta_seconds?: number | null;
}

/**
 * Bytes moving over the network. Structurally separate from
 * {@link JobUnitProgress} on purpose — a caller cannot render a download as
 * segmentation progress, because the two never share a field.
 */
export interface JobDownloadProgress {
  current_bytes: number;
  total_bytes: number | null;
  percent: number | null;
}

export interface JobBatchRun {
  job_id: string;
  segmentation_id?: string | null;
  status: JobStatus;
  batch_seq: number;
  units_done: number;
  units_total: number;
  stage: string;
}

/**
 * Every run of one image, rolled into one answer.
 *
 * `percent` is `units_done / units_total`: **the whole of the work the user
 * asked for**, including a queued run that has not started and the tiles a
 * failed or cancelled run will never walk. Those unwalked tiles are also
 * reported on their own in `units_abandoned` (and `units_reachable` is what is
 * left), which is what lets the UI say the wave will not finish whole — the bar
 * stopping short is the other half of saying it.
 *
 * `percent` is null only when `runs_unplanned` is not zero: a run in the wave
 * cannot say how big it is, so any fraction would be a fraction of an unknown.
 */
export interface JobBatchProgress {
  batch_id: string;
  unit_label: string;
  units_done: number;
  units_total: number;
  units_abandoned: number;
  units_reachable: number;
  percent: number | null;
  runs_total: number;
  runs_unplanned: number;
  runs_pending: number;
  runs_running: number;
  runs_succeeded: number;
  runs_failed: number;
  runs_cancelled: number;
  complete: boolean;
  eta_seconds: number | null;
  runs: JobBatchRun[];
}

/** The exact picker value used by one segmentation inference run. */
export interface JobModelRun {
  segmentation_id: string;
  source_model: string;
  adapter_id: string | null;
}

/** The three kinds of progress, as every job-bearing endpoint reports them. */
export interface JobProgressFields {
  progress_stage?: JobProgressStage | string;
  unit_progress?: JobUnitProgress | null;
  download?: JobDownloadProgress | null;
  batch_id?: string;
  batch_progress?: JobBatchProgress | null;
  /**
   * The organelles inside one run, when this job covers several.
   *
   * Ticking three organelles produces one job with one tile count, which is
   * what makes the whole-image bar honest — and it would collapse the three
   * per-organelle lines into one, because the run panel builds one line per
   * job. This is where those lines come from instead. Null (the normal case)
   * means this job is one piece of work and its own numbers are the row.
   */
  run_legs?: RunLeg[] | null;
}

export interface Job extends JobProgressFields {
  id: string;
  type: string;
  priority: JobPriority;
  status: JobStatus;
  progress: number;
  message?: string;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  attempts: number;
  max_attempts: number;
  next_run_at: string;
  payload_json: Record<string, unknown>;
  result_json?: Record<string, unknown> | null;
  error_traceback?: string;
  cancel_requested: boolean;
  resource_class: JobResourceClass;
  queue_name: string;
  tags: string[];
}

export interface JobQueueItem extends JobProgressFields {
  id: string;
  type: string;
  task_label: string;
  task_category?: "analysis" | "display" | "processing";
  status: JobStatus;
  progress: number;
  message?: string;
  cancel_requested: boolean;
  queue_name: string;
  resource_class: JobResourceClass;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  image?: { id: string; display_name: string } | null;
  segmentation?: {
    id: string;
    name: string;
    internal_name?: string;
    short_name?: string;
    long_name?: string;
  } | null;
  /**
   * The model a download job is fetching, titled the way the Models screen
   * titles it. Present only on `install_model_pack`, which is what lets the
   * download indicator be a visibly different kind of row.
   */
  model_pack?: { id: string; title: string } | null;
  /**
   * Model identity for each segmentation this inference job carries.
   *
   * Single-organelle jobs contain one entry; an image-wide batch contains one
   * per organelle. This stays separate from `run_legs`, which is progress and
   * may not exist yet while a run is queued.
   */
  model_runs?: JobModelRun[] | null;
}

export interface JobQueueStatus {
  running: JobQueueItem[];
  queues: Array<{
    queue_name: string;
    display_name: string;
    pending: JobQueueItem[];
  }>;
  failed: JobQueueItem[];
  completed: JobQueueItem[];
  /**
   * Whether anything in the server process will actually drain the queue.
   *
   * QuantEM runs the scheduler on a thread inside the server process: there is
   * no separate worker process, so no PID and no liveness probe. The previous
   * `{is_alive, pid}` shape did not exist on the wire, so the "worker
   * unavailable" banner could never fire however dead the scheduler was.
   */
  worker: {
    scheduler_in_process: boolean;
  };
  generated_at: string;
}

export interface SubmitJobPayload {
  type: string;
  payload: Record<string, unknown>;
  priority?: JobPriority;
  resource_class?: JobResourceClass;
  queue_name?: string;
  max_attempts?: number;
  tags?: string[];
}

export interface RetryJobResponse {
  status: "queued";
  job_id: string;
}

export interface ClearDoneJobsResponse {
  deleted: number;
  cleared_statuses: string[];
}

// `JobWorkerRestartResponse` was deleted with the "Restart worker" button:
// `POST /api/jobs/worker/restart/` is not a route (see the backend's
// `test_worker_restart_endpoint_does_not_exist`). Restarting a worker is
// operator tooling and a single-user desktop app has no operator.
