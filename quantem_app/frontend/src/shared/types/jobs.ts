import type { JobPriority, JobResourceClass, JobStatus } from "@/shared/types/common";

export interface SystemStatus {
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

export interface Job {
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

export interface JobQueueItem {
  id: string;
  type: string;
  task_label: string;
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
