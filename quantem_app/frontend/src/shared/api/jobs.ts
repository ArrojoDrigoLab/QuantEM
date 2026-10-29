import { apiRequest, ApiRequestError } from "@/shared/api/core/http";
import type {
  ClearDoneJobsResponse,
  Job,
  JobQueueStatus,
  RetryJobResponse,
  SubmitJobPayload,
  SystemStatus,
} from "@/shared/types/jobs";

export function getSystemStatus(): Promise<SystemStatus> {
  return apiRequest<SystemStatus>("/api/system/status/");
}

export function submitJob(payload: SubmitJobPayload): Promise<Job> {
  return apiRequest<Job>("/api/jobs/", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getJob(jobId: string): Promise<Job> {
  return apiRequest<Job>(`/api/jobs/${jobId}/`);
}

export function cancelJob(jobId: string): Promise<{ status: string }> {
  return apiRequest<{ status: string }>(`/api/jobs/${jobId}/cancel/`, {
    method: "POST",
  });
}

export function retryJob(jobId: string): Promise<RetryJobResponse> {
  return apiRequest<RetryJobResponse>(`/api/jobs/${jobId}/retry/`, {
    method: "POST",
  });
}

export function deleteJob(jobId: string): Promise<void> {
  return apiRequest<void>(`/api/jobs/${jobId}/`, {
    method: "DELETE",
  });
}

/** Queued jobs are removed; running jobs stop at a worker checkpoint. */
export async function stopJob(jobId: string, status: string): Promise<"removed" | "requested" | "settled"> {
  const stop = async (current: string) => {
    if (current === "RUNNING") { await cancelJob(jobId); return "requested" as const; }
    if (current === "PENDING" || current === "RETRY") { await deleteJob(jobId); return "removed" as const; }
    return "settled" as const;
  };
  try { return await stop(status); }
  catch (error) {
    if (error instanceof ApiRequestError && error.status === 404) return "removed";
    if (!(error instanceof ApiRequestError) || error.status !== 409) throw error;
    // The worker may have claimed the job since the last progress response.
    try { return await stop((await getJob(jobId)).status); }
    catch (latestError) {
      if (latestError instanceof ApiRequestError && latestError.status === 404) return "removed";
      throw latestError;
    }
  }
}

export function clearDoneJobs(): Promise<ClearDoneJobsResponse> {
  return apiRequest<ClearDoneJobsResponse>("/api/jobs/clear-done/", {
    method: "POST",
  });
}

// There is no `restartJobWorker`. `POST /api/jobs/worker/restart/` was never
// mounted -- the backend asserts its absence in
// `JobWorkerRestartEndpointTests` -- so the client function only ever produced
// a 404 that the sidebar swallowed silently.

export function getJobQueueStatus(): Promise<JobQueueStatus> {
  return apiRequest<JobQueueStatus>("/api/jobs/queue-status/");
}

export interface UpdateApplyLockResponse {
  ready: boolean;
  open_jobs: number;
  reason: "jobs_running" | "already_applying" | null;
}

export function acquireUpdateApplyLock(): Promise<UpdateApplyLockResponse> {
  return apiRequest<UpdateApplyLockResponse>("/api/update-maintenance/acquire/", {
    method: "POST",
  });
}

export function releaseUpdateApplyLock(): Promise<{ released: boolean }> {
  return apiRequest<{ released: boolean }>("/api/update-maintenance/release/", {
    method: "POST",
  });
}
