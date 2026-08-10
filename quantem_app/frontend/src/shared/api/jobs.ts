import { apiRequest } from "@/shared/api/core/http";
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
