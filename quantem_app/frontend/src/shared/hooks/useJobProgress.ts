import { getJob } from "@/shared/api/jobs";
import type { Job } from "@/shared/types/jobs";
import type { JobStatus } from "@/shared/types/common";
import { usePollingResource } from "./usePollingResource";

export function isTerminalJobStatus(status: JobStatus | undefined): boolean {
  return status === "SUCCESS" || status === "FAILED" || status === "CANCELLED";
}

export interface UseJobProgressResult {
  job: Job | null;
  error: Error | null;
  retrying?: boolean;
  settled: boolean;
  refresh: () => Promise<void>;
}

export function useJobProgress(jobId: string | null, intervalMs = 1000): UseJobProgressResult {
  const { data: job, ...state } = usePollingResource(jobId, getJob,
    (value) => isTerminalJobStatus(value.status), intervalMs);
  return { job, ...state };
}
