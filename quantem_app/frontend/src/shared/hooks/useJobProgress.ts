/**
 * Poll one background job until it settles.
 *
 * Long work in QuantEM returns `202` with a `job_id` and is watched through
 * `GET /api/jobs/<id>/`. Polling stops the moment the job reaches a terminal
 * status so a screen left open overnight is not still talking to the server.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { getJob } from "@/shared/api/jobs";
import type { Job } from "@/shared/types/jobs";
import type { JobStatus } from "@/shared/types/common";

const TERMINAL_STATUSES: JobStatus[] = ["SUCCESS", "FAILED", "CANCELLED"];

export function isTerminalJobStatus(status: JobStatus | undefined): boolean {
  return status !== undefined && TERMINAL_STATUSES.includes(status);
}

export interface UseJobProgressResult {
  job: Job | null;
  error: Error | null;
  /** True once the job reached SUCCESS, FAILED or CANCELLED. */
  settled: boolean;
  refresh: () => Promise<void>;
}

export function useJobProgress(
  jobId: string | null,
  intervalMs = 1000
): UseJobProgressResult {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<Error | null>(null);
  // Read inside the interval callback so a settled job stops the timer without
  // re-creating it on every progress tick.
  const settledRef = useRef(false);

  useEffect(() => {
    setJob(null);
    setError(null);
    settledRef.current = false;
  }, [jobId]);

  const poll = useCallback(async () => {
    if (!jobId) return;
    try {
      const next = await getJob(jobId);
      setJob(next);
      setError(null);
      if (isTerminalJobStatus(next.status)) {
        settledRef.current = true;
      }
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Unknown error"));
    }
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return undefined;
    let cancelled = false;
    const tick = async () => {
      if (cancelled || settledRef.current) return;
      await poll();
    };
    void tick();
    const timer = window.setInterval(() => {
      if (settledRef.current) {
        window.clearInterval(timer);
        return;
      }
      void tick();
    }, intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [jobId, intervalMs, poll]);

  return {
    job,
    error,
    settled: job !== null && isTerminalJobStatus(job.status),
    refresh: poll,
  };
}
