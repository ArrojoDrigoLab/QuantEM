/**
 * Poll one fine-tune run until it settles.
 *
 * Deliberately `GET /api/finetune/runs/<id>/progress/` and not the generic
 * `useJobProgress`: the job row carries the step count, but the round count and
 * the server-computed percent that folds rounds into steps live only on this
 * endpoint, and a bar that had to reconstruct them would be a second opinion
 * about how far along the run is.
 *
 * Same discipline as `useJobProgress` otherwise — one second while it runs,
 * stop dead on a terminal status, so a dialog left open overnight is not still
 * talking to the server.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { getFineTuneProgress } from "@/shared/api/finetune";
import type { FineTuneProgress } from "@/shared/types/finetune";

export function isTerminalFineTuneStatus(status: string | undefined): boolean {
  return status === "SUCCESS" || status === "FAILED";
}

export interface UseFineTuneProgressResult {
  progress: FineTuneProgress | null;
  error: Error | null;
  settled: boolean;
}

export function useFineTuneProgress(
  adapterId: string | null,
  intervalMs = 1000
): UseFineTuneProgressResult {
  const [progress, setProgress] = useState<FineTuneProgress | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const settledRef = useRef(false);

  useEffect(() => {
    setProgress(null);
    setError(null);
    settledRef.current = false;
  }, [adapterId]);

  const poll = useCallback(async () => {
    if (!adapterId) return;
    try {
      const next = await getFineTuneProgress(adapterId);
      setProgress(next);
      setError(null);
      if (isTerminalFineTuneStatus(next.status)) settledRef.current = true;
    } catch (err) {
      // Kept on screen rather than thrown: one dropped poll during a
      // twenty-minute run is not a reason to replace the bar with an error.
      setError(err instanceof Error ? err : new Error("Unknown error"));
    }
  }, [adapterId]);

  useEffect(() => {
    if (!adapterId) return undefined;
    let cancelled = false;
    const tick = () => {
      if (cancelled || settledRef.current) return;
      void poll();
    };
    tick();
    const timer = window.setInterval(() => {
      if (settledRef.current) {
        window.clearInterval(timer);
        return;
      }
      tick();
    }, intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [adapterId, intervalMs, poll]);

  return {
    progress,
    error,
    settled: progress !== null && isTerminalFineTuneStatus(progress.status),
  };
}
