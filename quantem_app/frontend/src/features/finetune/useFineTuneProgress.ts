import { getFineTuneProgress } from "@/shared/api/finetune";
import { usePollingResource } from "@/shared/hooks/usePollingResource";

export function isTerminalFineTuneStatus(status: string | undefined): boolean {
  return status === "SUCCESS" || status === "FAILED" || status === "CANCELLED";
}

/** Keep the server's round-aware progress while sharing job lifecycle rules. */
export function useFineTuneProgress(adapterId: string | null, intervalMs = 1000) {
  const { data: progress, ...state } = usePollingResource(adapterId, getFineTuneProgress,
    (value) => isTerminalFineTuneStatus(value.status), intervalMs);
  return { progress, ...state };
}
