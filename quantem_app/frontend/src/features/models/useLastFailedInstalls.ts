/**
 * The last FAILED install job per pack, read from the jobs API.
 *
 * Why this exists: a failed download used to be visible only while the Models
 * screen stayed mounted. The failure lived in the card's local state; navigate
 * away and back and the card showed "Not installed yet" plus a fresh Download
 * button, as if nothing had happened. The PermissionError (or digest mismatch,
 * or full disk) that explained the failure existed only in the jobs API, on a
 * row nothing rendered — so the obvious user response was to click Download
 * again, forever.
 *
 * Why the jobs API and not `GET /api/models/`: the catalogue carries no
 * last-failure information (checked against `quantem.registry.catalogue` as of
 * this writing — `pack_entry()` reports install/runnability facts only). If
 * the catalogue ever grows a `last_failed_install` block per pack, prefer it
 * and delete this hook; until then the jobs API is the only source. Two reads:
 * `GET /api/jobs/queue-status/` lists terminal jobs but not their payloads, so
 * the pack a failure belongs to needs a `GET /api/jobs/<id>/` per candidate —
 * bounded by {@link MAX_DETAIL_READS}, and install failures are rare enough
 * that the newest handful always covers every pack on screen.
 */

import { useCallback, useEffect, useState } from "react";
import { getJob, getJobQueueStatus } from "@/shared/api/jobs";
import type { Job, JobQueueItem } from "@/shared/types/jobs";

/** Mirrors `quantem.jobs.constants.JOB_TYPE_INSTALL_MODEL_PACK`. */
export const INSTALL_JOB_TYPE = "install_model_pack";

/** Newest failed install jobs worth a detail read. */
const MAX_DETAIL_READS = 16;

export interface LastFailedInstall {
  jobId: string;
  /**
   * `Job.message`, verbatim. It is the only text that says whether the network
   * died, a digest mismatched, the disk filled, or the cache promote hit a
   * PermissionError — and each of those wants a different response.
   */
  message: string | null;
  finishedAt: string | null;
}

/**
 * Which pack an install job was for.
 *
 * `payload_json.pack_id` is written by both install routes; the
 * `model:<pack_id>` tag is the fallback for a row written by an older build.
 */
export function packIdOfInstallJob(
  job: Pick<Job, "payload_json" | "tags">
): string | null {
  const raw = job.payload_json?.["pack_id"];
  if (typeof raw === "string" && raw) return raw;
  for (const tag of job.tags ?? []) {
    if (tag.startsWith("model:") && tag.length > "model:".length) {
      return tag.slice("model:".length);
    }
  }
  return null;
}

/**
 * The queue-status rows worth a detail read: FAILED installs only.
 *
 * The server's `failed` list also carries CANCELLED rows; a cancellation is
 * the user's own act and not something to warn them about later. Order is the
 * server's (newest first), so the first row per pack is the last failure.
 */
export function selectFailedInstallCandidates(
  failed: JobQueueItem[],
  limit: number = MAX_DETAIL_READS
): JobQueueItem[] {
  return failed
    .filter((job) => job.type === INSTALL_JOB_TYPE && job.status === "FAILED")
    .slice(0, limit);
}

export function useLastFailedInstalls(): {
  /** pack id → its most recent failed install job. Empty while loading. */
  failures: ReadonlyMap<string, LastFailedInstall>;
  refresh: () => void;
} {
  const [failures, setFailures] = useState<ReadonlyMap<string, LastFailedInstall>>(
    () => new Map()
  );
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const status = await getJobQueueStatus();
        const candidates = selectFailedInstallCandidates(status.failed);
        const details = await Promise.all(
          // A row can vanish between the two reads ("Clear done", a retry);
          // that is not this hook's problem, so a failed read drops the row.
          candidates.map((item) => getJob(item.id).catch(() => null))
        );
        if (cancelled) return;
        const next = new Map<string, LastFailedInstall>();
        for (const job of details) {
          // Re-checked on the detail read: a retry can flip the row to
          // PENDING between the list and the detail.
          if (!job || job.status !== "FAILED") continue;
          const packId = packIdOfInstallJob(job);
          if (!packId || next.has(packId)) continue;
          next.set(packId, {
            jobId: job.id,
            message: job.message?.trim() || null,
            finishedAt: job.finished_at ?? null,
          });
        }
        setFailures(next);
      } catch {
        // The screen must render without the jobs API — a card then simply
        // shows no install history, exactly as it did before this hook.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [nonce]);

  const refresh = useCallback(() => setNonce((current) => current + 1), []);

  return { failures, refresh };
}
