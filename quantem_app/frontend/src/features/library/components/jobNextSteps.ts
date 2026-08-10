/**
 * What a finished job says to do next, and how the queue gets hold of it.
 *
 * The backend writes actionable sentences into `Job.result_json.next_steps`
 * (`quantem.jobs.handlers._segmentation_run_outcome`), and until now nothing
 * rendered them. A run that finished having added no objects reported
 * `completed: no new objects` — true, and not enough. The three sentences that
 * explain *why* that is the expected outcome over a proofread image, and what
 * to do instead of lowering the threshold, went into the job row and stopped
 * there.
 *
 * The labeling screen has its own copy of that advice through the
 * segmentation's `run_notice`, which is the right place for a user who is
 * looking at that image. This is for the user who is not: someone who queued
 * four whole-image runs from the import form and then went to the library has
 * the queue sidebar and nothing else.
 *
 * The queue-status payload deliberately does not carry `result_json` — it is
 * one row per job for up to a hundred jobs — so the detail is fetched per job,
 * once, and only for the rows actually on screen. See `useJobNextSteps`.
 */

import { useEffect, useRef, useState } from "react";

import { getJob } from "@/shared/api/jobs";
import type { Job } from "@/shared/types/jobs";

/**
 * The `next_steps` a job recorded, or an empty list.
 *
 * `result_json` is an untyped JSON blob written by whichever handler ran, so
 * every level is checked rather than asserted: a build newer than this screen
 * must not be able to crash it, and a malformed entry must not render as
 * `[object Object]` beside real advice.
 */
export function readJobNextSteps(job: Job | null | undefined): string[] {
  const raw = (job?.result_json ?? {})["next_steps"];
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((step): step is string => typeof step === "string")
    .map((step) => step.trim())
    .filter(Boolean);
}

/**
 * Fetch the next steps for the jobs currently on screen, once each.
 *
 * `jobIdsKey` is a comma-joined list rather than an array so the effect does
 * not re-run on every poll: the queue refetches every three seconds and hands
 * back a new array of the same ids each time.
 *
 * Bounded by what is rendered, not by what the endpoint returned — the queue
 * sends up to a hundred completed jobs and shows six. Ids already asked about
 * are never asked again, including the ones that answered with nothing, so a
 * job with no advice costs exactly one request for as long as the panel is
 * open.
 */
export function useJobNextSteps(jobIdsKey: string): Record<string, string[]> {
  const [byJobId, setByJobId] = useState<Record<string, string[]>>({});
  const asked = useRef<Set<string>>(new Set());

  useEffect(() => {
    const ids = jobIdsKey ? jobIdsKey.split(",") : [];
    const todo = ids.filter((id) => id && !asked.current.has(id));
    if (todo.length === 0) return undefined;
    for (const id of todo) asked.current.add(id);

    let cancelled = false;
    void Promise.all(
      todo.map(async (id): Promise<[string, string[]]> => {
        try {
          return [id, readJobNextSteps(await getJob(id))];
        } catch {
          // A detail fetch that fails must not disturb the queue view. The
          // row's own message is already on screen and is the load-bearing
          // part; this is the paragraph under it.
          return [id, []];
        }
      })
    ).then((pairs) => {
      if (cancelled) return;
      const found = pairs.filter(([, steps]) => steps.length > 0);
      if (found.length === 0) return;
      setByJobId((prev) => ({ ...prev, ...Object.fromEntries(found) }));
    });

    return () => {
      cancelled = true;
    };
  }, [jobIdsKey]);

  return byJobId;
}
