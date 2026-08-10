/**
 * The adaptation run in flight for a segmentation, remembered across reloads.
 *
 * Step 4 of the wizard says "The run goes on the job queue, so you can leave
 * this screen and come back to it." That was not true: every id the wizard
 * needed to come back to a run -- the job it is watching and the adapter that
 * job is writing -- lived in component state, so a reload dropped them and left
 * steps 2-6 disabled with no way to reach the Results or Apply pages. A head
 * training the wizard itself estimates at "minutes to tens of minutes" was
 * wasted unless the user sat through it without refreshing.
 *
 * `localStorage`, not `sessionStorage`: the promise the step makes is that you
 * can leave, and leaving includes closing the tab.
 *
 * This is a convenience, not the source of truth. A cleared store loses nothing
 * permanently -- every successful adapter is still listed by `GET /api/models/`
 * with its `segmentation_id`, which is what the wizard's resume list reads.
 */

const ADAPT_RUNS_STORAGE_KEY = "quantem-adapt-runs-v1";

export interface AdaptRunHandle {
  adapterId: string;
  /** Null once the run has settled; only needed to reattach a progress poll. */
  jobId: string | null;
}

type AdaptRunStore = Record<string, AdaptRunHandle>;

function readStore(): AdaptRunStore {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(ADAPT_RUNS_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    const store: AdaptRunStore = {};
    for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (!value || typeof value !== "object") continue;
      const candidate = value as Record<string, unknown>;
      if (typeof candidate.adapterId !== "string") continue;
      store[key] = {
        adapterId: candidate.adapterId,
        jobId: typeof candidate.jobId === "string" ? candidate.jobId : null,
      };
    }
    return store;
  } catch {
    // A blocked, absent or corrupt store must never stop the wizard rendering.
    return {};
  }
}

function writeStore(store: AdaptRunStore): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(ADAPT_RUNS_STORAGE_KEY, JSON.stringify(store));
  } catch {
    // Ignored on purpose: see readStore. The resume list still works.
  }
}

export function loadAdaptRun(segmentationId: string | null | undefined): AdaptRunHandle | null {
  if (!segmentationId) return null;
  return readStore()[segmentationId] ?? null;
}

export function rememberAdaptRun(
  segmentationId: string,
  handle: AdaptRunHandle
): void {
  writeStore({ ...readStore(), [segmentationId]: handle });
}

export function forgetAdaptRun(segmentationId: string): void {
  const store = readStore();
  if (!(segmentationId in store)) return;
  delete store[segmentationId];
  writeStore(store);
}
