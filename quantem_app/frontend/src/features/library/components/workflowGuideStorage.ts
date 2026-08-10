/**
 * Whether the first-run workflow guide has been dismissed on this machine.
 *
 * Kept out of `WorkflowGuide.tsx` so that file exports a component and nothing
 * else -- a module that mixes the two breaks React Fast Refresh (and trips
 * `react-refresh/only-export-components`).
 */

const GUIDE_DISMISSED_STORAGE_KEY = "quantem-workflow-guide-dismissed-v1";

export function hasSeenWorkflowGuide(): boolean {
  if (typeof window === "undefined") return true;
  try {
    return window.localStorage.getItem(GUIDE_DISMISSED_STORAGE_KEY) === "1";
  } catch {
    // A blocked or absent localStorage must not stop the library rendering;
    // showing the guide again is the harmless side of that failure.
    return false;
  }
}

export function rememberWorkflowGuideDismissed(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(GUIDE_DISMISSED_STORAGE_KEY, "1");
  } catch {
    // Ignored on purpose: see hasSeenWorkflowGuide.
  }
}
