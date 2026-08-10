/**
 * Rendering an install the catalogue reports as already underway.
 *
 * uat13 #1: while installer-requested downloads were RUNNING, every pack card
 * said "not installed" with a live Download button, and clicking it queued a
 * real duplicate gigabyte download. The backend now reports the active job on
 * each pack (`active_install`) and refuses a second POST with a 409; these
 * helpers turn that snapshot into the card's installing state for the moment
 * before the job poll first answers.
 */

import { formatBytes } from "@/shared/ui/format";
import type { ModelPackActiveInstall } from "@/shared/types/finetune";

/**
 * The headline for a catalogue-reported install, before the job poll answers.
 *
 * "Queued" while nothing has moved; "Installing — 214.0 MB of 1.2 GB" (the
 * em-dash convention) once bytes exist. Total-less progress still shows what
 * has arrived rather than pretending to know the denominator.
 */
export function describeActiveInstall(install: ModelPackActiveInstall): string {
  const current = install.progress_current_bytes;
  const total = install.progress_total_bytes;
  if (current === null) {
    return install.status === "RUNNING" ? "Installing…" : "Queued";
  }
  if (total === null) {
    return `Installing — ${formatBytes(current)}`;
  }
  return `Installing — ${formatBytes(current)} of ${formatBytes(total)}`;
}

/** Progress as a 0–100 percentage, or null when the bytes cannot say. */
export function activeInstallPercent(
  install: ModelPackActiveInstall
): number | null {
  const current = install.progress_current_bytes;
  const total = install.progress_total_bytes;
  if (current === null || total === null || total <= 0) return null;
  return Math.min(100, Math.max(0, (current / total) * 100));
}
