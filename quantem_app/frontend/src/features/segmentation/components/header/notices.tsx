/**
 * Every `role="status"` sentence the labeling header prints under its controls.
 *
 * Moved out of `SegmentationHeader.tsx`. The header used to hold roughly twenty
 * distinct surfaces in one 966-line file, and five separate packages wanted to
 * edit it at once; this file is the failure-and-state copy, so the package that
 * owns the wording owns one file rather than a slice of a component everyone
 * else is also editing.
 *
 * Failure-class copy was added after the split for errors the header could not
 * previously explain:
 *
 *  - **`FailureCopyNotice`.** When the server names the *class* of failure
 *    (`error_code`, catalogued in `quantem/core/error_codes.py`), the header
 *    renders the class's own wording and its in-app control alongside the
 *    server's sentence. A red sentence with no way forward is what made users
 *    conclude the fault was theirs; the code is what lets the app offer a
 *    button instead.
 *
 * Everything else is unchanged and unreworded, and `HeaderNotices` still
 * renders as a fragment, so it adds no element of its own.
 */

import {
  formatThreshold,
  type AppliedAdapterState,
} from "@/features/models/appliedAdapter";
import type { ImageSegmentation } from "@/shared/types";
import {
  failureCopy,
  readFailureCode,
  type FailureCopy,
} from "@/shared/copy/failures";
import "./notices.css";

/** Every object on the segmentation, whatever its label state. */
function countAllObjects(segmentation: ImageSegmentation): number {
  const counts = segmentation.segment_counts;
  if (!counts) return 0;
  return Object.values(counts).reduce(
    (sum, value) => sum + (typeof value === "number" ? value : 0),
    0
  );
}

/**
 * A run that stopped without saving anything, said out loud.
 *
 * `status_error` was reaching no screen at all. The stage went FAILED, the
 * message was written, and the labeling header carried on describing the
 * objects from the *previous* run in its ordinary neutral chip -- so a
 * cancelled ER re-run read "190 confirmed of 190 from QuantEM" and a re-run at
 * a corrected pixel size left the wrongly-scaled objects looking finished.
 *
 * The object count is repeated here rather than left to the chip because the
 * sentence that matters is not "the run failed" on its own; it is that the
 * objects still on screen predate it. `status_error` is reproduced verbatim
 * (`quantem.jobs.failure_reconcile` writes it) since it is the only text that
 * separates a dead worker from a cancellation from a job removed from the
 * queue, and each of those wants a different response.
 */
export function FailedRunNotice({
  error,
  objectCount,
  copy,
}: {
  error?: string | null;
  objectCount: number;
  /**
   * The catalogued copy for this failure's class, when the server named one.
   * Absent for every failure that has no code, which is most of them and the
   * behaviour this notice had before codes existed.
   */
  copy?: FailureCopy | null;
}) {
  const reason = error?.trim();
  return (
    <div className="header-failed-notice" role="status">
      <strong>
        {/* The class's own headline when there is one: "This model is not
            installed on this computer." is a better first line than "the run
            failed", because it is already the answer. */}
        {copy ? copy.headline : "The last run on this segmentation failed."}
      </strong>{" "}
      {copy ? `${copy.body} ` : null}
      {reason || "The server recorded no reason for it."}{" "}
      {objectCount === 0
        ? "It saved no objects, and there are none here from an earlier run."
        : `It saved no objects: the ${objectCount} shown here ${
            objectCount === 1 ? "was" : "were"
          } already on this segmentation before it started.`}
      {copy?.action.href ? (
        <>
          {" "}
          <a className="header-failure-action" href={copy.action.href}>
            {copy.action.label}
          </a>
        </>
      ) : null}
    </div>
  );
}

/**
 * The queue's note about the *newest* attempt, on a stage that is not FAILED.
 *
 * The retry path writes "Attempt N of M failed; retrying automatically.
 * <error>" onto the segmentation after every failed attempt without touching
 * the stage, and the abandoned-run repair leaves its own sentence the same way
 * -- so this header must render `status_error` whenever it is present, or it
 * goes back to showing an older error (or nothing) while newer failures accrue
 * only in Tasks & Queues. The words are the server's, verbatim: they are the
 * only text that says which attempt failed and whether another is coming, and a
 * successful attempt clears the field.
 */
export function LatestAttemptNotice({ message }: { message: string }) {
  return (
    <div
      className="header-failed-notice"
      role="status"
      data-testid="latest-attempt-notice"
    >
      {message}
    </div>
  );
}

/**
 * That an adapted model is in play, on the screen where the run is started.
 *
 * Two states, and the second is the one worth the pixels: an adapter applied to
 * this segmentation is only used when the selected source model is the one it
 * was fitted on. Pick anything else and `apply_active_adapter` falls back to
 * the released pack at its published threshold, with nothing on screen to say
 * so -- your numbers change and the header still reads the same.
 */
export function AppliedAdapterNotice({
  state,
  selectedLabel,
}: {
  state: AppliedAdapterState;
  selectedLabel: string;
}) {
  const { adapter, active, publishedThreshold, trainedHead } = state;
  const calibrated = formatThreshold(adapter.calibrated_threshold);
  const published = formatThreshold(publishedThreshold);
  const name = adapter.name || adapter.base;

  if (!active) {
    return (
      <span className="header-adapter-notice is-inactive" role="status">
        <strong>Adapter not in use.</strong> "{name}" was fitted on{" "}
        {adapter.base}, and this run is set to {selectedLabel}, so it will use
        the published model
        {published ? ` at threshold ${published}` : ""}.
      </span>
    );
  }

  return (
    <span className="header-adapter-notice" role="status">
      <strong>Adapted model: {name}.</strong> Run model will use{" "}
      {trainedHead ? "your fine-tuned head" : "your calibration"}
      {calibrated ? ` at threshold ${calibrated}` : ""}
      {published && calibrated && published !== calibrated
        ? `, not the published ${published}`
        : ""}
      .
    </span>
  );
}

/**
 * The completion lock, made visible.
 *
 * The same reasoning as the blocked-model notice: a disabled control with no
 * sentence beside it reads as a bug, and a tooltip is unreachable by keyboard.
 * This is also the only place the app says how to get the segmentation back.
 */
export function LockedNotice() {
  return (
    <span className="header-locked-notice" role="status">
      <strong>This segmentation is locked.</strong> It was marked done, so
      objects cannot be added, edited or removed and no new run can be started.
      Use "Unlock segmentation" to change it again.
    </span>
  );
}

/**
 * The notice stack, in the order the header has always printed it.
 *
 * A fragment, so it adds no element of its own: what reaches the DOM is the
 * same sequence of siblings inside `.header-controls` that the header emitted
 * before this file existed.
 */
export function HeaderNotices({
  currentSegmentation,
  isOrganelle,
  appliedAdapter,
  runTargetLabel,
  isComplete,
}: {
  currentSegmentation: ImageSegmentation | null;
  isOrganelle: boolean;
  appliedAdapter: AppliedAdapterState | null;
  runTargetLabel: string;
  isComplete: boolean;
}) {
  // Read defensively rather than off a declared field: an `error_code` reaches
  // this screen on several differently-shaped payloads, and a code this build
  // has no copy for is the same as no code at all -- the server's sentence
  // still renders on its own, which is what happened before codes existed.
  const failure = failureCopy(readFailureCode(currentSegmentation));
  return (
    <>
      {currentSegmentation?.status_stage === "FAILED" ? (
        // Not gated on `isOrganelle`, and not left to the chip's tooltip. A
        // failed run is the one state where every other number on this
        // header belongs to a different run, and `status_error` is the only
        // text that says whether the worker died, the user cancelled, or the
        // job was removed before it started.
        <FailedRunNotice
          error={currentSegmentation.status_error}
          objectCount={countAllObjects(currentSegmentation)}
          copy={failure}
        />
      ) : currentSegmentation?.status_error?.trim() ? (
        // A non-empty error on a stage that is not FAILED is the queue's note
        // about the newest attempt -- see `LatestAttemptNotice`.
        <LatestAttemptNotice message={currentSegmentation.status_error.trim()} />
      ) : null}
      {isOrganelle && appliedAdapter && (
        <AppliedAdapterNotice
          state={appliedAdapter}
          selectedLabel={runTargetLabel}
        />
      )}
      {isComplete && <LockedNotice />}
    </>
  );
}
