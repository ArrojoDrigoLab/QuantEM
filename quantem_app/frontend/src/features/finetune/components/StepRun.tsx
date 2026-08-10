/**
 * Step 4 — the run itself.
 *
 * The job's own messages are shown verbatim ("sweeping the threshold",
 * "verifying the saved head") rather than being replaced with a spinner: they
 * are the only place the verification step is visible while it happens.
 *
 * A run that stops without producing an adapter has to leave a way forward.
 * This step used to render a progress bar and a CANCELLED badge for a run the
 * user had cancelled themselves, with no control on the screen and steps 5 and
 * 6 greyed out behind `status === "SUCCESS"` — the wizard was simply over for
 * that segmentation. Cancelling and crashing are now both recoverable, and
 * worded apart: a cancellation is a decision, not a fault, and telling someone
 * their deliberate cancel "failed" invites them to go looking for a bug.
 *
 * The wizard could recover from a cancellation before it could *cause* one. The
 * only Cancel in the application was in the Library's queue sidebar, which is
 * not reachable from here — so a screen that invites you to walk away from work
 * it describes as "minutes to tens of minutes" had no way to change your mind.
 */

import { Link } from "react-router-dom";
import { Badge, Button, Panel } from "@/shared/ui/design";
import { formatDuration } from "@/shared/ui/format";
import { resolveAdaptRunOutcome } from "@/features/finetune/adaptRunOutcome";
import type { Job } from "@/shared/types/jobs";
import type { Adapter, AdaptMode } from "@/shared/types/finetune";
import type { Runnability } from "@/features/models/runnable";

export interface StepRunProps {
  job: Job | null;
  adapter: Adapter | null;
  startError: string | null;
  onStart: () => void;
  starting: boolean;
  canStart: boolean;
  /**
   * Clear the concluded run so a new one can be started.
   *
   * Only offered once the run has stopped without an adapter: it forgets the
   * remembered ids, so calling it on a live training would orphan it.
   */
  onStartAgain: () => void;
  /**
   * Stop the run in flight.
   *
   * Omitted when there is nothing this screen can cancel — a wizard reattached
   * to an adapter whose job row is gone still shows progress, and offering a
   * button that cannot do anything is worse than offering none.
   */
  onCancel?: () => void;
  cancelling?: boolean;
  /** What went wrong trying to cancel, if anything. */
  cancelError?: string | null;
  /** Whether the chosen base model can be loaded on this machine. */
  baseModelRunnability: Runnability;
  baseModel: string;
  mode: AdaptMode;
  /**
   * Why the budget on step 3 cannot be run, if it cannot.
   *
   * `min={1}` on a number input stops nothing, and the server reads the budget
   * as `int(steps or 300)` — a typed zero is replaced rather than refused, and
   * the adapter then reports a budget nobody chose.
   */
  budgetError?: string | null;
}

/**
 * Head training loads the base model; threshold calibration does not.
 *
 * `mode: "threshold_only"` sweeps a stored probability map and never calls
 * `engine.load_model`, so it stays available on a machine where the pack cannot
 * be built. Blocking it there would remove the one rung that still works.
 */
function blocksThisRun(runnability: Runnability, mode: AdaptMode): boolean {
  return mode === "head" && runnability.state === "blocked";
}

export function StepRun({
  job,
  adapter,
  startError,
  onStart,
  starting,
  canStart,
  onStartAgain,
  onCancel,
  cancelling = false,
  cancelError = null,
  baseModelRunnability,
  baseModel,
  mode,
  budgetError = null,
}: StepRunProps) {
  const blocked = blocksThisRun(baseModelRunnability, mode);
  const outcome = resolveAdaptRunOutcome(job, adapter);
  const progress = Math.min(100, Math.max(0, job?.progress ?? 0));
  // A concluded run has nothing to report on, so the step goes back to being
  // the place a run is started from.
  const showStartForm = (!job && !adapter) || outcome.concluded;
  // Only a job the queue can still act on: `POST /cancel/` refuses anything
  // that is not RUNNING, and a queued job leaves through DELETE. A request
  // already sent is not repeated.
  const canCancel =
    Boolean(onCancel) &&
    outcome.running &&
    job !== null &&
    !job.cancel_requested &&
    (job.status === "PENDING" || job.status === "RUNNING");

  return (
    <Panel className="p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="m-0 text-base font-semibold text-slate-950">Run</h2>
        <div className="flex items-center gap-2">
          {outcome.status ? (
            <Badge
              tone={
                outcome.status === "FAILED" || outcome.cancelled ? "warning" : "info"
              }
            >
              {outcome.status}
            </Badge>
          ) : null}
          {canCancel ? (
            <Button size="sm" onClick={onCancel} disabled={cancelling}>
              {cancelling ? "Cancelling…" : "Cancel run"}
            </Button>
          ) : null}
        </div>
      </div>

      {outcome.concluded ? (
        <div
          className={
            outcome.cancelled
              ? "mt-3 rounded-md border border-amber-200 bg-amber-50 p-3"
              : "mt-3 rounded-md border border-red-200 bg-red-50 p-3"
          }
          role="status"
        >
          <p
            className={
              outcome.cancelled
                ? "m-0 text-sm font-semibold text-amber-900"
                : "m-0 text-sm font-semibold text-red-900"
            }
          >
            {outcome.cancelled
              ? "You cancelled this run."
              : "This run failed."}
          </p>
          <p
            className={
              outcome.cancelled
                ? "m-0 mt-1 text-sm text-amber-900"
                : "m-0 mt-1 text-sm text-red-800"
            }
          >
            {outcome.message}
          </p>
          {outcome.cancelled ? (
            <p className="m-0 mt-1 text-sm text-amber-900">
              Nothing about the segmentation changed and your annotations are
              untouched, so starting again costs only the time.
            </p>
          ) : null}
        </div>
      ) : null}

      {showStartForm ? (
        <>
          {!outcome.concluded ? (
            <p className="m-0 mt-2 text-sm text-slate-600">
              Nothing has been started yet. The run goes on the job queue, so you
              can leave this screen and come back to it.
            </p>
          ) : null}
          {blocked ? (
            // Refusing here rather than letting the queue swallow the failure:
            // a head run on an unloadable pack raises
            // ModelArchitectureUnavailable seconds in, and the banner then
            // replaces the message that said why.
            <div className="mt-3 rounded-md border border-amber-200 bg-amber-50 p-3">
              <p className="m-0 text-sm font-semibold text-amber-900">
                {baseModel} cannot run on this machine, so head training would
                fail.
              </p>
              <p className="m-0 mt-1 text-sm text-amber-900">
                {baseModelRunnability.reason}
              </p>
              <p className="m-0 mt-1 text-sm text-amber-900">
                <Link className="underline" to="/models">
                  Install or repair it on the models screen
                </Link>
                , or go back and choose threshold calibration, which needs no
                model weights.
              </p>
            </div>
          ) : null}
          {budgetError ? (
            <p
              className="m-0 mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800"
              role="alert"
            >
              {budgetError} Fix it on "What to fit" — the server would otherwise
              substitute its own default and record that as what you asked for.
            </p>
          ) : null}
          <div className="mt-3">
            <Button
              variant="primary"
              onClick={outcome.concluded ? onStartAgain : onStart}
              disabled={starting || !canStart || blocked || budgetError !== null}
            >
              {starting
                ? "Starting…"
                : outcome.concluded
                  ? "Start again"
                  : "Start adaptation"}
            </Button>
          </div>
        </>
      ) : (
        <>
          <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-slate-200">
            <div
              className="h-full rounded-full bg-cyan-600 transition-[width]"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="m-0 mt-2 text-sm text-slate-600">
            {job?.message || (outcome.running ? "Queued." : "Finished.")}
          </p>
          {job?.cancel_requested && outcome.running ? (
            <p className="m-0 mt-1 text-sm text-amber-800">
              Cancellation requested. Training stops at its next checkpoint,
              usually within a few seconds; nothing is saved.
            </p>
          ) : null}
          {adapter?.train_seconds ? (
            <p className="m-0 mt-1 text-xs text-slate-500">
              Trained in {formatDuration(adapter.train_seconds)}.
            </p>
          ) : null}
        </>
      )}

      {cancelError ? (
        <p
          className="m-0 mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800"
          role="alert"
        >
          {cancelError}
        </p>
      ) : null}

      {startError ? (
        <p className="m-0 mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
          {startError}
        </p>
      ) : null}
    </Panel>
  );
}
