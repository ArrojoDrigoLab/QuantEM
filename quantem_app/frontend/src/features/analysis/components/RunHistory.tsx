/**
 * Runs already started for this segmentation.
 *
 * The caveat count is on the row rather than hidden behind a click: a run with
 * four caveats and a run with none are not interchangeable, and someone
 * choosing which numbers to quote needs to see that in the list.
 *
 * A failed fetch is never rendered as an empty list. "0 runs -- nothing has
 * been analysed yet" in response to a 404 tells the user their previous runs
 * are gone, which is a different and much worse claim than "the endpoint did
 * not answer".
 *
 * The badge is `displayStatus`, not `status`. The row is only written when the
 * worker moves the run on, so a run mid-write showed `PENDING` here beside a
 * panel reading "writing export bundle" -- one run, two claims, a hand's width
 * apart, and the permanent-looking one was the wrong one. `reconcileRunHistory`
 * hands the selected row the same reconciled state the panel renders; every
 * other row is the server's own.
 */

import { Badge, Panel } from "@/shared/ui/design";
import { cx } from "@/shared/ui/cx";
import { formatInteger, formatTimestamp } from "@/shared/ui/format";
import { extractApiErrorMessage, isApiNotFoundError } from "@/utils/apiErrors";
import type { AnalysisRunStatus, AnalysisRunSummary } from "@/shared/types/analysis";

/** A row's status once the live job has had its say. */
export type DisplayRunStatus = AnalysisRunStatus | "CANCELLED";

export type RunHistoryRow = AnalysisRunSummary & {
  displayStatus: DisplayRunStatus;
};

function statusTone(status: DisplayRunStatus): "default" | "good" | "warning" {
  if (status === "SUCCESS") return "good";
  // A cancellation is not a fault, but it is not a result either: amber says
  // "there is nothing here", which is the thing worth knowing at a glance.
  if (status === "FAILED" || status === "CANCELLED") return "warning";
  return "default";
}

export interface RunHistoryProps {
  runs: RunHistoryRow[];
  selectedRunId: string | null;
  onSelect: (runId: string) => void;
  loading: boolean;
  /** Set when the list could not be fetched at all. */
  error?: Error | null;
}

export function RunHistory({
  runs,
  selectedRunId,
  onSelect,
  loading,
  error = null,
}: RunHistoryProps) {
  const failed = Boolean(error);
  return (
    <Panel className="p-4">
      <div className="flex items-center justify-between gap-2">
        <h2 className="m-0 text-base font-semibold text-slate-950">Run history</h2>
        {loading ? (
          <span className="text-xs text-slate-500">Loading…</span>
        ) : failed ? (
          <span className="text-xs text-amber-700">unavailable</span>
        ) : (
          <span className="text-xs text-slate-500">{runs.length} runs</span>
        )}
      </div>

      {failed ? (
        <div className="mt-2 rounded-md border border-amber-200 bg-amber-50 p-3">
          <p className="m-0 text-sm font-semibold text-amber-900">
            {isApiNotFoundError(error)
              ? "The run-history endpoint did not answer."
              : "Run history could not be loaded."}
          </p>
          <p className="m-0 mt-1 text-xs text-amber-800">
            {extractApiErrorMessage(
              error,
              "This is a failed request, not an empty history — previous runs may still exist."
            )}
          </p>
        </div>
      ) : null}

      {runs.length === 0 && !loading && !failed ? (
        <p className="m-0 mt-2 text-sm text-slate-500">
          Nothing has been analysed for this segmentation yet.
        </p>
      ) : null}

      <ul className="m-0 mt-3 flex list-none flex-col gap-2 p-0">
        {runs.map((run) => (
          <li key={run.id}>
            <button
              type="button"
              onClick={() => onSelect(run.id)}
              className={cx(
                "w-full rounded-md border px-3 py-2 text-left transition-colors",
                run.id === selectedRunId
                  ? "border-cyan-500 bg-cyan-50"
                  : "border-slate-200 bg-white hover:bg-slate-50"
              )}
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-sm font-medium text-slate-900">
                  {formatTimestamp(run.created_at)}
                </span>
                <Badge tone={statusTone(run.displayStatus)}>
                  {run.displayStatus}
                </Badge>
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                {run.group ? <span>group: {run.group}</span> : null}
                <span>{formatInteger(run.n_objects)} objects</span>
                {run.calibrated === false ? (
                  <span className="text-amber-700">uncalibrated</span>
                ) : null}
                {run.n_caveats > 0 ? (
                  <span className="text-amber-700">
                    {run.n_caveats} caveat{run.n_caveats === 1 ? "" : "s"}
                  </span>
                ) : null}
              </div>
            </button>
          </li>
        ))}
      </ul>
    </Panel>
  );
}
