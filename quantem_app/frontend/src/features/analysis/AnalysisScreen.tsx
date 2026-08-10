/**
 * The analysis screen: configure a run, watch it, read what it measured.
 *
 * Scoped to one segmentation because that is what the endpoint is scoped to —
 * `POST /api/segmentations/<id>/analysis/`. The route is per asset, so the
 * screen picks the segmentation itself and accepts `?seg=<id>` for a deep link
 * from the proofreading screen.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, Navigate, useParams, useSearchParams } from "react-router-dom";
import { getAsset, getAssetSegmentations } from "@/shared/api/assets";
import { getAnalysisRun, getAnalysisRuns, startAnalysisRun } from "@/shared/api/analysis";
import { cancelJob, deleteJob } from "@/shared/api/jobs";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { useJobProgress } from "@/shared/hooks/useJobProgress";
import { Badge, Button, PageState, Panel } from "@/shared/ui/design";
import { cx } from "@/shared/ui/cx";
import { PixelSizeEditor } from "@/shared/ui/PixelSize";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import type { AnalysisRun } from "@/shared/types/analysis";
import {
  buildAnalysisPayload,
  defaultFormState,
  type AnalysisFormState,
} from "@/features/analysis/analysisOptions";
import { AnalysisConfigForm } from "@/features/analysis/components/AnalysisConfigForm";
import { AnalysisResults } from "@/features/analysis/components/AnalysisResults";
import { RunHistory } from "@/features/analysis/components/RunHistory";
import {
  reconcileRunHistory,
  resolveAnalysisRunState,
} from "@/features/analysis/runState";

export function AnalysisScreen() {
  const { assetId } = useParams<{ assetId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedSegmentationId = searchParams.get("seg");

  const [segmentationId, setSegmentationId] = useState<string>("");
  const [form, setForm] = useState<AnalysisFormState | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);

  const { data: asset, refetch: refetchAsset } = useApiQuery(
    () => (assetId ? getAsset(assetId) : Promise.resolve(null)),
    [assetId]
  );
  const { data: segmentations, loading: segmentationsLoading } = useApiQuery(
    () => (assetId ? getAssetSegmentations(assetId) : Promise.resolve([])),
    [assetId]
  );

  // Settle on a segmentation whenever the list or the URL changes: the deep
  // link wins whenever it names a segmentation that exists, otherwise the
  // first one on the image. The URL must win on SPA navigation too — this
  // used to keep the first value for the life of the component
  // (`current ? current : wanted`), so `?seg=` was honoured only on a full
  // page load: an in-app link to this screen with a different `?seg=`, or
  // back/forward across two analysis URLs, changed the address bar and
  // nothing on the screen.
  useEffect(() => {
    if (!segmentations || segmentations.length === 0) return;
    const requested = requestedSegmentationId
      ? segmentations.find((seg) => seg.id === requestedSegmentationId)?.id ?? null
      : null;
    const next = requested ?? (segmentationId || segmentations[0].id);
    if (next === segmentationId) return;
    if (segmentationId) {
      // The screen was already showing another segmentation: the run panel
      // and job poller belong to it, so they reset exactly as the picker's
      // own handler resets them.
      setActiveRunId(null);
      setJobId(null);
    }
    setSegmentationId(next);
  }, [segmentations, requestedSegmentationId, segmentationId]);

  useEffect(() => {
    if (!segmentations || !segmentationId) return;
    setForm(defaultFormState(segmentations, segmentationId));
    setFormError(null);
  }, [segmentations, segmentationId]);

  const {
    data: runs,
    loading: runsLoading,
    error: runsError,
    refetch: refetchRuns,
  } = useApiQuery(
    () => (segmentationId ? getAnalysisRuns(segmentationId) : Promise.resolve([])),
    [segmentationId]
  );

  const {
    data: run,
    refetch: refetchRun,
  } = useApiQuery<AnalysisRun | null>(
    () => (activeRunId ? getAnalysisRun(activeRunId) : Promise.resolve(null)),
    [activeRunId]
  );

  const { job, refresh: refreshJob } = useJobProgress(jobId);
  const jobStatus = job?.status;

  // The run row is only filled in when the job finishes, so the detail fetch
  // waits for the job rather than polling the run on its own timer.
  useEffect(() => {
    if (!jobStatus) return;
    if (jobStatus === "SUCCESS" || jobStatus === "FAILED" || jobStatus === "CANCELLED") {
      void refetchRun();
      void refetchRuns();
    }
  }, [jobStatus, refetchRun, refetchRuns]);

  const selectedSegmentation = useMemo(
    () => segmentations?.find((seg) => seg.id === segmentationId) ?? null,
    [segmentations, segmentationId]
  );

  const handleSelectSegmentation = useCallback(
    (nextId: string) => {
      setSegmentationId(nextId);
      setActiveRunId(null);
      setJobId(null);
      const next = new URLSearchParams(searchParams);
      next.set("seg", nextId);
      setSearchParams(next, { replace: true });
    },
    [searchParams, setSearchParams]
  );

  const handleStart = useCallback(async () => {
    if (!form || !segmentationId) return;
    const { payload, error } = buildAnalysisPayload(form);
    if (error || !payload) {
      // Never `setFormError(error)` on its own: a refusal that carries no
      // sentence is a button that does nothing, and this is the only place
      // that decides not to make the request.
      setFormError(
        error ?? "This configuration could not be turned into a run."
      );
      return;
    }
    setFormError(null);
    setStarting(true);
    try {
      const response = await startAnalysisRun(segmentationId, payload);
      setActiveRunId(response.analysis_run_id);
      setJobId(response.job_id);
      setCancelError(null);
      void refetchRuns();
    } catch (err) {
      setFormError(
        extractApiErrorMessage(err, "The analysis could not be started.")
      );
    } finally {
      setStarting(false);
    }
  }, [form, segmentationId, refetchRuns]);

  /**
   * Stop the run this screen started, from this screen.
   *
   * The only Cancel in the application was in the Library's queue sidebar,
   * which is unreachable from here -- so somebody who had just started a
   * twenty-minute analysis by mistake had to navigate away to stop it, and the
   * screen they were on offered no hint that it could be stopped at all.
   *
   * Two endpoints, because a queued job and a running one leave by different
   * doors: `POST /cancel/` refuses anything that is not RUNNING with a 409, and
   * `DELETE /api/jobs/<id>/` is the *only* exit a queued job has. Both
   * reconcile the `AnalysisRun` behind them, so either way the run concludes
   * rather than sitting at PENDING for ever.
   */
  const handleCancel = useCallback(async () => {
    if (!jobId) return;
    setCancelling(true);
    setCancelError(null);
    try {
      if (jobStatus === "RUNNING") {
        await cancelJob(jobId);
      } else {
        await deleteJob(jobId);
      }
      await refreshJob();
      void refetchRun();
      void refetchRuns();
    } catch (err) {
      setCancelError(
        extractApiErrorMessage(err, "The run could not be cancelled.")
      );
    } finally {
      setCancelling(false);
    }
  }, [jobId, jobStatus, refreshJob, refetchRun, refetchRuns]);

  if (!assetId) {
    return <Navigate to="/" replace />;
  }

  // An undefined status means the first poll has not landed yet; that still
  // counts as running, or the Run button unblocks for a second and a second
  // job gets queued on the same configuration.
  const isRunning =
    jobId !== null &&
    (jobStatus === undefined || ["PENDING", "RUNNING", "RETRY"].includes(jobStatus));

  /**
   * The single state this screen renders, reconciled from the run and its job.
   *
   * The screen used to render both independently and contradict itself: a
   * history row saying PENDING, a panel saying FAILED with the raw error
   * printed twice, and "This run is pending. Results appear when it finishes."
   * all at once. Everything below reads this and nothing else.
   */
  const runState = resolveAnalysisRunState(run, job);
  const showProgress = runState.active && jobId !== null;
  // Same reconciler, same rows: the history badge for the selected run is the
  // status the panel above it is showing, not the value the worker last wrote.
  const historyRows = reconcileRunHistory(runs ?? [], activeRunId, runState);
  // A cancellation request is already in flight once the queue has it, and a
  // job that has left PENDING/RUNNING cannot be stopped at all.
  const canCancel =
    jobId !== null &&
    runState.active &&
    job !== null &&
    !job.cancel_requested &&
    (job.status === "PENDING" || job.status === "RUNNING");

  return (
    <div className="min-h-screen px-5 py-5 text-slate-900 lg:px-8">
      <div className="mx-auto flex max-w-[1500px] flex-col gap-5">
        <header className="flex flex-wrap items-center justify-between gap-4 rounded-lg border border-slate-200 bg-white px-4 py-3 shadow-sm">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-cyan-700">
              QuantEM
            </p>
            <h1 className="m-0 text-2xl font-semibold tracking-normal text-slate-950">
              Quantitative analysis
            </h1>
            <p className="m-0 mt-1 text-sm text-slate-500">
              {asset ? asset.display_name : "Loading image…"}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {/* Editable here, not just reported: an uncalibrated asset is
                refused physical units outright, and this is the screen where
                the user finds that out. */}
            {asset ? (
              <PixelSizeEditor
                asset={asset}
                onSaved={() => {
                  void refetchAsset();
                }}
              />
            ) : null}
            <Link
              className="inline-flex h-10 items-center rounded-md border border-slate-300 bg-white px-4 text-sm font-medium text-slate-800 shadow-sm hover:bg-slate-50"
              to={`/assets/${assetId}/adapt${segmentationId ? `?seg=${segmentationId}` : ""}`}
            >
              Adapt a model
            </Link>
            <Link
              className="inline-flex h-10 items-center rounded-md border border-slate-300 bg-white px-4 text-sm font-medium text-slate-800 shadow-sm hover:bg-slate-50"
              to={`/assets/${assetId}/viewer`}
            >
              Back to viewer
            </Link>
          </div>
        </header>

        {segmentationsLoading ? <PageState title="Loading segmentations…" /> : null}

        {!segmentationsLoading && (segmentations?.length ?? 0) === 0 ? (
          <PageState
            title="Nothing to analyse"
            detail="This image has no segmentations yet. Create one and confirm some objects first."
          />
        ) : null}

        {segmentations && segmentations.length > 0 ? (
          <Panel className="flex flex-wrap items-end gap-4 p-4">
            <div className="min-w-[280px] flex-1">
              <label
                className="block text-xs font-semibold uppercase tracking-wide text-slate-500"
                htmlFor="analysis-segmentation"
              >
                Segmentation
              </label>
              <select
                id="analysis-segmentation"
                className="mt-1 h-10 w-full rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                value={segmentationId}
                onChange={(event) => handleSelectSegmentation(event.target.value)}
              >
                {segmentations.map((seg) => (
                  <option key={seg.id} value={seg.id}>
                    {seg.segmentation_type.long_name}
                  </option>
                ))}
              </select>
            </div>
            {selectedSegmentation ? (
              <p className="m-0 text-xs text-slate-500">
                Runs, exports and the Monte-Carlo seed are recorded against this
                segmentation.
              </p>
            ) : null}
          </Panel>
        ) : null}

        <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(320px,380px)_1fr]">
          <div className="flex flex-col gap-5">
            {form && segmentations ? (
              <AnalysisConfigForm
                segmentations={segmentations}
                selectedSegmentation={selectedSegmentation}
                state={form}
                onChange={setForm}
                onSubmit={() => {
                  void handleStart();
                }}
                submitting={starting || isRunning}
                error={formError}
              />
            ) : null}

            <RunHistory
              runs={historyRows}
              selectedRunId={activeRunId}
              onSelect={(runId) => {
                setActiveRunId(runId);
                setJobId(null);
                setCancelError(null);
              }}
              loading={runsLoading}
              error={runsError}
            />
          </div>

          <div className="flex flex-col gap-5">
            {/* Exactly one panel describes the run's state, and it is the only
                place the failure is printed. There used to be two -- a job
                panel here and a status panel inside AnalysisResults -- which
                is how the same Windows exit code appeared twice above a
                sentence claiming the run was still pending. */}
            {runState.status && runState.status !== "SUCCESS" ? (
              <Panel
                className={cx(
                  "p-4",
                  runState.status === "FAILED" && "border-red-200 bg-red-50",
                  // Amber, not red. A cancellation is a decision, not a fault,
                  // and colouring it like a crash is half of why the same click
                  // read "You cancelled this run" in the Adapt wizard and "This
                  // run failed" here.
                  runState.cancelled && "border-amber-200 bg-amber-50"
                )}
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h2
                    className={cx(
                      "m-0 text-base font-semibold",
                      runState.cancelled
                        ? "text-amber-900"
                        : runState.status === "FAILED"
                          ? "text-red-900"
                          : "text-slate-950"
                    )}
                  >
                    {/* Word for word what StepRun says about the same event. */}
                    {runState.cancelled
                      ? "You cancelled this run."
                      : runState.status === "FAILED"
                        ? "This run failed"
                        : runState.active
                          ? "Running"
                          : "Run finished"}
                  </h2>
                  <div className="flex items-center gap-2">
                    <Badge
                      tone={
                        runState.status === "FAILED" || runState.cancelled
                          ? "warning"
                          : "info"
                      }
                    >
                      {runState.status}
                    </Badge>
                    {canCancel ? (
                      // The only Cancel used to be in the Library's queue
                      // sidebar, which is unreachable from this screen.
                      <Button size="sm" onClick={() => void handleCancel()} disabled={cancelling}>
                        {cancelling ? "Cancelling…" : "Cancel run"}
                      </Button>
                    ) : null}
                  </div>
                </div>

                {showProgress ? (
                  <>
                    <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-slate-200">
                      <div
                        className="h-full rounded-full bg-cyan-600 transition-[width]"
                        style={{
                          width: `${Math.min(100, Math.max(0, job?.progress ?? 0))}%`,
                        }}
                      />
                    </div>
                    <p className="m-0 mt-2 text-sm text-slate-600">
                      {job?.message || "Queued."}
                    </p>
                  </>
                ) : null}

                {job?.cancel_requested && runState.active ? (
                  <p className="m-0 mt-2 text-sm text-amber-800">
                    Cancellation requested. The worker stops at its next
                    checkpoint, usually within a few seconds.
                  </p>
                ) : null}

                {cancelError ? (
                  <p className="m-0 mt-2 text-sm text-red-800" role="alert">
                    {cancelError}
                  </p>
                ) : null}

                {runState.error ? (
                  <p
                    className={cx(
                      "m-0 mt-2 text-sm",
                      runState.cancelled ? "text-amber-900" : "text-red-800"
                    )}
                  >
                    {runState.error}
                  </p>
                ) : null}

                {runState.cancelled ? (
                  <p className="m-0 mt-2 text-sm text-amber-900">
                    Nothing about the segmentation changed and your annotations
                    are untouched, so starting again costs only the time.
                  </p>
                ) : null}

                {runState.reconciledFromJob ? (
                  // The run row itself has not caught up, so the history list
                  // beside this panel may still read PENDING. Say so rather
                  // than letting the two disagree in silence.
                  <p
                    className={cx(
                      "m-0 mt-2 text-xs",
                      runState.cancelled ? "text-amber-900" : "text-red-800"
                    )}
                  >
                    The job that was producing this run stopped. The run's own
                    record may still read PENDING in the history until the
                    server reconciles it; no results were written.
                  </p>
                ) : null}

                {runState.status !== "FAILED" &&
                !runState.cancelled &&
                !showProgress ? (
                  <p className="m-0 mt-2 text-sm text-slate-600">
                    Results appear when it finishes.
                  </p>
                ) : null}
              </Panel>
            ) : null}

            {runState.status === "SUCCESS" && run ? (
              <AnalysisResults run={run} />
            ) : null}

            {runState.status === null ? (
              <Panel className="p-6 text-center">
                <p className="m-0 text-sm text-slate-600">
                  Configure a run on the left, or pick one from the history to
                  read its results.
                </p>
                <div className="mt-3">
                  <Button
                    onClick={() => {
                      void refetchRuns();
                    }}
                  >
                    Refresh history
                  </Button>
                </div>
              </Panel>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
