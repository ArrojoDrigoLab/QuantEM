/**
 * The analysis screen: configure a run, watch it, read what it measured.
 *
 * Scoped to one segmentation because that is what the endpoint is scoped to —
 * `POST /api/segmentations/<id>/analysis/`. The route is per asset, so the
 * screen picks the segmentation itself and accepts `?seg=<id>` for a deep link
 * from the proofreading screen.
 */

import { useCallback, useEffect, useMemo, useState, useRef } from "react";
import { Link, Navigate, useParams, useSearchParams } from "react-router-dom";
import { getAsset, getAssetSegmentations } from "@/shared/api/assets";
import { segmentationDisplayName } from "@/shared/segmentationNames";
import {
  getAnalysisRun,
  getAnalysisRuns,
  startAnalysisRun,
} from "@/shared/api/analysis";
import { stopJob } from "@/shared/api/jobs";
import { useWorkspaceState } from "@/shared/hooks/useWorkspaceState";
import { usePollingResource } from "@/shared/hooks/usePollingResource";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { useJobProgress } from "@/shared/hooks/useJobProgress";
import { Badge, Button, PageState, Panel } from "@/shared/ui/design";
import { cx } from "@/shared/ui/cx";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import type { AnalysisRun } from "@/shared/types/analysis";
import {
  buildAnalysisPayload,
  defaultFormState,
  reconcileAnalysisForm,
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
  // Isolate requests across images; restore workflow choices from this session.
  return <ImageAnalysisScreen key={assetId} />;
}

function ImageAnalysisScreen() {
  const { assetId } = useParams<{ assetId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedSegmentationId = searchParams.get("seg");

  const [segmentationId, setSegmentationId] = useWorkspaceState<string>(`analysis.${assetId}.segmentation`, "");
  const [form, setForm] = useWorkspaceState<AnalysisFormState | null>(`analysis.${assetId}.${segmentationId}.form`, null);
  const [formError, setFormError] = useState<string | null>(null);
  const [queuedJob, setQueuedJob] = useState<{ runId: string; jobId: string } | null>(null);
  const [removedJobIds, setRemovedJobIds] = useState<Set<string>>(() => new Set());
  const [activeRunId, setActiveRunId] = useWorkspaceState<string | null>(`analysis.${assetId}.${segmentationId}.run`, null);
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [globalMaskIds, setGlobalMaskIds] = useState<string[]>([]);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const [measuringGlobal, setMeasuringGlobal] = useState(false);

  const { data: asset, error: assetError, refetch: refetchAsset } = useApiQuery(
    () => (assetId ? getAsset(assetId) : Promise.resolve(null)),
    [assetId]
  );
  const { data: segmentations, loading: segmentationsLoading, error: segmentationsError, refetch: refetchSegmentations } = useApiQuery(
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
    const organelles = segmentations.filter(
      (seg) => (seg.segmentation_type.category ?? "organelle") === "organelle"
    );
    const choices = organelles.length > 0 ? organelles : segmentations;
    const requested = requestedSegmentationId
      ? segmentations.find((seg) => seg.id === requestedSegmentationId)?.id ?? null
      : null;
    const next =
      requested ??
      segmentations.find((seg) => seg.id === segmentationId)?.id ??
      choices.find((seg) => seg.is_complete)?.id ?? choices[0].id;
    if (next === segmentationId) return;
    setSegmentationId(next);
  }, [segmentations, requestedSegmentationId, segmentationId, setSegmentationId]);

  useEffect(() => {
    if (!segmentations || !segmentationId) return;
    setForm((current) => current ? reconcileAnalysisForm(current, segmentations, segmentationId) : defaultFormState(segmentations, segmentationId));
    setFormError(null);
  }, [segmentations, segmentationId, setForm]);

  const {
    data: runHistory,
    loading: runsLoading,
    error: runsError,
    refetch: refetchRuns,
  } = useApiQuery(
    async () => ({ segmentationId, runs: segmentationId ? await getAnalysisRuns(segmentationId) : [] }),
    [segmentationId]
  );

  const runs = runHistory?.segmentationId === segmentationId ? runHistory.runs : [];

  const {
    data: run,
    error: runError,
    retrying: retryingRun,
    refresh: refetchRun,
  } = usePollingResource<AnalysisRun>(activeRunId, getAnalysisRun,
    (value) => value.status === "SUCCESS" || value.status === "FAILED", 2000);

  const candidateJobId = run && "job_id" in run ? run.job_id ?? null
    : queuedJob?.runId === activeRunId ? queuedJob.jobId : null;
  const jobId = candidateJobId && !removedJobIds.has(candidateJobId) ? candidateJobId : null;
  const selectedRunRef = useRef(activeRunId);
  useEffect(() => {
    selectedRunRef.current = activeRunId;
    setCancelling(false);
    setCancelError(null);
  }, [activeRunId]);
  useEffect(() => () => { selectedRunRef.current = null; }, []);
  const contextRef = useRef(segmentationId);
  useEffect(() => { contextRef.current = segmentationId; }, [segmentationId]);
  useEffect(() => () => { contextRef.current = ""; }, []);

  const { job, error: jobError, retrying: retryingJob, refresh: refreshJob } = useJobProgress(jobId);
  const jobStatus = job?.status;

  // Job completion refreshes both the detail and history immediately.
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
  const isGlobal = selectedSegmentation?.segmentation_type.category === "area";
  const availableAnalysisMasks = useMemo(
    () =>
      (segmentations ?? []).filter(
        (segmentation) =>
          segmentation.id !== segmentationId &&
          segmentation.segmentation_type.category === "area"
      ),
    [segmentationId, segmentations]
  );

  useEffect(() => {
    setGlobalMaskIds([]);
    setGlobalError(null);
  }, [segmentationId]);

  const handleGlobalMeasure = useCallback(async () => {
    if (!segmentationId) return;
    setMeasuringGlobal(true);
    setGlobalError(null);
    try {
      const response = await startAnalysisRun(segmentationId, { analysis_mask_ids: globalMaskIds });
      if (contextRef.current !== segmentationId) return;
      setActiveRunId(response.analysis_run_id);
      setQueuedJob({ runId: response.analysis_run_id, jobId: response.job_id });
      setCancelError(null);
      void refetchRuns();
    } catch (error) {
      if (contextRef.current !== segmentationId) return;
      setGlobalError(
        extractApiErrorMessage(error, "The foreground-area report could not be measured.")
      );
    } finally {
      setMeasuringGlobal(false);
    }
  }, [globalMaskIds, segmentationId, refetchRuns, setActiveRunId]);

  const handleSelectSegmentation = useCallback(
    (nextId: string) => {
      setSegmentationId(nextId);
      const next = new URLSearchParams(searchParams);
      next.set("seg", nextId);
      setSearchParams(next, { replace: true });
    },
    [searchParams, setSearchParams, setSegmentationId]
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
      if (contextRef.current !== segmentationId) return;
      setActiveRunId(response.analysis_run_id);
      setQueuedJob({ runId: response.analysis_run_id, jobId: response.job_id });
      setCancelError(null);
      void refetchRuns();
    } catch (err) {
      if (contextRef.current !== segmentationId) return;
      setFormError(
        extractApiErrorMessage(err, "The analysis could not be started.")
      );
    } finally {
      setStarting(false);
    }
  }, [form, segmentationId, refetchRuns, setActiveRunId]);

  const handleCancel = useCallback(async () => {
    if (!jobId) return;
    setCancelling(true);
    setCancelError(null);
    try {
      const outcome = await stopJob(jobId, jobStatus ?? "PENDING");
      if (selectedRunRef.current !== activeRunId) return;
      if (outcome === "removed") setRemovedJobIds((ids) => new Set([...ids, jobId]));
      else await refreshJob();
      void refetchRun();
      void refetchRuns();
    } catch (err) {
      if (selectedRunRef.current !== activeRunId) return;
      setCancelError(
        extractApiErrorMessage(err, "The run could not be cancelled.")
      );
    } finally {
      if (selectedRunRef.current === activeRunId) setCancelling(false);
    }
  }, [activeRunId, jobId, jobStatus, refreshJob, refetchRun, refetchRuns]);

  if (!assetId) {
    return <Navigate to="/" replace />;
  }

  // An undefined status means the first poll has not landed yet; that still
  // counts as running, or the Run button unblocks for a second and a second
  // job gets queued on the same configuration.
  const isRunning =
    jobId !== null &&
    run?.status !== "SUCCESS" && run?.status !== "FAILED" &&
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
    ["PENDING", "RETRY", "RUNNING"].includes(job.status);

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
            <Link
              className="inline-flex h-10 items-center rounded-md border border-slate-300 bg-white px-4 text-sm font-medium text-slate-800 shadow-sm hover:bg-slate-50"
              to={`/assets/${assetId}/viewer`}
            >
              Back to viewer
            </Link>
          </div>
        </header>

        {(assetError || segmentationsError) && <div role="alert">
          <p>Image details could not be loaded. {(assetError || segmentationsError)?.message}</p>
          <Button onClick={() => { void refetchAsset(); void refetchSegmentations(); }}>Retry image details</Button>
        </div>}
        {(runError || jobError) && <div role="status">
          <p>Progress could not be refreshed. {(retryingRun || retryingJob) ? "Retrying automatically." : "The saved run or task may no longer be available."} {(runError || jobError)?.message}</p>
          <Button onClick={() => { void refetchRun(); void refreshJob(); }}>Retry progress</Button>
        </div>}
        {segmentationsLoading ? <PageState title="Loading segmentations…" /> : null}

        {!segmentationsLoading && !segmentationsError && (segmentations?.length ?? 0) === 0 ? (
          <PageState
            title="Nothing to analyse"
            detail="This image has no segmentations yet. Apply a model or finish whole-image manual annotation first."
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
                disabled={starting || measuringGlobal}
                className="mt-1 h-10 w-full rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                value={segmentationId}
                onChange={(event) => handleSelectSegmentation(event.target.value)}
              >
                {(["organelle", "area"] as const).map((category) => <optgroup key={category} label={category === "organelle" ? "Organelles" : "Area coverage"}>
                  {segmentations.filter((seg) => (seg.segmentation_type.category ?? "organelle") === category).map((seg) => <option key={seg.id} value={seg.id}>{segmentationDisplayName(seg)}</option>)}
                </optgroup>)}
              </select>
            </div>
          </Panel>
        ) : null}

        <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(320px,380px)_1fr]">
          <div className="flex flex-col gap-5">
            {isGlobal ? (
              <Panel className="p-4">
                <h2 className="m-0 text-base font-semibold text-slate-950">
                  Foreground area
                </h2>
                {availableAnalysisMasks.length > 0 ? (
                  <fieldset className="mt-3 space-y-2">
                    <legend className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                      Measurement areas
                    </legend>
                    {availableAnalysisMasks.map((mask) => (
                      <label key={mask.id} className="flex items-center gap-2 text-sm">
                        <input
                          type="checkbox"
                          checked={globalMaskIds.includes(mask.id)}
                          onChange={(event) =>
                            setGlobalMaskIds((current) =>
                              event.target.checked
                                ? [...current, mask.id]
                                : current.filter((id) => id !== mask.id)
                            )
                          }
                        />
                        {segmentationDisplayName(mask)}
                      </label>
                    ))}
                  </fieldset>
                ) : null}
                <Button
                  className="mt-4"
                  onClick={() => void handleGlobalMeasure()}
                  disabled={measuringGlobal || isRunning}
                >
                  {measuringGlobal ? "Measuring…" : "Measure foreground area"}
                </Button>
                {globalError ? (
                  <p className="mt-3 text-sm text-red-800" role="alert">{globalError}</p>
                ) : null}
              </Panel>
            ) : form && segmentations ? (
              <AnalysisConfigForm
                segmentations={segmentations}
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
                    <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-slate-200" role="progressbar" aria-label="Analysis progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(job?.progress ?? 0)}>
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
                  {activeRunId ? runError ? "The selected run could not be loaded." : "Loading selected run…" : "No run selected."}
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
