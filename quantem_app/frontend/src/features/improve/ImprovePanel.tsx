/**
 * "Make it better" — one panel, one button.
 *
 * This replaces a six-step wizard (`AdaptWizard.tsx`, 898 lines, plus five
 * `Step*.tsx` files) whose entire purpose, for the rung anyone actually uses,
 * was to run a 0.35 s numpy threshold sweep. Six screens and five questions —
 * base model, seed, learning rate, step budget, split mode — none of which a
 * microscopist can answer, in front of one operation that takes less time than
 * the first screen took to read.
 *
 * Four defects the wizard shipped, and how each is closed here:
 *
 * 1. **It was single-shot.** `StepRun.tsx` only offered a start button when
 *    `(!job && !adapter) || outcome.concluded`, and a *succeeded* run is
 *    neither — so after one successful improvement the button was gone. A
 *    `localStorage` pointer written on start and never cleared on success meant
 *    a reload came straight back to the same finished run. "Improve again" was
 *    literally unreachable without clearing browser storage. Reattachment now
 *    comes from the server (`GET .../adapt/latest/`), a finished run is a
 *    *result* rather than a lock, and the button below it says "Learn from my
 *    fixes again".
 * 2. **It queued runs that could not work.** A threshold run with no stored
 *    probability map, and head training on a checked area too small to cut one
 *    training window from, both reached the queue and died there. Both are
 *    knowable from the crops response, and both are now refused before the
 *    request is sent *and* again by the server before it is queued.
 * 3. **It never said what would survive.** A model pass has never destroyed
 *    manual work — the extraction code deletes only its own previous guesses
 *    and drops any new guess landing on a kept or removed object — but nothing
 *    on screen said so, at the one moment the user is deciding whether to risk
 *    it. Now it is said before the button, again in the result, and again
 *    before the re-run.
 * 4. **The honest statistics were the price of admission.** Held-out Dice,
 *    split mode, the oracle ceiling, the sweep curve and the ground-truth
 *    composition were four wizard steps deep and unavoidable. They are all
 *    still here, unchanged, behind one disclosure.
 *
 * What is deliberately *not* here: seed, learning rate, step budget and the
 * step ladder. They were never answerable, and `adaptBudget.ts` existed only to
 * validate numbers nobody should have been typing.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, Navigate, useParams, useSearchParams } from "react-router-dom";
import { getAsset, getAssetSegmentations } from "@/shared/api/assets";
import { segmentationDisplayName } from "@/shared/segmentationNames";
import {
  applyAdapter,
  getAdaptCrops,
  getAdapter,
  getLatestAdaptRun,
  getModelCatalogue,
  startAdaptation,
} from "@/shared/api/finetune";
import { runFullSegmentation } from "@/shared/api/segmentations/overlays";
import { cancelJob, deleteJob } from "@/shared/api/jobs";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { useJobProgress } from "@/shared/hooks/useJobProgress";
import { Badge, Button, PageState, Panel } from "@/shared/ui/design";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import { runnabilityForPackId } from "@/features/models/runnable";
import { suggestBaseModel, toModelChoices } from "@/features/improve/modelChoices";
import { resolveAdaptRunOutcome } from "@/features/improve/runOutcome";
import { useGroundTruthProvenance } from "@/features/improve/useGroundTruthProvenance";
import { AboutTheNumbers } from "@/features/improve/AboutTheNumbers";
import {
  calibrationReport,
  checkedAreasSentence,
  costSentence,
  formatIncludeLevel,
  levelChanged,
  summariseCheckedAreas,
  PRESERVATION_SENTENCE,
} from "@/features/improve/copy";
import type {
  Adapter,
  AdapterJobResult,
  AdaptMode,
  AdapterSweep,
  ModelCatalogue,
} from "@/shared/types/finetune";

/** How often to re-read the run row while one is in flight. */
const ADAPTER_POLL_MS = 5000;

function sweepFromJobResult(
  result: unknown,
  key: "sweep" | "base_sweep"
): AdapterSweep | null {
  if (typeof result !== "object" || result === null) return null;
  const candidate = (result as AdapterJobResult)[key];
  if (
    typeof candidate === "object" &&
    candidate !== null &&
    Array.isArray((candidate as AdapterSweep).thresholds)
  ) {
    return candidate as AdapterSweep;
  }
  return null;
}

export function ImprovePanel() {
  const { assetId } = useParams<{ assetId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedSegmentationId = searchParams.get("seg");

  const [segmentationId, setSegmentationId] = useState("");
  const [adapterId, setAdapterId] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  /**
   * The run this panel is *presenting*.
   *
   * Distinct from "the latest run the server knows about", which is what
   * reattachment finds. Pressing the button again clears this so the panel
   * shows the new run rather than the old result — the thing the wizard could
   * not do at all.
   */
  const [reattached, setReattached] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [applying, setApplying] = useState(false);
  const [applyError, setApplyError] = useState<string | null>(null);
  const [rerunJobId, setRerunJobId] = useState<string | null>(null);
  const [rerunError, setRerunError] = useState<string | null>(null);
  const [showNumbers, setShowNumbers] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState<number | null>(null);
  const startedAt = useRef<number | null>(null);

  const { data: asset } = useApiQuery(
    () => (assetId ? getAsset(assetId) : Promise.resolve(null)),
    [assetId]
  );
  const { data: segmentations, loading: segmentationsLoading } = useApiQuery(
    () => (assetId ? getAssetSegmentations(assetId) : Promise.resolve([])),
    [assetId]
  );

  // Settle on a segmentation. The URL wins whenever it names one that exists,
  // including on in-app navigation, so a deep link from the labeling header
  // opens on the segmentation that header was describing.
  useEffect(() => {
    if (!segmentations || segmentations.length === 0) return;
    const requested = requestedSegmentationId
      ? segmentations.find((seg) => seg.id === requestedSegmentationId)?.id ?? null
      : null;
    const next = requested ?? (segmentationId || segmentations[0].id);
    if (next === segmentationId) return;
    setSegmentationId(next);
    setAdapterId(null);
    setJobId(null);
    setReattached(null);
    setShowNumbers(false);
    setElapsedSeconds(null);
  }, [segmentations, requestedSegmentationId, segmentationId]);

  const { data: catalogue } = useApiQuery<ModelCatalogue | null>(
    () => getModelCatalogue(),
    []
  );
  const choices = useMemo(() => toModelChoices(catalogue ?? null), [catalogue]);

  const selectedSegmentation = useMemo(
    () => segmentations?.find((seg) => seg.id === segmentationId) ?? null,
    [segmentations, segmentationId]
  );

  /**
   * Which model this is about — decided, not asked.
   *
   * The wizard's first screen was a base-model picker. There is exactly one
   * right answer: the pack that produced the objects on this screen. Asking was
   * also how an ER segmentation's improvement got fitted against the
   * mitochondria pack.
   */
  const baseModel = useMemo(() => {
    if (!selectedSegmentation || choices.length === 0) return "";
    return (
      suggestBaseModel(
        selectedSegmentation.segmentation_type.internal_name,
        choices
      ) ?? choices[0].id
    );
  }, [selectedSegmentation, choices]);

  const baseModelRunnability = useMemo(
    () => runnabilityForPackId(catalogue ?? null, baseModel),
    [catalogue, baseModel]
  );

  const {
    data: crops,
    loading: cropsLoading,
    error: cropsError,
    refetch: refetchCrops,
  } = useApiQuery(
    () =>
      segmentationId
        ? getAdaptCrops(segmentationId, baseModel || null)
        : Promise.resolve(null),
    [segmentationId, baseModel]
  );

  const { provenance, provenanceLoading, provenanceError } =
    useGroundTruthProvenance(crops ?? null);

  // Reattach to whatever this segmentation already has, from the server. A
  // reload during a head training used to lose the run; a *finished* run used
  // to make the next one impossible to start.
  const { data: latest } = useApiQuery(
    () =>
      segmentationId ? getLatestAdaptRun(segmentationId) : Promise.resolve(null),
    [segmentationId]
  );
  useEffect(() => {
    if (!segmentationId || !latest?.adapter) return;
    if (reattached === segmentationId || adapterId) return;
    setReattached(segmentationId);
    setAdapterId(latest.adapter.id);
    if (latest.adapter.status === "PENDING" || latest.adapter.status === "RUNNING") {
      setJobId(latest.job_id);
    }
  }, [latest, segmentationId, reattached, adapterId]);

  const {
    data: fetchedAdapter,
    refetch: refetchAdapter,
  } = useApiQuery<Adapter | null>(
    () => (adapterId ? getAdapter(adapterId) : Promise.resolve(null)),
    [adapterId]
  );
  /**
   * The run being presented, and only that run.
   *
   * `useApiQuery` keeps the previous body while the next request is in flight,
   * so the instant "Learn from my fixes again" is pressed the query still holds
   * the *finished* previous run — which would flash its result, and its
   * "Done in 0.4 seconds", over a run that has only just started. Pinning the
   * id is one comparison and removes the whole class.
   */
  const adapter =
    fetchedAdapter && fetchedAdapter.id === adapterId ? fetchedAdapter : null;

  const { job, refresh: refreshJob } = useJobProgress(jobId);
  const jobStatus = job?.status;

  useEffect(() => {
    if (!jobStatus) return;
    if (jobStatus === "SUCCESS" || jobStatus === "FAILED" || jobStatus === "CANCELLED") {
      void refetchAdapter();
    }
  }, [jobStatus, refetchAdapter]);

  // The run row is the authority on whether the run is over: the job poll is
  // the fast path and can miss the transition (a throttled tab, a dropped
  // request, a trimmed job row), which is how the wizard sat on a progress bar
  // for a run that had finished.
  const adapterStatus = adapter?.status;
  useEffect(() => {
    if (adapterStatus !== "SUCCESS" && adapterStatus !== "FAILED") return;
    if (jobId) setJobId(null);
    if (startedAt.current !== null) {
      setElapsedSeconds((Date.now() - startedAt.current) / 1000);
      startedAt.current = null;
    }
  }, [adapterStatus, jobId]);

  useEffect(() => {
    if (!adapterId) return undefined;
    if (adapterStatus === "SUCCESS" || adapterStatus === "FAILED") return undefined;
    const timer = window.setInterval(() => {
      void refetchAdapter();
    }, ADAPTER_POLL_MS);
    return () => window.clearInterval(timer);
  }, [adapterId, adapterStatus, refetchAdapter]);

  const outcome = resolveAdaptRunOutcome(job, adapter ?? null);
  const running = adapter !== null && !outcome.concluded && adapter.status !== "SUCCESS";
  const succeeded = adapter?.status === "SUCCESS";

  const headOffered = (crops?.modes ?? []).includes("head");

  /**
   * Why a rung cannot run, as one sentence, or null.
   *
   * Everything that stops the whole page comes first — no checked area is not a
   * threshold problem — then the reason specific to this rung, then the reason
   * this machine cannot load the model. First one wins: a stack of four
   * refusals teaches nothing.
   */
  const refusalFor = useCallback(
    (mode: AdaptMode): string | null => {
      if (!crops) return null;
      if (!baseModel) {
        // No catalogue, so no pack to fit against. A live button that silently
        // does nothing is worse than a greyed one that says why.
        return "I could not read the list of installed models, so I do not know which one made these objects.";
      }
      if (crops.blockers.length > 0) return crops.blockers[0];
      const perMode = (crops.mode_blockers ?? {})[mode] ?? [];
      if (perMode.length > 0) return perMode[0];
      if (mode === "head") {
        if (!headOffered) {
          return "This build cannot train a model. Matching my cut-off to your marks works everywhere.";
        }
        if (baseModelRunnability.state === "blocked") {
          return (
            baseModelRunnability.reason ??
            "This model cannot be loaded on this machine, so training it would fail."
          );
        }
      }
      return null;
    },
    [crops, baseModel, headOffered, baseModelRunnability]
  );

  const start = useCallback(
    async (mode: AdaptMode) => {
      if (!segmentationId || !baseModel) return;
      const refusal = refusalFor(mode);
      if (refusal) {
        setStartError(refusal);
        return;
      }
      setStarting(true);
      setStartError(null);
      setApplyError(null);
      setRerunError(null);
      setRerunJobId(null);
      setShowNumbers(false);
      setElapsedSeconds(null);
      startedAt.current = Date.now();
      try {
        const response = await startAdaptation(segmentationId, {
          base_model: baseModel,
          mode,
          name: asset ? `${asset.display_name}` : "",
        });
        // Point the panel at the new run before the old one is polled again,
        // so a second press never shows the previous result.
        setAdapterId(response.adapter_id);
        setJobId(response.job_id);
      } catch (err) {
        startedAt.current = null;
        setStartError(
          extractApiErrorMessage(err, "This could not be started.")
        );
      } finally {
        setStarting(false);
      }
    },
    [segmentationId, baseModel, refusalFor, asset]
  );

  const handleCancel = useCallback(async () => {
    if (!jobId) return;
    setCancelling(true);
    setCancelError(null);
    try {
      // A queued job and a running one leave by different doors: POST /cancel/
      // refuses anything that is not RUNNING, and DELETE is a queued job's only
      // exit. Both reconcile the run row behind them.
      if (jobStatus === "RUNNING") {
        await cancelJob(jobId);
      } else {
        await deleteJob(jobId);
      }
      await refreshJob();
      void refetchAdapter();
    } catch (err) {
      setCancelError(extractApiErrorMessage(err, "This could not be stopped."));
    } finally {
      setCancelling(false);
    }
  }, [jobId, jobStatus, refreshJob, refetchAdapter]);

  const handleApply = useCallback(async () => {
    if (!adapterId) return;
    setApplying(true);
    setApplyError(null);
    try {
      await applyAdapter(adapterId);
      await refetchAdapter();
    } catch (err) {
      setApplyError(
        extractApiErrorMessage(err, "This include level could not be used.")
      );
    } finally {
      setApplying(false);
    }
  }, [adapterId, refetchAdapter]);

  const handleRerun = useCallback(async () => {
    if (!segmentationId) return;
    setRerunError(null);
    try {
      const response = await runFullSegmentation(segmentationId, baseModel || null);
      setRerunJobId(String(response.job_id ?? ""));
    } catch (err) {
      setRerunError(
        extractApiErrorMessage(err, "The model could not be run again.")
      );
    }
  }, [segmentationId, baseModel]);

  const handleSelectSegmentation = useCallback(
    (nextId: string) => {
      const next = new URLSearchParams(searchParams);
      next.set("seg", nextId);
      setSearchParams(next, { replace: true });
    },
    [searchParams, setSearchParams]
  );

  if (!assetId) {
    return <Navigate to="/" replace />;
  }

  const summary = summariseCheckedAreas(crops?.crops);
  const report = succeeded && adapter ? calibrationReport(adapter, elapsedSeconds) : null;
  const applied = Boolean(adapter?.applied_at);
  const worthRerunning =
    applied &&
    levelChanged(
      adapter?.calibrated_threshold ?? null,
      adapter?.default_threshold ?? null
    );
  const baseSweep = sweepFromJobResult(job?.result_json, "base_sweep");
  const thresholdRefusal = refusalFor("threshold_only");
  const headRefusal = refusalFor("head");
  const labelingHref = selectedSegmentation
    ? `#/assets/${assetId}/labeling/${encodeURIComponent(
        selectedSegmentation.segmentation_type.long_name
      )}`
    : `#/assets/${assetId}/viewer`;

  return (
    <div className="min-h-screen px-5 py-5 text-slate-900 lg:px-8">
      <div className="mx-auto flex max-w-[900px] flex-col gap-5">
        <header className="flex flex-wrap items-center justify-between gap-4 rounded-lg border border-slate-200 bg-white px-4 py-3 shadow-sm">
          <div>
            <p className="m-0 text-xs font-semibold uppercase tracking-wide text-cyan-700">
              QuantEM
            </p>
            <h1 className="m-0 text-2xl font-semibold tracking-normal text-slate-950">
              Make it better
            </h1>
            <p className="m-0 mt-1 text-sm text-slate-500">
              {asset ? asset.display_name : "Loading image…"}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Link
              className="inline-flex h-10 items-center rounded-md border border-slate-300 bg-white px-4 text-sm font-medium text-slate-800 shadow-sm hover:bg-slate-50"
              to={`/assets/${assetId}/analysis${segmentationId ? `?seg=${segmentationId}` : ""}`}
            >
              Analysis
            </Link>
            <Link
              className="inline-flex h-10 items-center rounded-md border border-slate-300 bg-white px-4 text-sm font-medium text-slate-800 shadow-sm hover:bg-slate-50"
              to={`/assets/${assetId}/viewer`}
            >
              Back to viewer
            </Link>
          </div>
        </header>

        {segmentationsLoading ? <PageState title="Loading…" /> : null}

        {!segmentationsLoading && (segmentations?.length ?? 0) === 0 ? (
          <PageState
            title="Nothing to improve yet"
            detail="This image has no objects. Run the model on it first, then mark an area as checked."
          />
        ) : null}

        {segmentations && segmentations.length > 1 ? (
          <Panel className="flex flex-wrap items-end gap-4 p-4">
            <div className="min-w-[280px] flex-1">
              <label
                className="block text-xs font-semibold uppercase tracking-wide text-slate-500"
                htmlFor="improve-segmentation"
              >
                Which objects
              </label>
              <select
                id="improve-segmentation"
                className="mt-1 h-10 w-full rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                value={segmentationId}
                onChange={(event) => handleSelectSegmentation(event.target.value)}
              >
                {segmentations.map((seg) => (
                  <option key={seg.id} value={seg.id}>
                    {segmentationDisplayName(seg)}
                  </option>
                ))}
              </select>
            </div>
          </Panel>
        ) : null}

        {cropsLoading && !crops ? (
          <Panel className="p-4">
            <p className="m-0 text-sm text-slate-600">
              Reading the areas you have checked…
            </p>
          </Panel>
        ) : null}

        {cropsError ? (
          <Panel className="border-red-200 bg-red-50 p-4">
            <p className="m-0 text-sm text-red-800">
              {extractApiErrorMessage(
                cropsError,
                "The areas you have checked could not be read."
              )}
            </p>
            <div className="mt-3">
              <Button
                size="sm"
                onClick={() => {
                  void refetchCrops();
                }}
              >
                Try again
              </Button>
            </div>
          </Panel>
        ) : null}

        {/* ---- The one button ------------------------------------------- */}
        {crops ? (
          <Panel className="p-4" data-testid="improve-primary">
            <h2 className="m-0 text-base font-semibold text-slate-950">
              Learn from my fixes
            </h2>
            <p className="m-0 mt-2 text-sm text-slate-700">
              {checkedAreasSentence(summary)}
            </p>
            <p className="m-0 mt-1 text-sm text-slate-500">
              {costSentence("threshold_only")}
            </p>

            {crops.warnings.length > 0 ? (
              <ul className="m-0 mt-3 list-disc space-y-1 rounded-md border border-amber-200 bg-amber-50 p-3 pl-8 text-sm text-amber-900">
                {crops.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            ) : null}

            {thresholdRefusal ? (
              // Refused here, before the request is sent, and refused again by
              // the server before anything is queued. A queue slot spent on a
              // run that cannot work is a failure the user watches happen.
              <p
                className="m-0 mt-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900"
                role="status"
                data-testid="threshold-refusal"
              >
                {thresholdRefusal}
              </p>
            ) : null}

            <div className="mt-3 flex flex-wrap items-center gap-3">
              <Button
                variant="primary"
                data-testid="learn-from-my-fixes"
                disabled={starting || running || thresholdRefusal !== null}
                onClick={() => {
                  void start("threshold_only");
                }}
              >
                {starting
                  ? "Starting…"
                  : succeeded
                    ? "Learn from my fixes again"
                    : "Learn from my fixes"}
              </Button>
              {running ? (
                <Badge tone="info">{outcome.status ?? "running"}</Badge>
              ) : null}
              {running && job && !job.cancel_requested ? (
                <Button size="sm" onClick={() => void handleCancel()} disabled={cancelling}>
                  {cancelling ? "Stopping…" : "Stop"}
                </Button>
              ) : null}
            </div>

            {running ? (
              <p className="m-0 mt-2 text-sm text-slate-600" role="status">
                {job?.message || "Queued."}
              </p>
            ) : null}

            {startError ? (
              <p
                className="m-0 mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800"
                role="alert"
              >
                {startError}
              </p>
            ) : null}
            {cancelError ? (
              <p
                className="m-0 mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800"
                role="alert"
              >
                {cancelError}
              </p>
            ) : null}
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
                  {outcome.cancelled ? "You stopped this." : "This did not finish."}
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
                <p className="m-0 mt-1 text-sm text-slate-700">
                  {PRESERVATION_SENTENCE}
                </p>
              </div>
            ) : null}
          </Panel>
        ) : null}

        {/* ---- What happened -------------------------------------------- */}
        {report && adapter ? (
          <Panel className="p-4" data-testid="improve-result">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="m-0 text-base font-semibold text-slate-950">
                {report.timing ?? "Done."}
              </h2>
              {applied ? <Badge tone="good">in use</Badge> : null}
            </div>
            <p className="m-0 mt-2 text-sm font-medium text-slate-900">
              {report.level}
            </p>
            <p className="m-0 mt-1 text-sm text-slate-700">{report.verdict}</p>
            {report.adjustment ? (
              <p className="m-0 mt-1 text-sm text-slate-700">{report.adjustment}</p>
            ) : null}
            <p className="m-0 mt-1 text-sm text-slate-500">{report.evidence}</p>
            <p
              className="m-0 mt-2 rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-900"
              data-testid="preservation-note"
            >
              {report.preservation}
            </p>

            <div className="mt-3 flex flex-wrap items-center gap-3">
              {!applied ? (
                <Button
                  variant="primary"
                  data-testid="use-include-level"
                  disabled={applying}
                  onClick={() => {
                    void handleApply();
                  }}
                >
                  {applying
                    ? "Setting…"
                    : `Use include level ${formatIncludeLevel(adapter.calibrated_threshold)}`}
                </Button>
              ) : null}
              <Button
                data-testid="about-the-numbers"
                aria-expanded={showNumbers}
                onClick={() => setShowNumbers((open) => !open)}
              >
                {showNumbers ? "Hide the numbers" : "About the numbers"}
              </Button>
              <a
                className="inline-flex h-10 items-center rounded-md border border-slate-300 bg-white px-4 text-sm font-medium text-slate-800 shadow-sm hover:bg-slate-50"
                href={labelingHref}
              >
                Back to the objects
              </a>
            </div>

            {applyError ? (
              <p
                className="m-0 mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800"
                role="alert"
              >
                {applyError}
              </p>
            ) : null}

            {applied ? (
              <div className="mt-3 rounded-md border border-slate-200 p-3">
                <p className="m-0 text-sm text-slate-700">
                  {worthRerunning
                    ? "The objects on screen were found at the old include level. Running the model again finds them at the new one."
                    : "This is the include level your objects were already found at, so running the model again would find the same objects."}
                </p>
                {worthRerunning ? (
                  <>
                    <p className="m-0 mt-1 text-sm text-slate-500">
                      {PRESERVATION_SENTENCE}
                    </p>
                    <div className="mt-2 flex flex-wrap items-center gap-3">
                      <Button
                        disabled={rerunJobId !== null}
                        data-testid="find-objects-again"
                        onClick={() => {
                          void handleRerun();
                        }}
                      >
                        {rerunJobId ? "Running…" : "Find the objects again"}
                      </Button>
                      {rerunJobId ? (
                        <span className="text-sm text-slate-600">
                          Running over the whole image. You can leave this
                          screen; it carries on.
                        </span>
                      ) : null}
                    </div>
                  </>
                ) : null}
                {rerunError ? (
                  <p
                    className="m-0 mt-2 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800"
                    role="alert"
                  >
                    {rerunError}
                  </p>
                ) : null}
              </div>
            ) : null}
          </Panel>
        ) : null}

        {/* ---- The statistics layer, one click away ---------------------- */}
        {showNumbers && adapter ? (
          <div data-testid="about-the-numbers-panel">
            <AboutTheNumbers
              adapter={adapter}
              baseSweep={baseSweep}
              provenance={provenance}
              provenanceLoading={provenanceLoading}
              provenanceError={
                provenanceError
                  ? extractApiErrorMessage(provenanceError, "no response")
                  : null
              }
            />
          </div>
        ) : null}

        {/* ---- The deeper rung, offered honestly ------------------------- */}
        {crops ? (
          <Panel className="p-4" data-testid="improve-deeper">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="m-0 text-sm font-semibold text-slate-900">
                Or train on my fixes
              </h3>
              {headRefusal ? <Badge tone="warning">not available here</Badge> : null}
            </div>
            <p className="m-0 mt-1 text-sm text-slate-600">
              Matching my cut-off is one number. This retrains the part of the
              model that draws the outlines, on the same checked areas.
            </p>
            <p className="m-0 mt-1 text-sm text-slate-500">{costSentence("head")}</p>
            {headRefusal ? (
              <p
                className="m-0 mt-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900"
                role="status"
                data-testid="head-refusal"
              >
                {headRefusal}
              </p>
            ) : null}
            <div className="mt-3">
              <Button
                data-testid="train-on-my-fixes"
                disabled={starting || running || headRefusal !== null}
                onClick={() => {
                  void start("head");
                }}
              >
                Train on my fixes
              </Button>
            </div>
          </Panel>
        ) : null}
      </div>
    </div>
  );
}
