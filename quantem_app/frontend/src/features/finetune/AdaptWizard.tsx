/**
 * Guided fine-tuning, in six steps.
 *
 * The wizard exists because the honest version of "make the model better on my
 * data" has preconditions that are invisible otherwise: you need a completed
 * ROI for background to mean anything, you need annotations on a second image
 * before a held-out score means generalisation, and the cheap rung (calibrate
 * the threshold) fixes most of what people reach for training to fix. Each step
 * is a place to say one of those things at the moment it applies.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, Navigate, useParams, useSearchParams } from "react-router-dom";
import { getAsset, getAssetSegmentations } from "@/shared/api/assets";
import {
  applyAdapter,
  getAdaptCrops,
  getAdapter,
  getModelCatalogue,
  startAdaptation,
} from "@/shared/api/finetune";
import { ApiRequestError } from "@/shared/api/core/http";
import { cancelJob, deleteJob } from "@/shared/api/jobs";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { useJobProgress } from "@/shared/hooks/useJobProgress";
import { Button, PageState, Panel } from "@/shared/ui/design";
import { cx } from "@/shared/ui/cx";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import type {
  Adapter,
  AdapterJobResult,
  AdapterSweep,
  AdaptMode,
  ModelCatalogue,
} from "@/shared/types/finetune";
import {
  adaptedForBase,
  suggestBaseModel,
  toModelChoices,
} from "@/features/finetune/models";
import {
  forgetAdaptRun,
  loadAdaptRun,
  rememberAdaptRun,
} from "@/features/finetune/adaptRunStorage";
import { runnabilityForPackId } from "@/features/models/runnable";
import { resolveAdaptRunOutcome } from "@/features/finetune/adaptRunOutcome";
import { StepApply } from "@/features/finetune/components/StepApply";
import { StepBaseModel } from "@/features/finetune/components/StepBaseModel";
import { StepCrops } from "@/features/finetune/components/StepCrops";
import { StepMode } from "@/features/finetune/components/StepMode";
import {
  adaptBudgetError,
  type AdaptBudget,
} from "@/features/finetune/adaptBudget";
import { StepResults } from "@/features/finetune/components/StepResults";
import { StepRun } from "@/features/finetune/components/StepRun";
import { GroundTruthProvenancePanel } from "@/features/finetune/components/GroundTruthProvenance";
import { useGroundTruthProvenance } from "@/features/finetune/useGroundTruthProvenance";

const STEP_LABELS = [
  "Base model",
  "Your annotations",
  "What to fit",
  "Run",
  "Results",
  "Apply",
];

/**
 * How often to re-read the adapter row while a run is in flight.
 *
 * Deliberately slower than the job poll (1 s): this is the backstop that says
 * "finished", not the thing that draws the progress bar.
 */
const ADAPTER_POLL_MS = 5000;

const DEFAULT_BUDGET: AdaptBudget = {
  steps: 300,
  lr: 0.0001,
  seed: 0,
  name: "",
};

function sweepFromJobResult(result: unknown, key: "sweep" | "base_sweep"): AdapterSweep | null {
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

export function AdaptWizard() {
  const { assetId } = useParams<{ assetId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedSegmentationId = searchParams.get("seg");

  const [segmentationId, setSegmentationId] = useState("");
  const [step, setStep] = useState(1);
  /**
   * The furthest step reached for this segmentation, which is what gates the
   * step buttons -- not the step currently shown.
   *
   * Reachability used to be `number <= step`, so stepping *back* to "What to
   * fit" while a head training was running greyed out Run, Results and Apply.
   * The run was still going and the only route back to it was another reload.
   * Going backwards to re-read a step must never revoke access to the ones
   * already passed.
   */
  const [furthestStep, setFurthestStep] = useState(1);
  const [baseModel, setBaseModel] = useState("");
  /**
   * True once the user has chosen a base model themselves.
   *
   * The preselect below has to be allowed to change its mind. It used to bail
   * on `if (baseModel) return`, and the catalogue answers before the
   * segmentation list does, so the first guess was made with no organelle to
   * guess from -- it fell through to `choices[0]`, i.e. QuantEM — Mitochondria,
   * and then froze. Arriving at the wizard from an ER segmentation's "Adapt
   * model" link put you on the mitochondria pack, and the run's auto-name
   * followed it: an ER adapter saved as "mito @ liver_HFD2_ROI15_5nm".
   */
  const [baseModelPinned, setBaseModelPinned] = useState(false);
  const [mode, setMode] = useState<AdaptMode>("threshold_only");
  const [budget, setBudget] = useState<AdaptBudget>(DEFAULT_BUDGET);
  /** True once the user has typed a name; the auto-name stops tracking then. */
  const [nameEdited, setNameEdited] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [adapterId, setAdapterId] = useState<string | null>(null);
  const [restoredSegmentationId, setRestoredSegmentationId] = useState<string | null>(
    null
  );
  /**
   * The adapter whose settings have already been copied into this form.
   *
   * Guards the restore below against re-running on every poll, which would
   * fight anyone editing the form while a run is in flight.
   */
  const [adapterSyncedId, setAdapterSyncedId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [applying, setApplying] = useState(false);
  const [applyError, setApplyError] = useState<string | null>(null);

  const { data: asset } = useApiQuery(
    () => (assetId ? getAsset(assetId) : Promise.resolve(null)),
    [assetId]
  );
  const { data: segmentations, loading: segmentationsLoading } = useApiQuery(
    () => (assetId ? getAssetSegmentations(assetId) : Promise.resolve([])),
    [assetId]
  );

  /**
   * Adopt a different segmentation: reset everything scoped to the previous
   * one. Shared by the picker and the `?seg=` deep link so the two behave
   * identically — see the settle effect below.
   */
  const adoptSegmentation = useCallback((nextId: string) => {
    setSegmentationId(nextId);
    setStep(1);
    setFurthestStep(1);
    setJobId(null);
    setAdapterId(null);
    // The next segmentation's adapter, if it has one, has to be allowed to
    // write its settings into the form in turn.
    setAdapterSyncedId(null);
    // Let the preselect and the auto-name follow the new organelle again.
    setBaseModelPinned(false);
    setNameEdited(false);
    // The restore effect keys off this, so clearing it lets the newly
    // selected segmentation reattach to its own run.
    setRestoredSegmentationId(null);
  }, []);

  // Settle on a segmentation whenever the list or the URL changes. The URL is
  // the deep link and it wins whenever it names a segmentation that exists —
  // including on SPA navigation. This used to keep the first value for the
  // life of the component (`current ? current : wanted`), so `?seg=` was
  // honoured only on a full page load: navigating here in-app with a
  // different `?seg=`, or using back/forward across two wizard URLs, changed
  // the address bar and nothing else.
  useEffect(() => {
    if (!segmentations || segmentations.length === 0) return;
    const requested = requestedSegmentationId
      ? segmentations.find((seg) => seg.id === requestedSegmentationId)?.id ?? null
      : null;
    const next = requested ?? (segmentationId || segmentations[0].id);
    if (next === segmentationId) return;
    if (segmentationId) {
      // A mounted wizard being pointed at a different segmentation resets
      // exactly as it does through the picker; anything less leaves the form,
      // the step ladder and the run panel on the previous one.
      adoptSegmentation(next);
      return;
    }
    // First settle after mount: plain selection, so the reload-restore effect
    // can still reattach an in-flight run.
    setSegmentationId(next);
  }, [adoptSegmentation, segmentations, requestedSegmentationId, segmentationId]);

  // The catalogue is allowed to be missing; the wizard degrades rather than
  // dying, and says which parts it no longer knows.
  const { data: catalogue, error: catalogueError } = useApiQuery<ModelCatalogue | null>(
    () => getModelCatalogue(),
    []
  );

  const {
    data: crops,
    loading: cropsLoading,
    error: cropsError,
    refetch: refetchCrops,
  } = useApiQuery(
    () => (segmentationId ? getAdaptCrops(segmentationId) : Promise.resolve(null)),
    [segmentationId]
  );

  // Where the annotations came from: model candidates the user confirmed vs
  // regions drawn by hand. Derived from the crop list so it covers every
  // segmentation the run actually trains on, not just the selected one.
  const { provenance, provenanceLoading, provenanceError } =
    useGroundTruthProvenance(crops ?? null);

  const {
    data: adapter,
    error: adapterError,
    refetch: refetchAdapter,
  } = useApiQuery<Adapter | null>(
    () => (adapterId ? getAdapter(adapterId) : Promise.resolve(null)),
    [adapterId]
  );

  // A remembered pointer can outlive the adapter it names -- a reset data
  // directory, a row removed by hand. Only a 404 clears it: dropping the run on
  // any error would throw away a live training because the server hiccuped.
  useEffect(() => {
    if (!adapterId || !segmentationId) return;
    if (!(adapterError instanceof ApiRequestError) || adapterError.status !== 404) {
      return;
    }
    forgetAdaptRun(segmentationId);
    setAdapterId(null);
    setJobId(null);
  }, [adapterError, adapterId, segmentationId]);

  const { job, refresh: refreshJob } = useJobProgress(jobId);
  const jobStatus = job?.status;

  useEffect(() => {
    if (!jobStatus) return;
    if (jobStatus === "SUCCESS" || jobStatus === "FAILED" || jobStatus === "CANCELLED") {
      void refetchAdapter();
    }
    if (jobStatus === "SUCCESS") {
      setStep((current) => (current < 5 ? 5 : current));
    }
  }, [jobStatus, refetchAdapter]);

  /**
   * A run that has settled needs no job poll to describe it.
   *
   * The wizard used to sit on `RUNNING` with `step 300/300 loss 0.084 ETA ~0s`
   * forever after the job had reached SUCCESS, because the only thing that ever
   * moved the wizard off the Run step was a *transition* observed by the poll.
   * Reading the adapter's own terminal status covers every way the poll can
   * miss it: a reload that reattached after the job finished, a dropped
   * request, or a job row the queue has already trimmed.
   */
  const adapterStatus = adapter?.status;
  useEffect(() => {
    if (adapterStatus !== "SUCCESS" && adapterStatus !== "FAILED") return;
    if (jobId) setJobId(null);
    if (adapterStatus === "SUCCESS") {
      setStep((current) => (current < 5 ? 5 : current));
    }
  }, [adapterStatus, jobId]);

  // A second, slower clock, on the adapter rather than the job. The job poll is
  // the fast path for progress and stops itself the moment the job settles --
  // but it was also the *only* thing that ever moved the wizard off the Run
  // step, so any way it can miss the transition (a dropped request, a tab the
  // browser throttled while a head trained, a job row the queue has trimmed,
  // or a reload that reattached after the job was already gone) left the screen
  // reading RUNNING on a run that had finished. The adapter row is the
  // authority on whether the run is over, so ask it directly.
  useEffect(() => {
    if (!adapterId) return undefined;
    if (adapterStatus === "SUCCESS" || adapterStatus === "FAILED") return undefined;
    const timer = window.setInterval(() => {
      void refetchAdapter();
    }, ADAPTER_POLL_MS);
    return () => window.clearInterval(timer);
  }, [adapterId, adapterStatus, refetchAdapter]);

  // Reconnect to whatever this segmentation had in flight. Without this a
  // reload during a head training -- "minutes to tens of minutes" by the
  // wizard's own estimate -- lost the run entirely.
  useEffect(() => {
    if (!segmentationId || restoredSegmentationId === segmentationId) return;
    setRestoredSegmentationId(segmentationId);
    const stored = loadAdaptRun(segmentationId);
    if (!stored) return;
    setAdapterId(stored.adapterId);
    setJobId(stored.jobId);
    // Steps are gated on `number <= step`, so this is also what makes Results
    // and Apply reachable again.
    setStep((current) => (current < 4 ? 4 : current));
  }, [segmentationId, restoredSegmentationId]);

  useEffect(() => {
    if (!segmentationId || !adapterId) return;
    rememberAdaptRun(segmentationId, { adapterId, jobId });
  }, [segmentationId, adapterId, jobId]);

  /**
   * A run the wizard did not start in this session describes itself.
   *
   * Reattaching after a reload restored the *ids* but left every form field on
   * its defaults, so step 3 reported "Calibrate the threshold" while a head
   * training was actually running -- with an enabled "Go to run" beside it that
   * would have started the other kind of run. The adapter row is the authority
   * on what was asked for, so read the mode, the pack and the budget off it
   * rather than off `DEFAULT_BUDGET`.
   *
   * Pinning the base model and the name is the point, not a side effect: it
   * stops the preselect and the auto-name from immediately overwriting what the
   * run actually used with a fresh guess.
   */
  useEffect(() => {
    if (!adapter || adapterSyncedId === adapter.id) return;
    setAdapterSyncedId(adapter.id);
    setMode(adapter.mode);
    if (adapter.base_model) {
      setBaseModel(adapter.base_model);
      setBaseModelPinned(true);
    }
    if (adapter.name) setNameEdited(true);
    setBudget((current) => ({
      ...current,
      steps: adapter.steps > 0 ? adapter.steps : current.steps,
      name: adapter.name || current.name,
    }));
  }, [adapter, adapterSyncedId]);

  useEffect(() => {
    setFurthestStep((current) => (step > current ? step : current));
  }, [step]);

  /**
   * Every adaptation this segmentation already has, from the server.
   *
   * The catalogue lists successful adapters with their `segmentation_id`, so
   * this survives a cleared browser store, another machine, or a run started
   * before the wizard learned to remember anything. It is also the only route
   * back to Results and Apply for a run finished in a previous session.
   */
  const previousAdapters = useMemo(
    () =>
      (catalogue?.adapted ?? [])
        .filter((entry) => entry.segmentation_id === segmentationId)
        .sort((left, right) => right.created_at.localeCompare(left.created_at)),
    [catalogue, segmentationId]
  );

  const openAdapter = useCallback((catalogueId: string) => {
    // `GET /api/adapters/<id>/` takes the bare uuid; the catalogue prefixes it.
    setAdapterId(catalogueId.replace(/^adapted:/, ""));
    setJobId(null);
    setStep(5);
  }, []);

  const choices = useMemo(() => toModelChoices(catalogue ?? null), [catalogue]);

  // Whether the selected base model can actually be loaded here. Consulted by
  // step 4 before it will start a head run, and by the step gating below.
  const baseModelRunnability = useMemo(
    () => runnabilityForPackId(catalogue ?? null, baseModel),
    [catalogue, baseModel]
  );

  const selectedSegmentation = useMemo(
    () => segmentations?.find((seg) => seg.id === segmentationId) ?? null,
    [segmentations, segmentationId]
  );

  // Preselect the pack that matches this segmentation's organelle -- and keep
  // preselecting until the user picks one, because the segmentation this is
  // derived from arrives after the catalogue and can change from the dropdown.
  useEffect(() => {
    if (baseModelPinned || choices.length === 0) return;
    // No organelle to suggest from yet: leave the field alone rather than
    // guessing `choices[0]` and having that guess stick.
    if (!selectedSegmentation) return;
    const next =
      suggestBaseModel(
        selectedSegmentation.segmentation_type.internal_name,
        choices
      ) ?? choices[0].id;
    setBaseModel((current) => (current === next ? current : next));
  }, [baseModelPinned, choices, selectedSegmentation]);

  // The run's name follows the base model until the user types one. It used to
  // be written once and never revisited, so it recorded whatever pack happened
  // to be selected at that instant rather than the one actually trained.
  useEffect(() => {
    if (nameEdited || !baseModel || !asset) return;
    const organelle = baseModel.split(":")[1] ?? baseModel;
    const autoName = `${organelle} @ ${asset.display_name}`;
    setBudget((current) =>
      current.name === autoName ? current : { ...current, name: autoName }
    );
  }, [baseModel, asset, nameEdited]);

  const handleBudgetChange = useCallback(
    (next: AdaptBudget) => {
      if (next.name !== budget.name) setNameEdited(true);
      setBudget(next);
    },
    [budget.name]
  );

  const handleBaseModelChange = useCallback((packId: string) => {
    setBaseModelPinned(true);
    setBaseModel(packId);
  }, []);

  /**
   * The single state of whatever run this wizard is attached to.
   *
   * `concluded` is the one that matters here: a cancelled or failed run has no
   * results and never will, so everything the wizard locked down "because a run
   * exists" has to open back up. Leaving step 3 read-only after a cancel is how
   * a cancelled head run on a machine that cannot load the pack became
   * unrecoverable a second time -- the mode could not be changed and the run
   * could not be restarted.
   */
  const runOutcome = resolveAdaptRunOutcome(job, adapter ?? null);
  const runInProgressOrDone = adapter !== null && !runOutcome.concluded;

  // Head training is only offered where torch exists AND the chosen pack can
  // actually be built; drop back rather than letting the run fail seconds in.
  // Threshold calibration never loads the model, so it survives both.
  useEffect(() => {
    // Only the *pending* choice gets coerced. While a run is live or finished,
    // `mode` is a report of what it was started as, and rewriting that to
    // threshold_only would make step 3 describe a different run from the one on
    // Results -- the same lie this screen was fixed to stop telling. A run that
    // concluded with nothing is not a report of anything, so it does not hold
    // the form hostage.
    if (runInProgressOrDone) return;
    if (mode !== "head") return;
    const torchMissing = crops ? !crops.modes.includes("head") : false;
    if (torchMissing || baseModelRunnability.state === "blocked") {
      setMode("threshold_only");
    }
  }, [runInProgressOrDone, crops, mode, baseModelRunnability.state]);

  const handleStart = useCallback(async () => {
    if (!segmentationId || !baseModel) return;
    // The last gate before the request. `min={1}` on the budget fields stops
    // nothing, and the server reads them as `int(steps or 300)` -- a typed zero
    // is substituted, not refused, and the adapter row then reports a budget
    // nobody chose. Refusing with the reason on screen beats a run whose
    // recorded settings are not the ones that were asked for.
    const budgetProblem = adaptBudgetError(mode, budget);
    if (budgetProblem) {
      setStartError(budgetProblem);
      return;
    }
    setStarting(true);
    setStartError(null);
    try {
      const response = await startAdaptation(segmentationId, {
        base_model: baseModel,
        mode,
        steps: budget.steps,
        lr: budget.lr,
        seed: budget.seed,
        name: budget.name,
      });
      setAdapterId(response.adapter_id);
      setJobId(response.job_id);
    } catch (err) {
      setStartError(
        extractApiErrorMessage(err, "The adaptation could not be started.")
      );
    } finally {
      setStarting(false);
    }
  }, [segmentationId, baseModel, mode, budget]);

  /**
   * Throw away a run that concluded without an adapter, and start a fresh one.
   *
   * Cancelling used to be a dead end: the wizard held the ids of a run that
   * would never finish, step 4 rendered a progress bar for it, and steps 5 and
   * 6 stayed shut behind `status === "SUCCESS"`. Forgetting the stored pointer
   * is the part that matters — without it a reload reattaches to the same dead
   * run — and starting immediately is what the backend's own message tells the
   * user to do. Settings are unchanged, so this really is the same request
   * again; step 3 is still reachable if they want to change one first.
   */
  /**
   * Stop the training run, from the step that started it.
   *
   * The wizard could already *recover* from a cancellation and could not cause
   * one: the only Cancel in the application is in the Library's queue sidebar,
   * which this screen has no route to. That is backwards for the one screen
   * that tells you the work takes "minutes to tens of minutes".
   *
   * Two endpoints because a queued job and a running one leave by different
   * doors -- `POST /cancel/` 409s on anything that is not RUNNING, and DELETE
   * is a queued job's only exit. Both reconcile the `Adapter` behind them, so
   * `resolveAdaptRunOutcome` sees a cancellation either way and step 4 offers
   * "Start again" rather than sitting on a progress bar for ever.
   */
  const handleCancelRun = useCallback(async () => {
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
      void refetchAdapter();
    } catch (err) {
      setCancelError(
        extractApiErrorMessage(err, "The run could not be cancelled.")
      );
    } finally {
      setCancelling(false);
    }
  }, [jobId, jobStatus, refreshJob, refetchAdapter]);

  const handleStartAgain = useCallback(() => {
    if (segmentationId) forgetAdaptRun(segmentationId);
    setAdapterId(null);
    setJobId(null);
    setStartError(null);
    // Results and Apply were never reached by this run; leaving them lit would
    // offer a page about an adapter that no longer exists.
    setFurthestStep((current) => (current > 4 ? 4 : current));
    void handleStart();
  }, [handleStart, segmentationId]);

  const handleApply = useCallback(async () => {
    if (!adapterId) return;
    setApplying(true);
    setApplyError(null);
    try {
      await applyAdapter(adapterId);
      await refetchAdapter();
    } catch (err) {
      setApplyError(
        extractApiErrorMessage(err, "This adapter could not be applied.")
      );
    } finally {
      setApplying(false);
    }
  }, [adapterId, refetchAdapter]);

  const handleSelectSegmentation = useCallback(
    (nextId: string) => {
      adoptSegmentation(nextId);
      const next = new URLSearchParams(searchParams);
      next.set("seg", nextId);
      setSearchParams(next, { replace: true });
    },
    [adoptSegmentation, searchParams, setSearchParams]
  );

  if (!assetId) {
    return <Navigate to="/" replace />;
  }

  const baseSweep = sweepFromJobResult(job?.result_json, "base_sweep");
  // Only a *pending* budget can be wrong. While a run is live or finished these
  // fields are a report of what it was started with, and refusing to move on
  // because of a number the server already recorded would be a wall with
  // nothing behind it.
  const budgetError = runInProgressOrDone
    ? null
    : adaptBudgetError(mode, budget);
  const canAdvance =
    step === 1
      ? Boolean(baseModel)
      : step === 2
        ? Boolean(crops?.ready)
        : step === 3
          ? Boolean(baseModel && crops?.ready)
          : step === 4
            ? adapter?.status === "SUCCESS"
            : step === 5
              ? adapter?.status === "SUCCESS"
              : false;

  const segmentationHref = selectedSegmentation
    ? `#/assets/${assetId}/labeling/${encodeURIComponent(
        selectedSegmentation.segmentation_type.long_name
      )}`
    : `#/assets/${assetId}/viewer`;

  return (
    <div className="min-h-screen px-5 py-5 text-slate-900 lg:px-8">
      <div className="mx-auto flex max-w-[1100px] flex-col gap-5">
        <header className="flex flex-wrap items-center justify-between gap-4 rounded-lg border border-slate-200 bg-white px-4 py-3 shadow-sm">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-cyan-700">
              QuantEM
            </p>
            <h1 className="m-0 text-2xl font-semibold tracking-normal text-slate-950">
              Adapt a model
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

        {segmentationsLoading ? <PageState title="Loading segmentations…" /> : null}

        {!segmentationsLoading && (segmentations?.length ?? 0) === 0 ? (
          <PageState
            title="Nothing to adapt"
            detail="This image has no segmentations. Create one, annotate a region and mark it complete first."
          />
        ) : null}

        {segmentations && segmentations.length > 0 ? (
          <Panel className="flex flex-wrap items-end gap-4 p-4">
            <div className="min-w-[280px] flex-1">
              <label
                className="block text-xs font-semibold uppercase tracking-wide text-slate-500"
                htmlFor="adapt-segmentation"
              >
                Segmentation
              </label>
              <select
                id="adapt-segmentation"
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
            <p className="m-0 max-w-[520px] text-xs text-slate-500">
              Annotated regions are gathered across every image that has this
              same organelle segmented, which is what makes an image-disjoint
              split possible at all.
            </p>
          </Panel>
        ) : null}

        {/* Everything this segmentation has already been adapted with, read
            from the server rather than from memory. This is the resume route:
            a run finished in a previous session used to be unreachable from
            here, and the models screen listed it read-only, so the only way to
            see a held-out score or apply an adapter was to train it again. */}
        {previousAdapters.length > 0 ? (
          <Panel className="p-4">
            <h2 className="m-0 text-base font-semibold text-slate-950">
              Adaptations already run on this segmentation
            </h2>
            <ul className="m-0 mt-3 flex list-none flex-col gap-2 p-0">
              {previousAdapters.map((entry) => {
                const isOpen = adapterId !== null && entry.id.endsWith(adapterId);
                return (
                  <li
                    key={entry.id}
                    className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-slate-200 px-3 py-2"
                  >
                    <div className="min-w-0">
                      <p className="m-0 truncate text-sm font-medium text-slate-900">
                        {entry.name || entry.id}
                      </p>
                      <p className="m-0 text-xs text-slate-500">
                        from {entry.base}
                        {entry.calibrated_threshold !== null
                          ? ` · threshold ${entry.calibrated_threshold.toFixed(2)}`
                          : ""}
                        {entry.applied_at ? " · applied" : ""}
                      </p>
                    </div>
                    <Button
                      onClick={() => openAdapter(entry.id)}
                      disabled={isOpen}
                    >
                      {isOpen ? "Open" : "Open results"}
                    </Button>
                  </li>
                );
              })}
            </ul>
          </Panel>
        ) : null}

        <ol className="m-0 flex list-none flex-wrap gap-2 p-0">
          {STEP_LABELS.map((label, index) => {
            const number = index + 1;
            // A finished adapter makes Results and Apply reachable outright,
            // however you arrived: reopening one from the list above, or coming
            // back after a reload. Walking the wizard again to reach a page
            // that only reports on a run that already happened is busywork.
            const reachable =
              number <= Math.max(step, furthestStep) ||
              (adapter?.status === "SUCCESS" && number >= 5);
            return (
              <li key={label}>
                <button
                  type="button"
                  disabled={!reachable}
                  onClick={() => setStep(number)}
                  className={cx(
                    "rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                    number === step
                      ? "border-cyan-500 bg-cyan-50 text-cyan-900"
                      : reachable
                        ? "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                        : "cursor-not-allowed border-slate-100 bg-slate-50 text-slate-400"
                  )}
                >
                  {number}. {label}
                </button>
              </li>
            );
          })}
        </ol>

        {step === 1 ? (
          <StepBaseModel
            choices={choices}
            catalogue={catalogue ?? null}
            catalogueError={
              catalogueError
                ? extractApiErrorMessage(catalogueError, "no response")
                : null
            }
            value={baseModel}
            onChange={handleBaseModelChange}
            adapted={adaptedForBase(catalogue ?? null, baseModel)}
          />
        ) : null}

        {step === 2 ? (
          <StepCrops
            crops={crops ?? null}
            loading={cropsLoading}
            error={
              cropsError
                ? extractApiErrorMessage(
                    cropsError,
                    "Your annotations could not be read."
                  )
                : null
            }
            onRefresh={() => {
              void refetchCrops();
            }}
          />
        ) : null}

        {/* Also on the annotations step, where it is still actionable: this is
            the last point at which the user can go and draw an independent
            region before the score exists. */}
        {step === 2 && crops?.ready ? (
          <GroundTruthProvenancePanel
            provenance={provenance}
            loading={provenanceLoading}
            error={
              provenanceError
                ? extractApiErrorMessage(provenanceError, "no response")
                : null
            }
          />
        ) : null}

        {step === 3 ? (
          <StepMode
            crops={crops ?? null}
            mode={mode}
            onModeChange={setMode}
            budget={budget}
            onBudgetChange={handleBudgetChange}
            baseModelRunnability={baseModelRunnability}
            baseModel={baseModel}
            existingRun={
              // A run that concluded with nothing is not something to report
              // on, and treating it as one leaves the mode read-only with no
              // run to read it against.
              adapter && runInProgressOrDone
                ? { status: adapter.status, mode: adapter.mode }
                : null
            }
          />
        ) : null}

        {step === 4 ? (
          <StepRun
            job={job}
            adapter={adapter ?? null}
            startError={startError}
            onStart={() => {
              void handleStart();
            }}
            onStartAgain={handleStartAgain}
            onCancel={() => {
              void handleCancelRun();
            }}
            cancelling={cancelling}
            cancelError={cancelError}
            starting={starting}
            canStart={Boolean(baseModel && crops?.ready)}
            baseModelRunnability={baseModelRunnability}
            baseModel={baseModel}
            mode={mode}
            budgetError={budgetError}
          />
        ) : null}

        {step === 5 && adapter ? (
          <StepResults
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
        ) : null}
        {step === 5 && !adapter ? (
          <Panel className="p-4">
            <p className="m-0 text-sm text-slate-600">
              No adaptation has been run in this session yet.
            </p>
          </Panel>
        ) : null}

        {step === 6 && adapter ? (
          <StepApply
            adapter={adapter}
            onApply={() => {
              void handleApply();
            }}
            applying={applying}
            applyError={applyError}
            segmentationHref={segmentationHref}
          />
        ) : null}

        <div className="flex items-center justify-between gap-3">
          <Button onClick={() => setStep((current) => Math.max(1, current - 1))} disabled={step === 1}>
            Back
          </Button>
          <Button
            variant="primary"
            onClick={() => setStep((current) => Math.min(STEP_LABELS.length, current + 1))}
            disabled={!canAdvance || step === STEP_LABELS.length}
          >
            {step === 3 ? "Go to run" : "Next"}
          </Button>
        </div>
      </div>
    </div>
  );
}
