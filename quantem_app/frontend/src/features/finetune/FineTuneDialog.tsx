/**
 * Train on selected annotations, then optionally apply a saved model.
 *
 * One dialog, two ways in — a Fine-Tune button beside one organelle in the
 * labeling view, and one on the library header — and the same six decisions
 * either way: what to call it, what it applies to, how many annotations that
 * comes to, how to spend them, and then whether to run the result.
 *
 * Three things here are load-bearing and easy to lose in a refactor:
 *
 * * **the count is annotations, not tiles.** A ten-image dataset in which two
 *   images carry three annotations and a third carries one reads *7*. The tile
 *   count is a second, smaller line, because it is what the mode default is
 *   decided on and a reader who sees only one number will assume it is the one
 *   the sentence above it named.
 * * **the server decides eligibility and the default mode.** The
 *   same-experiment rule is enforced with a 400, and re-implementing it here
 *   would be a second copy of it free to drift; the submit button is disabled
 *   unless the preview says `eligible`, and the blockers are printed as sent.
 * * **applying is opt-in.** Success offers all images in the model's experiment,
 *   with its training images initially selected. Nothing is queued until asked.
 */

import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  fineTuneErrorMessage,
  getFineTuneRun,
  getFineTuneScope,
  getModelCatalogue,
  isFineTuneNameConflict,
  MODEL_NAME_RESERVED,
  listFineTuneAdapters,
  startFineTuneRun,
} from "@/shared/api/finetune";
import { getSegmentationTypes } from "@/shared/api/assets";
import { RunProgressList } from "@/shared/progress/RunProgressList";
import { Button } from "@/shared/ui/design";
import { suggestBaseModel, toModelChoices } from "@/features/improve/modelChoices";
import { ModelApplication } from "./ModelApplication";
import { useTrainingPreview } from "./useTrainingPreview";
import { useModalLifecycle } from "@/shared/hooks/useModalLifecycle";
import { HelpDisclosure } from "@/shared/ui/HelpDisclosure";
import { ScopePicker } from "@/features/finetune/components/ScopePicker";
import { fineTuneProgressRows } from "@/features/finetune/fineTuneProgressRows";
import { parseTrainingInputs } from "@/features/finetune/trainingOptions";
import {
  FINE_TUNE_MODE_OPTIONS,
  TRAINING_MODE_HELP,
  TRAINING_MODE_HELP_TITLE,
  minimumAnnotationsForMode,
  modeChoiceIsAvailable,
  modeChoiceFromDefault,
  modeChoicePayload,
  type FineTuneModeChoice,
} from "@/features/finetune/trainingModes";
import {
  buildScopeTree,
  emptySelection,
  filterScopeTree,
  isSelectionEmpty,
  selectionTotals,
  setImageSelected,
  toSelectionPayload,
  type ScopeSelection,
} from "@/features/finetune/scopeTree";
import { useFineTuneProgress } from "@/features/finetune/useFineTuneProgress";
import type { SegmentationType } from "@/shared/types/images";
import type {
  FineTuneAdapterSummary,
  FineTuneAppliedImageEvent,
  ModelCatalogue,
} from "@/shared/types/finetune";
import "./FineTuneDialog.css";

export interface FineTuneDialogProps {
  open: boolean;
  onClose: () => void;
  /**
   * The organelle, when the dialog was opened from a labeling view. Null on the
   * library route, where the first thing the dialog asks is which one.
   */
  segmentationType?: SegmentationType | null;
  /** Images to start with ticked — the one the labeling view was open on. */
  initialAssetIds?: string[];
  initialExperimentId?: string;
  initialParentAdapterId?: string;
  initialSavedAdapterId?: string;
  /** The released pack selected in the labeling view's model picker. */
  initialBaseModel?: string | null;
  /** Labeling-view origin retained on the resulting adapter for compatibility. */
  segmentationId?: string | null;
  /** Notify an open labeling view when one applied image finishes. */
  onAppliedImageCompleted?: (event: FineTuneAppliedImageEvent) => void;
  /** Notify it as soon as the image run is queued, so it can keep watching. */
  onAppliedImageQueued?: (event: FineTuneAppliedImageEvent) => void;
}

export function FineTuneDialog(props: FineTuneDialogProps) {
  // Unmounted while closed, so every reopen starts from a clean form rather
  // than from the last run's success panel.
  if (!props.open) return null;
  return <FineTuneDialogBody {...props} />;
}

function FineTuneDialogBody({
  onClose,
  segmentationType = null,
  initialAssetIds,
  initialExperimentId,
  initialParentAdapterId,
  initialSavedAdapterId,
  initialBaseModel = null,
  segmentationId = null,
  onAppliedImageCompleted,
  onAppliedImageQueued,
}: FineTuneDialogProps) {
  const titleId = useId();
  const nameId = useId();
  const overwriteId = useId();
  const savedRunId = useId();
  const searchId = useId();
  const baseModelId = useId();
  const organelleId = useId();
  const panelRef = useRef<HTMLDivElement | null>(null);
  useModalLifecycle(panelRef, onClose);

  const [types, setTypes] = useState<SegmentationType[] | null>(null);
  const [chosenTypeId, setChosenTypeId] = useState<string>(
    segmentationType?.id ?? ""
  );
  const [catalogue, setCatalogue] = useState<ModelCatalogue | null>(null);
  const [loadErrors, setLoadErrors] = useState<Record<string, string>>({});
  const [loadAttempt, setLoadAttempt] = useState(0);
  const [adapters, setAdapters] = useState<FineTuneAdapterSummary[]>([]);
  const [dialogMode, setDialogMode] = useState<"train" | "apply">(initialSavedAdapterId ? "apply" : "train");
  const [scopeGroups, setScopeGroups] = useState(() => buildScopeTree(null));
  const [scopeError, setScopeError] = useState<string | null>(null);
  const [scopeLoading, setScopeLoading] = useState(false);

  const [query, setQuery] = useState("");
  const [selection, setSelection] = useState<ScopeSelection>(() => {
    let next = emptySelection();
    for (const assetId of initialAssetIds ?? []) {
      next = setImageSelected(next, assetId, true);
    }
    return next;
  });

  const [name, setName] = useState("");
  const [parentAdapterId, setParentAdapterId] = useState(initialParentAdapterId ?? "");
  const [includePrevious, setIncludePrevious] = useState(true);
  const [savedAdapterId, setSavedAdapterId] = useState(initialSavedAdapterId ?? "");
  const [baseModel, setBaseModel] = useState(initialBaseModel ?? "");
  const [modeChoice, setModeChoice] = useState<FineTuneModeChoice>("use_all");
  const [schedule, setSchedule] = useState<"quick" | "extended">("quick");
  const [steps, setSteps] = useState("");
  const [lr, setLr] = useState("0.0001");
  const [seed, setSeed] = useState("0");
  const [validation, setValidation] = useState(false);
  const [evalInterval, setEvalInterval] = useState("100");
  const [patience, setPatience] = useState("5");
  const trainingInputs = useMemo(() => parseTrainingInputs({
    schedule, steps, lr, seed, validation, evalInterval, patience,
  }), [schedule, steps, lr, seed, validation, evalInterval, patience]);
  // A ref, rather than state, so a preview already in flight cannot overwrite
  // a radio choice made before its response arrives.
  const modeTouchedRef = useRef(false);

  const [submitError, setSubmitError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [trainingRunId, setTrainingRunId] = useState<string | null>(null);
  const { progress, settled } = useFineTuneProgress(trainingRunId);

  // --- lifecycle ---------------------------------------------------------

  useEffect(() => {
    if (segmentationType) return;
    let cancelled = false;
    void getSegmentationTypes()
      .then((rows) => {
        if (!cancelled) setTypes(rows);
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setTypes([]);
          setLoadErrors((current) => ({ ...current, types: fineTuneErrorMessage(error, "Organelles could not be loaded.") }));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [segmentationType, loadAttempt]);

  useEffect(() => {
    let cancelled = false;
    void getModelCatalogue()
      .then((next) => {
        if (!cancelled) setCatalogue(next);
      })
      .catch((error: unknown) => {
        // Fine-tuning must never offer a pack whose downloaded state cannot be
        // verified, so a failed catalogue request leaves the picker empty.
        if (!cancelled) {
          setCatalogue(null);
          setLoadErrors((current) => ({ ...current, catalogue: fineTuneErrorMessage(error, "Model availability could not be checked.") }));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [loadAttempt]);

  useEffect(() => {
    if (!chosenTypeId) {
      setScopeGroups(buildScopeTree(null));
      return undefined;
    }
    let cancelled = false;
    setScopeLoading(true);
    setScopeError(null);
    void getFineTuneScope(chosenTypeId)
      .then((response) => {
        if (cancelled) return;
        const groups = buildScopeTree(response);
        setScopeGroups(initialExperimentId ? groups.filter((group) => group.key === initialExperimentId) : groups);
        if (!initialAssetIds?.length && initialExperimentId) {
          const group = groups.find((entry) => entry.key === initialExperimentId);
          let next = emptySelection();
          for (const image of [...(group?.images ?? []), ...(group?.datasets.flatMap((dataset) => dataset.images) ?? [])]) {
            next = setImageSelected(next, image.id, true);
          }
          setSelection(next);
        }
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setScopeGroups(buildScopeTree(null));
        setScopeError(
          fineTuneErrorMessage(error, "Could not read the image library.")
        );
      })
      .finally(() => {
        if (!cancelled) setScopeLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [chosenTypeId, initialExperimentId, initialAssetIds, loadAttempt]);

  useEffect(() => {
    if (!chosenTypeId) {
      setAdapters([]);
      return undefined;
    }
    let cancelled = false;
    setAdapters([]);
    void listFineTuneAdapters(chosenTypeId)
      .then((rows) => {
        if (!cancelled) {
          setAdapters(initialExperimentId ? rows.filter((row) => typeof row.experiment === "object" && row.experiment?.id === initialExperimentId) : rows);
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setAdapters([]);
          setLoadErrors((current) => ({ ...current, adapters: fineTuneErrorMessage(error, "Saved fine-tunes could not be loaded.") }));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [chosenTypeId, initialExperimentId, loadAttempt]);

  useEffect(() => {
    const parent = adapters.find((adapter) => adapter.id === parentAdapterId);
    if (!parent) return;
    setBaseModel(parent.base_model);
    setName((current) => current || `${parent.name} · next version`);
  }, [adapters, parentAdapterId]);

  const selectionEmpty = isSelectionEmpty(selection);
  const { livePreview, previewPending, previewError } = useTrainingPreview({
    typeId: chosenTypeId, selection, baseModel, parentId: parentAdapterId,
    includePrevious, enabled: dialogMode === "train",
  });
  useEffect(() => {
    if (livePreview && !modeTouchedRef.current) setModeChoice(modeChoiceFromDefault(livePreview.default_mode));
  }, [livePreview]);

  const modelChoices = useMemo(
    () =>
      toModelChoices(catalogue).filter(
        (choice) => choice.pack?.installed === true
      ),
    [catalogue]
  );
  const resolvedType = useMemo(
    () =>
      segmentationType ??
      (types ?? []).find((type) => type.id === chosenTypeId) ??
      null,
    [segmentationType, types, chosenTypeId]
  );
  const compatibleModelChoices = useMemo(() => {
    const suggested = suggestBaseModel(
      resolvedType?.internal_name,
      modelChoices
    );
    const organelle = modelChoices.find((choice) => choice.id === suggested)?.organelle;
    return organelle
      ? modelChoices.filter((choice) => choice.organelle === organelle)
      : [];
  }, [resolvedType, modelChoices]);

  useEffect(() => {
    if (compatibleModelChoices.some((choice) => choice.id === baseModel)) return;
    const labelingChoice = compatibleModelChoices.find(
      (choice) =>
        choice.id === initialBaseModel && choice.pack?.runnable !== false
    );
    const suggestion = suggestBaseModel(
      resolvedType?.internal_name,
      compatibleModelChoices
    );
    const runnable = compatibleModelChoices.find(
      (choice) => choice.id === suggestion && choice.pack?.runnable !== false
    );
    setBaseModel(
      labelingChoice?.id ??
        runnable?.id ??
        suggestion ??
        compatibleModelChoices[0]?.id ??
        ""
    );
  }, [baseModel, resolvedType, compatibleModelChoices, initialBaseModel]);

  useEffect(() => {
    if (!trainingRunId || !settled) return;
    if (progress?.status !== "SUCCESS") return;
    let cancelled = false;
    void getFineTuneRun(trainingRunId)
      .then((detail) => {
        if (cancelled) return;
        setName((current) => detail.name || current);
        setAdapters((current) => {
          if (current.some((adapter) => adapter.id === trainingRunId)) return current;
          return [
            {
              id: trainingRunId,
              name: detail.name || "Saved fine-tune",
              base_model: detail.base_model,
              status: "SUCCESS",
              created_at: detail.created_at,
              experiment: detail.experiment ?? null,
              asset_count: detail.asset_ids?.length ?? 0,
              asset_ids: detail.asset_ids ?? [],
            },
            ...current,
          ];
        });
      })
      .catch(() => {
        // The Apply panel loads details independently and offers a retry.
      })
      .finally(() => {
        if (cancelled) return;
        setSavedAdapterId(trainingRunId);
        setDialogMode("apply");
      });
    return () => {
      cancelled = true;
    };
  }, [trainingRunId, settled, progress?.status]);

  // --- derived -----------------------------------------------------------

  const filteredGroups = useMemo(
    () => filterScopeTree(scopeGroups, query),
    [scopeGroups, query]
  );
  const localTotals = useMemo(
    () => selectionTotals(scopeGroups, selection),
    [scopeGroups, selection]
  );
  const annotationCount = livePreview
    ? livePreview.annotation_count
    : localTotals.annotationCount;
  const imageCount = livePreview ? livePreview.asset_count : localTotals.imageCount;

  useEffect(() => {
    if (modeChoiceIsAvailable(modeChoice, annotationCount)) return;
    modeTouchedRef.current = false;
    setModeChoice("use_all");
  }, [annotationCount, modeChoice]);

  const blockers = livePreview && !livePreview.eligible ? livePreview.blockers : [];
  const nameTaken =
    adapters.some(
      (adapter) => adapter.name.trim().toLowerCase() === name.trim().toLowerCase()
    );
  const canSubmit =
    Boolean(chosenTypeId) &&
    Boolean(name.trim()) &&
    Boolean(baseModel) &&
    !selectionEmpty &&
    Boolean(livePreview?.eligible) &&
    modeChoiceIsAvailable(modeChoice, annotationCount) &&
    !previewPending &&
    !starting &&
    !trainingInputs.error &&
    !nameTaken;

  const phase: "form" | "running" | "done" =
    trainingRunId === null ? "form" : settled ? "done" : "running";
  const hasSavedFineTune =
    adapters.some((adapter) => adapter.status === "SUCCESS") ||
    progress?.status === "SUCCESS" || Boolean(savedAdapterId);

  // --- actions -----------------------------------------------------------

  const pickParent = (value: string) => {
    setParentAdapterId(value);
    const chosen = adapters.find((adapter) => adapter.id === value);
    if (chosen) {
      setName(`${chosen.name} · next version`);
      setBaseModel(chosen.base_model);
    }
  };

  const selectSavedFineTune = (value: string) => setSavedAdapterId(value);

  const showApplyMode = () => {
    setDialogMode("apply");
    const chosen =
      adapters.find((adapter) => adapter.id === savedAdapterId && adapter.status === "SUCCESS") ??
      adapters.find((adapter) => adapter.status === "SUCCESS");
    if (chosen) selectSavedFineTune(chosen.id);
  };

  const showTrainMode = () => {
    setDialogMode("train");
    setTrainingRunId(null);
    setSubmitError(null);
  };

  const submit = useCallback(async () => {
    if (!chosenTypeId || !trainingInputs.options) return;
    setStarting(true);
    setSubmitError(null);
    const { mode, cv_benchmark } = modeChoicePayload(modeChoice);
    const scope = toSelectionPayload(chosenTypeId, selection);
    try {
      const response = await startFineTuneRun({
        name: name.trim(),
        overwrite_adapter_id: null,
        parent_adapter_id: parentAdapterId || undefined,
        include_previous_annotations: includePrevious,
        segmentation_type: chosenTypeId,
        base_model: baseModel,
        asset_ids: scope.asset_ids,
        dataset_ids: scope.dataset_ids,
        mode,
        cv_benchmark,
        segmentation_id: segmentationId || undefined,
        ...trainingInputs.options,
      });
      setTrainingRunId(response.adapter_id);
    } catch (error) {
      setSubmitError(
        isFineTuneNameConflict(error)
          ? MODEL_NAME_RESERVED
          : fineTuneErrorMessage(error, "Could not start the fine-tune.")
      );
    } finally {
      setStarting(false);
    }
  }, [
    chosenTypeId,
    modeChoice,
    selection,
    name,
    parentAdapterId,
    includePrevious,
    baseModel,
    segmentationId, trainingInputs,
  ]);

  // --- render ------------------------------------------------------------

  return (
    <div
      className="finetune-dialog-overlay"
      onClick={dialogMode === "train" && phase === "running" ? undefined : onClose}
    >
      <div
        ref={panelRef}
        className="finetune-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        data-testid="finetune-dialog"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 border-b border-slate-200 px-5 py-3">
          <div>
            <h2 id={titleId} className="m-0 text-lg font-semibold text-slate-950">
              Fine-tune a model
            </h2>
          </div>
          {/* Not just "Close": the footer has a Close too, and two controls
              with one accessible name is a screen reader reading the same word
              twice for two different things. */}
          <Button
            size="sm"
            variant="ghost"
            onClick={onClose}
            aria-label="Close the fine-tune dialog"
          >
            ✕
          </Button>
        </div>

        {hasSavedFineTune ? (
          <div
            className="flex gap-1 border-b border-slate-200 bg-slate-50 px-5 py-2"
            role="tablist"
            aria-label="Fine-tune action"
          >
            <button
              type="button"
              role="tab"
              aria-selected={dialogMode === "train"}
              className={`rounded-md px-4 py-1.5 text-sm font-semibold ${
                dialogMode === "train"
                  ? "bg-white text-slate-950 shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
              onClick={showTrainMode}
            >
              Train
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={dialogMode === "apply"}
              className={`rounded-md px-4 py-1.5 text-sm font-semibold ${
                dialogMode === "apply"
                  ? "bg-white text-slate-950 shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
              onClick={showApplyMode}
            >
              Apply
            </button>
          </div>
        ) : null}

        <div className="flex-1 overflow-y-auto px-5 py-4">
          {Object.keys(loadErrors).length > 0 && <div role="alert" className="mb-4 rounded border border-red-300 p-3 text-red-800">
            {Object.entries(loadErrors).map(([key, message]) => <p key={key}>{message}</p>)}
            <Button onClick={() => { setLoadErrors({}); setLoadAttempt((attempt) => attempt + 1); }}>Retry loading</Button>
          </div>}
          {dialogMode === "train" && phase === "form" ? (
            <div className="flex flex-col gap-5">
              {!segmentationType ? (
                <div>
                  <label
                    htmlFor={organelleId}
                    className="block text-sm font-semibold text-slate-900"
                  >
                    Organelle
                  </label>
                  <select
                    id={organelleId}
                    className="h-9 w-full rounded-md border border-slate-300 bg-white px-2 text-sm"
                    value={chosenTypeId}
                    onChange={(event) => {
                      setChosenTypeId(event.target.value);
                      setSelection(emptySelection());
                      setBaseModel("");
                      setParentAdapterId("");
                      setName("");
                    }}
                  >
                    <option value="">Choose an organelle…</option>
                    {(types ?? []).map((type) => (
                      <option key={type.id} value={type.id}>
                        {type.short_name || type.long_name}
                      </option>
                    ))}
                  </select>
                </div>
              ) : null}

              <div className="grid gap-3 sm:grid-cols-2">
                <div>
                  <label
                    htmlFor={nameId}
                    className="block text-sm font-semibold text-slate-900"
                  >
                    Name
                  </label>
                  <input
                    id={nameId}
                    className="mt-1 h-9 w-full rounded-md border border-slate-300 px-2 text-sm"
                    value={name}
                    placeholder="Fasted liver mitochondria"
                    onChange={(event) => {
                      setName(event.target.value);
                      setSubmitError(null);
                    }}
                  />
                  {nameTaken ? (
                    <p className="m-0 mt-1 text-xs text-amber-800" role="status">
                      {MODEL_NAME_RESERVED}
                    </p>
                  ) : null}
                </div>
                {adapters.length > 0 ? (
                  <div>
                    <label
                      htmlFor={overwriteId}
                      className="block text-sm font-semibold text-slate-900"
                    >
                      Continue a saved model
                    </label>
                    <select
                      id={overwriteId}
                      className="mt-1 h-9 w-full rounded-md border border-slate-300 bg-white px-2 text-sm"
                      value={parentAdapterId}
                      onChange={(event) => pickParent(event.target.value)}
                    >
                      <option value="">Start from a pretrained model</option>
                      {adapters.filter((adapter) => adapter.status === "SUCCESS").map((adapter) => (
                        <option key={adapter.id} value={adapter.id}>
                          {adapter.name}
                        </option>
                      ))}
                    </select>
                    {parentAdapterId ? (
                      <div className="mt-2 flex items-center gap-2 text-xs text-slate-600">
                        <label><input type="checkbox" checked={includePrevious}
                          onChange={(event) => setIncludePrevious(event.target.checked)} /> Include the parent's annotated images</label>
                        <HelpDisclosure label="continued training" title="Continue a saved model">
                          Saves a new version. Images used by the parent model cannot provide an independent test of improvement.
                        </HelpDisclosure>
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>

              <div>
                <label
                  htmlFor={baseModelId}
                  className="block text-sm font-semibold text-slate-900"
                >
                  Starting from
                </label>
                <select
                  id={baseModelId}
                  className="mt-1 h-9 w-full rounded-md border border-slate-300 bg-white px-2 text-sm"
                  value={baseModel}
                  disabled={Boolean(parentAdapterId)}
                  onChange={(event) => setBaseModel(event.target.value)}
                >
                  {compatibleModelChoices.length === 0 ? (
                    <option value="" disabled>
                      No downloaded models for this organelle
                    </option>
                  ) : null}
                  {compatibleModelChoices.map((choice) => (
                    <option
                      key={choice.id}
                      value={choice.id}
                      disabled={choice.pack?.runnable === false}
                    >
                      {choice.title}
                      {choice.pack?.runnable === false
                        ? " — cannot run on this machine"
                        : ""}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <div className="flex items-baseline justify-between gap-2">
                  <label
                    htmlFor={searchId}
                    className="text-sm font-semibold text-slate-900"
                  >
                    Training images
                  </label>
                </div>
                <input
                  id={searchId}
                  type="search"
                  className="mt-1 h-9 w-full rounded-md border border-slate-300 px-2 text-sm"
                  placeholder="Search datasets and images"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                />
                <div className="mt-2 max-h-56 overflow-y-auto rounded-md border border-slate-200">
                  {scopeLoading ? (
                    <p className="m-0 px-3 py-4 text-sm text-slate-500">
                      Reading the library…
                    </p>
                  ) : scopeError ? (
                    <p className="m-0 px-3 py-4 text-sm text-red-700" role="alert">
                      {scopeError}
                    </p>
                  ) : (
                    <ScopePicker
                      groups={filteredGroups}
                      selection={selection}
                      onChange={setSelection}
                      emptyMessage={
                        chosenTypeId
                          ? query.trim()
                            ? "Nothing matches that search."
                            : "No images in the library yet."
                          : "Choose an organelle first."
                      }
                    />
                  )}
                </div>

                <div
                  className="mt-2 rounded-md bg-slate-50 px-3 py-2"
                  data-testid="finetune-count"
                  role="status"
                >
                  <p className="m-0 text-sm font-semibold text-slate-900">
                    {annotationCount}{" "}
                    {annotationCount === 1 ? "annotation" : "annotations"} across{" "}
                    {imageCount} {imageCount === 1 ? "image" : "images"}
                  </p>
                  {livePreview ? (
                    <p className="m-0 mt-0.5 text-xs text-slate-600">
                      {livePreview.confirmed_areas} confirmed{" "}
                      {livePreview.confirmed_areas === 1 ? "area" : "areas"} and{" "}
                      {livePreview.done_rois} reviewed{" "}
                      {livePreview.done_rois === 1 ? "ROI" : "ROIs"}, cut into{" "}
                      {livePreview.tile_count}{" "}
                      {livePreview.tile_count === 1 ? "tile" : "tiles"}
                      {livePreview.experiment
                        ? ` in ${livePreview.experiment.name}`
                        : ""}
                      .
                    </p>
                  ) : previewPending ? (
                    <p className="m-0 mt-0.5 text-xs text-slate-500">Checking…</p>
                  ) : null}
                </div>

                {livePreview?.coverage && livePreview.coverage.areas_with_tiles < annotationCount ? (
                  <p className="text-xs text-amber-800">
                    {annotationCount - livePreview.coverage.areas_with_tiles} selected areas contain no usable training pixels.
                  </p>
                ) : null}
                {livePreview?.warnings?.map((warning) => <p key={warning} className="text-xs text-amber-800">{warning}</p>)}
                {previewError ? (
                  <p className="m-0 mt-1 text-sm text-red-700" role="alert">
                    {previewError}
                  </p>
                ) : null}
                {blockers.length > 0 ? (
                  <ul
                    className="m-0 mt-2 list-disc rounded-md border border-amber-200 bg-amber-50 py-2 pl-8 pr-3 text-xs text-amber-900"
                    data-testid="finetune-blockers"
                  >
                    {blockers.map((blocker) => (
                      <li key={blocker}>{blocker}</li>
                    ))}
                  </ul>
                ) : null}
              </div>

              <fieldset className="m-0 border-0 p-0">
                <legend className="flex items-center gap-2 p-0 text-sm font-semibold text-slate-900">
                  Training data
                  <HelpDisclosure
                    label="how the training data is used"
                    title={TRAINING_MODE_HELP_TITLE}
                  >
                    {TRAINING_MODE_HELP.map((paragraph) => (
                      <span key={paragraph} className="mt-1 block">
                        {paragraph}
                      </span>
                    ))}
                  </HelpDisclosure>
                </legend>
                <div className="mt-1 flex flex-col gap-2">
                  {FINE_TUNE_MODE_OPTIONS.map((option) => {
                    const minimum = minimumAnnotationsForMode(option.value);
                    const available = modeChoiceIsAvailable(
                      option.value,
                      annotationCount
                    );
                    return (
                      <label
                        key={option.value}
                        className={`flex items-start gap-2 text-sm ${
                          available ? "text-slate-800" : "text-slate-400"
                        }`}
                      >
                        <input
                          type="radio"
                          name="finetune-mode"
                          className="mt-1 h-4 w-4"
                          value={option.value}
                          checked={modeChoice === option.value}
                          disabled={!available}
                          onChange={() => {
                            modeTouchedRef.current = true;
                            setModeChoice(option.value);
                          }}
                        />
                        <span>
                          <span className="font-medium">{option.label}</span>
                          {!available ? (
                            <span className="block text-xs text-slate-500">
                              Requires at least {minimum} annotations.
                            </span>
                          ) : null}
                        </span>
                      </label>
                    );
                  })}
                </div>
              </fieldset>

              <details className="rounded border border-slate-200 p-3">
                <summary className="cursor-pointer text-sm font-semibold">Training options</summary>
                <div className="mt-2 grid grid-cols-2 gap-3 text-sm">
                  <label>Duration
                    <select className="block w-full rounded border p-1" value={schedule} onChange={(e) => setSchedule(e.target.value as "quick" | "extended")}>
                      <option value="quick">Quick (300–600 steps, or one complete pass)</option>
                      <option value="extended">Extended (20 steps per tile, up to 20,000)</option>
                    </select>
                  </label>
                  <label>Custom steps (blank for automatic)
                    <input type="number" min="1" max="20000" className="block w-full rounded border p-1" value={steps} onChange={(e) => setSteps(e.target.value)} />
                  </label>
                  <label>Learning rate
                    <input type="number" min="0.0000001" max="0.01" step="any" className="block w-full rounded border p-1" value={lr} onChange={(e) => setLr(e.target.value)} />
                  </label>
                  <label>Random seed
                    <input type="number" min="0" max="4294967295" step="1" className="block w-full rounded border p-1" value={seed} onChange={(e) => setSeed(e.target.value)} />
                  </label>
                  <div className="col-span-2 flex items-center gap-2">
                    <label><input type="checkbox" checked={validation} onChange={(e) => setValidation(e.target.checked)} /> Select the training duration using validation</label>
                    <HelpDisclosure label="validation" title="Training duration">
                      Uses 20% of training images to choose the duration, then trains on all of them. Requires at least two images and can take twice as long. Held-out test images remain separate.
                    </HelpDisclosure>
                  </div>
                  <label>Evaluate every (steps)
                    <input type="number" min="1" max="20000" className="block w-full rounded border p-1" value={evalInterval} onChange={(e) => setEvalInterval(e.target.value)} />
                  </label>
                  <label>Stop after unimproved evaluations (0 disables)
                    <input type="number" min="0" max="1000" disabled={!validation} className="block w-full rounded border p-1" value={patience} onChange={(e) => setPatience(e.target.value)} />
                  </label>
                </div>
              </details>
              {trainingInputs.error && <p role="alert" className="text-sm text-red-700">{trainingInputs.error}</p>}
              {submitError ? (
                <p className="m-0 text-sm text-red-700" role="alert">
                  {submitError}
                </p>
              ) : null}
            </div>
          ) : null}

          {dialogMode === "train" && phase === "running" ? (
            <div className="flex flex-col gap-3" data-testid="finetune-running">
              <RunProgressList
                className="finetune-progress"
                rows={fineTuneProgressRows(progress, name || "This fine-tune")}
                data-testid="finetune-progress"
              />
              {progress?.message ? (
                <p className="m-0 text-xs text-slate-600">{progress.message}</p>
              ) : (
                <p className="m-0 text-xs text-slate-600">
                  Getting ready. You can leave this open; the run carries on either
                  way and is listed in Tasks &amp; Queues.
                </p>
              )}
            </div>
          ) : null}

          {dialogMode === "train" && phase === "done" && progress?.status !== "SUCCESS" ? (
              <div className="flex flex-col gap-3" data-testid="finetune-failed">
                <p className="m-0 text-sm text-red-700" role="alert">
                  {progress?.error ||
                    "The fine-tune did not finish. Nothing on your images was changed."}
                </p>
                {parentAdapterId ? (
                  <p className="m-0 text-xs text-slate-600">
                    The parent model remains available.
                  </p>
                ) : null}
              </div>
          ) : null}

          {dialogMode === "apply" ? (
            <div className="flex flex-col gap-4" data-testid="finetune-apply-mode">
              <div>
                <label
                  htmlFor={savedRunId}
                  className="block text-sm font-semibold text-slate-900"
                >
                  Fine-tuned model
                </label>
                <p className="m-0 mt-1 text-xs text-slate-600">
                  Its cross-validation results and latest image runs stay with it.
                </p>
                <select
                  id={savedRunId}
                  className="mt-2 h-9 w-full rounded-md border border-slate-300 bg-white px-2 text-sm"
                  value={savedAdapterId}
                  onChange={(event) => selectSavedFineTune(event.target.value)}
                >
                  <option value="">Choose a saved fine-tune</option>
                  {adapters.map((adapter) => (
                    <option
                      key={adapter.id}
                      value={adapter.id}
                      disabled={adapter.status !== "SUCCESS"}
                    >
                      {adapter.name}
                      {adapter.status !== "SUCCESS"
                        ? ` (${adapter.status.toLowerCase()})`
                        : ""}
                    </option>
                  ))}
                </select>
              </div>

              {savedAdapterId && <ModelApplication key={savedAdapterId} adapterId={savedAdapterId}
                typeId={chosenTypeId} typeName={resolvedType?.long_name || resolvedType?.short_name || ""}
                onClose={onClose} onAppliedImageQueued={onAppliedImageQueued}
                onAppliedImageCompleted={onAppliedImageCompleted} />}
            </div>
          ) : null}
        </div>

        {dialogMode === "train" && phase === "form" ? (
          <div className="flex items-center justify-end gap-2 border-t border-slate-200 px-5 py-3">
            <Button onClick={onClose}>Cancel</Button>
            <Button variant="primary" disabled={!canSubmit} onClick={() => void submit()}>
              {starting ? "Starting…" : "Fine-tune"}
            </Button>
          </div>
        ) : null}
        {dialogMode === "train" && phase === "running" ? (
          <div className="flex items-center justify-end gap-2 border-t border-slate-200 px-5 py-3">
            <Button onClick={onClose}>Close and keep running</Button>
          </div>
        ) : null}
        {dialogMode === "train" && phase === "done" && progress?.status !== "SUCCESS" ? (
          // The success panel carries its own two buttons. A failure had none,
          // leaving the header cross as the only way out of a dialog that has
          // just told you something went wrong.
          <div className="flex items-center justify-end gap-2 border-t border-slate-200 px-5 py-3">
            <Button onClick={onClose}>Close</Button>
          </div>
        ) : null}
      </div>
    </div>
  );
}
