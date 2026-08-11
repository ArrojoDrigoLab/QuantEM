/**
 * Fine-tune a model on your own annotations. Owner ruling R13, end to end.
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
 * * **applying is opt-in.** Success offers to run the new model on the images
 *   it was scoped to. It queues nothing until the user picks and clicks.
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
  applyFineTuneRun,
  fineTuneErrorMessage,
  getFineTuneRun,
  getFineTuneScope,
  getModelCatalogue,
  isFineTuneNameConflict,
  listFineTuneAdapters,
  previewFineTuneScope,
  startFineTuneRun,
} from "@/shared/api/finetune";
import { getSegmentationTypes } from "@/shared/api/assets";
import { RunProgressList } from "@/shared/progress/RunProgressList";
import { Button } from "@/shared/ui/design";
import { TRAINING_ANNOTATION_SOURCES } from "@/shared/constants/confirmedArea";
import { suggestBaseModel, toModelChoices } from "@/features/improve/modelChoices";
import { FineTuneSuccess } from "@/features/finetune/components/FineTuneSuccess";
import { HelpDisclosure } from "@/features/finetune/components/HelpDisclosure";
import { ScopePicker } from "@/features/finetune/components/ScopePicker";
import { fineTuneProgressRows } from "@/features/finetune/fineTuneProgressRows";
import {
  FINE_TUNE_MODE_OPTIONS,
  TRAINING_MODE_HELP,
  TRAINING_MODE_HELP_TITLE,
  modeChoiceFromDefault,
  modeChoicePayload,
  type FineTuneModeChoice,
} from "@/features/finetune/trainingModes";
import {
  buildScopeTree,
  emptySelection,
  filterScopeTree,
  isSelectionEmpty,
  selectionKey,
  selectionTotals,
  setImageSelected,
  toSelectionPayload,
  type ScopeSelection,
} from "@/features/finetune/scopeTree";
import { useFineTuneProgress } from "@/features/finetune/useFineTuneProgress";
import type { SegmentationType } from "@/shared/types/images";
import type {
  FineTuneAdapterSummary,
  FineTuneApplyResponse,
  FineTunePreviewResponse,
  FineTuneRunDetail,
  ModelCatalogue,
} from "@/shared/types/finetune";
import "./FineTuneDialog.css";

const PREVIEW_DEBOUNCE_MS = 250;

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
}: FineTuneDialogProps) {
  const titleId = useId();
  const nameId = useId();
  const overwriteId = useId();
  const searchId = useId();
  const baseModelId = useId();
  const organelleId = useId();
  const panelRef = useRef<HTMLDivElement | null>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);

  const [types, setTypes] = useState<SegmentationType[] | null>(null);
  const [chosenTypeId, setChosenTypeId] = useState<string>(
    segmentationType?.id ?? ""
  );
  const [catalogue, setCatalogue] = useState<ModelCatalogue | null>(null);
  const [adapters, setAdapters] = useState<FineTuneAdapterSummary[]>([]);
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

  const [preview, setPreview] = useState<FineTunePreviewResponse | null>(null);
  const [previewKey, setPreviewKey] = useState<string>("");
  const [previewPending, setPreviewPending] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);

  const [name, setName] = useState("");
  const [overwriteAdapterId, setOverwriteAdapterId] = useState("");
  const [baseModel, setBaseModel] = useState("");
  const [modeChoice, setModeChoice] = useState<FineTuneModeChoice>("use_all");
  const [modeTouched, setModeTouched] = useState(false);

  const [submitError, setSubmitError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [adapterId, setAdapterId] = useState<string | null>(null);
  const [runDetail, setRunDetail] = useState<FineTuneRunDetail | null>(null);
  const [applying, setApplying] = useState(false);
  const [applyError, setApplyError] = useState<string | null>(null);
  const [applyResult, setApplyResult] = useState<FineTuneApplyResponse | null>(null);

  const { progress, settled } = useFineTuneProgress(adapterId);

  // --- lifecycle ---------------------------------------------------------

  useEffect(() => {
    restoreFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      restoreFocusRef.current?.focus();
    };
  }, [onClose]);

  useEffect(() => {
    if (segmentationType) return;
    let cancelled = false;
    void getSegmentationTypes()
      .then((rows) => {
        if (!cancelled) setTypes(rows);
      })
      .catch(() => {
        if (!cancelled) setTypes([]);
      });
    return () => {
      cancelled = true;
    };
  }, [segmentationType]);

  useEffect(() => {
    let cancelled = false;
    void getModelCatalogue()
      .then((next) => {
        if (!cancelled) setCatalogue(next);
      })
      .catch(() => {
        // The picker still offers the fixed released ids, marked "install state
        // unknown", which is what `toModelChoices(null)` is for.
        if (!cancelled) setCatalogue(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

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
        setScopeGroups(buildScopeTree(response));
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
  }, [chosenTypeId]);

  useEffect(() => {
    if (!chosenTypeId) {
      setAdapters([]);
      return undefined;
    }
    let cancelled = false;
    void listFineTuneAdapters(chosenTypeId)
      .then((rows) => {
        if (!cancelled) setAdapters(rows);
      })
      .catch(() => {
        if (!cancelled) setAdapters([]);
      });
    return () => {
      cancelled = true;
    };
  }, [chosenTypeId]);

  const currentSelectionKey = chosenTypeId
    ? selectionKey(chosenTypeId, selection)
    : "";
  const selectionEmpty = isSelectionEmpty(selection);

  useEffect(() => {
    if (!chosenTypeId || selectionEmpty) {
      setPreview(null);
      setPreviewKey("");
      setPreviewPending(false);
      setPreviewError(null);
      return undefined;
    }
    let cancelled = false;
    setPreviewPending(true);
    // Debounced: ticking a group is one intent and several state updates, and
    // a request per tick would leave the count trailing the boxes.
    const timer = window.setTimeout(() => {
      void previewFineTuneScope(toSelectionPayload(chosenTypeId, selection))
        .then((response) => {
          if (cancelled) return;
          setPreview(response);
          setPreviewKey(currentSelectionKey);
          setPreviewError(null);
        })
        .catch((error: unknown) => {
          if (cancelled) return;
          setPreview(null);
          setPreviewKey(currentSelectionKey);
          setPreviewError(
            fineTuneErrorMessage(error, "Could not check this selection.")
          );
        })
        .finally(() => {
          if (!cancelled) setPreviewPending(false);
        });
    }, PREVIEW_DEBOUNCE_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [chosenTypeId, selection, selectionEmpty, currentSelectionKey]);

  // The preview on screen is only about the selection on screen. Without this
  // the count would keep showing the previous selection's total while the new
  // request was out, which is the one moment a user is watching it change.
  const livePreview = previewKey === currentSelectionKey ? preview : null;

  useEffect(() => {
    if (modeTouched || !livePreview) return;
    setModeChoice(modeChoiceFromDefault(livePreview.default_mode));
  }, [livePreview, modeTouched]);

  const modelChoices = useMemo(() => toModelChoices(catalogue), [catalogue]);
  const resolvedType = useMemo(
    () =>
      segmentationType ??
      (types ?? []).find((type) => type.id === chosenTypeId) ??
      null,
    [segmentationType, types, chosenTypeId]
  );

  useEffect(() => {
    if (baseModel) return;
    const suggestion = suggestBaseModel(resolvedType?.internal_name, modelChoices);
    const runnable = modelChoices.find(
      (choice) => choice.id === suggestion && choice.pack?.runnable !== false
    );
    setBaseModel(runnable?.id ?? suggestion ?? modelChoices[0]?.id ?? "");
  }, [baseModel, resolvedType, modelChoices]);

  useEffect(() => {
    if (!adapterId || !settled) return;
    if (progress?.status !== "SUCCESS") return;
    let cancelled = false;
    void getFineTuneRun(adapterId)
      .then((detail) => {
        if (!cancelled) setRunDetail(detail);
      })
      .catch(() => {
        // The run succeeded; only the extra detail is missing. The success
        // panel stands without it.
        if (!cancelled) setRunDetail(null);
      });
    return () => {
      cancelled = true;
    };
  }, [adapterId, settled, progress?.status]);

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

  const blockers = livePreview && !livePreview.eligible ? livePreview.blockers : [];
  const nameTaken =
    !overwriteAdapterId &&
    adapters.some(
      (adapter) => adapter.name.trim().toLowerCase() === name.trim().toLowerCase()
    );
  const canSubmit =
    Boolean(chosenTypeId) &&
    Boolean(name.trim()) &&
    Boolean(baseModel) &&
    !selectionEmpty &&
    Boolean(livePreview?.eligible) &&
    !previewPending &&
    !starting &&
    !nameTaken;

  const phase: "form" | "running" | "done" =
    adapterId === null ? "form" : settled ? "done" : "running";

  // --- actions -----------------------------------------------------------

  const pickOverwrite = (value: string) => {
    setOverwriteAdapterId(value);
    const chosen = adapters.find((adapter) => adapter.id === value);
    if (chosen) setName(chosen.name);
  };

  const submit = useCallback(async () => {
    if (!chosenTypeId) return;
    setStarting(true);
    setSubmitError(null);
    const { mode, cv_benchmark } = modeChoicePayload(modeChoice);
    const scope = toSelectionPayload(chosenTypeId, selection);
    try {
      const response = await startFineTuneRun({
        name: name.trim(),
        overwrite_adapter_id: overwriteAdapterId || null,
        segmentation_type: chosenTypeId,
        base_model: baseModel,
        asset_ids: scope.asset_ids,
        dataset_ids: scope.dataset_ids,
        mode,
        cv_benchmark,
      });
      setAdapterId(response.adapter_id);
    } catch (error) {
      setSubmitError(
        isFineTuneNameConflict(error)
          ? fineTuneErrorMessage(
              error,
              "A fine-tune of this organelle already has that name. Pick it from “Overwrite an existing fine-tune”, or choose another name."
            )
          : fineTuneErrorMessage(error, "Could not start the fine-tune.")
      );
    } finally {
      setStarting(false);
    }
  }, [chosenTypeId, modeChoice, selection, name, overwriteAdapterId, baseModel]);

  const apply = useCallback(
    (assetIds: string[]) => {
      if (!adapterId) return;
      setApplying(true);
      setApplyError(null);
      void applyFineTuneRun(adapterId, assetIds)
        .then(setApplyResult)
        .catch((error: unknown) => {
          setApplyError(
            fineTuneErrorMessage(error, "Could not queue those runs.")
          );
        })
        .finally(() => setApplying(false));
    },
    [adapterId]
  );

  // --- render ------------------------------------------------------------

  const organelleLabel = resolvedType?.short_name || resolvedType?.long_name || "";

  return (
    <div
      className="finetune-dialog-overlay"
      onClick={phase === "running" ? undefined : onClose}
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
            <p className="m-0 mt-0.5 text-xs text-slate-600">
              {organelleLabel
                ? `Train on your own ${organelleLabel.toLowerCase()} annotations.`
                : "Train on your own annotations, one organelle at a time."}
            </p>
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

        <div className="flex-1 overflow-y-auto px-5 py-4">
          {phase === "form" ? (
            <div className="flex flex-col gap-5">
              {!segmentationType ? (
                <div>
                  <label
                    htmlFor={organelleId}
                    className="block text-sm font-semibold text-slate-900"
                  >
                    Organelle
                  </label>
                  <p className="m-0 mb-1 text-xs text-slate-600">
                    One fine-tune trains one organelle.
                  </p>
                  <select
                    id={organelleId}
                    className="h-9 w-full rounded-md border border-slate-300 bg-white px-2 text-sm"
                    value={chosenTypeId}
                    onChange={(event) => {
                      setChosenTypeId(event.target.value);
                      setSelection(emptySelection());
                      setBaseModel("");
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
                      A fine-tune of this organelle is already called that. Pick it
                      below to replace it, or choose another name.
                    </p>
                  ) : null}
                </div>
                <div>
                  <label
                    htmlFor={overwriteId}
                    className="block text-sm font-semibold text-slate-900"
                  >
                    Overwrite an existing fine-tune
                  </label>
                  <select
                    id={overwriteId}
                    className="mt-1 h-9 w-full rounded-md border border-slate-300 bg-white px-2 text-sm"
                    value={overwriteAdapterId}
                    onChange={(event) => pickOverwrite(event.target.value)}
                  >
                    <option value="">Make a new one</option>
                    {adapters.map((adapter) => (
                      <option key={adapter.id} value={adapter.id}>
                        {adapter.name}
                      </option>
                    ))}
                  </select>
                  {overwriteAdapterId ? (
                    <p className="m-0 mt-1 text-xs text-slate-600">
                      The old weights stay in place until this run succeeds. If it
                      fails, nothing is lost.
                    </p>
                  ) : null}
                </div>
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
                  onChange={(event) => setBaseModel(event.target.value)}
                >
                  {modelChoices.map((choice) => (
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
                    Applies to
                  </label>
                  <span className="text-xs text-slate-600">
                    Everything picked has to be in one experiment.
                  </span>
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
                  <p className="m-0 mt-0.5 text-xs text-slate-600">
                    {TRAINING_ANNOTATION_SOURCES}
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
                  {FINE_TUNE_MODE_OPTIONS.map((option) => (
                    <label
                      key={option.value}
                      className="flex items-start gap-2 text-sm text-slate-800"
                    >
                      <input
                        type="radio"
                        name="finetune-mode"
                        className="mt-1 h-4 w-4"
                        value={option.value}
                        checked={modeChoice === option.value}
                        onChange={() => {
                          setModeChoice(option.value);
                          setModeTouched(true);
                        }}
                      />
                      <span>
                        <span className="font-medium">{option.label}</span>
                        <span className="block text-xs text-slate-600">
                          {option.summary}
                        </span>
                      </span>
                    </label>
                  ))}
                </div>
              </fieldset>

              {submitError ? (
                <p className="m-0 text-sm text-red-700" role="alert">
                  {submitError}
                </p>
              ) : null}
            </div>
          ) : null}

          {phase === "running" ? (
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

          {phase === "done" ? (
            progress?.status === "SUCCESS" ? (
              <FineTuneSuccess
                name={name}
                run={runDetail}
                scopedImages={livePreview?.per_image ?? preview?.per_image ?? []}
                applying={applying}
                applyError={applyError}
                applyResult={applyResult}
                onApply={apply}
                onClose={onClose}
              />
            ) : (
              <div className="flex flex-col gap-3" data-testid="finetune-failed">
                <p className="m-0 text-sm text-red-700" role="alert">
                  {progress?.error ||
                    "The fine-tune did not finish. Nothing on your images was changed."}
                </p>
                {overwriteAdapterId ? (
                  <p className="m-0 text-xs text-slate-600">
                    The fine-tune you were replacing is untouched and still usable.
                  </p>
                ) : null}
              </div>
            )
          ) : null}
        </div>

        {phase === "form" ? (
          <div className="flex items-center justify-end gap-2 border-t border-slate-200 px-5 py-3">
            <Button onClick={onClose}>Cancel</Button>
            <Button variant="primary" disabled={!canSubmit} onClick={() => void submit()}>
              {starting ? "Starting…" : "Fine-tune"}
            </Button>
          </div>
        ) : null}
        {phase === "running" ? (
          <div className="flex items-center justify-end gap-2 border-t border-slate-200 px-5 py-3">
            <Button onClick={onClose}>Close and keep running</Button>
          </div>
        ) : null}
        {phase === "done" && progress?.status !== "SUCCESS" ? (
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
