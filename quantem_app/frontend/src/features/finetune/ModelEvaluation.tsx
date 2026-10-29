import { useModelApplicationChoice } from "@/features/experiments/modelApplicationChoice";
import { preferredPackId } from "@/features/models/defaultModel";
import { useModelCatalogue } from "@/features/models/useModelCatalogue";
import { Link } from "react-router-dom";
import { BulkModelPanel } from "@/features/experiments/BulkModelPanel";
import { useCallback, useEffect, useMemo, useState, useRef } from "react";
import { apiRequest } from "@/shared/api/core/http";
import { getFineTuneScope } from "@/shared/api/finetune";
import type { SegmentationType } from "@/shared/types/images";
import type { FineTuneAdapterSummary, FineTuneScopeExperiment } from "@/shared/types/finetune";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { usePollingResource } from "@/shared/hooks/usePollingResource";
import { useWorkspaceState } from "@/shared/hooks/useWorkspaceState";
import { Button } from "@/shared/ui/design";
import { extractApiErrorMessage } from "@/utils/apiErrors";

type Roi = { id: string; name: string; image_id: string; training_overlap: boolean };
type Score = { name: string; dice: number | null; iou: number | null; training_overlap: boolean };
type EvaluationSettings = { experiment_id: string; segmentation_type: string; asset_ids: string[]; base_model: string; adapter_id: string | null; threshold: number; rois: Roi[] };
type Evaluation = { settings?: EvaluationSettings; id: string; status: string; message: string; created_at: string; results?: { per_roi: Score[]; notes: string[]; mean_dice: number | null } | null };

export function ModelEvaluation({ types, initialTypeId, initialAssetIds, initialExperimentId, initialBaseModel, initialAdapterId, adapters }: {
  initialBaseModel?: string | null; initialAdapterId?: string;
  types: SegmentationType[]; initialTypeId: string; initialAssetIds?: string[]; initialExperimentId?: string; adapters: FineTuneAdapterSummary[];
}) {
  const memoryKey = `evaluation.${JSON.stringify([initialExperimentId, initialTypeId, initialAssetIds, initialBaseModel, initialAdapterId])}`;
  const updateApplication = useModelApplicationChoice((state) => state.update);
  const [error, setError] = useState("");
  const [view, setView] = useWorkspaceState<"visual" | "scores">(`${memoryKey}.view`, "visual");
  const availableAdapters = adapters;
  const refresh = useCallback(() => {}, []);
  const supportedTypes = types.filter((entry) => ["mito", "er", "nucleus", "ld"].some((key) => entry.internal_name === `quantem_internal_${key}`));
  const initialApplication = useModelApplicationChoice.getState().choices[initialExperimentId ?? ""];
  const defaultTypeId = supportedTypes.find((entry) => entry.internal_name === `quantem_internal_${initialApplication?.organelles[0]}`)?.id ?? supportedTypes[0]?.id ?? "";
  const [typeId, setTypeId] = useWorkspaceState(`${memoryKey}.typeId`, initialTypeId || defaultTypeId);
  const [experimentId, setExperimentId] = useWorkspaceState(`${memoryKey}.experimentId`, initialExperimentId ?? "");

  const { catalogue } = useModelCatalogue();
  const [model, setModel] = useWorkspaceState(`${memoryKey}.model`, initialAdapterId || initialBaseModel?.split(":")[0] || initialApplication?.adapter || initialApplication?.family || "auto");
  const [rois, setRois] = useState<Roi[]>([]);

  const [savedThreshold, setThreshold] = useWorkspaceState<number | null>(`${memoryKey}.${typeId}.${experimentId}.${model}.threshold`, null);
  const threshold = savedThreshold ?? 0.5;
  const [previewKey, setPreviewKey] = useState("");
  const [evaluationId, setEvaluationId] = useWorkspaceState<string | null>(`${memoryKey}.${experimentId}.run`, null);
  const [submittedEvaluation, setSubmittedEvaluation] = useState<Evaluation | null>(null);
  const [submittedHistory, setSubmittedHistory] = useState<Evaluation[]>([]);
  const [busy, setBusy] = useState(false);
  const type = types.find((entry) => entry.id === typeId);
  useEffect(() => { if (!typeId && defaultTypeId) setTypeId(defaultTypeId); }, [defaultTypeId, typeId, setTypeId]);
  const { data: scopeResponse, error: scopeError, refetch: refetchScope } = useApiQuery(
    async () => ({ typeId, experiments: typeId ? (await getFineTuneScope(typeId)).experiments : [] as FineTuneScopeExperiment[] }), [typeId]);
  const scopeReady = scopeResponse?.typeId === typeId;
  const scope = useMemo(() => scopeReady ? scopeResponse.experiments : [], [scopeReady, scopeResponse]);
  useEffect(() => {
    if (!scopeReady) return;
    setExperimentId((id) => scope.some((entry) => entry.id === id) ? id : scope[0]?.id ?? "");
  }, [scope, scopeReady, setExperimentId]);
  const { data: historyResponse, error: historyError, refetch: refetchHistory } = useApiQuery(
    async () => ({ experimentId, rows: experimentId ? await apiRequest<Evaluation[]>(`/api/finetune/evaluations/?experiment=${experimentId}`) : [] }), [experimentId]);
  const history = [...new Map([
    ...(submittedHistory.filter((entry) => entry.settings?.experiment_id === experimentId)),
    ...(historyResponse?.experimentId === experimentId ? historyResponse.rows : []),
  ].map((entry) => [entry.id, entry])).values()];
  const images = useMemo(() => {
    const experiment = scope.find((entry) => entry.id === experimentId);
    return [...new Map([...(experiment?.ungrouped_images ?? []), ...(experiment?.datasets.flatMap((dataset) => dataset.images) ?? [])].map((image) => [image.id, image])).values()];
  }, [scope, experimentId]);
  const assetIds = images.filter((image) => !initialAssetIds?.length || (initialExperimentId && experimentId !== initialExperimentId) || initialAssetIds.includes(image.id)).map((image) => image.id);
  const organelle = type?.internal_name.replace("quantem_internal_", "") ?? "";
  const saved = availableAdapters.filter((adapter) => adapter.base_model.endsWith(`:${organelle}`) && (typeof adapter.experiment === "string" ? adapter.experiment : adapter.experiment?.id) === experimentId && adapter.status === "SUCCESS");
  const adapter = saved.find((entry) => entry.id === model);
  const payload = { segmentation_type: typeId, asset_ids: assetIds, base_model: !organelle ? null : adapter?.base_model ?? (model === "auto" ? preferredPackId(catalogue, organelle) : ["quantem", "omniem"].includes(model) ? `${model}:${organelle}` : null), adapter_id: adapter?.id };
  const key = JSON.stringify(payload);
  const [selectedRois, setSelected] = useWorkspaceState<string[] | null>(`${memoryKey}.rois.${key}`, null);
  const selected = (selectedRois ?? rois.map((roi) => roi.id)).filter((id) => rois.some((roi) => roi.id === id));
  const [visualInitialized, setVisualInitialized] = useWorkspaceState(`${memoryKey}.visualInitialized`, false);
  useEffect(() => {
    if (visualInitialized || !experimentId || !organelle || !payload.base_model) return;
    setVisualInitialized(true);
    updateApplication(experimentId, { family: payload.base_model.split(":")[0], organelles: [organelle],
      adapter: adapter?.id ?? "", threshold: initialApplication?.threshold ?? null });
  }, [visualInitialized, setVisualInitialized, experimentId, organelle, payload.base_model, adapter?.id, updateApplication, initialApplication?.threshold]);
  const currentKey = useRef(key);
  useEffect(() => { currentKey.current = key; }, [key]);
  useEffect(() => () => { currentKey.current = ""; }, []);
  const { data: polledEvaluation, error: progressError, retrying, refresh: refreshEvaluation } = usePollingResource(evaluationId,
    (id) => apiRequest<Evaluation>(`/api/finetune/evaluations/${id}/`),
    (value) => ["SUCCESS", "FAILED", "CANCELLED"].includes(value.status), 4000);
  const evaluation = polledEvaluation ?? (submittedEvaluation?.id === evaluationId ? submittedEvaluation : null)
    ?? history.find((entry) => entry.id === evaluationId) ?? null;
  const describeEvaluation = (entry: Evaluation) => {
    const settings = entry.settings;
    if (!settings) return "Settings unavailable";
    const name = availableAdapters.find((row) => row.id === settings.adapter_id)?.name ?? settings.adapter_id ?? settings.base_model;
    return `${name} · ${settings.base_model} · threshold ${settings.threshold} · ${settings.asset_ids.length} images`;
  };
  const applySettings = (settings: Pick<EvaluationSettings, "experiment_id" | "base_model" | "adapter_id" | "threshold">) => {
    setVisualInitialized(true);
    updateApplication(settings.experiment_id, { family: settings.base_model.split(":")[0],
      organelles: [settings.base_model.split(":")[1]], adapter: settings.adapter_id ?? "", threshold: settings.threshold });
    setExperimentId(settings.experiment_id);
    setView("visual");
  };
  const submit = async (preview: boolean) => {
    setBusy(true); setError("");
    try {
      if (preview) {
        const next = await apiRequest<{ rois: Roi[]; threshold: number }>("/api/finetune/evaluations/", { method: "POST", body: JSON.stringify({ ...payload, preview: true }) });
        if (currentKey.current !== key) return;
        setRois(next.rois); if (savedThreshold === null) setThreshold(next.threshold); setPreviewKey(key);
      } else {
        const next = await apiRequest<Evaluation>("/api/finetune/evaluations/", { method: "POST", body: JSON.stringify({ ...payload, roi_ids: selected, threshold }) });
        if (currentKey.current !== key) return;
        setSubmittedEvaluation(next); setEvaluationId(next.id); setSubmittedHistory((rows) => [next, ...rows]); void refetchHistory();
      }
    } catch (cause) { if (currentKey.current === key) setError(extractApiErrorMessage(cause, "Evaluation could not be started.")); }
    finally { setBusy(false); }
  };
  return <section className="space-y-4" aria-label="Evaluate a model">
    <div className="flex gap-2" role="group" aria-label="Evaluation method">
      <Button aria-pressed={view === "visual"} variant={view === "visual" ? "primary" : "secondary"} onClick={() => setView("visual")}>Inspect predictions</Button>
      <Button aria-pressed={view === "scores"} variant={view === "scores" ? "primary" : "secondary"} onClick={() => setView("scores")}>Score against ROIs</Button>
    </div>
    <label>Experiment <select disabled={busy} className="rounded border p-1" value={experimentId} onChange={(e) => { setExperimentId(e.target.value); setModel("auto"); }}>{scope.map((entry) => <option key={entry.id} value={entry.id}>{entry.name}</option>)}</select></label>
    {view === "visual" && <>
      <p>Try a model on a representative selection, then open each image to inspect or compare saved results. Annotation is optional. Return to the image table to apply your preferred model more broadly or measure the results.</p>
      {experimentId && <BulkModelPanel key={experimentId} experimentId={experimentId} selectedIds={assetIds} onChanged={refresh} />}
      <div className="max-h-48 overflow-auto">{images.filter((image) => assetIds.includes(image.id)).map((image) => <Link className="block text-cyan-800 underline" key={image.id} to={`/assets/${image.id}`} target="_blank" rel="noreferrer">Inspect {image.name}</Link>)}</div>
    </>}
    {view === "scores" && <>
    <p>Evaluate a released or saved model against completed ROIs. Training-set overlap is flagged and evaluation remains available.</p>
    <div className="flex flex-wrap gap-3">
      <label>Organelle <select disabled={busy} className="rounded border p-1" value={typeId} onChange={(e) => { setTypeId(e.target.value); setModel("auto"); }}>{types.filter((entry) => ["mito", "er", "nucleus", "ld"].some((key) => entry.internal_name === `quantem_internal_${key}`)).map((entry) => <option key={entry.id} value={entry.id}>{entry.long_name}</option>)}</select></label>
      <label>Model <select disabled={busy} className="rounded border p-1" value={model} onChange={(e) => setModel(e.target.value)}><option value="auto">Recommended available model</option><option value="quantem">QuantEM default</option><option value="omniem">OmniEM default</option>{saved.map((entry) => <option key={entry.id} value={entry.id}>{entry.name}</option>)}</select></label>
    </div>
    <Button disabled={busy || !assetIds.length || !payload.base_model} onClick={() => void submit(true)}>Choose ROIs from {assetIds.length} images</Button>
    {previewKey === key && <>
      {!rois.length && <p>No completed ROIs are available in these images. Create an ROI, correct its annotations, and mark it done to include it here.</p>}
      <fieldset className="max-h-64 overflow-auto"><legend>Evaluation ROIs</legend>{rois.map((roi) => <label key={roi.id} className="block py-1"><input type="checkbox" checked={selected.includes(roi.id)} disabled={busy} onChange={(e) => setSelected(e.target.checked ? [...selected, roi.id] : selected.filter((id) => id !== roi.id))} /> {roi.name} {roi.training_overlap && <span className="text-amber-800">⚑ Image used in training; evaluation is allowed</span>}</label>)}</fieldset>
      <label>Threshold <input disabled={busy} className="rounded border p-1" type="number" min="0" max="1" step="0.01" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} /></label>
      <Button variant="primary" disabled={busy || !selected.length || !Number.isFinite(threshold) || threshold < 0 || threshold > 1} onClick={() => void submit(false)}>Evaluate selected ROIs</Button>
    </>}

    <Button disabled={busy || !experimentId || !payload.base_model || !Number.isFinite(threshold) || threshold < 0 || threshold > 1} onClick={() => {
      if (payload.base_model) applySettings({ experiment_id: experimentId, base_model: payload.base_model,
        adapter_id: adapter?.id ?? null, threshold });
    }}>Use current settings for application</Button>
    {history.length > 0 && <label>Saved evaluations <select disabled={busy} className="rounded border p-1" value={evaluation?.id ?? ""} onChange={(e) => setEvaluationId(e.target.value || null)}><option value="">Choose an evaluation</option>{history.map((entry) => <option key={entry.id} value={entry.id}>{new Date(entry.created_at).toLocaleString()} · {describeEvaluation(entry)}</option>)}</select></label>}
    {evaluation && <div className="space-y-2" aria-label="Saved evaluation result">
      <p className="font-semibold">Evaluated settings: {describeEvaluation(evaluation)}</p>
      <p className="text-sm">These scores belong to the saved settings above. Changing the controls requires a new evaluation.</p>
      {evaluation.status === "SUCCESS" && evaluation.settings && <Button onClick={() => applySettings(evaluation.settings!)}>Use evaluated settings for application</Button>}
      <p role="status">{evaluation.status} · {evaluation.message}</p>
      {evaluation.results && <><p>Mean ROI Dice: {evaluation.results.mean_dice?.toFixed(3) ?? "Undefined"}</p><table className="w-full text-left text-sm"><thead><tr><th>ROI</th><th>Dice</th><th>IoU</th><th>Training overlap</th></tr></thead><tbody>{evaluation.results.per_roi.map((row, index) => <tr key={index}><td>{row.name}</td><td>{row.dice?.toFixed(3) ?? "Undefined"}</td><td>{row.iou?.toFixed(3) ?? "Undefined"}</td><td>{row.training_overlap ? "Image used in training" : "None recorded"}</td></tr>)}</tbody></table>{evaluation.results.notes.map((note) => <p className="text-sm text-amber-900" key={note}>{note}</p>)}</>}
    </div>}
    </>}
    {progressError && <div role="status">Evaluation progress could not be refreshed. {retrying ? "Retrying automatically." : "This saved evaluation may no longer be available."} <Button onClick={() => void refreshEvaluation()}>Retry progress</Button></div>}
    {(scopeError || historyError) && <div role="alert">
      {scopeError?.message || historyError?.message}
      <Button onClick={() => { void refetchScope(); void refetchHistory(); }}>Retry loading evaluation</Button>
    </div>}
    {error && <p role="alert" className="text-red-700">{error}</p>}
  </section>;
}
