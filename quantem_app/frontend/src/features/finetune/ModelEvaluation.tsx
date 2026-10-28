import { useModelApplicationChoice } from "@/features/experiments/modelApplicationChoice";
import { preferredPackId } from "@/features/models/defaultModel";
import { useModelCatalogue } from "@/features/models/useModelCatalogue";
import { Link } from "react-router-dom";
import { BulkModelPanel } from "@/features/experiments/BulkModelPanel";
import { useCallback, useEffect, useMemo, useState } from "react";
import { apiRequest } from "@/shared/api/core/http";
import { getFineTuneScope } from "@/shared/api/finetune";
import type { SegmentationType } from "@/shared/types/images";
import type { FineTuneAdapterSummary, FineTuneScopeExperiment } from "@/shared/types/finetune";
import { Button } from "@/shared/ui/design";
import { extractApiErrorMessage } from "@/utils/apiErrors";

type Roi = { id: string; name: string; image_id: string; training_overlap: boolean };
type Score = { name: string; dice: number | null; iou: number | null; training_overlap: boolean };
type Evaluation = { id: string; status: string; message: string; created_at: string; results?: { per_roi: Score[]; notes: string[]; mean_dice: number | null } | null };

export function ModelEvaluation({ types, initialTypeId, initialAssetIds, initialExperimentId, adapters }: {
  types: SegmentationType[]; initialTypeId: string; initialAssetIds?: string[]; initialExperimentId?: string; adapters: FineTuneAdapterSummary[];
}) {
  const updateApplication = useModelApplicationChoice((state) => state.update);
  const [error, setError] = useState("");
  const [view, setView] = useState<"visual" | "scores">("visual");
  const [availableAdapters, setAvailableAdapters] = useState(adapters);
  const refresh = useCallback(() => {}, []);
  useEffect(() => {
    let current = true;
    void apiRequest<FineTuneAdapterSummary[]>("/api/finetune/adapters/").then((rows) => { if (current) setAvailableAdapters(rows); })
      .catch((cause) => { if (current) setError(extractApiErrorMessage(cause, "Saved models could not be loaded.")); });
    return () => { current = false; };
  }, []);
  const supportedTypes = types.filter((entry) => ["mito", "er", "nucleus", "ld"].some((key) => entry.internal_name === `quantem_internal_${key}`));
  const defaultTypeId = supportedTypes[0]?.id ?? "";
  const [typeId, setTypeId] = useState(initialTypeId || defaultTypeId);
  const [experimentId, setExperimentId] = useState(initialExperimentId ?? "");
  const [scope, setScope] = useState<FineTuneScopeExperiment[]>([]);
  const { catalogue } = useModelCatalogue();
  const [model, setModel] = useState("auto");
  const [rois, setRois] = useState<Roi[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [threshold, setThreshold] = useState(0.5);
  const [previewKey, setPreviewKey] = useState("");
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [history, setHistory] = useState<Evaluation[]>([]);
  const [busy, setBusy] = useState(false);
  const type = types.find((entry) => entry.id === typeId);
  useEffect(() => { if (!typeId && defaultTypeId) setTypeId(defaultTypeId); }, [defaultTypeId, typeId]);
  useEffect(() => {
    if (!typeId) return;
    let current = true;
    void getFineTuneScope(typeId).then((value) => { if (current) { setScope(value.experiments); setExperimentId((id) => value.experiments.some((entry) => entry.id === id) ? id : value.experiments[0]?.id ?? ""); } })
      .catch((cause) => { if (current) setError(extractApiErrorMessage(cause, "Annotated images could not be loaded.")); });
    return () => { current = false; };
  }, [typeId]);
  useEffect(() => {
    if (!experimentId) return;
    let current = true;
    void apiRequest<Evaluation[]>(`/api/finetune/evaluations/?experiment=${experimentId}`).then((rows) => { if (current) setHistory(rows); })
      .catch((cause) => { if (current) setError(extractApiErrorMessage(cause, "Evaluation history could not be loaded.")); });
    return () => { current = false; };
  }, [experimentId]);
  const images = useMemo(() => {
    const experiment = scope.find((entry) => entry.id === experimentId);
    return [...new Map([...(experiment?.ungrouped_images ?? []), ...(experiment?.datasets.flatMap((dataset) => dataset.images) ?? [])].map((image) => [image.id, image])).values()];
  }, [scope, experimentId]);
  const assetIds = images.filter((image) => !initialAssetIds?.length || (initialExperimentId && experimentId !== initialExperimentId) || initialAssetIds.includes(image.id)).map((image) => image.id);
  const organelle = type?.internal_name.replace("quantem_internal_", "") ?? "mito";
  const saved = availableAdapters.filter((adapter) => adapter.base_model.endsWith(`:${organelle}`) && (typeof adapter.experiment === "string" ? adapter.experiment : adapter.experiment?.id) === experimentId && adapter.status === "SUCCESS");
  const adapter = saved.find((entry) => entry.id === model);
  const payload = { segmentation_type: typeId, asset_ids: assetIds, base_model: adapter?.base_model ?? (model === "auto" ? preferredPackId(catalogue, organelle) : `${model}:${organelle}`), adapter_id: adapter?.id };
  const key = JSON.stringify(payload);
  useEffect(() => {
    if (!evaluation || !["PENDING", "RETRY", "RUNNING"].includes(evaluation.status)) return;
    let current = true;
    const timer = window.setInterval(() => {
      void apiRequest<Evaluation>(`/api/finetune/evaluations/${evaluation.id}/`).then((next) => { if (current) setEvaluation(next); })
        .catch((cause) => { if (current) setError(extractApiErrorMessage(cause, "Evaluation status could not be refreshed.")); });
    }, 4000);
    return () => { current = false; clearInterval(timer); };
  }, [evaluation]);
  const submit = async (preview: boolean) => {
    setBusy(true); setError("");
    try {
      if (preview) {
        const next = await apiRequest<{ rois: Roi[]; threshold: number }>("/api/finetune/evaluations/", { method: "POST", body: JSON.stringify({ ...payload, preview: true }) });
        setRois(next.rois); setSelected(next.rois.map((roi) => roi.id)); setThreshold(next.threshold); setPreviewKey(key);
      } else {
        const next = await apiRequest<Evaluation>("/api/finetune/evaluations/", { method: "POST", body: JSON.stringify({ ...payload, roi_ids: selected, threshold }) });
        setEvaluation(next); setHistory((rows) => [next, ...rows]);
      }
    } catch (cause) { setError(extractApiErrorMessage(cause, "Evaluation could not be started.")); }
    finally { setBusy(false); }
  };
  return <section className="space-y-4" aria-label="Evaluate a model">
    <div className="flex gap-2" role="group" aria-label="Evaluation method">
      <Button variant={view === "visual" ? "primary" : "secondary"} onClick={() => setView("visual")}>Inspect predictions</Button>
      <Button variant={view === "scores" ? "primary" : "secondary"} onClick={() => setView("scores")}>Score against ROIs</Button>
    </div>
    <label>Experiment <select className="rounded border p-1" value={experimentId} onChange={(e) => { setExperimentId(e.target.value); setModel("auto"); }}>{scope.map((entry) => <option key={entry.id} value={entry.id}>{entry.name}</option>)}</select></label>
    {view === "visual" && <>
      <p>Try a model on a representative selection, then open each image to inspect or compare saved results. Annotation is optional. Return to the image table to apply your preferred model more broadly or measure the results.</p>
      {experimentId && <BulkModelPanel experimentId={experimentId} selectedIds={assetIds} onChanged={refresh} />}
      <div className="max-h-48 overflow-auto">{images.filter((image) => assetIds.includes(image.id)).map((image) => <Link className="block text-cyan-800 underline" key={image.id} to={`/assets/${image.id}`} target="_blank" rel="noreferrer">Inspect {image.name}</Link>)}</div>
    </>}
    {view === "scores" && <>
    <p>Evaluate a released or saved model against completed ROIs. Training-set overlap is flagged and evaluation remains available.</p>
    <div className="flex flex-wrap gap-3">
      <label>Organelle <select className="rounded border p-1" value={typeId} onChange={(e) => { setTypeId(e.target.value); setModel("auto"); }}>{types.filter((entry) => ["mito", "er", "nucleus", "ld"].some((key) => entry.internal_name === `quantem_internal_${key}`)).map((entry) => <option key={entry.id} value={entry.id}>{entry.long_name}</option>)}</select></label>
      <label>Model <select className="rounded border p-1" value={model} onChange={(e) => setModel(e.target.value)}><option value="auto">Recommended available model</option><option value="quantem">QuantEM default</option><option value="omniem">OmniEM default</option>{saved.map((entry) => <option key={entry.id} value={entry.id}>{entry.name}</option>)}</select></label>
    </div>
    <Button disabled={busy || !assetIds.length} onClick={() => void submit(true)}>Choose ROIs from {assetIds.length} images</Button>
    {previewKey === key && <>
      {!rois.length && <p>No completed ROIs are available in these images. Create an ROI, correct its annotations, and mark it done to include it here.</p>}
      <fieldset className="max-h-64 overflow-auto"><legend>Evaluation ROIs</legend>{rois.map((roi) => <label key={roi.id} className="block py-1"><input type="checkbox" checked={selected.includes(roi.id)} onChange={(e) => setSelected((values) => e.target.checked ? [...values, roi.id] : values.filter((id) => id !== roi.id))} /> {roi.name} {roi.training_overlap && <span className="text-amber-800">⚑ Image used in training; evaluation is allowed</span>}</label>)}</fieldset>
      <label>Threshold <input className="rounded border p-1" type="number" min="0" max="1" step="0.01" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} /></label>
      <Button variant="primary" disabled={busy || !selected.length} onClick={() => void submit(false)}>Evaluate selected ROIs</Button>
    </>}

    <Button disabled={!experimentId || !payload.base_model} onClick={() => {
      updateApplication(experimentId, { family: adapter?.base_model.split(":")[0] ?? model,
        organelles: [organelle], adapter: adapter?.id ?? "", threshold });
      setView("visual");
    }}>Use these settings for application</Button>
    {history.length > 0 && <label>Saved evaluations <select className="rounded border p-1" value={evaluation?.id ?? ""} onChange={(e) => setEvaluation(history.find((entry) => entry.id === e.target.value) ?? null)}><option value="">Choose an evaluation</option>{history.map((entry) => <option key={entry.id} value={entry.id}>{new Date(entry.created_at).toLocaleString()}</option>)}</select></label>}
    {evaluation && <div className="space-y-2"><p role="status">{evaluation.status} · {evaluation.message}</p>
      {evaluation.results && <><p>Mean ROI Dice: {evaluation.results.mean_dice?.toFixed(3) ?? "Undefined"}</p><table className="w-full text-left text-sm"><thead><tr><th>ROI</th><th>Dice</th><th>IoU</th><th>Training overlap</th></tr></thead><tbody>{evaluation.results.per_roi.map((row, index) => <tr key={index}><td>{row.name}</td><td>{row.dice?.toFixed(3) ?? "Undefined"}</td><td>{row.iou?.toFixed(3) ?? "Undefined"}</td><td>{row.training_overlap ? "Image used in training" : "None recorded"}</td></tr>)}</tbody></table>{evaluation.results.notes.map((note) => <p className="text-sm text-amber-900" key={note}>{note}</p>)}</>}
    </div>}
    </>}
    {error && <p role="alert" className="text-red-700">{error}</p>}
  </section>;
}
