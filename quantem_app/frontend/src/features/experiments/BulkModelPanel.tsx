import { DEFAULT_APPLICATION_CHOICE, useModelApplicationChoice } from "./modelApplicationChoice";
import { preferredPackId } from "@/features/models/defaultModel";
import { useModelCatalogue } from "@/features/models/useModelCatalogue";
import { useEffect, useState, useRef } from "react";
import { Link } from "react-router-dom";
import { apiRequest } from "@/shared/api/core/http";
import { useWorkspaceState } from "@/shared/hooks/useWorkspaceState";
import { usePollingResource } from "@/shared/hooks/usePollingResource";
import { stopJob, retryJob } from "@/shared/api/jobs";
import { Button } from "@/shared/ui/design";
import { extractApiErrorMessage } from "@/utils/apiErrors";

type Row = { asset_id: string; image: string; ready: boolean; reason: string; job_id?: string; status?: string; message?: string; progress?: number; tiles?: number; warnings?: string[]; download_bytes?: number };
type Batch = { id: string; items: Row[]; created_at: string };
type Model = { id: string; name: string; base_model: string; status: string; experiment: { id: string } | null };
const organelles = { mito: "Mitochondria", er: "ER", nucleus: "Nuclei", ld: "Lipid droplets" };

type Props = { experimentId: string; selectedIds: string[]; onChanged: () => void };

export function BulkModelPanel(props: Props) {
  return <ExperimentModelPanel key={props.experimentId} {...props} />;
}

function ExperimentModelPanel({ experimentId, selectedIds, onChanged }: Props) {
  const { catalogue } = useModelCatalogue();
  const choice = useModelApplicationChoice((state) => state.choices[experimentId] ?? DEFAULT_APPLICATION_CHOICE);
  const updateChoice = useModelApplicationChoice((state) => state.update);
  const { family, organelles: chosen, adapter, threshold } = choice;
  const setFamily = (family: string) => updateChoice(experimentId, { family, threshold: null });
  const setAdapter = (adapter: string) => updateChoice(experimentId, { adapter, threshold: null });
  const setChosen = (value: string[] | ((previous: string[]) => string[])) => updateChoice(experimentId,
    { organelles: typeof value === "function" ? value(chosen) : value });
  const [models, setModels] = useState<Model[]>([]);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [history, setHistory] = useState<Batch[]>([]);
  const [batchId, setBatchId] = useWorkspaceState<string | null>(`model-runs.${experimentId}.selected`, null);
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const [preview, setPreview] = useState<Row[] | null>(null);
  const [previewKey, setPreviewKey] = useState("");
  const [error, setError] = useState("");
  const [loadError, setLoadError] = useState("");
  const [loadAttempt, setLoadAttempt] = useState(0);
  const [busy, setBusy] = useState(false);
  const sourceModels = Object.fromEntries(chosen.map((key) => [key,
    family === "auto" ? preferredPackId(catalogue, key) : `${family}:${key}`]));
  const missingAdapter = Boolean(adapter) && modelsLoaded && !models.some((model) => model.id === adapter);
  const invalidThreshold = threshold !== null && (!Number.isFinite(threshold) || threshold < 0 || threshold > 1);
  const unavailable = adapter ? !modelsLoaded || missingAdapter : Object.values(sourceModels).some((value) => value === null);
  const payload = { asset_ids: selectedIds, organelles: chosen, family: family === "auto" ? "omniem" : family,
    source_models: sourceModels, adapter_id: adapter || undefined, threshold };
  const key = JSON.stringify(payload);
  useEffect(() => {
    let current = true;
    void Promise.all([apiRequest<Model[]>("/api/finetune/adapters/"), apiRequest<Batch[]>(`/api/experiments/${experimentId}/model-runs/`)])
      .then(([saved, batches]) => { if (current) { setModels(saved.filter((model) => model.status === "SUCCESS" && model.experiment?.id === experimentId)); setHistory(batches); setModelsLoaded(true); setLoadError(""); } })
      .catch((cause) => { if (current) setLoadError(extractApiErrorMessage(cause, "Application history could not be loaded.")); });
    return () => { current = false; };
  }, [experimentId, loadAttempt]);
  const { data: polledBatch, error: pollError, retrying, refresh: refreshBatch } = usePollingResource(
    batchId, (id) => apiRequest<Batch>(`/api/model-runs/${id}/`),
    (value) => !value.items.some((row) => ["PENDING", "RETRY", "RUNNING"].includes(row.status ?? "")), 5000);
  const batch = polledBatch ?? history.find((entry) => entry.id === batchId) ?? null;
  const active = batch?.items.some((row) => ["PENDING", "RETRY", "RUNNING"].includes(row.status ?? ""));
  const previousPoll = useRef("");
  useEffect(() => {
    if (!polledBatch) return;
    setHistory((rows) => rows.map((row) => row.id === polledBatch.id ? polledBatch : row));
    // Tile counts/progress do not change the image table. Refresh it only when
    // a task changes status or a different saved batch is selected.
    const signature = JSON.stringify([polledBatch.id, polledBatch.items.map((row) => [row.job_id, row.status])]);
    if (previousPoll.current !== signature) onChanged();
    previousPoll.current = signature;
  }, [polledBatch, onChanged]);
  const start = async (check: boolean) => {
    setBusy(true); setError("");
    try {
      const response = await apiRequest<Batch>(`/api/experiments/${experimentId}/model-runs/`, { method: "POST", body: JSON.stringify({ ...payload, preview: check }) });
      if (!mounted.current) return;
      if (check) { setPreview(response.items); setPreviewKey(key); }
      else { setBatchId(response.id); setHistory((rows) => [response, ...rows]); setPreview(null); onChanged(); }
    } catch (cause) { if (mounted.current) setError(extractApiErrorMessage(cause, "The models could not be queued.")); }
    finally { if (mounted.current) setBusy(false); }
  };
  const act = async (action: "cancel" | "retry") => {
    if (!batch) return;
    setBusy(true); setError("");
    try {
      const eligible = action === "cancel" ? ["PENDING", "RETRY", "RUNNING"] : ["FAILED", "CANCELLED"];
      const targets = batch.items.filter((row) => row.job_id && eligible.includes(row.status ?? ""));
      const failures: string[] = [];
      // Large experiments must not flood the desktop backend with requests.
      for (let offset = 0; offset < targets.length; offset += 4) {
        const group = targets.slice(offset, offset + 4);
        const outcomes = await Promise.allSettled(group.map((row) => action === "cancel"
          ? stopJob(row.job_id!, row.status!) : retryJob(row.job_id!)));
        failures.push(...outcomes.flatMap((outcome, index) => outcome.status === "rejected"
          ? [`${group[index].image}: ${extractApiErrorMessage(outcome.reason, "Could not update this task.")}`] : []));
      }
      if (!mounted.current) return;
      if (failures.length) setError(failures.join("; "));
      await refreshBatch(); onChanged();
    } catch (cause) { if (mounted.current) setError(extractApiErrorMessage(cause, "Some tasks could not be updated. Refresh and try again.")); }
    finally { if (mounted.current) setBusy(false); }
  };
  const rows = preview && previewKey === key ? preview : batch?.items ?? [];
  return <section className="space-y-3 rounded border bg-white p-4" aria-label="Apply models to selected images">
    <h2 className="font-semibold">Apply models · {selectedIds.length} selected images</h2>
    <p>Start with released models and their defaults. Full-image results are ready to measure automatically; completed manual areas are preserved.</p>
    <div className="flex flex-wrap gap-4">
      <label>Model <select disabled={busy} className="rounded border p-1" value={adapter} onChange={(e) => { setAdapter(e.target.value); const model = models.find((row) => row.id === e.target.value); if (model) setChosen([model.base_model.split(":")[1]]); }}>
        {missingAdapter && <option value={adapter} disabled>Unavailable saved model</option>}
        <option value="">Released model</option>{models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
      </select></label>
      {!adapter && <label>Family <select disabled={busy} className="rounded border p-1" value={family} onChange={(e) => setFamily(e.target.value)}><option value="auto">Recommended available models</option><option value="quantem">QuantEM</option><option value="omniem">OmniEM</option></select></label>}
    </div>
    <label>Threshold <input disabled={busy} type="number" min="0" max="1" step="0.01" placeholder="Model default"
      value={threshold ?? ""} onChange={(event) => updateChoice(experimentId, { threshold: event.target.value === "" ? null : Number(event.target.value) })} /></label>
    {!adapter && <fieldset className="flex gap-4"><legend>Structures</legend>{Object.entries(organelles).map(([id, name]) => <label key={id}>
      <input disabled={busy} type="checkbox" checked={chosen.includes(id)} onChange={(e) => setChosen((values) => e.target.checked ? [...values, id] : values.filter((value) => value !== id))} /> {name}
    </label>)}</fieldset>}
    <div className="flex gap-2"><Button disabled={busy || unavailable || invalidThreshold || !selectedIds.length || !chosen.length} onClick={() => void start(true)}>Check selected images</Button>
      <Button variant="primary" disabled={busy || unavailable || invalidThreshold || previewKey !== key || !preview?.some((row) => row.ready)} onClick={() => void start(false)}>Apply to ready images</Button></div>
    {history.length > 0 && <label>Saved applications <select disabled={busy} className="rounded border p-1" value={batch?.id ?? ""} onChange={(e) => { const saved = history.find((entry) => entry.id === e.target.value); setPreview(null); setBatchId(saved?.id ?? null); }}>
      <option value="">Choose an application</option>{history.map((entry) => <option key={entry.id} value={entry.id}>{new Date(entry.created_at).toLocaleString()} · {entry.items.length} images</option>)}</select></label>}
    {batch && !preview && <div className="flex gap-2"><Button disabled={busy || !active} onClick={() => void act("cancel")}>Cancel remaining</Button><Button disabled={busy || !batch.items.some((row) => ["FAILED", "CANCELLED"].includes(row.status ?? ""))} onClick={() => void act("retry")}>Retry failed</Button></div>}
    {unavailable && !adapter && <p role="alert">No usable installed model is available for a selected organelle. Choose another model family or update its model pack.</p>}
    {missingAdapter && <p role="alert">The selected saved model is no longer available. Choose another model.</p>}
    {invalidThreshold && <p role="alert">Threshold must be between 0 and 1.</p>}
    {pollError && <div role="status">Application progress could not be refreshed. {retrying ? "Retrying automatically." : "This saved application may no longer be available."} <Button onClick={() => void refreshBatch()}>Retry progress</Button></div>}
    {loadError && <div role="alert">{loadError} <Button onClick={() => setLoadAttempt((value) => value + 1)}>Retry application history</Button></div>}
    {error && <p role="alert" className="text-red-700">{error}</p>}
    {rows.length > 0 && <div className="max-h-80 overflow-auto"><table className="w-full text-left text-sm"><thead><tr><th>Image</th><th>Status</th><th>Details</th></tr></thead><tbody>{rows.map((row) => <tr key={row.asset_id}>
      <td className="border-b p-2"><Link className="text-cyan-800 underline" to={`/assets/${row.asset_id}`}>{row.image}</Link></td>
      <td className="border-b p-2">{row.status ?? (row.ready ? "Ready" : "Skipped")}{row.status === "RUNNING" && ` · ${Math.round(row.progress ?? 0)}%`}</td>
      <td className="border-b p-2">{row.message || row.reason || (row.tiles != null ? `${row.tiles.toLocaleString()} tiles` : "")}
        {!!row.download_bytes && <span> · Model download required</span>}
        {row.warnings?.map((warning) => <p className="text-amber-800" key={warning}>{warning}</p>)}
      </td>
    </tr>)}</tbody></table></div>}
  </section>;
}
