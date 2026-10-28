import { DEFAULT_APPLICATION_CHOICE, useModelApplicationChoice } from "./modelApplicationChoice";
import { preferredPackId } from "@/features/models/defaultModel";
import { useModelCatalogue } from "@/features/models/useModelCatalogue";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { apiRequest } from "@/shared/api/core/http";
import { Button } from "@/shared/ui/design";
import { extractApiErrorMessage } from "@/utils/apiErrors";

type Row = { asset_id: string; image: string; ready: boolean; reason: string; job_id?: string; status?: string; message?: string; progress?: number; tiles?: number; warnings?: string[]; download_bytes?: number };
type Batch = { id: string; items: Row[]; created_at: string };
type Model = { id: string; name: string; base_model: string; status: string; experiment: { id: string } | null };
const organelles = { mito: "Mitochondria", er: "ER", nucleus: "Nuclei", ld: "Lipid droplets" };

export function BulkModelPanel({ experimentId, selectedIds, onChanged }: { experimentId: string; selectedIds: string[]; onChanged: () => void }) {
  const { catalogue } = useModelCatalogue();
  const choice = useModelApplicationChoice((state) => state.choices[experimentId] ?? DEFAULT_APPLICATION_CHOICE);
  const updateChoice = useModelApplicationChoice((state) => state.update);
  const { family, organelles: chosen, adapter, threshold } = choice;
  const setFamily = (family: string) => updateChoice(experimentId, { family, threshold: null });
  const setAdapter = (adapter: string) => updateChoice(experimentId, { adapter, threshold: null });
  const setChosen = (value: string[] | ((previous: string[]) => string[])) => updateChoice(experimentId,
    { organelles: typeof value === "function" ? value(chosen) : value });
  const [models, setModels] = useState<Model[]>([]);
  const [history, setHistory] = useState<Batch[]>([]);
  const [batch, setBatch] = useState<Batch | null>(null);
  const [preview, setPreview] = useState<Row[] | null>(null);
  const [previewKey, setPreviewKey] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const sourceModels = Object.fromEntries(chosen.map((key) => [key,
    family === "auto" ? preferredPackId(catalogue, key) : `${family}:${key}`]));
  const unavailable = !adapter && Object.values(sourceModels).some((value) => value === null);
  const payload = { asset_ids: selectedIds, organelles: chosen, family: family === "auto" ? "omniem" : family,
    source_models: sourceModels, adapter_id: adapter || undefined, threshold };
  const key = JSON.stringify(payload);
  useEffect(() => {
    let current = true;
    void Promise.all([apiRequest<Model[]>("/api/finetune/adapters/"), apiRequest<Batch[]>(`/api/experiments/${experimentId}/model-runs/`)])
      .then(([saved, batches]) => { if (current) { setModels(saved.filter((model) => model.status === "SUCCESS" && model.experiment?.id === experimentId)); setHistory(batches); } })
      .catch((cause) => { if (current) setError(extractApiErrorMessage(cause, "Application history could not be loaded.")); });
    return () => { current = false; };
  }, [experimentId]);
  const active = batch?.items.some((row) => ["PENDING", "RETRY", "RUNNING"].includes(row.status ?? ""));
  useEffect(() => {
    if (!batch?.id || !active) return;
    let stopped = false;
    const timer = window.setInterval(() => {
      void apiRequest<Batch>(`/api/model-runs/${batch.id}/`).then((next) => {
        if (!stopped) { setBatch(next); if (!next.items.some((row) => ["PENDING", "RETRY", "RUNNING"].includes(row.status ?? ""))) onChanged(); }
      }).catch((cause) => { if (!stopped) setError(extractApiErrorMessage(cause, "Could not refresh this application.")); });
    }, 5000);
    return () => { stopped = true; clearInterval(timer); };
  }, [batch?.id, active, onChanged]);
  const start = async (check: boolean) => {
    setBusy(true); setError("");
    try {
      const response = await apiRequest<Batch>(`/api/experiments/${experimentId}/model-runs/`, { method: "POST", body: JSON.stringify({ ...payload, preview: check }) });
      if (check) { setPreview(response.items); setPreviewKey(key); }
      else { setBatch(response); setHistory((rows) => [response, ...rows]); setPreview(null); onChanged(); }
    } catch (cause) { setError(extractApiErrorMessage(cause, "The models could not be queued.")); }
    finally { setBusy(false); }
  };
  const act = async (action: "cancel" | "retry") => {
    if (!batch) return;
    setBusy(true); setError("");
    try {
      for (const row of batch.items) {
        const eligible = action === "cancel" ? ["PENDING", "RETRY", "RUNNING"] : ["FAILED", "CANCELLED"];
        if (row.job_id && eligible.includes(row.status ?? "")) await apiRequest(`/api/jobs/${row.job_id}/${action}/`, { method: "POST" });
      }
      setBatch(await apiRequest<Batch>(`/api/model-runs/${batch.id}/`)); onChanged();
    } catch (cause) { setError(extractApiErrorMessage(cause, "Some tasks could not be updated. Refresh and try again.")); }
    finally { setBusy(false); }
  };
  const rows = preview && previewKey === key ? preview : batch?.items ?? [];
  return <section className="space-y-3 rounded border bg-white p-4" aria-label="Apply models to selected images">
    <h2 className="font-semibold">Apply models · {selectedIds.length} selected images</h2>
    <p>Start with released models and their defaults. Full-image results are ready to measure automatically; completed manual areas are preserved.</p>
    <div className="flex flex-wrap gap-4">
      <label>Model <select className="rounded border p-1" value={adapter} onChange={(e) => { setAdapter(e.target.value); const model = models.find((row) => row.id === e.target.value); if (model) setChosen([model.base_model.split(":")[1]]); }}>
        <option value="">Released model</option>{models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
      </select></label>
      {!adapter && <label>Family <select className="rounded border p-1" value={family} onChange={(e) => setFamily(e.target.value)}><option value="auto">Recommended available models</option><option value="quantem">QuantEM</option><option value="omniem">OmniEM</option></select></label>}
    </div>
    <label>Threshold <input type="number" min="0" max="1" step="0.01" placeholder="Model default"
      value={threshold ?? ""} onChange={(event) => updateChoice(experimentId, { threshold: event.target.value === "" ? null : Number(event.target.value) })} /></label>
    {!adapter && <fieldset className="flex gap-4"><legend>Structures</legend>{Object.entries(organelles).map(([id, name]) => <label key={id}>
      <input type="checkbox" checked={chosen.includes(id)} onChange={(e) => setChosen((values) => e.target.checked ? [...values, id] : values.filter((value) => value !== id))} /> {name}
    </label>)}</fieldset>}
    <div className="flex gap-2"><Button disabled={busy || unavailable || !selectedIds.length || !chosen.length} onClick={() => void start(true)}>Check selected images</Button>
      <Button variant="primary" disabled={busy || previewKey !== key || !preview?.some((row) => row.ready)} onClick={() => void start(false)}>Apply to ready images</Button></div>
    {history.length > 0 && <label>Saved applications <select className="rounded border p-1" value={batch?.id ?? ""} onChange={(e) => { const saved = history.find((entry) => entry.id === e.target.value); setPreview(null); setBatch(saved ?? null); }}>
      <option value="">Choose an application</option>{history.map((entry) => <option key={entry.id} value={entry.id}>{new Date(entry.created_at).toLocaleString()} · {entry.items.length} images</option>)}</select></label>}
    {batch && !preview && <div className="flex gap-2"><Button disabled={busy || !active} onClick={() => void act("cancel")}>Cancel remaining</Button><Button disabled={busy || !batch.items.some((row) => ["FAILED", "CANCELLED"].includes(row.status ?? ""))} onClick={() => void act("retry")}>Retry failed</Button></div>}
    {unavailable && <p role="alert">No usable installed model is available for a selected organelle. Choose another model family or update its model pack.</p>}
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
