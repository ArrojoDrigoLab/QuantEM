import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { apiRequest, resolveApiUrl } from "@/shared/api/core/http";
import type { WorkspaceImage } from "@/shared/api/workspace";
import { Button } from "@/shared/ui/design";
import { extractApiErrorMessage } from "@/utils/apiErrors";

interface Identity { image: string; asset_id: string; organelle: string; sample_id: string; group: string; measurement_area: string; pixel_size_nm: number | null }
interface Item { identity: Identity; reason: string; ready?: boolean; status?: string; current?: boolean | null; reused?: boolean; run_id?: string; metrics?: Record<string, number | null> }
interface Batch { id: string; name: string; created_at: string; items: Item[] }
interface History { id: string; name: string; created_at: string; count: number }
export function BatchAnalysisPanel({ experimentId, images, selectedIds }: { experimentId: string; images: WorkspaceImage[]; selectedIds: string[] }) {
  const [types, setTypes] = useState<string[]>([]);
  const [name, setName] = useState("Experiment measurements");
  const [mask, setMask] = useState("");
  const [complete, setComplete] = useState(true);
  const [reuse, setReuse] = useState(true);
  const [objects, setObjects] = useState(false);
  const [preview, setPreview] = useState<Item[] | null>(null);
  const [previewKey, setPreviewKey] = useState("");
  const [batch, setBatch] = useState<Batch | null>(null);
  const [history, setHistory] = useState<History[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [metric, setMetric] = useState("n_objects");
  const options = useMemo(() => [...new Map(images.flatMap((image) => image.segmentations)
    .filter((seg) => seg.type !== "quantem_internal_analysis_mask").map((seg) => [seg.type, seg])).values()], [images]);
  const masks = useMemo(() => [...new Set(images.flatMap((image) => image.segmentations)
    .filter((seg) => seg.type === "quantem_internal_analysis_mask").map((seg) => seg.name))].sort(), [images]);
  const payload = useMemo(() => ({ name, asset_ids: selectedIds, segmentation_types: types,
    mask_name: mask, require_complete: complete, reuse }), [name, selectedIds, types, mask, complete, reuse]);
  const payloadKey = JSON.stringify(payload);
  const loadHistory = useCallback(async () => {
    try { setHistory(await apiRequest<History[]>(`/api/experiments/${experimentId}/analysis-batches/`)); }
    catch (cause) { setError(extractApiErrorMessage(cause, "Batch history could not be loaded.")); }
  }, [experimentId]);
  useEffect(() => { void loadHistory(); }, [loadHistory]);
  useEffect(() => { setPreview(null); }, [payload]);
  const active = batch?.items.some((item) => ["PENDING", "RUNNING"].includes(item.status ?? ""));
  const batchId = batch?.id;
  useEffect(() => {
    if (!batchId || !active) return;
    let stopped = false;
    let timer: number;
    const refresh = async () => {
      try {
        const next = await apiRequest<Batch>(`/api/analysis-batches/${batchId}/`);
        if (!stopped) setBatch(next);
      } catch (cause) {
        if (!stopped) setError(extractApiErrorMessage(cause, "Could not refresh this batch. Use Refresh."));
      } finally {
        if (!stopped) timer = window.setTimeout(() => void refresh(), 5000);
      }
    };
    timer = window.setTimeout(() => void refresh(), 5000);
    return () => { stopped = true; clearTimeout(timer); };
  }, [batchId, active]);
  const run = async (check: boolean) => {
    setBusy(true); setError("");
    try {
      if (check) {
        const response = await apiRequest<{ items: Item[] }>(`/api/experiments/${experimentId}/analysis-batches/`, { method: "POST", body: JSON.stringify({ ...payload, preview: true }) });
        setPreview(response.items); setPreviewKey(payloadKey);
      } else {
        setBatch(await apiRequest<Batch>(`/api/experiments/${experimentId}/analysis-batches/`, { method: "POST", body: JSON.stringify(payload) }));
        setPreview(null); await loadHistory();
      }
    } catch (cause) { setError(extractApiErrorMessage(cause, "Measurements could not be started.")); }
    finally { setBusy(false); }
  };
  const open = async (id: string, action?: string) => {
    setBusy(true); setError("");
    try { setBatch(await apiRequest<Batch>(`/api/analysis-batches/${id}/`, action ? { method: "POST", body: JSON.stringify({ action }) } : {})); setPreview(null); }
    catch (cause) { setError(extractApiErrorMessage(cause, "The batch could not be loaded.")); }
    finally { setBusy(false); }
  };
  const currentPreview = previewKey === payloadKey ? preview : null;
  const rows = currentPreview ?? batch?.items ?? [];
  const metricOptions = [...new Set(rows.flatMap((item) => Object.keys(item.metrics ?? {})))].sort();
  const selectedMetric = metricOptions.includes(metric) ? metric : metricOptions[0];
  const totals = rows.reduce<Record<string, number>>((result, item) => {
    const key = item.ready === true ? "Ready" : item.ready === false ? "Skipped" : item.current === false ? "Outdated" : item.status ?? "";
    result[key] = (result[key] ?? 0) + 1; return result;
  }, {});
  return <section className="space-y-4 rounded-lg border bg-white p-4" aria-label="Experiment analysis">
    <h2 className="text-lg font-semibold">Measure selected images</h2>
    <p className="text-sm text-slate-600">{selectedIds.length} unique images selected. Change the selection in Images. Object analyses use confirmed objects; global segmentations provide area measurements only. Review completion and prediction completion are separate.</p>
    <div className="flex flex-wrap items-center gap-4">
      <label>Batch name <input className="rounded border px-2 py-1" value={name} onChange={(e) => setName(e.target.value)} /></label>
      <label>Measurement area <select className="rounded border px-2 py-1" value={mask} onChange={(e) => setMask(e.target.value)}>
        <option value="">Whole image</option>{masks.map((entry) => <option key={entry}>{entry}</option>)}</select></label>
    </div>
    <fieldset className="flex flex-wrap gap-4"><legend className="mb-2 font-medium">Organelles</legend>
      {options.map((seg) => <label key={seg.type}><input type="checkbox" checked={types.includes(seg.type)}
        onChange={(e) => setTypes((current) => e.target.checked ? [...current, seg.type] : current.filter((value) => value !== seg.type))} /> {seg.name} {seg.measurement_mode === "global" ? "(area only)" : ""}</label>)}
      {!options.length && <p>Run a segmentation on an image before measuring this experiment.</p>}
    </fieldset>
    <div className="flex flex-wrap gap-4">
      <label><input type="checkbox" checked={complete} onChange={(e) => setComplete(e.target.checked)} /> Require review marked complete</label>
      <label><input type="checkbox" checked={reuse} onChange={(e) => setReuse(e.target.checked)} /> Reuse matching current results</label>
    </div>
    {!complete && <p className="text-sm text-amber-900">Unreviewed images will be included. Their confirmed objects may be an incomplete subset; exports record review status.</p>}
    <p className="text-sm text-slate-600">A missing named mask skips the image. Missing calibration is recorded; physical units are unavailable. Samples are entered explicitly in Images; unnamed samples stay in image results but are excluded from group statistics.</p>
    <div className="flex flex-wrap gap-2">
      <Button disabled={busy || !selectedIds.length || !types.length} onClick={() => void run(true)}>Check readiness</Button>
      <Button variant="primary" disabled={busy || previewKey !== payloadKey || !preview?.some((item) => item.ready)} onClick={() => void run(false)}>Run ready images</Button>
    </div>
    {error && <p className="text-red-700" role="alert">{error}</p>}
    <label className="block">Reopen a batch <select className="ml-2 max-w-full rounded border p-1" value={batch?.id ?? ""} onChange={(e) => { if (e.target.value) void open(e.target.value); }}>
      <option value="">Choose saved batch</option>{history.map((entry) => <option key={entry.id} value={entry.id}>{entry.name} · {new Date(entry.created_at).toLocaleString()} · {entry.count} measurements</option>)}
    </select></label>
    {rows.length > 0 && <p role="status">{Object.entries(totals).map(([state, count]) => `${count} ${state.toLowerCase()}`).join(" · ")}</p>}
    {metricOptions.length > 0 && <label className="block">Compare metric <select className="ml-2 rounded border p-1" value={selectedMetric} onChange={(e) => setMetric(e.target.value)}>
      {metricOptions.map((entry) => <option key={entry} value={entry}>{entry.replaceAll("_", " ")}</option>)}</select></label>}
    {batch && !currentPreview && <div className="flex flex-wrap items-center gap-3">
      <strong>{batch.name}</strong><Button disabled={busy} onClick={() => void open(batch.id)}>Refresh</Button>
      <Button disabled={busy || !batch.items.some((item) => ["FAILED", "CANCELLED"].includes(item.status ?? ""))} onClick={() => void open(batch.id, "retry_failed")}>Retry failed</Button>
      <Button disabled={busy || !active} onClick={() => void open(batch.id, "cancel")}>Cancel remaining</Button>
      <label><input type="checkbox" checked={objects} onChange={(e) => setObjects(e.target.checked)} /> Include object tables</label>
      <a className="font-medium text-cyan-800 underline" href={resolveApiUrl(`/api/analysis-batches/${batch.id}/export/?objects=${objects}`)} download>Export measurements ZIP</a>
    </div>}
    {rows.length > 0 && <div className="max-h-96 overflow-auto"><table className="w-full text-left text-sm">
      <thead><tr>{["Image", "Organelle", "Area", "Sample / group", "Status", selectedMetric?.replaceAll("_", " ") ?? "Measurement", "Details"].map((label) => <th key={label} className="border-b p-2">{label}</th>)}</tr></thead>
      <tbody>{rows.map((item, index) => <tr key={index}>
        <td className="border-b p-2"><Link className="text-cyan-800 underline" to={`/assets/${item.identity.asset_id}/analysis`}>{item.identity.image}</Link></td>
        <td className="border-b p-2">{item.identity.organelle}</td><td className="border-b p-2">{item.identity.measurement_area}</td>
        <td className="border-b p-2">{item.identity.sample_id || "No sample ID"} / {item.identity.group || "No group"}</td>
        <td className="border-b p-2">{item.ready === true ? "Ready" : item.ready === false ? "Skipped" : item.status}{item.reused ? " · reused" : ""}{item.current === false ? " · outdated" : ""}</td>
        <td className="border-b p-2">{selectedMetric && item.metrics?.[selectedMetric] != null ? item.metrics[selectedMetric]?.toLocaleString(undefined, { maximumSignificantDigits: 6 }) : "—"}</td>
        <td className="border-b p-2">{item.reason}{!item.identity.pixel_size_nm ? " Physical units unavailable." : ""}</td>
      </tr>)}</tbody></table></div>}
    <p className="text-xs text-slate-600">The ZIP contains image measurements, explicit sample means, group statistics and a manifest of settings, model identity and skipped/failed items. Group statistics average images within each sample, then average samples with equal weight; outdated results are excluded. Partial exports retain their current status.</p>
  </section>;
}
