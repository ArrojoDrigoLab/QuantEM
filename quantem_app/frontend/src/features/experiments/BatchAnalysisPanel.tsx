import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { apiRequest, resolveApiUrl } from "@/shared/api/core/http";
import type { WorkspaceImage } from "@/shared/api/workspace";
import { Button } from "@/shared/ui/design";
import { HelpDisclosure } from "@/shared/ui/HelpDisclosure";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import type { AnalysisBatch as Batch, BatchItem as Item } from "@/shared/api/analysisHistory";
import { AnalysisImageSelection } from "./AnalysisImageSelection";

export function BatchAnalysisPanel({ experimentId, images, selectedIds, onSelectionChange, onHistoryChange, onMetadataSaved }: {
  experimentId: string; images: WorkspaceImage[]; selectedIds: string[];
  onSelectionChange: (ids: string[]) => void; onHistoryChange: () => void; onMetadataSaved: () => void;
}) {
  const [types, setTypes] = useState<string[]>([]);
  const [name, setName] = useState("Experiment measurements");
  const [mask, setMask] = useState("");
  const [complete, setComplete] = useState(true);
  const [reuse, setReuse] = useState(true);
  const [objects, setObjects] = useState(false);
  const [preview, setPreview] = useState<Item[] | null>(null);
  const [previewKey, setPreviewKey] = useState("");
  const [batch, setBatch] = useState<Batch | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [metric, setMetric] = useState("n_objects");
  const options = useMemo(() => [...new Map(images.flatMap((image) => image.segmentations)
    .filter((seg) => (seg.category ?? "organelle") === "organelle").map((seg) => [seg.type_id, seg])).values()], [images]);
  const masks = useMemo(() => [...new Map(images.flatMap((image) => image.segmentations)
    .filter((seg) => seg.type === "quantem_internal_tissue").map((seg) => [seg.type_id, seg])).values()], [images]);
  const payload = useMemo(() => ({ name, asset_ids: selectedIds, segmentation_type_ids: types,
    mask_definition_id: mask || null, require_complete: complete, reuse }), [name, selectedIds, types, mask, complete, reuse]);
  const payloadKey = JSON.stringify(payload);
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
        if (!stopped && !next.items.some((item) => ["PENDING", "RUNNING"].includes(item.status ?? ""))) onHistoryChange();
      } catch (cause) {
        if (!stopped) setError(extractApiErrorMessage(cause, "Could not refresh this batch. Use Refresh."));
      } finally {
        if (!stopped) timer = window.setTimeout(() => void refresh(), 5000);
      }
    };
    timer = window.setTimeout(() => void refresh(), 5000);
    return () => { stopped = true; clearTimeout(timer); };
  }, [batchId, active, onHistoryChange]);
  const run = async (check: boolean) => {
    setBusy(true); setError("");
    try {
      if (check) {
        const response = await apiRequest<{ items: Item[] }>(`/api/experiments/${experimentId}/analysis-batches/`, { method: "POST", body: JSON.stringify({ ...payload, preview: true }) });
        setPreview(response.items); setPreviewKey(payloadKey);
      } else {
        setBatch(await apiRequest<Batch>(`/api/experiments/${experimentId}/analysis-batches/`, { method: "POST", body: JSON.stringify(payload) }));
        setPreview(null); onHistoryChange();
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
    <h2 className="text-lg font-semibold">Measure selected images ({selectedIds.length})</h2>
    <AnalysisImageSelection experimentId={experimentId} images={images} selectedIds={selectedIds} onChange={onSelectionChange} onMetadataSaved={onMetadataSaved} />
    <div className="flex flex-wrap items-center gap-4">
      <label>Batch name <input className="rounded border px-2 py-1" value={name} onChange={(e) => setName(e.target.value)} /></label>
      <label>Measurement area <select className="rounded border px-2 py-1" value={mask} onChange={(e) => setMask(e.target.value)}>
        <option value="">Whole image</option>{masks.map((entry) => <option key={entry.type_id} value={entry.type_id}>{entry.name}</option>)}</select></label>
    </div>
    <fieldset className="flex flex-wrap gap-4"><legend className="mb-2 font-medium">Organelles</legend>
      {options.map((seg) => <label key={seg.type_id}><input type="checkbox" checked={types.includes(seg.type_id)}
        onChange={(e) => {
          const checked = e.currentTarget.checked;
          setTypes((current) => checked ? [...current, seg.type_id] : current.filter((value) => value !== seg.type_id));
        }} /> {seg.name} {seg.representation === "semantic" ? "(semantic mask)" : ""}</label>)}
      {!options.length && <p>Run a segmentation on an image before measuring this experiment.</p>}
    </fieldset>
    <div className="flex flex-wrap gap-4">
      <div className="flex items-center gap-2">
        <label><input type="checkbox" checked={complete} onChange={(e) => setComplete(e.target.checked)} /> Require review marked complete</label>
        <HelpDisclosure label="review requirements" title="Unfinished reviews">
          Only confirmed objects are counted. Including unfinished reviews may give lower counts.
        </HelpDisclosure>
      </div>
      <label><input type="checkbox" checked={reuse} onChange={(e) => setReuse(e.target.checked)} /> Reuse matching current results</label>
    </div>
    <div className="flex flex-wrap gap-2">
      <Button disabled={busy || !selectedIds.length || !types.length} onClick={() => void run(true)}>Check readiness</Button>
      <Button variant="primary" disabled={busy || previewKey !== payloadKey || !preview?.some((item) => item.ready)} onClick={() => void run(false)}>Run ready images</Button>
    </div>
    {error && <p className="text-red-700" role="alert">{error}</p>}
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
        <td className="border-b p-2">{options.find((seg) => seg.type_id === item.identity.segmentation_type_id)?.name ?? item.identity.structure ?? item.identity.organelle}</td><td className="border-b p-2">{item.identity.measurement_area}</td>
        <td className="border-b p-2">{item.identity.sample_id || item.identity.image} / {item.identity.group || "Ungrouped"}</td>
        <td className="border-b p-2">{item.ready === true ? "Ready" : item.ready === false ? "Skipped" : item.status}{item.reused ? " · reused" : ""}{item.current === false ? " · outdated" : ""}</td>
        <td className="border-b p-2">{selectedMetric && item.metrics?.[selectedMetric] != null ? item.metrics[selectedMetric]?.toLocaleString(undefined, { maximumSignificantDigits: 6 }) : "—"}</td>
        <td className="border-b p-2">{item.reason}</td>
      </tr>)}</tbody></table></div>}
  </section>;
}
