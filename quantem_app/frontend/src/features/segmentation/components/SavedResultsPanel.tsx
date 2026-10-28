import { useEffect, useState } from "react";
import { apiRequest, resolveApiUrl } from "@/shared/api/core/http";
import { getAssetNgffThumbnailUrl } from "@/shared/api/assets";
import { Button } from "@/shared/ui/design";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { extractApiErrorMessage } from "@/utils/apiErrors";

type SavedResult = { id: string; name: string; source_model: string; threshold: number; is_canonical: boolean; available: boolean; created_at: string; preview_url: string };

export function SavedResultsPanel({ segmentationId, assetId, revision, onQueued }: {
  segmentationId: string; assetId: string; revision: string; onQueued: () => void;
}) {
  const [results, setResults] = useState<SavedResult[]>([]);
  const [left, setLeft] = useState("");
  const [right, setRight] = useState("");
  const [leftOpacity, setLeftOpacity] = useState(0.3);
  const [rightOpacity, setRightOpacity] = useState(0.3);
  const [compare, setCompare] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [aspectRatio, setAspectRatio] = useState(1);
  const [confirm, setConfirm] = useState<SavedResult | null>(null);
  const [deletions, setDeletions] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [queued, setQueued] = useState("");
  useEffect(() => {
    let stopped = false;
    let timer: number;
    const load = async () => {
      try {
        const response = await apiRequest<{ results: SavedResult[] }>(`/api/segmentations/${segmentationId}/results/`);
        if (stopped) return;
        setResults(response.results);
        if (queued) {
          const job = await apiRequest<{ status: string; message: string }>(`/api/jobs/${queued}/`);
          if (stopped) return;
          if (!["PENDING", "RETRY", "RUNNING"].includes(job.status)) {
            setQueued("");
            if (job.status !== "SUCCESS") setError(job.message || "The result could not be applied. Retry from Tasks & Queues.");
            onQueued();
          }
        }
        setLeft((id) => response.results.some((row) => row.id === id) ? id : response.results.find((row) => row.is_canonical)?.id ?? response.results[0]?.id ?? "");
        setRight((id) => response.results.some((row) => row.id === id) ? id : response.results.find((row) => !row.is_canonical)?.id ?? "");
      } catch (cause) { if (!stopped) setError(extractApiErrorMessage(cause, "Saved results could not be loaded.")); }
      if (!stopped && queued) timer = window.setTimeout(() => void load(), 4000);
    };
    void load();
    return () => { stopped = true; clearTimeout(timer); };
  }, [segmentationId, revision, queued, onQueued]);
  const applyResult = async () => {
    if (!confirm) return;
    setBusy(true); setError("");
    try {
      const response = await apiRequest<{ job_id: string }>(`/api/segmentations/${segmentationId}/results/`, { method: "POST", body: JSON.stringify({ result_id: confirm.id, delete_result_ids: deletions }) });
      setConfirm(null); setQueued(response.job_id); onQueued();
    } catch (cause) { setError(extractApiErrorMessage(cause, "This result could not be selected.")); }
    finally { setBusy(false); }
  };
  const pane = (id: string, setId: (value: string) => void, label: string, opacity: number, setOpacity: (value: number) => void) => {
    const result = results.find((row) => row.id === id);
    return <div className="min-w-0 flex-1 space-y-2" role="group" aria-label={`${label} view`}>
      <div className="flex items-end gap-2">
        <label className="min-w-0 flex-1 text-sm">{label}
          <select className="block w-full rounded border p-1" value={id} onChange={(e) => setId(e.target.value)}>
            {results.map((row) => <option key={row.id} value={row.id}>{row.is_canonical ? "✓ " : ""}{row.name}{row.is_canonical ? " · Confirmed" : ""} · {new Date(row.created_at).toLocaleString()}</option>)}
          </select>
        </label>
        {result && !result.is_canonical && <Button className="shrink-0" disabled={busy || !!queued || !result.available} onClick={() => { setConfirm(result); setDeletions(results.filter((row) => row.id !== result.id).map((row) => row.id)); }}>Confirm this model</Button>}
      </div>
      {result && <>
        <label className="flex items-center gap-2 text-sm">Opacity
          <input aria-label={`${label} opacity`} type="range" min="0" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
          <span>{Math.round(opacity * 100)}%</span>
        </label>
        <div className="relative h-80 overflow-auto rounded bg-slate-900" onScroll={(event) => {
          const siblings = event.currentTarget.parentElement?.parentElement?.querySelectorAll<HTMLElement>("[data-comparison-image]");
          siblings?.forEach((node) => { if (node !== event.currentTarget) { node.scrollLeft = event.currentTarget.scrollLeft; node.scrollTop = event.currentTarget.scrollTop; } });
        }} data-comparison-image>
          <div className="relative mx-auto" style={{ width: `min(${zoom * 100}%, ${320 * aspectRatio * zoom}px)` }}>
            <img className="block w-full" src={getAssetNgffThumbnailUrl(assetId)} alt="Electron microscopy image" onLoad={(event) => { const image = event.currentTarget; if (image.naturalHeight) setAspectRatio(image.naturalWidth / image.naturalHeight); }} />
            <img className="absolute inset-0 h-full w-full" style={{ opacity }} src={resolveApiUrl(result.preview_url)} alt={`${result.name} prediction overlay`} />
          </div>
        </div>
      </>}
    </div>;
  };
  if (!results.length && !error) return null;
  return <details className="m-2 min-h-0 max-h-[60vh] overflow-auto rounded border bg-white p-3">
    <summary className="cursor-pointer font-medium">Model results · {results.length} saved {results.length === 1 ? "version" : "versions"}</summary>
    <p className="my-2 text-sm">Switch predictions to compare them. A check mark identifies the confirmed model used for analysis. Manually annotated areas in ROIs marked as ‘Done’ override its outputs.</p>
    <div className="my-2 flex gap-4">
      <label><input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} disabled={results.length < 2} /> Side by side</label>
      <label>Overview zoom <input type="range" min="1" max="6" step="0.25" value={zoom} onChange={(e) => setZoom(Number(e.target.value))} /></label>
    </div>
    <div className="flex flex-col gap-3 sm:flex-row">{pane(left, setLeft, "Prediction", leftOpacity, setLeftOpacity)}{compare && pane(right, setRight, "Compare with", rightOpacity, setRightOpacity)}</div>
    <p className="mt-2 text-xs text-slate-600">Overview previews show thresholded probability maps. Measurements use the full-resolution extracted result and completed manual areas.</p>
    {queued && <p role="status">Applying the selected result…</p>}
    {error && <p role="alert" className="text-red-700">{error}</p>}
    <ConfirmDialog isOpen={!!confirm} title="Confirm this model" message="Use this model's outputs as the default for analysis? Your manually annotated areas will still override model outputs in ROIs marked as 'Done'."
      details={<fieldset disabled={busy}><legend>Delete old model outputs</legend>{results.filter((row) => row.id !== confirm?.id).map((row) => <label className="block" key={row.id}>
        <input type="checkbox" checked={deletions.includes(row.id)} onChange={(e) => setDeletions((ids) => e.target.checked ? [...ids, row.id] : ids.filter((id) => id !== row.id))} /> {row.name}
      </label>)}<p>Clear a checkbox to keep that version for comparison. Checked outputs are deleted only after the selected model's outputs are successfully applied.</p>{error && <p role="alert">{error}</p>}</fieldset>}
      confirmText={busy ? "Queuing…" : "Confirm"} confirmDisabled={busy} cancelText="Cancel" onCancel={() => { if (!busy) setConfirm(null); }} onConfirm={() => { if (!busy) void applyResult(); }} />
  </details>;
}
