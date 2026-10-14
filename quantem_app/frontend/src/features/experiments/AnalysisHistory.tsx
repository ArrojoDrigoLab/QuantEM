import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { getAnalysisBatch, getAnalysisHistory, type AnalysisBatch, type AnalysisHistoryEntry, type BatchItem } from "@/shared/api/analysisHistory";
import { getAnalysisExportUrl, getAnalysisRun } from "@/shared/api/analysis";
import { apiRequest, resolveApiUrl } from "@/shared/api/core/http";
import type { WorkspaceImage } from "@/shared/api/workspace";
import type { AnalysisRun } from "@/shared/types/analysis";
import { Button } from "@/shared/ui/design";
import { formatTimestamp } from "@/shared/ui/format";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import { AnalysisResults } from "@/features/analysis/components/AnalysisResults";
import { AnalysisNotes } from "@/features/analysis/components/AnalysisNotes";
import { AnalysisSettings } from "./AnalysisSettings";
import { ImageAnalysisResultsDialog } from "./ImageAnalysisResultsDialog";

const statusName: Record<string, string> = { SUCCESS: "Complete", PARTIAL: "Partly complete", PENDING: "Queued", RUNNING: "Running", FAILED: "Failed", CANCELLED: "Cancelled", SKIPPED: "Skipped" };

function HistoryDetails({ entry, images, onChanged }: { entry: AnalysisHistoryEntry; images: WorkspaceImage[]; onChanged: () => void }) {
  const [detail, setDetail] = useState<AnalysisBatch | AnalysisRun | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [revision, setRevision] = useState(0);
  const [selectedItem, setSelectedItem] = useState<BatchItem | null>(null);
  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const refresh = async () => {
      try {
        const result = entry.kind === "batch" ? await getAnalysisBatch(entry.id) : await getAnalysisRun(entry.id);
        if (cancelled) return;
        setDetail(result);
        const active = "items" in result ? result.items.some((item) => ["PENDING", "RUNNING"].includes(item.status ?? "")) : ["PENDING", "RUNNING"].includes(result.status);
        if (active) timer = window.setTimeout(() => void refresh(), 5000);
      } catch (cause) { if (!cancelled) setError(extractApiErrorMessage(cause, "This analysis could not be loaded.")); }
    };
    void refresh();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [entry.id, entry.kind, revision]);
  const act = async (action: "retry_failed" | "cancel") => {
    setBusy(true); setError("");
    try {
      setDetail(await apiRequest<AnalysisBatch>(`/api/analysis-batches/${entry.id}/`, { method: "POST", body: JSON.stringify({ action }) }));
      setRevision((value) => value + 1); onChanged();
    } catch (cause) { setError(extractApiErrorMessage(cause, "This analysis could not be updated.")); }
    finally { setBusy(false); }
  };
  if (error && !detail) return <p role="alert" className="text-red-700">{error}</p>;
  if (!detail) return <p role="status">Loading analysis…</p>;
  if ("items" in detail) return <div className="space-y-4">
    <div className="flex flex-wrap gap-4">
      <a className="inline-flex h-10 items-center justify-center rounded-md border border-slate-300 bg-white px-4 text-sm font-medium text-slate-800 shadow-sm transition-colors hover:border-slate-400 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2" href={resolveApiUrl(`/api/analysis-batches/${detail.id}/export/?objects=true`)} download>Download</a>
      {detail.items.some((item) => ["FAILED", "CANCELLED"].includes(item.status ?? "")) && <Button disabled={busy} onClick={() => void act("retry_failed")}>Retry failed</Button>}
      {detail.items.some((item) => ["PENDING", "RUNNING"].includes(item.status ?? "")) && <Button disabled={busy} onClick={() => void act("cancel")}>Cancel remaining</Button>}
    </div>
    {error && <p role="alert" className="text-red-700">{error}</p>}
    <details><summary className="cursor-pointer font-medium">Analysis settings</summary><div className="mt-3"><AnalysisSettings values={detail.settings} images={images} /></div></details>
    <div className="overflow-x-auto"><table className="w-full text-left text-sm">
      <thead><tr>{["Image", "Structure", "Area", "Status", "Results"].map((name) => <th className="border-b p-2" key={name}>{name}</th>)}</tr></thead>
      <tbody>{detail.items.map((item, index) => {
        const structure = images.flatMap((image) => image.segmentations).find((seg) => seg.type === item.identity.organelle)?.name ?? item.identity.organelle;
        return <tr key={index}>
        <td className="border-b p-2">{item.identity.image}</td>
        <td className="border-b p-2">{structure}</td>
        <td className="border-b p-2">{item.identity.measurement_area}</td>
        <td className="border-b p-2">{statusName[item.status ?? ""] ?? item.status}{item.current === false ? " · inputs changed" : ""}{item.reason && <p>{item.reason}</p>}</td>
        <td className="border-b p-2"><Button size="sm" aria-label={`View results for ${item.identity.image} · ${structure}`} onClick={() => setSelectedItem(item)}>View results</Button></td>
      </tr>; })}</tbody>
    </table></div>
    <AnalysisNotes notes={detail.items.flatMap((item) => (item.notes ?? []).map((note) => `${item.identity.image}: ${note}`))} />
    {selectedItem && <ImageAnalysisResultsDialog item={selectedItem} images={images} onClose={() => setSelectedItem(null)} />}
  </div>;
  return <div className="space-y-4">
    <div className="flex flex-wrap gap-4">{detail.exports.map((file) => <a key={file} className="text-cyan-800 underline" href={getAnalysisExportUrl(detail.id, file)} download>{file}</a>)}</div>
    <details><summary className="cursor-pointer font-medium">Analysis settings</summary><div className="mt-3"><AnalysisSettings values={detail.params as unknown as Record<string, unknown>} images={images} /></div></details>
    {detail.error && <p role="alert" className="text-red-700">{detail.error}</p>}
    <AnalysisResults run={detail} />
  </div>;
}

export function AnalysisHistory({ experimentId, images, refreshKey = 0 }: { experimentId: string; images: WorkspaceImage[]; refreshKey?: number }) {
  const [params] = useSearchParams();
  const imageId = params.get("image") ?? undefined;
  const [entries, setEntries] = useState<AnalysisHistoryEntry[]>([]);
  const [open, setOpen] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const reload = () => { void getAnalysisHistory(experimentId, imageId).then(setEntries).catch((cause) => setError(extractApiErrorMessage(cause, "Analysis history could not be refreshed."))); };
  useEffect(() => {
    let cancelled = false;
    setLoading(true); setError(""); setOpen("");
    void getAnalysisHistory(experimentId, imageId).then((rows) => {
      if (!cancelled) setEntries(rows);
    }).catch((cause) => { if (!cancelled) setError(extractApiErrorMessage(cause, "Analysis history could not be loaded.")); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [experimentId, imageId, refreshKey]);
  const active = entries.some((entry) => ["PENDING", "RUNNING"].includes(entry.status));
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    const timer = window.setInterval(() => {
      void getAnalysisHistory(experimentId, imageId).then((rows) => { if (!cancelled) setEntries(rows); })
        .catch((cause) => { if (!cancelled) setError(extractApiErrorMessage(cause, "Analysis history could not be refreshed.")); });
    }, 5000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [active, experimentId, imageId]);
  const image = images.find((entry) => entry.id === imageId);
  return <section className="space-y-3" aria-label="Previous analyses">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <h2 className="text-lg font-semibold">{imageId ? `Analyses for ${image?.name ?? "image"}` : "Previous analyses"}</h2>
      {imageId && <Link className="text-cyan-800 underline" to={`/experiments/${experimentId}?tab=analysis`}>All experiment analyses</Link>}
    </div>
    {error && <p role="alert" className="text-red-700">{error}</p>}
    {loading ? <p role="status">Loading analyses…</p> : !entries.length && !error ? <p>No saved analyses.</p> :
      <div className="divide-y rounded border bg-white">{entries.map((entry) => {
        const key = `${entry.kind}:${entry.id}`;
        return <article key={key}>
          <div className="overflow-x-auto"><Button className="!flex !w-full min-w-[660px] !justify-start !rounded-none !border-0 !bg-transparent !p-3 !text-left" aria-expanded={open === key} onClick={() => setOpen(open === key ? "" : key)}>
            <span aria-hidden="true">{open === key ? "▾" : "▸"}</span>
            <span className="min-w-0 flex-1 truncate font-medium" title={entry.name}>{entry.name}</span>
            <span className="shrink-0 text-xs text-slate-600">{formatTimestamp(entry.created_at)}</span>
            <span className="shrink-0 text-xs">{entry.images.length} {entry.images.length === 1 ? "image" : "images"} · {entry.structures.join(", ")} · {statusName[entry.status] ?? entry.status}</span>
          </Button></div>
          {open === key && <div className="space-y-3 border-t p-4"><h3 className="font-semibold">{entry.name}</h3><p className="text-sm">{entry.measurement_area} · {entry.measurement_count} measurements</p><HistoryDetails entry={entry} images={images} onChanged={reload} /></div>}
        </article>;
      })}</div>}
  </section>;
}
