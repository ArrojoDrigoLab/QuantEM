import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { getAnalysisExportUrl, getAnalysisRun } from "@/shared/api/analysis";
import type { BatchItem } from "@/shared/api/analysisHistory";
import type { WorkspaceImage } from "@/shared/api/workspace";
import type { AnalysisRun } from "@/shared/types/analysis";
import { Button } from "@/shared/ui/design";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import { AnalysisTablePreview } from "@/features/analysis/components/AnalysisTablePreview";
import { AnalysisSettings } from "./AnalysisSettings";
import { SegmentationSettings } from "@/features/analysis/components/SegmentationSettings";
import { metricLabel } from "@/features/analysis/metricLabel";

const TABS = ["Spreadsheet", "Image averages", "Settings"] as const;
type Tab = typeof TABS[number];

export function ImageAnalysisResultsDialog({ item, images, onClose }: {
  item: BatchItem; images: WorkspaceImage[]; onClose: () => void;
}) {
  const id = useId();
  const [tab, setTab] = useState<Tab>("Spreadsheet");
  const [run, setRun] = useState<AnalysisRun | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const panelRef = useRef<HTMLElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const tabRefs = useRef<Partial<Record<Tab, HTMLButtonElement | null>>>({});
  const onCloseRef = useRef(onClose);
  useEffect(() => { onCloseRef.current = onClose; }, [onClose]);

  useEffect(() => {
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") { event.preventDefault(); onCloseRef.current(); return; }
      if (event.key !== "Tab") return;
      const elements = [...(panelRef.current?.querySelectorAll<HTMLElement>('button:not([disabled]), a[href], select:not([disabled]), [tabindex="0"]') ?? [])]
        .filter((element) => element.tabIndex >= 0 && !element.closest("[hidden]"));
      const first = elements[0];
      const last = elements.at(-1);
      if (!panelRef.current?.contains(document.activeElement)) { event.preventDefault(); (event.shiftKey ? last : first)?.focus(); }
      else if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("keydown", handleKey);
      document.body.style.overflow = previousOverflow;
      previousFocus?.focus({ preventScroll: true });
    };
  }, []);

  useEffect(() => {
    if (!item.run_id) return;
    let cancelled = false;
    setError("");
    void getAnalysisRun(item.run_id)
      .then((result) => { if (!cancelled) setRun(result); })
      .catch((cause) => { if (!cancelled) setError(extractApiErrorMessage(cause, "The saved results could not be loaded.")); });
    return () => { cancelled = true; };
  }, [item.run_id, retry]);

  const structure = images.flatMap((image) => image.segmentations).find((seg) => seg.type === item.identity.organelle)?.name ?? item.identity.organelle;
  const metrics = Object.entries(item.metrics ?? {});
  return createPortal(<div className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-950/45 p-3 sm:p-5" onClick={onClose}>
    <section ref={panelRef} role="dialog" aria-modal="true" aria-labelledby={`${id}-title`} onClick={(event) => event.stopPropagation()}
      className="flex h-[85vh] max-h-[900px] w-full max-w-6xl flex-col overflow-hidden rounded-xl bg-white shadow-2xl">
      <header className="flex shrink-0 items-center justify-between gap-3 border-b px-4 py-3">
        <h2 id={`${id}-title`} className="text-lg font-semibold">{item.identity.image} · {structure}</h2>
        <button ref={closeRef} type="button" onClick={onClose} aria-label="Close results" className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md text-slate-600 hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-cyan-500">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="m6 6 12 12M6 18 18 6" /></svg>
        </button>
      </header>
      <div role="tablist" aria-label="Image results" className="flex shrink-0 gap-1 overflow-x-auto border-b bg-slate-50 px-3 py-2">
        {TABS.map((name, index) => <button key={name} type="button" role="tab" id={`${id}-tab-${index}`} aria-controls={`${id}-panel-${index}`} aria-selected={tab === name} tabIndex={tab === name ? 0 : -1}
          ref={(element) => { tabRefs.current[name] = element; }}
          className={`shrink-0 rounded-md px-3 py-2 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-cyan-500 ${tab === name ? "bg-slate-900 text-white" : "text-slate-700 hover:bg-slate-200"}`}
          onClick={() => setTab(name)} onKeyDown={(event) => {
            const next = event.key === "ArrowRight" ? (index + 1) % TABS.length : event.key === "ArrowLeft" ? (index + TABS.length - 1) % TABS.length : event.key === "Home" ? 0 : event.key === "End" ? TABS.length - 1 : null;
            if (next === null) return;
            event.preventDefault(); setTab(TABS[next]); tabRefs.current[TABS[next]]?.focus();
          }}>{name}</button>)}
      </div>
      <div role="tabpanel" id={`${id}-panel-0`} aria-labelledby={`${id}-tab-0`} hidden={tab !== "Spreadsheet"} className="flex min-h-0 flex-1 flex-col p-4">
        {error ? <div className="flex items-center gap-3"><p role="alert" className="text-red-700">{error}</p><Button size="sm" onClick={() => setRetry((value) => value + 1)}>Retry</Button></div> :
          run ? <AnalysisTablePreview run={run} /> : item.run_id ? <p role="status">Loading results…</p> : <p className="text-sm text-slate-600">No spreadsheet is available for this image.</p>}
      </div>
      <div role="tabpanel" id={`${id}-panel-1`} aria-labelledby={`${id}-tab-1`} hidden={tab !== "Image averages"} className="min-h-0 flex-1 overflow-auto p-4">
        {metrics.length ? <table className="w-full text-left text-sm" aria-label="Image averages">
          <thead><tr><th scope="col" className="border-b px-3 py-2">Measurement</th><th scope="col" className="border-b px-3 py-2">Value</th></tr></thead>
          <tbody>{metrics.map(([metric, value]) => <tr key={metric} className="even:bg-slate-50">
            <th scope="row" className="border-b px-3 py-2 font-normal">{metricLabel(metric)}</th>
            <td className="border-b px-3 py-2 tabular-nums">{value?.toLocaleString(undefined, { maximumSignificantDigits: 8 }) ?? "—"}</td>
          </tr>)}</tbody>
        </table> : <p className="text-sm text-slate-600">No image averages were saved.</p>}
      </div>
      <div role="tabpanel" id={`${id}-panel-2`} aria-labelledby={`${id}-tab-2`} hidden={tab !== "Settings"} className="min-h-0 flex-1 space-y-4 overflow-auto p-4">
        <AnalysisSettings values={{ sample: item.identity.sample_id || item.identity.image, group: item.identity.group, resolution_nm_per_pixel: item.identity.pixel_size_nm, measurement_area: item.identity.measurement_area, ...item.params }} images={images} />
        {run && <SegmentationSettings settings={run.segmentation_settings ?? []} images={images} />}
        {item.run_id && item.status === "SUCCESS" && <a className="inline-flex h-10 items-center rounded-md border border-slate-300 px-4 text-sm font-medium text-slate-800 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-cyan-500" href={getAnalysisExportUrl(item.run_id, "manifest.json")} download>Download settings</a>}
      </div>
    </section>
  </div>, document.body);
}
