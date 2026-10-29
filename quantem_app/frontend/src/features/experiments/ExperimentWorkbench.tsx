import { BulkModelPanel } from "./BulkModelPanel";
import { FineTuneDialog } from "@/features/finetune/FineTuneDialog";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, useLocation, useSearchParams } from "react-router-dom";
import { getWorkspace, type WorkspaceImage, type WorkspaceSegmentation } from "@/shared/api/workspace";
import { getAssetNgffThumbnailUrl } from "@/shared/api/assets";
import type { Experiment } from "@/shared/types/common";
import { Button } from "@/shared/ui/design";
import { SavedModelsPanel } from "@/features/models/SavedModelsPanel";
import { BatchAnalysisPanel } from "./BatchAnalysisPanel";
import { AnalysisHistory } from "./AnalysisHistory";
import { readWorkspaceMemory as readMemory, rememberWorkspace as writeMemory } from "./workspaceMemory";
import { extractApiErrorMessage } from "@/utils/apiErrors";

interface Controls { query: string; dataset: string; sort: string; layout: string; readiness: string; selected: string[] }
const DEFAULTS: Controls = { query: "", dataset: "", sort: "name", layout: "table", readiness: "", selected: [] };

function ImageStatus({ stage }: { stage: string }) {
  if (stage === "DONE") return null;
  const status = `Status: ${stage}`;
  return <span className="inline-flex shrink-0 rounded focus-visible:outline focus-visible:outline-2 focus-visible:outline-amber-600"
    role="img" tabIndex={0} aria-label={status} title={status}>
    <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M10.3 3.9a2 2 0 0 1 3.4 0l8 14A2 2 0 0 1 20 21H4a2 2 0 0 1-1.7-3.1z" fill="#facc15" stroke="#a16207" strokeWidth="1.5" />
      <path d="M12 8v6" stroke="#713f12" strokeWidth="2" strokeLinecap="round" />
      <circle cx="12" cy="17" r="1" fill="#713f12" />
    </svg>
  </span>;
}

function ImageThumbnail({ image }: { image: WorkspaceImage }) {
  const [failed, setFailed] = useState(false);
  return <Link to={`/assets/${image.id}`} aria-label={`Open ${image.name}`} className="flex h-12 w-16 items-center justify-center overflow-hidden rounded bg-slate-100">
    {failed ? <svg className="text-slate-400" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2" /><path d="m3 17 6-6 4 4 3-3 5 5" /><circle cx="16" cy="8" r="1" /></svg> :
      <img className="h-full w-full object-contain" src={getAssetNgffThumbnailUrl(image.id)} alt="" loading="lazy" onError={() => setFailed(true)} />}
  </Link>;
}

function StructureLinks({ image }: { image: WorkspaceImage }) {
  const structures = new Map<string, WorkspaceSegmentation>();
  for (const seg of image.segmentations) {
    if (seg.type === "quantem_internal_analysis_mask" || !seg.coverage || seg.coverage === "none") continue;
    const previous = structures.get(seg.type);
    if (!previous || seg.coverage === "whole") structures.set(seg.type, seg);
  }
  return <div className="flex flex-wrap gap-x-3 gap-y-1">{[...structures.values()].map((seg) => {
    const count = seg.annotated_area_count ?? 0;
    const percent = seg.annotated_percent;
    const tooltip = seg.coverage === "whole" ? "Whole-image segmentation" : count ?
      `${count} annotated ${count === 1 ? "area" : "areas"}${percent != null ? ` (${percent.toLocaleString(undefined, { maximumFractionDigits: 1 })}% of image area)` : ""}` :
      "Manual annotations; no annotated areas marked";
    return <Link key={seg.type} className={`font-medium underline decoration-dotted underline-offset-4 ${seg.coverage === "partial" ? "text-yellow-700" : "text-cyan-800"}`}
      title={tooltip} aria-label={`${seg.short_name || seg.name}: ${tooltip}`} to={`/assets/${image.id}/labeling/${seg.type}?seg=${seg.id}`}>{seg.short_name || seg.name}</Link>;
  })}</div>;
}

export function ExperimentWorkbench({ experiment, children, refreshKey, onAddImages }: {
  experiment: Experiment; children: ReactNode; refreshKey: number;
  onAddImages?: () => void;
}) {
  const key = `${experiment.id}.controls`;
  const [controls, setControls] = useState<Controls>(() => ({ ...DEFAULTS, ...readMemory<Partial<Controls>>(key, {}) }));
  const [images, setImages] = useState<WorkspaceImage[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [newAnalysis, setNewAnalysis] = useState(false);
  const [applying, setApplying] = useState(false);
  const [fineTuning, setFineTuning] = useState(false);
  const [historyRevision, setHistoryRevision] = useState(0);
  const historyChanged = useCallback(() => setHistoryRevision((value) => value + 1), []);
  const [page, setPage] = useState(() => readMemory<number>(`${experiment.id}.page`, 0));
  const [params, setParams] = useSearchParams();
  const { key: navigationKey } = useLocation();
  const tab = params.get("tab") ?? "images";
  const restored = useRef(false);
  const change = (patch: Partial<Controls>) => {
    setControls((current) => ({ ...current, ...patch }));
    if (Object.keys(patch).some((entry) => entry !== "selected")) setPage(0);
  };
  const load = useCallback(async () => {
    setLoading(true);
    try { const response = await getWorkspace(experiment.id); setImages(response.images); setError(""); }
    catch (cause) { setError(extractApiErrorMessage(cause, "The experiment's images could not be loaded.")); }
    finally { setLoading(false); }
  }, [experiment.id]);
  useEffect(() => { void load(); }, [load, refreshKey, historyRevision]);
  useEffect(() => { writeMemory(key, controls); }, [controls, key]);
  useEffect(() => { writeMemory(`${experiment.id}.page`, page); }, [page, experiment.id]);
  useEffect(() => {
    if (loading || restored.current) return;
    restored.current = true;
    const frame = requestAnimationFrame(() => window.scrollTo(0, readMemory<number>(`${experiment.id}.scroll`, 0)));
    return () => cancelAnimationFrame(frame);
  }, [loading, experiment.id]);
  useEffect(() => {
    const save = () => writeMemory(`${experiment.id}.scroll`, window.scrollY);
    window.addEventListener("scroll", save, { passive: true });
    return () => window.removeEventListener("scroll", save);
  }, [experiment.id]);
  const filtered = useMemo(() => images.filter((image) =>
    image.name.toLowerCase().includes(controls.query.toLowerCase()) &&
    (!controls.dataset || (controls.dataset === "ungrouped" ? !image.datasets.length : image.datasets.some((dataset) => dataset.id === controls.dataset))) &&
    (!controls.readiness || (controls.readiness === "uncalibrated" ? !image.pixel_size_nm :
      controls.readiness === "reviewed" ? image.segmentations.some((seg) => seg.review_complete) :
      !image.segmentations.some((seg) => seg.review_complete)))
  ).sort((a, b) => controls.sort === "name-desc" ? b.name.localeCompare(a.name) :
    controls.sort === "sample" ? a.sample_id.localeCompare(b.sample_id) || a.name.localeCompare(b.name) :
    a.name.localeCompare(b.name)), [images, controls.query, controls.dataset, controls.readiness, controls.sort]);
  useEffect(() => { writeMemory(`${experiment.id}.imageOrder`, filtered.map((image) => image.id)); }, [filtered, experiment.id]);
  const pageCount = Math.max(1, Math.ceil(filtered.length / 50));
  const visible = filtered.slice(Math.min(page, pageCount - 1) * 50, (Math.min(page, pageCount - 1) + 1) * 50);
  const selectedIds = useMemo(() => controls.selected.filter((id) => images.some((image) => image.id === id)), [controls.selected, images]);
  const chooseTab = (next: string) => setParams(next === "images" ? {} : { tab: next });
  return <div className="space-y-4">
    <div className="flex flex-wrap items-center gap-2 border-b pb-3">
      <nav className="flex gap-2" aria-label="Experiment sections">{["images", "analysis", "models"].map((entry) =>
        <Button key={entry} variant={tab === entry ? "primary" : "secondary"} aria-current={tab === entry ? "page" : undefined}
          onClick={() => chooseTab(entry)}>{entry[0].toUpperCase() + entry.slice(1)}</Button>)}</nav>
      <Button size="icon" className="ml-auto shrink-0" aria-label="Refresh status" title="Refresh" onClick={historyChanged}>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M3 12a9 9 0 0 1 15-6.7L21 8M21 3v5h-5M21 12a9 9 0 0 1-15 6.7L3 16M8 16H3v5" />
        </svg>
      </Button>
    </div>
    {error && <p role="alert" className="text-red-700">{error}</p>}
    {tab === "images" && <>
      <p className="text-sm text-slate-600">Organize → Evaluate → Apply broadly → Measure. Annotate examples and fine-tune only when the default results need improvement.</p>
      <div className="flex flex-wrap items-center gap-2">
        <strong>{selectedIds.length} selected</strong>
        <Button disabled={!filtered.length} onClick={() => change({ selected: [...new Set([...selectedIds, ...filtered.map((image) => image.id)])] })}>Select all filtered images ({filtered.length})</Button>
        <Button disabled={!selectedIds.length} onClick={() => change({ selected: [] })}>Clear selection</Button>
        <Button disabled={!selectedIds.length} onClick={() => setApplying(!applying)}>Apply models…</Button>
        <Button disabled={!selectedIds.length} onClick={() => setFineTuning(true)}>Evaluate / Fine-tune…</Button>
        <Button disabled={!selectedIds.length} onClick={() => { chooseTab("analysis"); setNewAnalysis(true); }}>Measure selected…</Button>
        <Button className="ml-auto shadow-md" variant="primary" onClick={onAddImages}>
          <svg width="17" height="17" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
            <path d="M10 4v12M4 10h12" />
          </svg>
          Add images
        </Button>
      </div>
      {applying && <BulkModelPanel experimentId={experiment.id} selectedIds={selectedIds} onChanged={historyChanged} />}
      <FineTuneDialog initialMode="evaluate" open={fineTuning} onClose={() => setFineTuning(false)} initialAssetIds={selectedIds} initialExperimentId={experiment.id} />
      <div className="flex flex-wrap items-end gap-3 rounded-lg border bg-white p-4">
        {controls.layout === "table" && <>
        <label>Search images <input className="block rounded border px-2 py-1" type="search" value={controls.query} onChange={(e) => change({ query: e.target.value })} /></label>
        <label>Dataset <select className="block rounded border px-2 py-1" value={controls.dataset} onChange={(e) => change({ dataset: e.target.value })}>
          <option value="">All datasets</option><option value="ungrouped">Ungrouped</option>{experiment.datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}</select></label>
        <label>Image order <select className="block rounded border px-2 py-1" value={controls.sort} onChange={(e) => change({ sort: e.target.value })}>
          <option value="name">Name A–Z</option><option value="name-desc">Name Z–A</option><option value="sample">Sample ID</option></select></label>
        <label>Readiness <select className="block rounded border px-2 py-1" value={controls.readiness} onChange={(e) => change({ readiness: e.target.value })}>
          <option value="">All images</option><option value="uncalibrated">Missing resolution</option><option value="reviewed">Whole image segmented</option><option value="unreviewed">Needs full-image coverage</option></select></label>
        </>}
        <label>View <select className="block rounded border px-2 py-1" value={controls.layout} onChange={(e) => change({ layout: e.target.value })}>
          <option value="table">Workflow table</option><option value="cards">Dataset cards</option></select></label>
      </div>
      {controls.layout === "cards" ? children :
        loading ? <p role="status">Loading workflow status…</p> :
        <div className="overflow-x-auto rounded border bg-white"><table className="w-full text-left text-sm">
          <thead><tr><th scope="col" className="p-3">Select</th><th scope="col" aria-label="Thumbnail" className="border-b p-3" />{["Image", "Dataset", "Resolution", "Segmentations", "Analyses"].map((label) => <th scope="col" key={label} className="border-b p-3">{label}</th>)}</tr></thead>
          <tbody>{visible.map((image) => <tr key={image.id}>
            <td className="border-b p-3"><input type="checkbox" aria-label={`Select ${image.name}`} checked={selectedIds.includes(image.id)} onChange={(e) => change({ selected: e.target.checked ? [...selectedIds, image.id] : selectedIds.filter((id) => id !== image.id) })} /></td>
            <td className="border-b p-3"><ImageThumbnail image={image} /></td>
            <td className="border-b p-3"><div className="flex items-center gap-2"><Link className="font-medium text-cyan-800 underline" to={`/assets/${image.id}`}>{image.name}</Link><ImageStatus stage={image.stage} /></div></td>
            <td className="border-b p-3">{image.datasets.map((dataset) => dataset.name).join(", ") || "Ungrouped"}</td>
            <td className="border-b p-3">{image.pixel_size_nm ? `${image.pixel_size_nm.toFixed(2)} nm/px` : "Missing"}</td>
            <td className="border-b p-3"><StructureLinks image={image} /></td>
            <td className="border-b p-3">{image.analysis_count ? <Link className="font-medium text-cyan-800 underline" aria-label={`${image.analysis_count} ${image.analysis_count === 1 ? "analysis" : "analyses"} for ${image.name}`} to={`/experiments/${experiment.id}?tab=analysis&image=${image.id}`}>{image.analysis_count}</Link> : 0}</td>
          </tr>)}</tbody>
        </table>{!visible.length && !error && <p className="p-4">No images match these filters.</p>}
        <div className="flex items-center justify-between p-3"><Button disabled={page === 0} onClick={() => setPage(page - 1)}>Previous page</Button><span>Page {Math.min(page, pageCount - 1) + 1} of {pageCount} · {filtered.length} images</span><Button disabled={page >= pageCount - 1} onClick={() => setPage(page + 1)}>Next page</Button></div></div>}
    </>}
    {tab === "analysis" && <>
      {!params.has("image") && <Button variant="primary" aria-expanded={newAnalysis} onClick={() => setNewAnalysis(!newAnalysis)}>{newAnalysis ? "Close new analysis" : "New analysis"}</Button>}
      {newAnalysis && !params.has("image") && <BatchAnalysisPanel experimentId={experiment.id} images={images} selectedIds={selectedIds} onSelectionChange={(ids) => change({ selected: ids })} onHistoryChange={historyChanged} onMetadataSaved={historyChanged} />}
      <AnalysisHistory experimentId={experiment.id} images={images} refreshKey={historyRevision} />
    </>}
    {tab === "models" && <SavedModelsPanel experimentId={experiment.id} refreshKey={`${navigationKey}:${historyRevision}`} />}
  </div>;
}
