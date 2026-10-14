import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { getWorkspace, pinExperiment, saveSample, type WorkspaceImage } from "@/shared/api/workspace";
import type { Experiment } from "@/shared/types/common";
import { Button } from "@/shared/ui/design";
import { SavedModelsPanel } from "@/features/models/SavedModelsPanel";
import { BatchAnalysisPanel } from "./BatchAnalysisPanel";
import { readWorkspaceMemory as readMemory, rememberWorkspace as writeMemory } from "./workspaceMemory";
import { extractApiErrorMessage } from "@/utils/apiErrors";

interface Controls { query: string; dataset: string; sort: string; layout: string; readiness: string; selected: string[] }
const DEFAULTS: Controls = { query: "", dataset: "", sort: "name", layout: "table", readiness: "", selected: [] };
export function ExperimentWorkbench({ experiment, children, refreshKey, onPin }: {
  experiment: Experiment; children: ReactNode; refreshKey: number; onPin: (value: Experiment) => void;
}) {
  const key = `${experiment.id}.controls`;
  const [controls, setControls] = useState<Controls>(() => ({ ...DEFAULTS, ...readMemory<Partial<Controls>>(key, {}) }));
  const [images, setImages] = useState<WorkspaceImage[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [sample, setSample] = useState("");
  const [group, setGroup] = useState("");
  const [saving, setSaving] = useState(false);
  const [page, setPage] = useState(() => readMemory<number>(`${experiment.id}.page`, 0));
  const [params, setParams] = useSearchParams();
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
  useEffect(() => { void load(); }, [load, refreshKey]);
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
  const select = (ids: string[]) => change({ selected: [...new Set([...selectedIds, ...ids])] });
  const chooseTab = (next: string) => setParams(next === "images" ? {} : { tab: next });
  const setMetadata = async () => {
    setSaving(true); setError("");
    try { for (const id of selectedIds) await saveSample(experiment.id, id, sample, group); await load(); }
    catch (cause) { setError(extractApiErrorMessage(cause, "Some sample metadata could not be saved. Refresh and retry.")); }
    finally { setSaving(false); }
  };
  return <div className="space-y-4">
    <div className="flex flex-wrap items-center gap-2 border-b pb-3">
      <nav className="flex gap-2" aria-label="Experiment sections">{["images", "analysis", "models"].map((entry) =>
        <Button key={entry} variant={tab === entry ? "primary" : "secondary"} aria-current={tab === entry ? "page" : undefined}
          onClick={() => chooseTab(entry)}>{entry[0].toUpperCase() + entry.slice(1)}</Button>)}</nav>
      <Button onClick={() => { void pinExperiment(experiment.id, !experiment.pinned).then(onPin).catch(() => setError("The pin could not be saved.")); }}>{experiment.pinned ? "Unpin experiment" : "Pin experiment"}</Button>
      <Button onClick={() => void load()}>Refresh status</Button>
    </div>
    {error && <p role="alert" className="text-red-700">{error}</p>}
    {tab === "images" && <>
      <div className="flex flex-wrap items-end gap-3 rounded-lg border bg-white p-4">
        <label>Search images <input className="block rounded border px-2 py-1" type="search" value={controls.query} onChange={(e) => change({ query: e.target.value })} /></label>
        <label>Dataset <select className="block rounded border px-2 py-1" value={controls.dataset} onChange={(e) => change({ dataset: e.target.value })}>
          <option value="">All datasets</option><option value="ungrouped">Ungrouped</option>{experiment.datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}</select></label>
        <label>Image order <select className="block rounded border px-2 py-1" value={controls.sort} onChange={(e) => change({ sort: e.target.value })}>
          <option value="name">Name A–Z</option><option value="name-desc">Name Z–A</option><option value="sample">Sample ID</option></select></label>
        <label>Readiness <select className="block rounded border px-2 py-1" value={controls.readiness} onChange={(e) => change({ readiness: e.target.value })}>
          <option value="">All images</option><option value="uncalibrated">Missing calibration</option><option value="reviewed">Has completed review</option><option value="unreviewed">No completed review</option></select></label>
        <label>View <select className="block rounded border px-2 py-1" value={controls.layout} onChange={(e) => change({ layout: e.target.value })}>
          <option value="table">Workflow table</option><option value="cards">Dataset cards</option></select></label>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <strong>{selectedIds.length} selected</strong>
        <Button onClick={() => select(visible.map((image) => image.id))}>Select visible ({visible.length})</Button>
        <Button onClick={() => select(filtered.map((image) => image.id))}>Select all matching ({filtered.length})</Button>
        <Button onClick={() => change({ selected: [] })}>Clear selection</Button>
        <Button variant="primary" disabled={!selectedIds.length} onClick={() => chooseTab("analysis")}>Analyze selection</Button>
        <span className="text-xs text-slate-600">Selections persist across filters; each image is included once.</span>
      </div>
      {controls.layout === "cards" ? <><p className="text-sm text-slate-600">Dataset cards show the full experiment for editing. Search, readiness and selection above apply to the workflow table.</p>{children}</> :
        loading ? <p role="status">Loading workflow status…</p> :
        <div className="overflow-x-auto rounded border bg-white"><table className="w-full text-left text-sm">
          <thead><tr>{["Select", "Image", "Dataset", "Calibration", "Prediction / review", "Measurements", "Sample / group"].map((label) => <th key={label} className="border-b p-3">{label}</th>)}</tr></thead>
          <tbody>{visible.map((image) => <tr key={image.id}>
            <td className="border-b p-3"><input aria-label={`Select ${image.name}`} type="checkbox" checked={selectedIds.includes(image.id)} onChange={(e) => e.target.checked ? select([image.id]) : change({ selected: selectedIds.filter((id) => id !== image.id) })} /></td>
            <td className="border-b p-3"><Link className="font-medium text-cyan-800 underline" to={`/assets/${image.id}`}>{image.name}</Link><p className="text-xs text-slate-500">{image.stage}</p></td>
            <td className="border-b p-3">{image.datasets.map((dataset) => dataset.name).join(", ") || "Ungrouped"}</td>
            <td className="border-b p-3">{image.pixel_size_nm ? `${image.pixel_size_nm} nm/px` : "Missing"}</td>
            <td className="border-b p-3">{image.segmentations.filter((seg) => seg.type !== "quantem_internal_analysis_mask").map((seg) => <p key={seg.id}>
              <Link className="text-cyan-800 underline" to={`/assets/${image.id}/labeling/${seg.type}?seg=${seg.id}`}>{seg.name}</Link>: {seg.stage.toLowerCase().replaceAll("_", " ")} · {seg.review_complete ? "review complete" : "review incomplete"}{seg.measurement_mode !== "global" ? ` · ${seg.confirmed_count} confirmed` : " · area only"}
            </p>)}{!image.segmentations.length && "No segmentation"}</td>
            <td className="border-b p-3">{image.segmentations.filter((seg) => seg.latest_analysis).map((seg) => <p key={seg.id}>{seg.name}: {seg.latest_analysis?.current === false ? "outdated" : seg.latest_analysis?.current === null ? "unverified" : seg.latest_analysis?.status.toLowerCase()}</p>)}
              <Link className="text-cyan-800 underline" to={`/assets/${image.id}/analysis`}>Open analysis</Link></td>
            <td className="border-b p-3">{image.sample_id || image.name} / {image.group || "Ungrouped"}</td>
          </tr>)}</tbody>
        </table>{!visible.length && !error && <p className="p-4">No images match these filters.</p>}
        <div className="flex items-center justify-between p-3"><Button disabled={page === 0} onClick={() => setPage(page - 1)}>Previous page</Button><span>Page {Math.min(page, pageCount - 1) + 1} of {pageCount} · {filtered.length} images</span><Button disabled={page >= pageCount - 1} onClick={() => setPage(page + 1)}>Next page</Button></div></div>}
      <form className="flex flex-wrap items-end gap-3 rounded border bg-white p-4" onSubmit={(e) => { e.preventDefault(); void setMetadata(); }}>
        <div className="w-full"><strong>Group selected images</strong><p className="text-sm text-slate-600">Sample IDs default to image names. Use a shared ID to average images from the same sample together.</p></div>
        <label>Sample ID <input className="block rounded border px-2 py-1" maxLength={255} placeholder="Image name by default" value={sample} onChange={(e) => setSample(e.target.value)} /></label>
        <label>Group <input className="block rounded border px-2 py-1" maxLength={255} value={group} onChange={(e) => setGroup(e.target.value)} /></label>
        <Button type="submit" disabled={saving || !selectedIds.length}>Save for {selectedIds.length} images</Button>
      </form>
    </>}
    {tab === "analysis" && <BatchAnalysisPanel experimentId={experiment.id} images={images} selectedIds={selectedIds} />}
    {tab === "models" && <SavedModelsPanel experimentId={experiment.id} />}
  </div>;
}
