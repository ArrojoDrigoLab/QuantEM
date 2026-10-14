import { useEffect, useRef, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { getAsset, getAssetSegmentations, getExperiment } from "@/shared/api/assets";
import { getNavigationImages, recordVisit, type WorkspaceImage } from "@/shared/api/workspace";
import type { Experiment } from "@/shared/types/common";
import type { AssetDetail, SegmentationType } from "@/shared/types/images";
import { FineTuneDialog } from "@/features/finetune/FineTuneDialog";
import { JobQueueSidebar } from "@/features/library/components/JobQueueSidebar";
import { TaskQueueButton } from "@/features/library/components/TaskQueueButton";
import { Button } from "@/shared/ui/design";
import { readWorkspaceMemory } from "@/features/experiments/workspaceMemory";
import { CorrectionHistory } from "@/features/experiments/CorrectionHistory";

export function WorkspaceBar() {
  const location = useLocation();
  const assetId = /^\/assets\/([^/]+)/.exec(location.pathname)?.[1];
  const routeExperiment = /^\/experiments\/([^/]+)/.exec(location.pathname)?.[1];
  const [asset, setAsset] = useState<AssetDetail | null>(null);
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [images, setImages] = useState<Pick<WorkspaceImage, "id" | "name">[]>([]);
  const [type, setType] = useState<SegmentationType | null>(null);
  const [fineTune, setFineTune] = useState(false);
  const [tasks, setTasks] = useState(false);
  const [history, setHistory] = useState(false);
  const barRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const element = barRef.current;
    if (!element) { document.documentElement.style.removeProperty("--workspace-bar-height"); return; }
    const measure = () => document.documentElement.style.setProperty("--workspace-bar-height", `${element.getBoundingClientRect().height}px`);
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure); observer.observe(element);
    return () => observer.disconnect();
  }, [location.pathname]);
  useEffect(() => {
    let current = true;
    setAsset(null); setExperiment(null); setImages([]); setType(null);
    void (async () => {
      const nextAsset = assetId ? await getAsset(assetId) : null;
      const experimentId = routeExperiment || nextAsset?.experiment_id;
      if (!current) return;
      setAsset(nextAsset);
      if (!experimentId) return;
      const [nextExperiment, workspace] = await Promise.all([
        getExperiment(experimentId), getNavigationImages(experimentId),
      ]);
      if (!current) return;
      setExperiment(nextExperiment); setImages(workspace.images);
      if (assetId && location.pathname.includes("/labeling/")) {
        const segmentations = await getAssetSegmentations(assetId);
        const requested = new URLSearchParams(location.search).get("seg");
        const internal = decodeURIComponent(location.pathname.split("/")[4] ?? "");
        const segmentation = segmentations.find(s => s.id === requested)
          ?? segmentations.find(s => s.segmentation_type.internal_name === internal);
        if (current) setType(segmentation?.segmentation_type ?? null);
      }
      if (current) await recordVisit(experimentId, location.pathname + location.search);
    })().catch(() => { /* Individual pages retain their error and retry controls. */ });
    return () => { current = false; };
  }, [assetId, routeExperiment, location.pathname, location.search]);
  if (location.pathname === "/") return null;
  const order = experiment
    ? readWorkspaceMemory<string[]>(experiment.id + ".imageOrder", []) : [];
  const ordered = order.length
    ? order.map(id => images.find(image => image.id === id)).filter((image): image is Pick<WorkspaceImage, "id" | "name"> => Boolean(image))
    : images;
  const index = ordered.findIndex(image => image.id === assetId);
  const neighbor = (offset: number) => ordered[index + offset];
  return <div ref={barRef} className="border-b border-slate-200 bg-white px-4 py-2 text-sm" data-testid="workspace-bar">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <nav aria-label="Workspace" className="flex flex-wrap items-center gap-3">
        <Link to="/">Home</Link>
        {experiment && <Link to={`/experiments/${experiment.id}`}>{experiment.name}</Link>}
        {asset && <span className="max-w-72 truncate" title={asset.display_name}>{asset.display_name}</span>}
        {assetId && <>
          <Link to={`/assets/${assetId}/viewer`}>View</Link>
          <Link to={`/assets/${assetId}/labeling`}>Segment & review</Link>
          <Link to={`/assets/${assetId}/analysis`}>Analysis</Link>
          {index >= 0 && [-1, 1].map(offset => neighbor(offset) && <Link key={offset}
            to={location.pathname.replace(`/assets/${assetId}`, `/assets/${neighbor(offset).id}`)}
            title={neighbor(offset).name}>{offset < 0 ? "Previous image" : "Next image"}</Link>)}
        </>}
      </nav>
      <div className="flex items-center gap-2">
        <Button onClick={() => setFineTune(true)}>Fine-tune</Button>
        <Link to={experiment ? `/experiments/${experiment.id}?tab=models` : "/models"}>Models</Link>
        <TaskQueueButton isOpen={tasks} onClick={() => setTasks(true)} />
        {assetId && <Button onClick={() => setHistory(true)}>Correction history</Button>}
      </div>
    </div>
    <FineTuneDialog open={fineTune} onClose={() => setFineTune(false)}
      initialExperimentId={experiment?.id} initialAssetIds={assetId ? [assetId] : undefined}
      segmentationType={type} />
    <JobQueueSidebar isOpen={tasks} onClose={() => setTasks(false)} />
    {history && assetId && <CorrectionHistory assetId={assetId} onClose={() => setHistory(false)} />}
  </div>;
}
