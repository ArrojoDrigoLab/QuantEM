import { useEffect, useMemo, useState } from "react";
import { applyFineTuneRun, fineTuneErrorMessage, getFineTuneRun, getFineTuneScope } from "@/shared/api/finetune";
import type { FineTuneAppliedImageEvent, FineTuneApplyResponse, FineTuneRunDetail } from "@/shared/types/finetune";
import { Button } from "@/shared/ui/design";
import { buildScopeTree, type ScopeGroupNode } from "./scopeTree";
import { FineTuneSuccess } from "./components/FineTuneSuccess";

/** Application has its own experiment targets and never uses training eligibility. */
export function ModelApplication({ adapterId, typeId, typeName, onClose, onAppliedImageQueued, onAppliedImageCompleted }: {
  adapterId: string; typeId: string; typeName: string; onClose: () => void;
  onAppliedImageQueued?: (event: FineTuneAppliedImageEvent) => void;
  onAppliedImageCompleted?: (event: FineTuneAppliedImageEvent) => void;
}) {
  const [run, setRun] = useState<FineTuneRunDetail | null>(null);
  const [group, setGroup] = useState<ScopeGroupNode | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [applying, setApplying] = useState(false);
  const [applyError, setApplyError] = useState<string | null>(null);
  const [applyResult, setApplyResult] = useState<FineTuneApplyResponse | null>(null);
  useEffect(() => {
    let current = true;
    setLoading(true); setError(null);
    void Promise.all([getFineTuneRun(adapterId), getFineTuneScope(typeId)]).then(([detail, scope]) => {
      if (!current) return;
      setRun(detail);
      setGroup(buildScopeTree(scope).find((entry) => entry.key === detail.experiment?.id) ?? null);
    }).catch((cause) => { if (current) setError(fineTuneErrorMessage(cause, "The saved model and its target images could not be loaded.")); })
      .finally(() => { if (current) setLoading(false); });
    return () => { current = false; };
  }, [adapterId, typeId, attempt]);
  const images = useMemo(() => [...new Map([
    ...(group?.images ?? []), ...(group?.datasets.flatMap((dataset) => dataset.images) ?? []),
  ].map((image) => [image.id, { asset_id: image.id, name: image.name, confirmed_areas: image.confirmedAreas, done_rois: image.doneRois, tiles: 0 }])).values()], [group]);
  const apply = async (assetIds: string[], datasetIds: string[]) => {
    if (!run) return;
    setApplying(true); setApplyError(null);
    try {
      const result = await applyFineTuneRun(adapterId, assetIds, datasetIds);
      setApplyResult(result);
      for (const queued of result.queued) onAppliedImageQueued?.({ adapterId, baseModel: run.base_model,
        assetId: queued.asset_id, segmentationId: queued.segmentation_id });
    } catch (cause) { setApplyError(fineTuneErrorMessage(cause, "Could not queue those runs.")); }
    finally { setApplying(false); }
  };
  if (loading) return <p role="status">Loading the saved model and images…</p>;
  if (error) return <div><p role="alert">{error}</p><Button onClick={() => setAttempt((value) => value + 1)}>Retry</Button></div>;
  if (!run || run.status !== "SUCCESS") return <p role="alert">This model is not ready to apply.</p>;
  return <FineTuneSuccess adapterId={adapterId} name={run.name} run={run} baseModel={run.base_model}
    segmentationTypeName={typeName} scopedImages={images} initialAssetIds={(run.asset_ids ?? []).filter((id) => images.some((image) => image.asset_id === id))}
    availableDatasets={group?.datasets ?? []} applying={applying} applyError={applyError} applyResult={applyResult}
    onApply={(assetIds, datasetIds) => void apply(assetIds, datasetIds)} onClose={onClose}
    onAppliedImageCompleted={onAppliedImageCompleted} />;
}
