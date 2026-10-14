import { useCallback, useEffect, useState } from "react";
import { apiRequest } from "@/shared/api/core/http";
import { getSegmentationTypes } from "@/shared/api/assets";
import type { FineTuneRunDetail } from "@/shared/types/finetune";
import type { SegmentationType } from "@/shared/types/images";
import { FineTuneDialog } from "@/features/finetune/FineTuneDialog";
import { TrainingHistory } from "@/features/finetune/components/TrainingHistory";
import { Button } from "@/shared/ui/design";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { extractApiErrorMessage } from "@/utils/apiErrors";

type SavedModel = FineTuneRunDetail & { segmentation_type: string };
export function SavedModelsPanel({ experimentId, refreshKey = "" }: { experimentId?: string; refreshKey?: string }) {
  const [models, setModels] = useState<SavedModel[]>([]);
  const [types, setTypes] = useState<SegmentationType[]>([]);
  const [archived, setArchived] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [rename, setRename] = useState<{ id: string; name: string } | null>(null);
  const [deleting, setDeleting] = useState<SavedModel | null>(null);
  const [continuing, setContinuing] = useState<SavedModel | null>(null);
  const [applySaved, setApplySaved] = useState(false);
  const load = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ archived: String(archived) });
      if (experimentId) params.set("experiment", experimentId);
      const [rows, choices] = await Promise.all([
        apiRequest<SavedModel[]>(`/api/finetune/models/?${params}`), getSegmentationTypes(),
      ]);
      setModels(rows); setTypes(choices); setError("");
    } catch (cause) { setError(extractApiErrorMessage(cause, "Models could not be loaded.")); }
    finally { setLoading(false); }
  }, [experimentId, archived]);
  useEffect(() => { void load(); }, [load, refreshKey]);
  const update = async (model: SavedModel, values: { name?: string; archived?: boolean }) => {
    setBusy(true);
    try { await apiRequest(`/api/finetune/models/${model.id}/`, { method: "PATCH", body: JSON.stringify(values) }); setRename(null); await load(); }
    catch (cause) { setError(extractApiErrorMessage(cause, "The model could not be updated.")); }
    finally { setBusy(false); }
  };
  return <section className="rounded-lg border bg-white p-4" aria-label="Saved fine-tuned models">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <h2 className="text-lg font-semibold">Saved fine-tuned models</h2>
      <label><input type="checkbox" checked={archived} onChange={(e) => setArchived(e.target.checked)} /> Show archived</label>
    </div>
    {error && <p role="alert" className="text-red-700">{error}</p>}
    {loading ? <p role="status">Loading models…</p> : !models.length && !error ? <p>No saved fine-tunes here yet. Use Fine-tune to create one.</p> : null}
    <div className="space-y-3">{models.map((model) => <article key={model.id} className="rounded border p-3">
      <div className="flex flex-wrap items-center gap-3">
        <strong>{model.name || "Unnamed fine-tune"}</strong>
        <span>{model.status}{model.archived_at ? " · Archived" : ""}</span>
        <span className="text-sm text-slate-600">{types.find((type) => type.id === model.segmentation_type)?.short_name} · {model.base_model}</span>
        <Button size="sm" disabled={busy || model.status !== "SUCCESS" || Boolean(model.archived_at) || model.mode !== "head"} onClick={() => { setApplySaved(false); setContinuing(model); }}>Continue training</Button>
        <Button size="sm" disabled={busy || model.status !== "SUCCESS" || Boolean(model.archived_at)} onClick={() => { setApplySaved(true); setContinuing(model); }}>Apply to images</Button>
        <Button size="sm" disabled={busy} onClick={() => setRename({ id: model.id, name: model.name })}>Rename</Button>
        <Button size="sm" disabled={busy} onClick={() => void update(model, { archived: !model.archived_at })}>{model.archived_at ? "Restore to list" : "Archive"}</Button>
        <Button size="sm" disabled={busy} onClick={() => setDeleting(model)}>Delete model</Button>
      </div>
      <p className="my-2 text-xs text-slate-600">Created {new Date(model.created_at).toLocaleString()} · {model.asset_ids?.length ?? 0} training images
        {model.parent_id ? ` · Continued from ${models.find((row) => row.id === model.parent_id)?.name ?? model.parent_id}` : " · Pretrained starting point"}</p>
      {model.evaluation_note && <p className="text-sm text-amber-900">{model.evaluation_note}</p>}
      <TrainingHistory rounds={model.training_rounds} />
      {rename?.id === model.id && <form className="flex gap-2" onSubmit={(event) => { event.preventDefault(); void update(model, { name: rename.name }); }}>
        <input aria-label="Model name" className="rounded border px-2" maxLength={255} value={rename.name} onChange={(event) => setRename({ ...rename, name: event.target.value })} />
        <Button disabled={busy || !rename.name.trim()} type="submit">Save name</Button><Button onClick={() => setRename(null)}>Cancel</Button>
      </form>}
    </article>)}</div>
    <ConfirmDialog isOpen={Boolean(deleting)} title="Delete model?"
      message={`Delete weights for “${deleting?.name}”? This cannot be undone. Existing segmentations and measurement records stay available, but this model can no longer run. Models used by active jobs or saved descendants cannot be deleted.`}
      confirmText="Delete model" cancelText="Cancel" onCancel={() => setDeleting(null)} onConfirm={() => {
        const model = deleting; setDeleting(null); if (!model) return; setBusy(true);
        void apiRequest(`/api/finetune/models/${model.id}/`, { method: "DELETE" }).then(load)
          .catch((cause) => setError(extractApiErrorMessage(cause, "The model could not be deleted."))).finally(() => setBusy(false));
      }} />
    {continuing && <FineTuneDialog open onClose={() => { setContinuing(null); void load(); }}
      segmentationType={types.find((type) => type.id === continuing.segmentation_type)}
      initialExperimentId={continuing.experiment?.id}
      initialAssetIds={continuing.asset_ids} initialBaseModel={continuing.base_model}
      initialParentAdapterId={applySaved ? undefined : continuing.id}
      initialSavedAdapterId={applySaved ? continuing.id : undefined} />}
  </section>;
}
