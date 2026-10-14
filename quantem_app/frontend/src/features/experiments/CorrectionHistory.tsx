import { useEffect, useState } from "react";
import { apiRequest } from "@/shared/api/core/http";
import { getAssetSegmentations } from "@/shared/api/assets";
import type { ImageSegmentation } from "@/shared/types";
import { Button } from "@/shared/ui/design";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { extractApiErrorMessage } from "@/utils/apiErrors";

interface Revision { id: string; name: string; created_at: string; object_count: number }
export function CorrectionHistory({ assetId, onClose }: { assetId: string; onClose: () => void }) {
  const [segmentations, setSegmentations] = useState<ImageSegmentation[]>([]);
  const [segId, setSegId] = useState("");
  const [rows, setRows] = useState<Revision[]>([]);
  const [name, setName] = useState("Saved corrections");
  const [restoring, setRestoring] = useState<Revision | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  useEffect(() => { void getAssetSegmentations(assetId).then((values) => {
    setSegmentations(values); setSegId(new URLSearchParams(window.location.hash.split("?")[1]).get("seg") ?? values[0]?.id ?? "");
    if (!values.length) setLoading(false);
  }).catch((cause) => { setError(extractApiErrorMessage(cause, "Segmentations could not be loaded.")); setLoading(false); }); }, [assetId]);
  useEffect(() => {
    if (!segId) return;
    let current = true;
    setRows([]); setLoading(true); setError("");
    void apiRequest<Revision[]>(`/api/segmentations/${segId}/revisions/`).then((values) => { if (current) setRows(values); })
      .catch((cause) => { if (current) setError(extractApiErrorMessage(cause, "Revisions could not be loaded.")); })
      .finally(() => { if (current) setLoading(false); });
    return () => { current = false; };
  }, [segId, refresh]);
  const submit = async (revision?: Revision) => {
    setBusy(true); setError("");
    try {
      await apiRequest(`/api/segmentations/${segId}/revisions/`, { method: "POST", body: JSON.stringify(revision ? { restore: revision.id } : { name }) });
      if (revision) window.location.reload();
      else setRefresh((value) => value + 1);
    } catch (cause) { setError(extractApiErrorMessage(cause, "The revision could not be saved or restored.")); }
    finally { setBusy(false); }
  };
  return <div className="fixed inset-0 z-[1200] flex items-center justify-center bg-black/40 p-5">
    <section role="dialog" aria-modal="true" aria-label="Correction history" className="max-h-[85vh] w-full max-w-2xl overflow-auto rounded-xl bg-white p-5 shadow-xl">
      <div className="flex items-center justify-between"><h2 className="text-lg font-semibold">Correction history</h2><Button onClick={onClose}>Close</Button></div>
      <p className="my-3 text-sm text-slate-600">Saves objects, reviewed areas and masks. A revision is saved automatically before clearing corrections or rerunning segmentation. Probability previews are not restored. Restore saves the current state first and opens the restored result for review.</p>
      <label>Segmentation <select className="rounded border p-2" value={segId} onChange={(e) => setSegId(e.target.value)}>
        {segmentations.map((seg) => <option key={seg.id} value={seg.id}>{seg.display_name || seg.segmentation_type.long_name}</option>)}</select></label>
      <form className="my-4 flex gap-2" onSubmit={(e) => { e.preventDefault(); void submit(); }}>
        <input aria-label="Revision name" className="min-w-0 flex-1 rounded border px-2" value={name} maxLength={255} onChange={(e) => setName(e.target.value)} />
        <Button type="submit" disabled={busy || !segId}>Save revision</Button>
      </form>
      {error && <p role="alert" className="text-red-700">{error}</p>}
      {loading && <p role="status">Loading correction history…</p>}
      {!loading && !rows.length && !error && <p className="text-sm">No saved revisions for this segmentation yet.</p>}
      {rows.map((revision) => <div key={revision.id} className="flex items-center justify-between gap-3 border-t py-3">
        <div><strong>{revision.name}</strong><p className="text-xs text-slate-600">{new Date(revision.created_at).toLocaleString()} · {revision.object_count} objects</p></div>
        <Button disabled={busy} onClick={() => setRestoring(revision)}>Restore…</Button>
      </div>)}
      <ConfirmDialog isOpen={Boolean(restoring)} title="Restore correction revision?" message={`Restore “${restoring?.name}”? The current corrections will be saved as another revision first. Unlock completed segmentations and wait for active tasks before restoring.`}
        confirmText="Restore revision" cancelText="Cancel" onCancel={() => setRestoring(null)} onConfirm={() => { const revision = restoring; setRestoring(null); if (revision) void submit(revision); }} />
    </section>
  </div>;
}
