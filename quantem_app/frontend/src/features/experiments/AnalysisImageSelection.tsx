import { useState } from "react";
import { saveSample, type WorkspaceImage } from "@/shared/api/workspace";
import { Button } from "@/shared/ui/design";
import { HelpDisclosure } from "@/shared/ui/HelpDisclosure";
import { extractApiErrorMessage } from "@/utils/apiErrors";

export function AnalysisImageSelection({ experimentId, images, selectedIds, onChange, onMetadataSaved }: {
  experimentId: string; images: WorkspaceImage[]; selectedIds: string[];
  onChange: (ids: string[]) => void; onMetadataSaved: () => void;
}) {
  const [query, setQuery] = useState("");
  const [dataset, setDataset] = useState("");
  const [sample, setSample] = useState("");
  const [group, setGroup] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const datasets = [...new Map(images.flatMap((image) => image.datasets).map((entry) => [entry.id, entry])).values()];
  const matching = images.filter((image) => image.name.toLowerCase().includes(query.toLowerCase()) && (!dataset || image.datasets.some((entry) => entry.id === dataset)));
  const save = async () => {
    setSaving(true); setError("");
    try { for (const id of selectedIds) await saveSample(experimentId, id, sample, group); onMetadataSaved(); }
    catch (cause) { setError(extractApiErrorMessage(cause, "Sample groups could not be saved.")); }
    finally { setSaving(false); }
  };
  return <details className="rounded border p-3" open>
    <summary className="cursor-pointer font-medium">Images ({selectedIds.length} selected)</summary>
    <div className="mt-3 flex flex-wrap items-end gap-3">
      <label>Find images <input type="search" className="block rounded border px-2 py-1" value={query} onChange={(e) => setQuery(e.target.value)} /></label>
      <label>Dataset <select className="block rounded border px-2 py-1" value={dataset} onChange={(e) => setDataset(e.target.value)}><option value="">All datasets</option>{datasets.map((entry) => <option key={entry.id} value={entry.id}>{entry.name}</option>)}</select></label>
      <Button onClick={() => onChange([...new Set([...selectedIds, ...matching.map((image) => image.id)])])}>Select all matching ({matching.length})</Button>
      <Button onClick={() => onChange([])}>Clear selection</Button>
    </div>
    <div className="my-3 max-h-64 overflow-auto rounded border"><table className="w-full text-left text-sm"><tbody>{matching.map((image) => <tr key={image.id}>
      <td className="border-b p-2"><label className="flex items-center gap-2"><input type="checkbox" aria-label={`Select ${image.name}`} checked={selectedIds.includes(image.id)} onChange={(e) => onChange(e.target.checked ? [...selectedIds, image.id] : selectedIds.filter((id) => id !== image.id))} />{image.name}</label></td>
      <td className="border-b p-2">{image.datasets.map((entry) => entry.name).join(", ")}</td>
    </tr>)}</tbody></table>{!matching.length && <p className="p-3">No images match.</p>}</div>
    <details><summary className="cursor-pointer text-sm">Sample groups</summary>
      <div className="mt-3 flex flex-wrap items-end gap-3">
        <label>Sample ID <input className="block rounded border px-2 py-1" placeholder="Image name by default" maxLength={255} value={sample} onChange={(e) => setSample(e.target.value)} /></label>
        <HelpDisclosure label="sample IDs" title="Sample IDs">Images with the same sample ID are averaged together before calculating group statistics. By default, each image is a separate sample.</HelpDisclosure>
        <label>Group <input className="block rounded border px-2 py-1" maxLength={255} value={group} onChange={(e) => setGroup(e.target.value)} /></label>
        <Button disabled={saving || !selectedIds.length} onClick={() => void save()}>Save for {selectedIds.length} images</Button>
      </div>
      {error && <p role="alert" className="text-red-700">{error}</p>}
    </details>
  </details>;
}
