import type { WorkspaceImage } from "@/shared/api/workspace";

const settingNames: Record<string, string> = {
  mask_name: "Measurement area", require_complete: "Require completed review", reuse: "Reuse matching results",
  tissue_segmentation_id: "Measurement mask", analysis_mask_ids: "Measurement masks", compartments: "Structures",
  objects_in_measurement_area: "Only objects inside measurement area", points_source: "Point source",
  points_csv: "Imported points", distance_target: "Distance to", band_edges_nm: "Distance bands (nm)",
  replicates: "Monte Carlo repetitions", seed: "Random seed", resolution_nm_per_pixel: "Resolution (nm/px)",
};

export function AnalysisSettings({ values, images }: { values: Record<string, unknown>; images: WorkspaceImage[] }) {
  const names = new Map(images.flatMap((image) => [[image.id, image.name], ...image.segmentations.map((seg) => [seg.id, `${image.name}: ${seg.name}`]), ...image.segmentations.map((seg) => [seg.type, seg.name])] as [string, string][]));
  const display = (value: unknown): string => {
    if (value == null || value === "") return "None";
    if (typeof value === "boolean") return value ? "Yes" : "No";
    if (typeof value === "string") return names.get(value) ?? value;
    if (Array.isArray(value)) return value.map(display).join(", ") || "None";
    if (typeof value === "object") return Object.entries(value).map(([key, entry]) => `${key.replaceAll("_", " ")}: ${display(entry)}`).join("; ");
    return String(value);
  };
  return <dl className="grid gap-x-4 gap-y-2 text-sm sm:grid-cols-[auto_1fr]">
    {Object.entries(values).filter(([key]) => !["asset_ids", "dataset_ids", "name", "segmentation_types"].includes(key)).map(([key, value]) => <div key={key} className="contents">
      <dt className="font-medium">{settingNames[key] ?? key.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase())}</dt>
      <dd className="min-w-0 whitespace-pre-wrap break-words">{key === "mask_name" && !value ? "Whole image" : display(value)}</dd>
    </div>)}
  </dl>;
}
