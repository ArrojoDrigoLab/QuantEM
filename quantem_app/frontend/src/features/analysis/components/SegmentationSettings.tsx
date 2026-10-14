import type { SegmentationSettingsSnapshot } from "@/shared/types/analysis";
import { AnalysisSettings } from "@/features/experiments/AnalysisSettings";
import type { WorkspaceImage } from "@/shared/api/workspace";

export function SegmentationSettings({ settings, images }: { settings: SegmentationSettingsSnapshot[]; images: WorkspaceImage[] }) {
  const names = new Map(images.flatMap((image) => image.segmentations.map((seg) => [seg.id, seg.name])));
  return <section aria-label="Segmentation settings" className="space-y-3">
    <h3 className="font-semibold">Segmentation settings</h3>
    {!settings.length && <p className="text-sm text-slate-600">Unknown</p>}
    {settings.map((compartment, index) => <div key={`${compartment.segmentation_id}-${index}`} className="space-y-3 rounded-md border border-slate-200 p-3">
      <h4 className="font-medium">{names.get(compartment.segmentation_id ?? "") ?? compartment.compartment}</h4>
      {!!compartment.manual_objects && <p className="text-sm">Manually drawn objects: {compartment.manual_objects}</p>}
      {compartment.runs.map((run, index) => <div key={index} className="space-y-2 border-t pt-3">
        <AnalysisSettings unknownLabel="Unknown" images={images} values={{
          model: run.adapter?.name ?? run.pack_id,
          base_model: run.pack_id,
          fine_tuned_model: run.adapter_id ? run.adapter?.name ?? run.adapter_id : run.run_id ? "None" : null,
          inference_run: run.run_id, ...(run.n_objects != null ? { objects: run.n_objects } : {}),
          threshold: run.threshold, minimum_area_px: run.min_area,
          input_resolution_nm_per_pixel: run.native_pixel_size_nm,
          inference_resolution_nm_per_pixel: run.ran_at_nm ?? (run.resampled === false ? "Native pixels" : null),
          resized: run.resampled, scope: run.scope, device: run.device,
          probability_grid: run.prob_map_grid,
          ...(run.adapter ? { fine_tuning: Object.fromEntries(Object.entries(run.adapter).filter(([key]) => !["caveats", "unavailable", "note"].includes(key))) } : {}),
        }} />
      </div>)}
      {compartment.global_mask && <AnalysisSettings unknownLabel="Unknown" images={images} values={{ global_segmentation: compartment.global_mask }} />}
      {!!compartment.probability_measurements?.length && <AnalysisSettings unknownLabel="Unknown" images={images} values={{ outline_probability: compartment.probability_measurements }} />}
    </div>)}
  </section>;
}
