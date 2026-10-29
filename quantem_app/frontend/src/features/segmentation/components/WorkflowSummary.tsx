import type { AssetDetail, ImageSegmentation } from "@/shared/types";

export function WorkflowSummary({ image, segmentation, nextModel }: {
  image: AssetDetail; segmentation: ImageSegmentation; nextModel: string;
}) {
  const result = segmentation.final_result_provenance;
  const protection = segmentation.protection;
  const ready = segmentation.coverage?.ready ?? segmentation.is_complete;
  const readiness = ready ? "Ready for measurement"
    : segmentation.status_stage === "FAILED" ? "Run failed — retry or choose a saved result"
    : segmentation.coverage?.kind === "processing" ? "Processing — wait for completion"
    : "Needs full-image model output or completed manual coverage";
  return <section className="border-b border-slate-200 bg-slate-50 px-4 py-2 text-sm" aria-label="Workflow status" aria-live="polite">
    <dl className="flex flex-wrap gap-x-6 gap-y-1">
      <div><dt className="inline font-semibold">Model for next run: </dt><dd className="inline">{nextModel}</dd></div>
      <div><dt className="inline font-semibold">Measurement result: </dt><dd className="inline">
        {result ? <>{result.model_identifier || "Unknown model"}{result.adapter_identifier && ` · adapted ${result.adapter_identifier}`}
          {result.result_id && ` · result ${result.result_id.slice(0, 8)}`}
          {result.final_level != null && ` · threshold ${result.final_level}`}</>
          : segmentation.coverage?.kind === "manual" ? "Manual segmentation"
          : ready ? "Saved result — model details unavailable" : "No full-image result selected"}
      </dd></div>
      <div><dt className="inline font-semibold">Protected manual areas: </dt><dd className="inline">
        {protection?.whole_image ? "Whole image" : protection
          ? `${protection.completed_areas} completed areas, ${protection.done_rois} Done ROIs`
          : "Open completed areas and Done ROIs to inspect protection"}
      </dd></div>
      <div><dt className="inline font-semibold">Measurement readiness: </dt><dd className="inline">{readiness}
        {!image.pixel_size_nm ? " · Resolution missing; physical units unavailable" : segmentation.objects_pixel_size?.predates_calibration
          ? " · Objects predate calibration; review resolution warning" : ""}
      </dd></div>
    </dl>
    <p className="m-0 mt-1 text-xs text-slate-600">Confirmed objects outside completed manual areas can be replaced by another model result.</p>
  </section>;
}
