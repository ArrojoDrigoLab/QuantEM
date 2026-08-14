import type { SourceModelOption } from "@/shared/types/images";
import type {
  AdaptedModelEntry,
  ModelCatalogue,
} from "@/shared/types/finetune";

export interface LabelingModelOption {
  value: string;
  label: string;
  sourceModel: string;
  adapterId: string | null;
  packId: string | null;
  adapted: AdaptedModelEntry | null;
}

export function rawAdapterId(value: string): string | null {
  return value.startsWith("adapted:") ? value.slice("adapted:".length) || null : null;
}

function appliesTo(entry: AdaptedModelEntry, segmentationId: string): boolean {
  return (
    entry.segmentation_id === segmentationId ||
    entry.scope_segmentation_ids?.includes(segmentationId) === true ||
    entry.applied_segmentation_ids?.includes(segmentationId) === true ||
    entry.segmentation_ids?.includes(segmentationId) === true
  );
}

export function buildLabelingModelOptions(
  sourceOptions: SourceModelOption[],
  catalogue: ModelCatalogue | null | undefined,
  segmentationId: string | null
): LabelingModelOption[] {
  const released = sourceOptions
    .filter((option) => option.value === "manual" || option.model_family !== "manual")
    .map((option) => ({
      value: option.value,
      label: option.value === "manual" ? "Manual segmentation" : option.label,
      sourceModel: option.value,
      adapterId: null,
      packId: option.value === "manual" ? null : option.value,
      adapted: null,
    }));
  if (!catalogue || !segmentationId) return released;

  const releasedIds = new Set(sourceOptions.map((option) => option.value));
  const adapted = catalogue.adapted
    .filter(
      (entry) => releasedIds.has(entry.base) && appliesTo(entry, segmentationId)
    )
    .sort((left, right) => right.created_at.localeCompare(left.created_at))
    .map((entry) => ({
      value: entry.id,
      label: `${entry.name} (fine-tuned model)`,
      sourceModel: entry.base,
      adapterId: rawAdapterId(entry.id),
      packId: entry.base,
      adapted: entry,
    }));
  return [...released, ...adapted];
}

/** Default required by the labeling workflow, independent of picker history. */
export function defaultLabelingModel(
  options: LabelingModelOption[],
  catalogue: ModelCatalogue | null | undefined,
  segmentationId: string | null
): string | null {
  const adapted = options.filter((option) => option.adapted !== null);
  const lastRun = adapted
    .filter((option) =>
      Boolean(
        segmentationId &&
          option.adapted?.last_run_at_by_segmentation?.[segmentationId]
      )
    )
    .sort((left, right) =>
      String(
        right.adapted?.last_run_at_by_segmentation?.[segmentationId ?? ""] ?? ""
      ).localeCompare(
        String(
          left.adapted?.last_run_at_by_segmentation?.[segmentationId ?? ""] ?? ""
        )
      )
    )[0];
  if (lastRun) return lastRun.value;

  const newest = [...adapted].sort((left, right) =>
    String(right.adapted?.created_at ?? "").localeCompare(
      String(left.adapted?.created_at ?? "")
    )
  )[0];
  if (newest) return newest.value;

  const bases = options.filter(
    (option) => option.adapterId === null && option.packId !== null
  );
  const installed = bases.filter(
    (option) =>
      catalogue?.packs.find((pack) => pack.id === option.packId)?.installed === true
  );
  if (installed.length === 1) return installed[0].value;
  return (
    bases.find((option) => option.sourceModel.startsWith("omniem:"))?.value ??
    bases[0]?.value ??
    options.find((option) => option.sourceModel === "manual")?.value ??
    null
  );
}

export function resolveLabelingModel(
  options: LabelingModelOption[],
  value: string | null | undefined
): LabelingModelOption | null {
  if (!value) return null;
  return options.find((option) => option.value === value) ?? null;
}
