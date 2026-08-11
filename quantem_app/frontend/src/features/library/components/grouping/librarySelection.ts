import type { GroupingChoice } from "@/features/library/components/grouping/groupingChoices";
import type { HomeEntry } from "@/shared/types/images";

/** The experiment every selected image is in, or `""` when they differ. */
export function sharedExperimentId(entries: HomeEntry[]): string {
  const first = entries[0]?.experiment_id ?? "";
  return entries.every((entry) => (entry.experiment_id ?? "") === first) ? first : "";
}

/** Count images that this grouping change would remove from a dataset. */
export function datasetsLostBy(entries: HomeEntry[], experiment: GroupingChoice): number {
  if (experiment.kind === "keep") return 0;
  const targetId = experiment.kind === "existing" ? experiment.id : "";
  return entries.filter((entry) => {
    const inDatasets = (entry.dataset_ids ?? []).length > 0;
    if (!inDatasets) return false;
    if (experiment.kind === "new") return true;
    return (entry.experiment_id ?? "") !== targetId;
  }).length;
}
