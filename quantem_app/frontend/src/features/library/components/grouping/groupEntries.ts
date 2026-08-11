/**
 * Arrange library entries into dataset sections.
 *
 * Three kinds of section, in this order, because that is the order the
 * information narrows in: each experiment's datasets by name, then that
 * experiment's images which are in no dataset, then the images in no experiment
 * at all.
 *
 * **Unassigned is last, not absent.** It is where every image starts and where
 * an image returns to when its experiment is deleted, so a grouping that quietly
 * dropped it would hide most of most libraries.
 *
 * An image in two datasets appears in both sections. That is what grouping by
 * dataset means -- the alternative, picking one arbitrarily, would make an image
 * vanish from a dataset it is genuinely in.
 */

import type { HomeEntry } from "@/shared/types/images";

export interface EntryGroup {
  key: string;
  /** The dataset's name, or what this bucket is when there is no dataset. */
  label: string;
  /** The experiment the section sits under, when there is one. */
  sublabel: string | null;
  entries: HomeEntry[];
}

const UNASSIGNED_KEY = "unassigned";

export function groupEntriesByDataset(entries: HomeEntry[]): EntryGroup[] {
  const groups = new Map<string, EntryGroup>();
  // Sort keys, kept beside each group so the comparison below does not have to
  // re-derive them from the label (where "No dataset" would sort under N).
  const order = new Map<string, [string, number, string]>();

  const push = (
    key: string,
    label: string,
    sublabel: string | null,
    sortKey: [string, number, string],
    entry: HomeEntry
  ) => {
    const existing = groups.get(key);
    if (existing) {
      existing.entries.push(entry);
      return;
    }
    groups.set(key, { key, label, sublabel, entries: [entry] });
    order.set(key, sortKey);
  };

  for (const entry of entries) {
    const experimentName = entry.experiment_name ?? "";
    const experimentId = entry.experiment_id ?? "";
    const datasetIds = entry.dataset_ids ?? [];
    const datasetNames = entry.dataset_names ?? [];

    if (!experimentId) {
      push(
        UNASSIGNED_KEY,
        "Not in an experiment",
        null,
        // Sorts after every experiment, whatever they are called.
        ["￿", 2, ""],
        entry
      );
      continue;
    }
    if (datasetIds.length === 0) {
      push(
        `ungrouped:${experimentId}`,
        "No dataset",
        experimentName,
        [experimentName, 1, ""],
        entry
      );
      continue;
    }
    datasetIds.forEach((datasetId, index) => {
      const name = datasetNames[index] ?? "Dataset";
      push(
        `dataset:${datasetId}`,
        name,
        experimentName,
        [experimentName, 0, name],
        entry
      );
    });
  }

  return [...groups.values()].sort((left, right) => {
    const a = order.get(left.key) as [string, number, string];
    const b = order.get(right.key) as [string, number, string];
    return (
      a[0].localeCompare(b[0], undefined, { sensitivity: "base" }) ||
      a[1] - b[1] ||
      a[2].localeCompare(b[2], undefined, { sensitivity: "base" })
    );
  });
}
