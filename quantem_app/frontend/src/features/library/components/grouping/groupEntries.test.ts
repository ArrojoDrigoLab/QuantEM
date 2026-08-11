import { describe, expect, it } from "vitest";
import { groupEntriesByDataset } from "@/features/library/components/grouping/groupEntries";
import type { HomeEntry } from "@/shared/types/images";

function entry(overrides: Partial<HomeEntry> = {}): HomeEntry {
  return {
    id: "asset-1",
    display_name: "Scan",
    original_filename: "scan.tif",
    metadata_summary: "1024x1024",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    preprocess_stage: "DONE",
    preprocess_progress: 1,
    can_open: true,
    ...overrides,
  };
}

describe("groupEntriesByDataset", () => {
  it("puts an unorganised library in one section rather than none", () => {
    const groups = groupEntriesByDataset([
      entry({ id: "a" }),
      entry({ id: "b" }),
    ]);

    expect(groups).toHaveLength(1);
    expect(groups[0].label).toBe("Not in an experiment");
    expect(groups[0].entries).toHaveLength(2);
  });

  /**
   * Unassigned is where every image starts and where an image returns to when
   * its experiment is deleted. A grouping that dropped it would hide most of
   * most libraries.
   */
  it("keeps the unassigned section last, never absent", () => {
    const groups = groupEntriesByDataset([
      entry({ id: "loose" }),
      entry({
        id: "filed",
        experiment_id: "exp-1",
        experiment_name: "Fasted cohort",
        dataset_ids: ["set-1"],
        dataset_names: ["Liver 24h"],
      }),
    ]);

    expect(groups.map((group) => group.label)).toEqual([
      "Liver 24h",
      "Not in an experiment",
    ]);
  });

  it("gives an experiment's unfiled images their own section, after its datasets", () => {
    const groups = groupEntriesByDataset([
      entry({
        id: "loose-in-exp",
        experiment_id: "exp-1",
        experiment_name: "Fasted cohort",
      }),
      entry({
        id: "filed",
        experiment_id: "exp-1",
        experiment_name: "Fasted cohort",
        dataset_ids: ["set-1"],
        dataset_names: ["Liver 24h"],
      }),
    ]);

    expect(groups.map((group) => group.label)).toEqual([
      "Liver 24h",
      "No dataset",
    ]);
    expect(groups[1].sublabel).toBe("Fasted cohort");
  });

  /**
   * An image genuinely in two datasets is in both. Picking one arbitrarily
   * would make it vanish from a dataset it is a member of.
   */
  it("shows an image in every dataset it belongs to", () => {
    const groups = groupEntriesByDataset([
      entry({
        id: "both",
        experiment_id: "exp-1",
        experiment_name: "Fasted cohort",
        dataset_ids: ["set-1", "set-2"],
        dataset_names: ["Liver 24h", "Liver 48h"],
      }),
    ]);

    expect(groups.map((group) => group.label)).toEqual([
      "Liver 24h",
      "Liver 48h",
    ]);
  });

  it("orders sections by experiment, then by dataset name", () => {
    const groups = groupEntriesByDataset([
      entry({
        id: "b",
        experiment_id: "exp-2",
        experiment_name: "Fed cohort",
        dataset_ids: ["set-9"],
        dataset_names: ["Kidney"],
      }),
      entry({
        id: "a",
        experiment_id: "exp-1",
        experiment_name: "Fasted cohort",
        dataset_ids: ["set-2"],
        dataset_names: ["Liver 48h"],
      }),
      entry({
        id: "c",
        experiment_id: "exp-1",
        experiment_name: "Fasted cohort",
        dataset_ids: ["set-1"],
        dataset_names: ["Liver 24h"],
      }),
    ]);

    expect(groups.map((group) => group.label)).toEqual([
      "Liver 24h",
      "Liver 48h",
      "Kidney",
    ]);
  });
});
