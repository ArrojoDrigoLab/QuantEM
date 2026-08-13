/**
 * The experiment and dataset pair, wherever images are being filed.
 *
 * The two are not independent: a dataset lives inside exactly one experiment,
 * so the dataset picker offers the chosen experiment's datasets and nothing
 * else, and changing the experiment clears whatever dataset was chosen under
 * the old one. Enforcing that here rather than at each call site is what stops
 * the two disagreeing on one screen and not on another.
 *
 * Selecting no shared experiment means one experiment per image, named from
 * the image. It never creates an unassigned active image.
 */

import { useCallback } from "react";
import {
  GroupingPicker,
  type GroupingOption,
} from "@/features/library/components/grouping/GroupingPicker";
import {
  chosenId,
  type GroupingChoice,
} from "@/features/library/components/grouping/groupingChoices";
import { datasetsFor } from "@/features/library/components/grouping/useExperiments";
import type { Experiment } from "@/shared/types/common";

export function GroupingFields({
  experiments,
  experiment,
  dataset,
  onExperimentChange,
  onDatasetChange,
  disabled = false,
  showCounts = false,
  keepLabels = false,
  currentExperimentId = "",
  experimentHelp,
}: {
  experiments: Experiment[];
  experiment: GroupingChoice;
  dataset: GroupingChoice;
  onExperimentChange: (next: GroupingChoice) => void;
  onDatasetChange: (next: GroupingChoice) => void;
  disabled?: boolean;
  /** Show how many images each group holds. Useful when picking, noise when filing. */
  showCounts?: boolean;
  /**
   * Offer "leave as they are" on both pickers. On where there is something to
   * leave alone (a selection of existing images), off where there is not (an
   * import, whose images do not exist yet).
   */
  keepLabels?: boolean;
  /**
   * The experiment the images are already in, when they all share one.
   *
   * Only consulted while the experiment picker says "keep": that is the case
   * where the dataset list still has a well-defined scope even though the
   * request will not carry an experiment. Empty when the selection spans more
   * than one, which is exactly when there is no such list to show.
   */
  currentExperimentId?: string;
  experimentHelp?: string;
}) {
  const scopeExperimentId =
    experiment.kind === "keep" ? currentExperimentId : chosenId(experiment);
  const datasetOptions: GroupingOption[] = datasetsFor(
    experiments,
    scopeExperimentId
  ).map((row) => ({
    id: row.id,
    name: row.name,
    count: showCounts ? row.asset_count : undefined,
  }));

  const handleExperimentChange = useCallback(
    (next: GroupingChoice) => {
      onExperimentChange(next);
      // A dataset chosen under the previous experiment cannot survive the
      // change: it belongs to that experiment and to no other.
      onDatasetChange(keepLabels ? { kind: "keep" } : { kind: "none" });
    },
    [keepLabels, onDatasetChange, onExperimentChange]
  );

  // Separate per-image experiments leave no single scope in which a shared
  // dataset can be selected.
  const canChooseDataset = experiment.kind !== "none";

  return (
    <div className="flex flex-wrap gap-3">
      <GroupingPicker
        label="Experiment"
        options={experiments.map((row) => ({
          id: row.id,
          name: row.name,
          count: showCounts ? row.asset_count : undefined,
        }))}
        value={experiment}
        onChange={handleExperimentChange}
        disabled={disabled}
        noneLabel={
          keepLabels ? "Move to separate experiments" : "Use image name"
        }
        newLabel="New experiment…"
        newPlaceholder="e.g. Fasted cohort"
        keepLabel={keepLabels ? "Leave as they are" : undefined}
        help={experimentHelp}
      />
      <GroupingPicker
        label="Dataset"
        options={datasetOptions}
        value={canChooseDataset ? dataset : { kind: "none" }}
        onChange={onDatasetChange}
        disabled={disabled || !canChooseDataset}
        noneLabel="No dataset"
        newLabel="New dataset…"
        newPlaceholder="e.g. Liver 24h"
        keepLabel={keepLabels && canChooseDataset ? "Leave as they are" : undefined}
        help={
          canChooseDataset
            ? undefined
            : "Choose one shared experiment before selecting a dataset."
        }
      />
    </div>
  );
}
