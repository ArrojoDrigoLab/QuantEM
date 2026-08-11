/**
 * File the images the user has selected.
 *
 * A bar above the grid rather than a modal, because the selection it acts on is
 * the grid, and a dialog that covers it makes the user remember what they
 * picked. Everything it can do is one request:
 * {@link assignAssetGrouping}.
 *
 * The one thing this has to get right is **saying what will happen before it
 * happens**. A dataset belongs to exactly one experiment, so moving images to
 * another experiment takes them out of the datasets they are in. That is a loss
 * of information the user did not ask for, so the bar counts it from the
 * selection and states it above the button, then reports what actually
 * happened after the request lands.
 */

import { useMemo, useState } from "react";
import { assignAssetGrouping } from "@/shared/api/assets";
import { Button } from "@/shared/ui/design";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import { GroupingFields } from "@/features/library/components/grouping/GroupingFields";
import {
  KEEP_GROUP,
  chosenName,
  type GroupingChoice,
} from "@/features/library/components/grouping/groupingChoices";
import {
  datasetsLostBy,
  sharedExperimentId,
} from "@/features/library/components/grouping/librarySelection";
import type { Experiment } from "@/shared/types/common";
import type {
  AssetGroupingRequest,
  AssetGroupingResult,
  HomeEntry,
} from "@/shared/types/images";

/**
 * How many of these images would be taken out of a dataset by this change.
 *
 * Computed from what the cards already carry, so the warning is on screen
 * before the request rather than in the reply to it. An image is affected when
 * it is in at least one dataset and its experiment is about to become something
 * else -- including "no experiment", which empties the datasets too.
 */
function describeOutcome(result: AssetGroupingResult): string {
  const filed =
    result.assets_changed === 1
      ? "1 image was moved."
      : `${result.assets_changed} images were moved.`;
  if (result.dataset_links_dropped === 0) return filed;
  const left = result.datasets_left.join(", ");
  const images =
    result.assets_moved_out_of_datasets === 1 ? "1 image" : `${result.assets_moved_out_of_datasets} images`;
  return `${filed} ${images} left ${left}, which the new experiment does not contain.`;
}

export function LibrarySelectionBar({
  selected,
  experiments,
  onApplied,
  onClearSelection,
  onSelectAllShown,
  shownCount,
}: {
  selected: HomeEntry[];
  experiments: Experiment[];
  /** The change landed: refresh the library and the experiment catalogue. */
  onApplied: (result: AssetGroupingResult) => void;
  onClearSelection: () => void;
  onSelectAllShown: () => void;
  shownCount: number;
}) {
  const [experimentChoice, setExperimentChoice] =
    useState<GroupingChoice>(KEEP_GROUP);
  const [datasetChoice, setDatasetChoice] = useState<GroupingChoice>(KEEP_GROUP);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<string | null>(null);

  const currentExperimentId = useMemo(
    () => sharedExperimentId(selected),
    [selected]
  );
  const losing = datasetsLostBy(selected, experimentChoice);
  const changesSomething =
    experimentChoice.kind !== "keep" || datasetChoice.kind !== "keep";
  // A picker parked on "New …" with nothing typed is not a choice yet.
  const incomplete =
    (experimentChoice.kind === "new" && !chosenName(experimentChoice)) ||
    (datasetChoice.kind === "new" && !chosenName(datasetChoice));

  const apply = async () => {
    setApplying(true);
    setError(null);
    setOutcome(null);
    try {
      const payload: AssetGroupingRequest = {
        asset_ids: selected.map((entry) => entry.id),
      };
      if (experimentChoice.kind === "none") payload.experiment = null;
      else if (experimentChoice.kind === "existing") {
        payload.experiment = experimentChoice.id;
      } else if (experimentChoice.kind === "new") {
        payload.experiment_name = chosenName(experimentChoice);
      }
      if (datasetChoice.kind === "none") payload.datasets = [];
      else if (datasetChoice.kind === "existing") {
        payload.datasets = [datasetChoice.id];
      } else if (datasetChoice.kind === "new") {
        payload.dataset_name = chosenName(datasetChoice);
      }
      const result = await assignAssetGrouping(payload);
      setOutcome(describeOutcome(result));
      setExperimentChoice(KEEP_GROUP);
      setDatasetChoice(KEEP_GROUP);
      onApplied(result);
    } catch (err) {
      setError(
        extractApiErrorMessage(err, "Those images could not be organised.")
      );
    } finally {
      setApplying(false);
    }
  };

  return (
    <div
      className="rounded-lg border border-cyan-300 bg-cyan-50 p-4"
      data-testid="library-selection-bar"
    >
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex min-w-[240px] flex-1 flex-col gap-2">
          <p className="m-0 text-sm font-semibold text-slate-900">
            {selected.length === 1
              ? "1 image selected"
              : `${selected.length} images selected`}
          </p>
          <GroupingFields
            experiments={experiments}
            experiment={experimentChoice}
            dataset={datasetChoice}
            onExperimentChange={setExperimentChoice}
            onDatasetChange={setDatasetChoice}
            disabled={applying}
            showCounts
            keepLabels
            currentExperimentId={currentExperimentId}
          />
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            variant="primary"
            disabled={applying || !changesSomething || incomplete}
            onClick={() => void apply()}
          >
            {applying
              ? "Applying..."
              : selected.length === 1
                ? "Apply to this image"
                : `Apply to ${selected.length} images`}
          </Button>
          <Button
            disabled={applying || selected.length >= shownCount}
            onClick={onSelectAllShown}
          >
            Select all shown
          </Button>
          <Button disabled={applying} onClick={onClearSelection}>
            Clear selection
          </Button>
        </div>
      </div>

      {/* Said before the button is pressed, not discovered afterwards. */}
      {losing > 0 ? (
        <p className="mt-3 text-sm text-slate-900">
          {losing === 1 ? "1 of these images is" : `${losing} of these images are`}{" "}
          in a dataset of another experiment.{" "}
          {losing === 1 ? "It will leave it" : "They will leave those datasets"},
          because a dataset belongs to one experiment only. The images
          themselves are not touched.
        </p>
      ) : null}
      {outcome ? (
        <p className="mt-3 text-sm text-slate-900" role="status">
          {outcome}
        </p>
      ) : null}
      {error ? (
        <p className="mt-3 text-sm text-red-700" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
