import type { Experiment } from "@/shared/types/common";

export type FixedDatasetMode = "new" | "existing";

export function FixedExperimentGroupingFields({
  experiment,
  mode,
  newDatasetName,
  selectedDatasetIds,
  onModeChange,
  onNewDatasetNameChange,
  onSelectedDatasetIdsChange,
}: {
  experiment: Experiment;
  mode: FixedDatasetMode;
  newDatasetName: string;
  selectedDatasetIds: string[];
  onModeChange: (mode: FixedDatasetMode) => void;
  onNewDatasetNameChange: (name: string) => void;
  onSelectedDatasetIdsChange: (ids: string[]) => void;
}) {
  const toggleDataset = (datasetId: string, selected: boolean) => {
    onSelectedDatasetIdsChange(
      selected
        ? [...new Set([...selectedDatasetIds, datasetId])]
        : selectedDatasetIds.filter((id) => id !== datasetId)
    );
  };

  return (
    <div className="space-y-3">
      <div>
        <span className="text-sm font-medium text-slate-800">Experiment</span>
        <p className="m-0 mt-1 text-sm text-slate-700">{experiment.name}</p>
      </div>

      <fieldset className="space-y-2">
        <legend className="text-sm font-medium text-slate-800">Dataset</legend>
        <label className="flex items-center gap-2 text-sm text-slate-800">
          <input
            type="radio"
            name="import-dataset-mode"
            checked={mode === "new"}
            onChange={() => onModeChange("new")}
          />
          New dataset
        </label>
        {mode === "new" ? (
          <input
            className="h-9 w-full rounded-md border border-slate-300 px-3 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            value={newDatasetName}
            onChange={(event) => onNewDatasetNameChange(event.target.value)}
            aria-label="New dataset name"
            placeholder="e.g. Liver 24h"
          />
        ) : null}

        <label className="flex items-center gap-2 text-sm text-slate-800">
          <input
            type="radio"
            name="import-dataset-mode"
            checked={mode === "existing"}
            disabled={experiment.datasets.length === 0}
            onChange={() => onModeChange("existing")}
          />
          Existing datasets
        </label>
        {mode === "existing" ? (
          <div
            className="max-h-36 space-y-1 overflow-y-auto rounded-md border border-slate-200 bg-slate-50 p-2"
            aria-label="Existing datasets"
          >
            {experiment.datasets.map((dataset) => (
              <label
                key={dataset.id}
                className="flex items-center gap-2 text-sm text-slate-800"
              >
                <input
                  type="checkbox"
                  checked={selectedDatasetIds.includes(dataset.id)}
                  onChange={(event) =>
                    toggleDataset(dataset.id, event.target.checked)
                  }
                />
                {dataset.name}
              </label>
            ))}
          </div>
        ) : null}
        {experiment.datasets.length === 0 ? (
          <p className="m-0 text-xs text-slate-600">
            This experiment has no datasets yet, so a new one will be created.
          </p>
        ) : null}
      </fieldset>
    </div>
  );
}
