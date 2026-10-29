/** Fields shared by every image in the import queue. */

import { plural } from "@/features/library/components/import/importValidation";
import { GroupingFields } from "@/features/library/components/grouping/GroupingFields";
import type { GroupingChoice } from "@/features/library/components/grouping/groupingChoices";
import type { ChosenFile } from "@/features/library/components/import/importValidation";
import type { ImportScaleState } from "@/features/library/components/import/useImportScale";
import type { Experiment } from "@/shared/types/common";
import type { ReactNode } from "react";

export function ImportDetailsFields({
  files,
  scale,
  displayName,
  onDisplayNameChange,
  pixelSizeText,
  onPixelSizeTextChange,
  pixelSizeError,
  notesText,
  onNotesTextChange,
  replaceDeclaredPixelSizes,
  onReplaceDeclaredPixelSizesChange,
  experiments,
  experimentChoice,
  datasetChoice,
  onExperimentChoiceChange,
  onDatasetChoiceChange,
  groupingFields,
}: {
  files: ChosenFile[];
  scale: ImportScaleState;
  displayName: string;
  onDisplayNameChange: (next: string) => void;
  pixelSizeText: string;
  onPixelSizeTextChange: (next: string) => void;
  pixelSizeError: string | null;
  notesText: string;
  onNotesTextChange: (next: string) => void;
  replaceDeclaredPixelSizes: boolean;
  onReplaceDeclaredPixelSizesChange: (next: boolean) => void;
  experiments: Experiment[];
  experimentChoice: GroupingChoice;
  datasetChoice: GroupingChoice;
  onExperimentChoiceChange: (next: GroupingChoice) => void;
  onDatasetChoiceChange: (next: GroupingChoice) => void;
  /** A fixed-experiment importer supplies its own multi-dataset controls. */
  groupingFields?: ReactNode;
}) {
  const { singleFile, scaleCensus } = scale;
  const unresolvedMetadataCount =
    scaleCensus.silent + scaleCensus.unreadable;
  const missingPixelSize = unresolvedMetadataCount > 0 && !pixelSizeText.trim();

  return (
    <fieldset className="rounded-md border border-slate-200 p-3">
      <legend className="px-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
        Import details{files.length > 1 ? ` (${files.length} images)` : ""}
      </legend>

      {singleFile ? (
        <div className="flex flex-col gap-1">
          <label
            className="text-sm font-medium text-slate-800"
            htmlFor="display-name"
          >
            Display name
          </label>
          <input
            id="display-name"
            className="h-9 w-full rounded-md border border-slate-300 px-3 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            type="text"
            value={displayName}
            onChange={(event) => onDisplayNameChange(event.target.value)}
            placeholder="Enter display name..."
          />
        </div>
      ) : null}

      <div className="mt-3 flex flex-col gap-1">
        <div className="flex items-center gap-1">
          <label
            className="text-sm font-medium text-slate-800"
            htmlFor="upload-pixel-size"
          >
            Pixel size, nm per pixel
          </label>
          <span
            className="cursor-help text-slate-500"
            role="img"
            aria-label="Optional, you can change or set these later. Required for some analysis measurements"
            title="Optional, you can change or set these later. Required for some analysis measurements"
          >
            ⓘ
          </span>
        </div>
        <input
          id="upload-pixel-size"
          className="h-9 w-40 rounded-md border border-slate-300 px-3 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          type="number"
          min="0"
          step="any"
          inputMode="decimal"
          value={pixelSizeText}
          onChange={(event) => onPixelSizeTextChange(event.target.value)}
          placeholder="e.g. 4.2"
          aria-describedby={missingPixelSize ? "upload-pixel-size-help" : undefined}
        />
        {missingPixelSize ? <span className="text-sm text-slate-700" id="upload-pixel-size-help">
          Pixel size is missing for {unresolvedMetadataCount} {plural(unresolvedMetadataCount, "image", "images")}.
        </span> : null}
        {!singleFile && scaleCensus.declared > 0 ? (
          <label className="mt-1 flex items-start gap-2 text-sm text-slate-800">
            <input
              type="checkbox"
              className="mt-1"
              checked={replaceDeclaredPixelSizes}
              onChange={(event) =>
                onReplaceDeclaredPixelSizesChange(event.target.checked)
              }
            />
            <span>
              Use it for all {files.length} images, replacing the pixel size{" "}
              {scaleCensus.declared}{" "}
              {plural(scaleCensus.declared, "image", "of them")}{" "}
              {plural(scaleCensus.declared, "declares", "declare")}.
            </span>
          </label>
        ) : null}
        {pixelSizeError ? (
          <span className="text-sm text-red-700" role="alert">
            {pixelSizeError}
          </span>
        ) : null}
      </div>

      <div className="mt-3 flex flex-col gap-1">
        {groupingFields ?? (
          <GroupingFields
            experiments={experiments}
            experiment={experimentChoice}
            dataset={datasetChoice}
            onExperimentChange={onExperimentChoiceChange}
            onDatasetChange={onDatasetChoiceChange}
            defaultNewDatasetOnExperimentChange
          />
        )}
      </div>

      <div className="mt-3 flex flex-col gap-1">
        <label className="text-sm font-medium text-slate-800" htmlFor="notes">
          Notes (optional)
        </label>
        <textarea
          id="notes"
          className="w-full resize-y rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          rows={3}
          value={notesText}
          onChange={(event) => onNotesTextChange(event.target.value)}
          placeholder="e.g. PV, day 14, control"
        />
      </div>
    </fieldset>
  );
}
