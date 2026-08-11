/**
 * "Optional — you can do this later": display name, pixel size, notes.
 *
 * Split out of `ImageUploadPanel.tsx` unchanged. Everything in here is optional
 * and only one field ever raises its voice: the pixel size, because packs
 * resample before they look for anything, so it decides which objects exist and
 * not merely the units they are reported in. It is *asked* when the file does
 * not declare one and merely *stated* when it does, and nothing is pre-filled
 * from a header — a typed value is `user-entered` provenance forever, and
 * pre-filling would erase that distinction on day one.
 */

import { formatPixelSizeNm } from "@/shared/pixelSize";
import { plural } from "@/features/library/components/import/importValidation";
import { GroupingFields } from "@/features/library/components/grouping/GroupingFields";
import type { GroupingChoice } from "@/features/library/components/grouping/GroupingPicker";
import type { ChosenFile } from "@/features/library/components/import/importValidation";
import type { ImportScaleState } from "@/features/library/components/import/useImportScale";
import type { Experiment } from "@/shared/types/common";

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
  willRunUncalibrated,
  experiments,
  experimentChoice,
  datasetChoice,
  onExperimentChoiceChange,
  onDatasetChoiceChange,
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
  willRunUncalibrated: boolean;
  experiments: Experiment[];
  experimentChoice: GroupingChoice;
  datasetChoice: GroupingChoice;
  onExperimentChoiceChange: (next: GroupingChoice) => void;
  onDatasetChoiceChange: (next: GroupingChoice) => void;
}) {
  const { singleFile, draftPixelSizeNm, scaleCensus } = scale;
  return (
    <fieldset className="rounded-md border border-slate-200 p-3">
      <legend className="px-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
        Optional — you can do this later
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
            onChange={(e) => onDisplayNameChange(e.target.value)}
            placeholder="Enter display name..."
          />
        </div>
      ) : (
        <p className="m-0 text-sm text-slate-700">
          Each image keeps its own file name. You can rename any of them
          later.
        </p>
      )}

      <div className="mt-3 flex flex-col gap-1">
        <label
          className="text-sm font-medium text-slate-800"
          htmlFor="upload-pixel-size"
        >
          Pixel size, nm per pixel
        </label>
        {/* Answering the question the box asks, from the files
            themselves. Nothing is copied into the input: a value the user
            types is `user-entered` provenance forever, and pre-filling
            this box from a header would erase that distinction on day
            one. */}
        {singleFile ? (
          <SingleFileScaleHelp
            fileName={singleFile.file.name}
            declaredNm={
              singleFile.scale?.state === "declared"
                ? singleFile.scale.pixelSizeNm
                : null
            }
            silent={singleFile.scale?.state === "silent"}
            typedNm={draftPixelSizeNm}
            willRunUncalibrated={willRunUncalibrated}
          />
        ) : (
          <span className="text-sm text-slate-700" id="upload-pixel-size-help">
            {scaleCensus.declared > 0 ? (
              <>
                {scaleCensus.declared}{" "}
                {plural(scaleCensus.declared, "image")}{" "}
                {scaleCensus.declaredValues.length === 1
                  ? `${plural(scaleCensus.declared, "says", "say")} ${formatPixelSizeNm(
                      scaleCensus.declaredValues[0]
                    )} in the file.`
                  : `${plural(
                      scaleCensus.declared,
                      "carries",
                      "carry"
                    )} a pixel size of its own, from ${formatPixelSizeNm(
                      scaleCensus.declaredValues[0]
                    )} to ${formatPixelSizeNm(
                      scaleCensus.declaredValues[
                        scaleCensus.declaredValues.length - 1
                      ]
                    )}.`}{" "}
              </>
            ) : null}
            {scaleCensus.silent > 0 ? (
              <>
                {scaleCensus.silent} {plural(scaleCensus.silent, "image")}{" "}
                {plural(scaleCensus.silent, "does", "do")} not say.{" "}
                {draftPixelSizeNm === null
                  ? "Until you set one, they are measured in pixels."
                  : `They will be imported at ${formatPixelSizeNm(draftPixelSizeNm)}.`}{" "}
              </>
            ) : null}
            {scaleCensus.unreadable > 0 ? (
              <>
                {scaleCensus.unreadable}{" "}
                {plural(scaleCensus.unreadable, "image")} could not be read
                here; the server will use whatever{" "}
                {plural(scaleCensus.unreadable, "it carries", "they carry")}
                .{" "}
              </>
            ) : null}
            {willRunUncalibrated
              ? "You can set or change these later, but that does not re-run anything segmented now."
              : "You can set or change these later."}
          </span>
        )}
        <input
          id="upload-pixel-size"
          className="h-9 w-40 rounded-md border border-slate-300 px-3 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          type="number"
          min="0"
          step="any"
          inputMode="decimal"
          value={pixelSizeText}
          onChange={(e) => onPixelSizeTextChange(e.target.value)}
          placeholder="e.g. 4.2"
          aria-describedby="upload-pixel-size-help"
        />
        {/* The batch's one dangerous act, and it takes a deliberate
            tick. Only offered when there is something to overwrite. */}
        {!singleFile && scaleCensus.declared > 0 ? (
          <label className="mt-1 flex items-start gap-2 text-sm text-slate-800">
            <input
              type="checkbox"
              className="mt-1"
              checked={replaceDeclaredPixelSizes}
              onChange={(e) =>
                onReplaceDeclaredPixelSizesChange(e.target.checked)
              }
            />
            <span>
              Use it for all {files.length} images, replacing the pixel
              size {scaleCensus.declared}{" "}
              {plural(scaleCensus.declared, "image", "of them")}{" "}
              {plural(scaleCensus.declared, "declares", "declare")}.
            </span>
          </label>
        ) : null}
        {pixelSizeError && (
          <span className="text-sm text-red-700" role="alert">
            {pixelSizeError}
          </span>
        )}
      </div>

      {/* This was "Tags (comma-separated, optional)". It collected text,
          posted it as `tag_names`, and the server never read it -- there
          is no tag field on `Asset` and no tag anywhere in the Python
          tree. Notes is the field that already exists, is PATCHable, and
          is included in the library search alongside the display name and
          the filename. */}
      <div className="mt-3 flex flex-col gap-1">
        <label className="text-sm font-medium text-slate-800" htmlFor="notes">
          Notes
        </label>
        <input
          id="notes"
          className="h-9 w-full rounded-md border border-slate-300 px-3 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          type="text"
          value={notesText}
          onChange={(e) => onNotesTextChange(e.target.value)}
          placeholder="e.g. PV, day 14, control"
          aria-describedby="notes-help"
        />
        <span className="text-xs text-slate-600" id="notes-help">
          Saved with{" "}
          {files.length === 1
            ? "the image"
            : `all ${files.length} images`}
          . The library&apos;s search box matches notes as well as names
          and filenames, so a word typed here is how you find{" "}
          {files.length === 1 ? "this image" : "these images"} again.
        </span>
      </div>

      {/* Where these images go in the library. Optional like everything else
          in this fieldset, and it stays quiet when it is left alone: an import
          that names neither behaves exactly as it did before these two
          controls existed. Both are here rather than on a separate organising
          screen because filing at import is the one moment the user already
          knows what the images are. */}
      <div className="mt-3 flex flex-col gap-1">
        <GroupingFields
          experiments={experiments}
          experiment={experimentChoice}
          dataset={datasetChoice}
          onExperimentChange={onExperimentChoiceChange}
          onDatasetChange={onDatasetChoiceChange}
        />
        <span className="text-xs text-slate-600">
          Applied to{" "}
          {files.length === 1 ? "this image" : `all ${files.length} images`}. You
          can leave both blank and organise later, or never.
        </span>
      </div>
    </fieldset>
  );
}

/**
 * The pixel-size sentence for exactly one file.
 *
 * Kept verbatim from the single-image form: this is the wording that was
 * written against the owner's report and verified on his own images, and a
 * batch-shaped rewrite of it would change three sentences nobody asked to
 * change.
 */
export function SingleFileScaleHelp({
  fileName,
  declaredNm,
  silent,
  typedNm,
  willRunUncalibrated,
}: {
  fileName: string;
  declaredNm: number | null;
  silent: boolean;
  typedNm: number | null;
  willRunUncalibrated: boolean;
}) {
  if (declaredNm !== null) {
    return (
      <span
        className="text-sm text-slate-700"
        id="upload-pixel-size-help"
        role="status"
      >
        {fileName} declares {formatPixelSizeNm(declaredNm)}
        {typedNm === null
          ? ", which this import will use. Leave this blank unless you know it is wrong."
          : ". The value typed below is used instead."}
      </span>
    );
  }
  return (
    <span className="text-sm text-slate-700" id="upload-pixel-size-help">
      {silent
        ? `${fileName} does not say. Until you set one, everything is measured in pixels.`
        : "If this file does not carry one, everything is measured in pixels until you set one."}{" "}
      {/* "You can set or change it later" is true of the number and false of
          everything downstream of it once a run has been queued, and read on
          its own it is what made an uncalibrated import look reversible. */}
      {willRunUncalibrated
        ? "You can set or change it later, but that does not re-run anything segmented now."
        : "You can set or change it later."}
    </span>
  );
}
