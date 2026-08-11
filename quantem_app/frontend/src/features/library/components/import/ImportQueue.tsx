/**
 * The files that are about to be imported, and where each of them has got to.
 *
 * Split out of `ImageUploadPanel.tsx` unchanged. It answers two questions with
 * one list: "did it take the ones I meant?" before anything is uploaded, and
 * "where has it got to?" once the queue is running. Each row states the pixel
 * size it will actually be imported with, so "what is about to be applied to
 * what" is never a guess, and a failure is printed against its own row in the
 * server's words -- one corrupt TIFF in a plate of forty is a fact about that
 * TIFF, not about the batch.
 */

import { Button } from "@/shared/ui/design";
import { cx } from "@/shared/ui/cx";
import { formatBytes } from "@/shared/ui/format";
import { formatPixelSizeNm } from "@/shared/pixelSize";
import {
  plural,
  type BatchSummary,
  type ChosenFile,
  type FileImportState,
} from "@/features/library/components/import/importValidation";
import type { AppliedPixelSize } from "@/features/library/components/import/useImportScale";

export function ImportQueue({
  files,
  imports,
  importing,
  highlightDropZone,
  batchSummary,
  totalBytes,
  appliedPixelSize,
  onAddMoreFiles,
  onChooseDifferentFile,
  onClearChosenFiles,
  onRemoveFile,
}: {
  files: ChosenFile[];
  imports: Record<string, FileImportState>;
  importing: boolean;
  highlightDropZone: boolean;
  batchSummary: BatchSummary | null;
  totalBytes: number;
  appliedPixelSize: (entry: ChosenFile) => AppliedPixelSize;
  onAddMoreFiles: () => void;
  onChooseDifferentFile: () => void;
  onClearChosenFiles: () => void;
  onRemoveFile: (key: string) => void;
}) {
  return (
    <>
      {/* The files, named and sized, so "did it take the ones I meant?" is
          answered before anything is uploaded -- and, once the queue is
          running, so is "where has it got to?". */}
      {files.length > 1 ? (
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="m-0 text-sm font-semibold text-slate-900">
            {files.length} images · {formatBytes(totalBytes)}
          </p>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              onClick={() => onAddMoreFiles()}
              disabled={importing}
            >
              Add more files
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={onClearChosenFiles}
              disabled={importing}
            >
              Remove all
            </Button>
          </div>
        </div>
      ) : null}

      <ul
        className="m-0 flex list-none flex-col gap-2 p-0"
        data-testid="import-file-list"
      >
        {files.map((entry) => {
          const state = imports[entry.key];
          const applied = appliedPixelSize(entry);
          const declared =
            entry.scale?.state === "declared" ? entry.scale.pixelSizeNm : null;
          return (
            <li
              key={entry.key}
              className={cx(
                "flex flex-wrap items-center justify-between gap-3 rounded-md border px-3 py-2",
                state?.kind === "failed"
                  ? "border-red-200 bg-red-50"
                  : state?.kind === "imported"
                    ? "border-emerald-200 bg-emerald-50"
                    : highlightDropZone
                      ? "border-cyan-500 bg-cyan-50"
                      : "border-slate-200 bg-slate-50"
              )}
              data-testid={
                files.length === 1 ? "import-chosen-file" : "import-file-row"
              }
            >
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold text-slate-900">
                  {entry.file.name}
                </p>
                <p className="text-xs text-slate-600">
                  {formatBytes(entry.file.size)}
                  {declared !== null
                    ? ` · declares ${formatPixelSizeNm(declared)}`
                    : ""}
                  {/* What this row will actually be imported with. Only
                      for a batch: with one file the sentence under the
                      pixel-size box already says it in full, and saying it
                      twice in two different shapes is how two answers to
                      one question start. */}
                  {files.length > 1
                    ? ` · ${
                        applied.source === "typed"
                          ? `importing at ${formatPixelSizeNm(applied.valueNm)} (you typed it)`
                          : applied.source === "file"
                            ? `importing at ${formatPixelSizeNm(applied.valueNm)} (from the file)`
                            : applied.source === "none"
                              ? "no pixel size: measured in pixels"
                              : "pixel size not read here; the server will use whatever the file carries"
                      }`
                    : ""}
                </p>
                {state?.kind === "failed" ? (
                  <p className="mt-1 text-sm text-red-700" role="alert">
                    {state.message}
                  </p>
                ) : null}
              </div>
              <div className="flex items-center gap-2">
                {state && state.kind !== "failed" ? (
                  <span className="text-xs font-medium text-slate-700">
                    {state.kind === "imported"
                      ? "Imported"
                      : state.kind === "uploading"
                        ? "Uploading…"
                        : "Waiting"}
                  </span>
                ) : null}
                {files.length === 1 ? (
                  <Button
                    size="sm"
                    onClick={() => onChooseDifferentFile()}
                    disabled={importing}
                  >
                    Choose a different file
                  </Button>
                ) : null}
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() =>
                    files.length === 1
                      ? onClearChosenFiles()
                      : onRemoveFile(entry.key)
                  }
                  disabled={importing}
                >
                  Remove
                </Button>
              </div>
            </li>
          );
        })}
      </ul>

      {batchSummary && batchSummary.failed > 0 ? (
        <p
          className="m-0 text-sm text-slate-800"
          role="status"
          data-testid="import-batch-summary"
        >
          Imported {batchSummary.imported} of {batchSummary.attempted}{" "}
          {plural(batchSummary.attempted, "image")}. {batchSummary.failed}{" "}
          {plural(batchSummary.failed, "was", "were")} not imported and{" "}
          {batchSummary.failed === 1 ? "is" : "are"} still listed above, with
          the reason.
        </p>
      ) : null}
    </>
  );
}
