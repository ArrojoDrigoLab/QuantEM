/**
 * The empty state: a drop target that is also the file picker.
 *
 * Split out of `ImageUploadPanel.tsx` unchanged. It is a `<label>` for the
 * panel's file input, so a click anywhere in it opens the OS dialog through the
 * browser's own native path -- no JavaScript, nothing to go wrong, and it works
 * with the keyboard. The same area takes a drop, through the same validation
 * and the same state, with a visible hover state while a file is over it.
 */

import { cx } from "@/shared/ui/cx";
import {
  formatExtensionList,
  formatFormatFamilies,
  plural,
  type BatchSummary,
} from "@/features/library/components/import/importValidation";

export function ImportDropZone({
  acceptedExtensions,
  highlightDropZone,
  batchSummary,
  variant = "initial",
  compact = false,
}: {
  acceptedExtensions: string[];
  highlightDropZone: boolean;
  batchSummary: BatchSummary | null;
  variant?: "initial" | "additional";
  compact?: boolean;
}) {
  const additional = variant === "additional";
  return (
    <>
      <label
        htmlFor="file-input"
        data-testid={additional ? "import-add-more-drop-zone" : "import-drop-zone"}
        data-drag-active={highlightDropZone ? "true" : "false"}
        className={cx(
          "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-6 text-center transition-colors",
          additional ? "py-4" : compact ? "mt-2 py-4" : "mt-3 py-10",
          "peer-focus-visible:ring-2 peer-focus-visible:ring-cyan-500 peer-focus-visible:ring-offset-2",
          highlightDropZone
            ? "border-cyan-500 bg-cyan-50"
            : "border-slate-300 bg-slate-50 hover:border-cyan-400 hover:bg-cyan-50/40"
        )}
      >
        <span className="text-base font-semibold text-slate-900">
          {additional ? "Drop more images here" : "Drop your images here"}
        </span>
        <span className="text-sm text-slate-700">
          or{" "}
          <span className="inline-flex h-9 items-center rounded-md border border-slate-900 bg-slate-900 px-4 text-sm font-medium text-white shadow-sm">
            Choose files…
          </span>
        </span>
        <span
          className={compact ? "sr-only" : "text-sm text-slate-600"}
          id={additional ? undefined : "file-input-help"}
        >
          {formatFormatFamilies(acceptedExtensions)} from this computer.
          Nothing leaves this machine.
        </span>
        {!additional && !compact ? (
          <span className="text-xs text-slate-500">
            Accepted here: {formatExtensionList(acceptedExtensions)}, reported
            by the server. As many as you like, at once.
          </span>
        ) : null}
      </label>
      {/* A batch that finished with nothing left to look at. One image is
          already obvious from its new card, but "did all forty go?" is a
          question only a count can answer. */}
      {batchSummary && batchSummary.attempted > 1 ? (
        <p
          className="mt-3 text-sm text-slate-700"
          role="status"
          data-testid="import-batch-summary"
        >
          Imported {batchSummary.imported} of {batchSummary.attempted}{" "}
          {plural(batchSummary.attempted, "image")}.
        </p>
      ) : null}
    </>
  );
}
