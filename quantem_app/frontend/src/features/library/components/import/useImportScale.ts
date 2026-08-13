/**
 * What pixel size each queued file will actually be imported with.
 *
 * Split out of `ImageUploadPanel.tsx` unchanged. Only one question on the
 * import form can silently ruin every number downstream, and this is it: packs
 * resample before they look for anything, so the scale decides which objects
 * exist, not just the units they are reported in. Everything the form says
 * about calibration — the census above the box, the per-row resolution tag,
 * the warning under the run options and the wording of the submit button — is
 * derived here, once, so those four surfaces cannot disagree.
 */

import { useCallback, useMemo } from "react";
import { parsePixelSizeInput } from "@/shared/pixelSize";
import type { ChosenFile } from "@/features/library/components/import/importValidation";

export interface AppliedPixelSize {
  valueNm: number | null;
  source: "typed" | "file" | "none" | "unknown";
}

export function useImportScale({
  files,
  pixelSizeText,
  replaceDeclaredPixelSizes,
}: {
  files: ChosenFile[];
  pixelSizeText: string;
  replaceDeclaredPixelSizes: boolean;
}) {
  /**
   * What the typed pixel size is *right now*, by the same rule the submit uses.
   *
   * Read from the draft text rather than from the saved asset, because the
   * whole point is to warn while the field is still empty and still editable.
   * A half-typed or invalid value counts as no pixel size: it is what would be
   * sent, and `parsePixelSizeInput` is the same check the server applies.
   */
  const draftPixelSizeNm = useMemo(() => {
    const parsed = parsePixelSizeInput(pixelSizeText);
    return parsed.error ? null : parsed.value;
  }, [pixelSizeText]);

  const singleFile = files.length === 1 ? files[0] : null;

  /**
   * Whether a typed value also overrides what a file declares.
   *
   * With one file, yes, always: the box sits under a sentence naming the value
   * the file declared and saying the typed one is used instead, which is the
   * documented way to correct a wrong tag. With a batch it is the checkbox, and
   * it is off until it is ticked.
   */
  const typedOverridesDeclared = files.length === 1 || replaceDeclaredPixelSizes;

  /**
   * The pixel size one file will actually be imported with.
   *
   * `"file"` and `"typed"` carry a number; `"none"` means the file was read and
   * declares nothing and nothing was typed for it, so the import is
   * uncalibrated; `"unknown"` means this browser could not read the header
   * (BigTIFF, an unfamiliar layout) and the server may still find one.
   */
  const appliedPixelSize = useCallback(
    (entry: ChosenFile): AppliedPixelSize => {
      const declared =
        entry.scale?.state === "declared" ? entry.scale.pixelSizeNm : null;
      if (draftPixelSizeNm !== null && (declared === null || typedOverridesDeclared)) {
        return { valueNm: draftPixelSizeNm, source: "typed" };
      }
      if (declared !== null) return { valueNm: declared, source: "file" };
      if (entry.scale?.state === "silent") return { valueNm: null, source: "none" };
      return { valueNm: null, source: "unknown" };
    },
    [draftPixelSizeNm, typedOverridesDeclared]
  );

  /** How the batch's pixel sizes break down, for the sentence above the box. */
  const scaleCensus = useMemo(() => {
    const declaredValues: number[] = [];
    let silent = 0;
    let unreadable = 0;
    let pending = 0;
    for (const entry of files) {
      if (entry.scale === null) pending += 1;
      else if (entry.scale.state === "declared") declaredValues.push(entry.scale.pixelSizeNm);
      else if (entry.scale.state === "silent") silent += 1;
      else unreadable += 1;
    }
    const unique = Array.from(new Set(declaredValues)).sort((a, b) => a - b);
    return {
      declared: declaredValues.length,
      declaredValues: unique,
      silent,
      unreadable,
      pending,
    };
  }, [files]);

  /** Files that will be imported with no pixel size at all, if nothing changes. */
  const uncalibratedCount = useMemo(
    () =>
      files.filter((entry) => appliedPixelSize(entry).source === "none").length,
    [appliedPixelSize, files]
  );
  const unknownScaleCount = useMemo(
    () =>
      files.filter((entry) => appliedPixelSize(entry).source === "unknown").length,
    [appliedPixelSize, files]
  );

  /**
   * The packs that would resample and cannot. Empty when every image is
   * calibrated, when nothing that declares a working resolution is ticked (ER
   * runs at native scale by design), or when the catalogue has not answered --
   * an unknown pack must not be claimed to declare a resolution.
   *
   * A batch reports the worst case: if *any* of these images could end up
   * without a pixel size -- read and silent, or not readable here -- the runs
   * about to be queued include an uncalibrated one, and the warning belongs on
   * screen. How strongly it is worded is `uncalibratedIsCertain`'s job, not this
   * value's.
   */
  const worstCasePixelSizeNm =
    uncalibratedCount > 0 || unknownScaleCount > 0
      ? null
      : files[0]
        ? appliedPixelSize(files[0]).valueNm
        : null;

  /**
   * Whether "this will run uncalibrated" is a fact or a possibility.
   *
   * Only files that were read and found silent earn the flat statement. A file
   * that could not be parsed here (BigTIFF, an unfamiliar layout) leaves it a
   * conditional -- which is weaker, and correct.
   */
  const uncalibratedIsCertain = uncalibratedCount > 0 && unknownScaleCount === 0;

  return {
    draftPixelSizeNm,
    singleFile,
    typedOverridesDeclared,
    appliedPixelSize,
    scaleCensus,
    uncalibratedCount,
    unknownScaleCount,
    worstCasePixelSizeNm,
    uncalibratedIsCertain,
  };
}

export type ImportScaleState = ReturnType<typeof useImportScale>;
