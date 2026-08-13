/**
 * Whether a segmentation's objects were made at the pixel size the image
 * records now — and the warning to show when they were not.
 *
 * The gap this closes was reported end to end. The labeling header showed a
 * current "5 nm/px" tag and an ordinary objects chip over a set produced before
 * that number existed; nothing on the screen where a user decides the work is
 * finished said so, and neither did the Analysis screen before a run was spent.
 * It surfaced in the finished bundle, as blank micron columns and
 * `calibrated: false`, after the run had already cost its minutes.
 *
 * The verdict is the server's (`objects_pixel_size.predates_calibration`, the
 * same `calibrated_after_the_fact` predicate `run_analysis` blanks its
 * physical units on), never re-derived here: a screen that says the objects
 * are fine over a bundle that refuses to convert them is a disagreement nobody
 * sees until they compare the two. This module only puts it into sentences,
 * and both screens read the same sentences so they cannot drift.
 */

import { formatPixelSizeNm } from "@/shared/pixelSize";
import type {
  ImageSegmentation,
  SegmentationObjectsPixelSize,
} from "@/shared/types/images";

export interface ObjectsPixelSizeWarning {
  /** One short line for a chip: the claim, no rationale. */
  summary: string;
  /** The full explanation, fit to stand as body text on the Analysis screen. */
  detail: string;
  /** What an analysis run will actually do about it. */
  consequence: string;
}

/** One produced_nm entry as words. `null` is a real member, not a gap. */
function describeProducedEntry(value: unknown): string {
  if (value === null) return "no pixel size";
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    return formatPixelSizeNm(value);
  }
  // A damaged stamp can hold anything; the server reports it unchanged rather
  // than failing the read, and so does this.
  return String(value);
}

/**
 * The warning the two screens share, or null when there is nothing to say.
 *
 * Null covers: no payload field (older backend), a segmentation with no
 * objects (the server sends null), and objects that were produced at the
 * scale the image records (`predates_calibration: false`). Only the state the
 * analysis will actually refuse to convert gets a warning — firing on every
 * calibrated image is the crying-wolf failure the import form already had
 * fixed once.
 */
export function describeObjectsPixelSize(
  segmentation: Pick<ImageSegmentation, "objects_pixel_size"> | null | undefined
): ObjectsPixelSizeWarning | null {
  const info: SegmentationObjectsPixelSize | null | undefined =
    segmentation?.objects_pixel_size;
  if (!info || !info.predates_calibration) return null;

  const produced = Array.isArray(info.produced_nm) ? info.produced_nm : [];
  const producedNumbers = produced.filter((value) => value !== null);

  // `predates_calibration` asserts at least one null-stamped object; whether
  // the whole set is uncalibrated or only part of it changes the sentence.
  const scope =
    producedNumbers.length > 0
      ? `Some of the objects here were produced while this image had no pixel size (runs here recorded: ${produced
          .map(describeProducedEntry)
          .join(", ")}).`
      : "The objects here were produced while this image had no pixel size.";

  const detail =
    `${scope} The pixel size the image records now was set after they were ` +
    "made, so they were not measured at it — a number typed in afterwards " +
    "does not re-measure objects that already exist.";

  const unstamped =
    typeof info.unstamped_count === "number" && info.unstamped_count > 0
      ? ` ${info.unstamped_count} object(s) here carry no run record at all ` +
        "(drawn by hand, or made before runs were recorded); that says " +
        "nothing about their scale."
      : "";

  return {
    summary: "Objects predate the pixel size",
    detail: detail + unstamped,
    consequence:
      "An analysis of this segmentation reports these objects in pixels, not " +
      "µm² or nm, and its bundle carries the wrong-scale caveat.",
  };
}
