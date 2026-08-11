import type { HomeEntry } from "@/shared/types/images";

/**
 * What the card says an image is doing, in the user's words.
 *
 * Two things were wrong with the previous version and both of them made the
 * card lie for the entire import.
 *
 * **The order.** The first test was `is_workable && !ngff_ready`, and
 * `is_workable` is `openable is not None` (`assets/serializers.py:203`) -- true
 * the instant the staged rendition row is written, which is within a second of
 * the upload finishing. So the branch that reports a percentage was
 * unreachable, and a 475 MP import read "NGFF pending" for its whole 100 s,
 * even though `preprocess_progress` was being written once a second by
 * `set_stage` and was sitting in the very payload this function is reading.
 * The percentage branch is now first.
 *
 * **The words.** "NGFF pending" names an on-disk format the user has never
 * heard of and cannot act on; the plan's vocabulary table puts NGFF in the
 * drawer or nowhere. The states a person needs are the three in §1.2:
 * queued, preparing (with its number), ready -- plus the two ways it can stop.
 *
 * `preprocess_progress` is a float 0-100 written by
 * `quantem.assets.preprocess_status.set_stage`. It is clamped here rather than
 * trusted: a percentage over 100 on a card is the kind of detail that makes a
 * user stop believing the other numbers.
 */
export function getStageDisplay(image: HomeEntry): string {
  if (image.preprocess_stage === "FAILED") return "Failed";
  if (image.preprocess_stage === "CANCELLED") return "Cancelled";
  if (image.preprocess_stage === "DONE" || image.preprocess_stage === "SKIPPED") {
    return "Ready";
  }
  // Openable wins over "still busy", and the two really do overlap: the
  // pyramid becomes valid partway through `ENCODING` (observed live at 50% on
  // a 475 MP import), and from that moment `ngff_ready`, `can_view` and
  // `can_segment` are all true and the viewer will open the image. What is
  // left running is the preview rendition and the coarser levels, which block
  // nothing the user can ask for. Reporting "Preparing 50%" beside a green
  // badge, next to a strip saying the image is ready, is two answers to one
  // question -- and the answer the user acts on is this one.
  if (image.ngff_ready) return "Ready";
  if (
    image.preprocess_stage === "ENCODING" ||
    image.preprocess_stage === "SAM" ||
    image.preprocess_stage === "FEATURES"
  ) {
    return `Preparing ${formatPreprocessPercent(image.preprocess_progress)}%`;
  }
  // "NONE": the asset row exists and nothing has picked the work up yet. That
  // is a real, distinct state -- a stalled queue looks exactly like this and
  // "Preparing" would hide it.
  return image.ngff_ready ? "Ready" : "Queued";
}

function formatPreprocessPercent(progress: number | null | undefined): number {
  if (progress === null || progress === undefined || !Number.isFinite(progress)) {
    return 0;
  }
  return Math.min(100, Math.max(0, Math.round(progress)));
}

/** True while the server is still working on this image. */
export function isLibraryEntryProcessing(image: HomeEntry): boolean {
  return !["DONE", "FAILED", "CANCELLED", "NONE", "SKIPPED"].includes(
    image.preprocess_stage
  );
}

/**
 * True while the library still has to keep asking about this image.
 *
 * Wider than {@link isLibraryEntryProcessing} by exactly one case: an asset
 * sitting at `NONE` with no pyramid. Nothing is running on it, so it is not
 * "processing", but it is also not finished -- that is what a just-created
 * asset looks like in the seconds before the scheduler picks its job up, and
 * polling that stopped there would freeze a fresh import on "Queued" until the
 * user reloaded. Every terminal stage (DONE, SKIPPED, FAILED, CANCELLED) still
 * ends the polling, so a genuinely stuck queue does not spin forever either.
 */
export function isLibraryEntryUnfinished(image: HomeEntry): boolean {
  if (isLibraryEntryProcessing(image)) return true;
  return image.preprocess_stage === "NONE" && !image.ngff_ready;
}
