/**
 * Every sentence the include-level dial can show, in one place.
 *
 * The vocabulary rule this feature inherits (see `features/improve/copy.ts`):
 * **include level**, never "threshold"; **kept** and **removed**, never
 * "confirmed" and "excluded". The word "threshold" appears in this directory's
 * name and in the backend, and nowhere a user can read.
 *
 * No sentence here names a route, an HTTP verb, a job type or a file path
 * (invariant I-12). Where a failure has a way out, the sentence names the
 * *control* that takes it.
 */

/** How a level is written wherever one is shown. Two decimals, matching Improve. */
export function formatIncludeLevel(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : value.toFixed(2);
}

/**
 * Below this, two levels are the same level.
 *
 * The slider steps in hundredths, so this only has to be small enough not to
 * swallow one step and large enough to absorb float noise from a round trip
 * through JSON.
 */
export const LEVEL_EPSILON = 0.005;

export function levelsDiffer(
  a: number | null | undefined,
  b: number | null | undefined
): boolean {
  if (a === null || a === undefined) return b !== null && b !== undefined;
  if (b === null || b === undefined) return true;
  return Math.abs(a - b) > LEVEL_EPSILON;
}

export const DIAL_TITLE = "Include level";

/**
 * What the dial *is*, in one line, before the user moves anything.
 *
 * Says which way is which, because a bare 0-1 number over a picture of cells
 * carries no direction and guessing wrong costs a re-run.
 */
export const DIAL_EXPLANATION =
  "How sure I have to be before I call something an object. Lower finds more " +
  "and includes shakier ones; higher finds fewer and keeps only the clear ones.";

/** Shown when no dial move has been recorded against this image. */
export const LEVEL_NOT_SET =
  "These objects came straight from the model, at its own level.";

/**
 * What it costs. The honest headline of the whole feature: it is not a re-run.
 */
export const DIAL_COST =
  "Takes a few seconds. The model does not run again — I re-read the result it " +
  "already saved.";

/**
 * The preservation promise, in this feature's own words.
 *
 * Not a reassurance: it is a description of what the extraction code does, and
 * `segmentation/tests/test_annotation_preservation_invariant.py` is what keeps
 * it true on this path as well as on the model path.
 */
export const DIAL_PRESERVATION =
  "Nothing you have kept, removed or drawn by hand changes, and areas you have " +
  "marked finished are left exactly as they are. Only my own guesses are redone.";

/** While the re-extract is queued or running. */
export const DIAL_WORKING = "Finding the objects at the new level…";

/** After it lands. */
export function dialDone(objectCount: number): string {
  return objectCount === 1
    ? "1 object at this include level."
    : `${objectCount} objects at this include level.`;
}

/**
 * What a user can do when the dial cannot move.
 *
 * The server's own sentence says *why* -- and the two reasons are different
 * futures, so it is never replaced with a generic one. This is appended after
 * it, naming the control rather than the request.
 */
export const DIAL_BLOCKED_ACTION =
  "Run the model on this image from the labeling header, and the dial works " +
  "from then on.";

export const DIAL_FAILED_FALLBACK =
  "The objects could not be redone at that include level.";
