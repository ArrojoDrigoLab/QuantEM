/** User-facing copy for the threshold control. */

/** How a threshold is written wherever one is shown. */
export function formatIncludeLevel(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : value.toFixed(2);
}

/** The slider steps in hundredths; this only absorbs JSON float noise. */
export const LEVEL_EPSILON = 0.005;

export function levelsDiffer(
  a: number | null | undefined,
  b: number | null | undefined
): boolean {
  if (a === null || a === undefined) return b !== null && b !== undefined;
  if (b === null || b === undefined) return true;
  return Math.abs(a - b) > LEVEL_EPSILON;
}

export const DIAL_TITLE = "Threshold";

/** Available on hover without taking space from the control. */
export const DIAL_TOOLTIP =
  "Adjust the saved model result before creating candidates. Lower values include " +
  "weaker evidence; higher values keep only stronger evidence.";

export const DIAL_BLOCKED_TOOLTIP =
  "No stored result is kept for this image, so the include level cannot be moved " +
  "without running the model again. Running it once saves one, and the level can " +
  "be moved freely from then on. Run the model on this image from the labeling " +
  "header, and the threshold can be adjusted afterwards.";

/** While the re-extract is queued or running. */
export const DIAL_WORKING = "Applying threshold…";

/** Appended to the server's specific reason when the control is unavailable. */
export const DIAL_FAILED_FALLBACK =
  "The objects could not be re-extracted at that threshold.";
