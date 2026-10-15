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

export function thresholdTooltip(hasResult: boolean, defaultThreshold: number | null | undefined): string {
  const explanation = "Probability threshold for objects.";
  if (!hasResult) return `${explanation} Run the model first to view results and adjust this value.`;
  return defaultThreshold == null ? explanation
    : `${explanation} The default threshold for this model is ${formatIncludeLevel(defaultThreshold)}.`;
}

export const DIAL_WORKING = "Recalculating…";

/** Appended to the server's specific reason when the control is unavailable. */
export const DIAL_FAILED_FALLBACK =
  "The objects could not be re-extracted at that threshold.";
