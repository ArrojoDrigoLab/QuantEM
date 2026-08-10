/**
 * Number formatting for reportable values.
 *
 * The one rule: a value that was not computed renders as an em dash, never as
 * `0`, `NaN` or a blank cell. An undefined enrichment ratio and a measured
 * enrichment of zero are different findings, and a table that shows them the
 * same way is a defect.
 */

export const NOT_MEASURED = "—";

export function formatNumber(
  value: number | null | undefined,
  digits = 3
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return NOT_MEASURED;
  }
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e6)) {
    return value.toExponential(2);
  }
  return value.toFixed(digits);
}

export function formatInteger(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return NOT_MEASURED;
  }
  return Math.round(value).toLocaleString();
}

export function formatPercent(
  value: number | null | undefined,
  digits = 1
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return NOT_MEASURED;
  }
  return `${(value * 100).toFixed(digits)}%`;
}

/**
 * P-values from a Monte-Carlo null are bounded below by the replicate count:
 * with 20 replicates nothing can be smaller than 1/21, so a "< 0.001" here
 * would be a fiction. Reported at three decimals, with the floor stated by the
 * caller.
 */
export function formatPValue(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return NOT_MEASURED;
  }
  return value.toFixed(3);
}

export function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return NOT_MEASURED;
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let scaled = value;
  let unit = 0;
  while (scaled >= 1024 && unit < units.length - 1) {
    scaled /= 1024;
    unit += 1;
  }
  return `${scaled.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function formatTimestamp(value: string | null | undefined): string {
  if (!value) return NOT_MEASURED;
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) return value;
  return new Date(parsed).toLocaleString();
}

/** "2 min 05 s", for a training run or an analysis. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return NOT_MEASURED;
  }
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes} min ${rest.toFixed(0).padStart(2, "0")} s`;
}
