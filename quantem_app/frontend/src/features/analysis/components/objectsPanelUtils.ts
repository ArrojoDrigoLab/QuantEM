import type { AnalysisMetricSummary } from "@/shared/types/analysis";

const CIRCULARITY_FAILURE =
  "Their 4*pi*area/perimeter^2 came out above 1.015. The theoretical ceiling is 1, " +
  "and the estimator's measured envelope for a genuinely round object reaches about " +
  "1.011 — so a value beyond 1.015 measures the estimator failing on a small object, " +
  "not the object, and is left blank rather than exported as a roundness.";

/** Return the concise partial-coverage note shown beside one metric. */
export function metricNote(
  metric: string,
  row: AnalysisMetricSummary
): string | null {
  const total = row.n_objects ?? null;
  const missing = row.n_missing ?? 0;
  if (!total || missing <= 0) return null;

  const measured = row.n;
  if (metric === "mean_prob") {
    return (
      `Measured on ${measured} of ${total} confirmed objects; ${missing} ` +
      `carr${missing === 1 ? "ies" : "y"} no stored value for this metric. ` +
      "User-drawn or defined objects have no model probability behind them."
    );
  }

  const unreportable = row.n_unreportable ?? 0;
  const absent = Math.max(0, missing - unreportable);
  const clauses: string[] = [];
  if (absent > 0) {
    clauses.push(
      `${absent} carr${absent === 1 ? "ies" : "y"} no stored value for this metric`
    );
  }
  if (unreportable > 0) {
    clauses.push(
      `${unreportable} ${unreportable === 1 ? "was" : "were"} measured and could not be reported`
    );
  }

  const reason =
    metric === "circularity" && unreportable > 0
      ? CIRCULARITY_FAILURE
      : (row.unreportable_reason ?? "").trim();
  return (
    `Measured on ${measured} of ${total} confirmed objects; ${clauses.join(" and ")}.` +
    (reason ? ` ${reason}` : "")
  );
}
