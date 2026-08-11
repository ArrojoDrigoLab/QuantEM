import type { AnalysisMetricSummary } from "@/shared/types/analysis";

/** Return the complete, deduplicated note for one analysis metric. */
export function metricNote(row: AnalysisMetricSummary): string | null {
  const coverage = (row.note ?? "").trim();
  const estimator = (row.estimator_note ?? "").trim();
  if (!estimator) return coverage || null;
  if (!coverage) return estimator;
  return coverage.includes(estimator) ? coverage : `${coverage} ${estimator}`;
}
