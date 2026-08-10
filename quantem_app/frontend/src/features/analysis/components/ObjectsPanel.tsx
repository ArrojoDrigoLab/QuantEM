/**
 * Object count, density, and the morphometric distributions.
 *
 * Every mean is shown with its sd, median, IQR, range and n. A mean quoted
 * without its spread is the single easiest way to mislead with this table, and
 * the backend already computes all of it, so there is no reason to hide it.
 *
 * Only CONFIRMED objects are counted — the backend excludes inferred and
 * candidate ones — and the panel says so, because "n" here is not the number of
 * things the model found.
 *
 * Any metric that arrives with a note prints it under its own row. That is not
 * decoration: the caveat block above this table promises the reader that a
 * partly-measured metric "carries its own n and the reason for it in the
 * summary table", and an estimator note applies to a column that is fully
 * populated and has no n to explain it. See `metricNote`.
 */

import { Fragment } from "react";

import { Badge, Button, Panel } from "@/shared/ui/design";
import { formatInteger, formatNumber, NOT_MEASURED } from "@/shared/ui/format";
import { downloadCsv } from "@/utils/downloadText";
import type {
  AnalysisMetricSummary,
  AnalysisObjects,
} from "@/shared/types/analysis";

/** Metric rows we never show: constants, not distributions. */
const HIDDEN_METRICS = new Set(["pixel_size_nm"]);

/**
 * Everything this metric's own row has to say, deduplicated.
 *
 * Two keys arrive and they overlap. `note` is coverage ("measured on 4 of 90")
 * with the estimator paragraph already appended when the metric has one;
 * `estimator_note` is that paragraph on its own, and it is published whether or
 * not anything was blanked. Printing both concatenated repeats the estimator
 * text on every partly-measured row, and printing only `note` was not an option
 * either — a metric may carry the estimator note without a coverage sentence.
 * So: take `note`, and append `estimator_note` only when it is not already in
 * there.
 *
 * Why this row exists at all: the caveat block above the table tells a reader
 * that a metric "carries its own n and the reason for it in the summary table",
 * and until now the summary table carried the n and not the reason. Worse, a
 * fully populated circularity column shipped with no word of an estimator bias
 * that is monotone in object size — eight real mitochondrial outlines scaled to
 * 0.6x, a pure size change with identical shapes, move mean circularity
 * 0.6186 -> 0.6409, paired t = 3.596, p = 0.0088. That is a publishable
 * "mitochondria became more circular" out of a correct segmentation, and the
 * sentence that prevents it has to be beside the number.
 */
export function metricNote(row: AnalysisMetricSummary): string | null {
  const coverage = (row.note ?? "").trim();
  const estimator = (row.estimator_note ?? "").trim();
  if (!estimator) return coverage || null;
  if (!coverage) return estimator;
  return coverage.includes(estimator) ? coverage : `${coverage} ${estimator}`;
}

export interface ObjectsPanelProps {
  objects: AnalysisObjects;
  calibrated: boolean;
  /** Where the full per-object table can be downloaded, if the bundle exists. */
  objectsCsvUrl: string | null;
  downloadStem: string;
}

export function ObjectsPanel({
  objects,
  calibrated,
  objectsCsvUrl,
  downloadStem,
}: ObjectsPanelProps) {
  const metricNames = Object.keys(objects.summary)
    .filter((name) => !HIDDEN_METRICS.has(name))
    .filter((name) => (objects.summary[name]?.n ?? 0) > 0)
    .sort();

  const handleDownloadSummary = () => {
    downloadCsv(
      `object-summary-${downloadStem}.csv`,
      // `note` travels with the numbers. A mean lifted out of this file into a
      // spreadsheet looks the same whether or not its estimator is biased, and
      // the sentence that says it is was on a screen the file does not carry.
      ["metric", "n", "mean", "sd", "median", "iqr", "min", "max", "note"],
      metricNames.map((name) => {
        const row = objects.summary[name];
        return [
          name,
          row.n,
          row.mean ?? "",
          row.sd ?? "",
          row.median ?? "",
          row.iqr ?? "",
          row.min ?? "",
          row.max ?? "",
          metricNote(row) ?? "",
        ];
      })
    );
  };

  return (
    <Panel className="p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="m-0 text-sm font-semibold text-slate-900">Objects</h3>
        <Badge tone="info">Confirmed objects only</Badge>
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">Count</dt>
          <dd className="m-0 text-sm font-semibold tabular-nums text-slate-900">
            {formatInteger(objects.n)}
          </dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">
            Density
          </dt>
          <dd className="m-0 text-sm font-semibold tabular-nums text-slate-900">
            {calibrated && objects.density.per_um2 !== null
              ? `${formatNumber(objects.density.per_um2, 4)} / µm²`
              : NOT_MEASURED}
          </dd>
        </div>
        {calibrated ? (
          <div>
            <dt className="text-xs uppercase tracking-wide text-slate-500">
              Tissue area
            </dt>
            <dd className="m-0 text-sm font-semibold tabular-nums text-slate-900">
              {formatNumber(objects.density.tissue_um2 ?? null, 2)} µm²
            </dd>
          </div>
        ) : null}
      </dl>

      {!calibrated ? (
        <p className="mt-2 mb-0 text-xs text-amber-700">
          This image has no pixel size, so density and every physical-unit
          measurement are unavailable. The pixel-unit metrics below are still
          exact.
        </p>
      ) : null}

      {metricNames.length === 0 ? (
        <p className="mt-3 mb-0 text-sm text-slate-500">
          No confirmed objects carry stored features, so there is nothing to
          summarise.
        </p>
      ) : (
        <>
          <div className="mt-3 max-h-[420px] overflow-auto">
            <table className="w-full min-w-[600px] border-collapse text-sm">
              <thead className="sticky top-0 bg-white">
                <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
                  <th className="py-1 pr-3 font-semibold">Metric</th>
                  <th className="py-1 pr-3 font-semibold">n</th>
                  <th className="py-1 pr-3 font-semibold">Mean</th>
                  <th className="py-1 pr-3 font-semibold">SD</th>
                  <th className="py-1 pr-3 font-semibold">Median</th>
                  <th className="py-1 pr-3 font-semibold">IQR</th>
                  <th className="py-1 pr-3 font-semibold">Min</th>
                  <th className="py-1 font-semibold">Max</th>
                </tr>
              </thead>
              <tbody>
                {metricNames.map((name) => {
                  const row = objects.summary[name];
                  const note = metricNote(row);
                  return (
                    <Fragment key={name}>
                      <tr
                        className={
                          note
                            ? "border-b border-slate-100/0"
                            : "border-b border-slate-100"
                        }
                      >
                        <td className="py-1 pr-3 text-slate-700">{name}</td>
                        <td className="py-1 pr-3 tabular-nums text-slate-900">
                          {formatInteger(row.n)}
                        </td>
                        <td className="py-1 pr-3 tabular-nums text-slate-900">
                          {formatNumber(row.mean)}
                        </td>
                        <td className="py-1 pr-3 tabular-nums text-slate-900">
                          {formatNumber(row.sd)}
                        </td>
                        <td className="py-1 pr-3 tabular-nums text-slate-900">
                          {formatNumber(row.median)}
                        </td>
                        <td className="py-1 pr-3 tabular-nums text-slate-900">
                          {formatNumber(row.iqr)}
                        </td>
                        <td className="py-1 pr-3 tabular-nums text-slate-900">
                          {formatNumber(row.min)}
                        </td>
                        <td className="py-1 tabular-nums text-slate-900">
                          {formatNumber(row.max)}
                        </td>
                      </tr>
                      {/* Under the row, spanning it, not folded into a title
                          attribute: a bias warning nobody hovers over is a
                          warning nobody reads. */}
                      {note ? (
                        <tr className="border-b border-slate-100">
                          <td colSpan={8} className="pb-2 pr-3">
                            <p className="m-0 text-xs leading-relaxed text-amber-800">
                              <span className="font-semibold">{name}:</span>{" "}
                              {note}
                            </p>
                          </td>
                        </tr>
                      ) : null}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Button size="sm" onClick={handleDownloadSummary}>
              Download this table
            </Button>
            {objectsCsvUrl ? (
              <a
                className="inline-flex h-8 items-center rounded-md border border-slate-300 bg-white px-3 text-xs font-medium text-slate-800 shadow-sm hover:bg-slate-50"
                href={objectsCsvUrl}
                download
              >
                Download objects.csv (one row per object)
              </a>
            ) : null}
            <p className="m-0 text-xs text-slate-500">
              SD is the sample standard deviation; it is blank for n = 1.
            </p>
          </div>
        </>
      )}
    </Panel>
  );
}
