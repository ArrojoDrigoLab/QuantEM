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
 * Partly measured metrics carry a compact tooltip beside the metric name.
 */

import { Button, Panel } from "@/shared/ui/design";
import { formatInteger, formatNumber, NOT_MEASURED } from "@/shared/ui/format";
import { downloadCsv } from "@/utils/downloadText";
import type { AnalysisObjects } from "@/shared/types/analysis";
import { metricNote } from "@/features/analysis/components/objectsPanelUtils";

/** Metric rows we never show: constants, not distributions. */
const HIDDEN_METRICS = new Set(["pixel_size_nm"]);

function MetricNoteIndicator({ metric, note }: { metric: string; note: string }) {
  const neutral = metric === "mean_prob";
  return (
    <button
      type="button"
      className={`ml-1 inline-flex h-4 w-4 items-center justify-center rounded-full border text-[11px] font-bold leading-none ${
        neutral
          ? "border-slate-400 text-slate-600"
          : "border-amber-500 bg-amber-50 text-amber-700"
      }`}
      aria-label={`${metric} measurement note: ${note}`}
      title={note}
    >
      !
    </button>
  );
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
      // Keep the same concise coverage explanation in the downloaded summary.
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
          metricNote(name, row) ?? "",
        ];
      })
    );
  };

  return (
    <Panel className="p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="m-0 text-sm font-semibold text-slate-900">Objects</h3>
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" onClick={handleDownloadSummary}>
            Download Summary Table
          </Button>
          {objectsCsvUrl ? (
            <a
              className="inline-flex h-8 items-center rounded-md border border-slate-300 bg-white px-3 text-xs font-medium text-slate-800 shadow-sm hover:bg-slate-50"
              href={objectsCsvUrl}
              download
            >
              Download All Objects
            </a>
          ) : null}
        </div>
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
                  const note = metricNote(name, row);
                  return (
                    <tr key={name} className="border-b border-slate-100">
                        <td className="py-1 pr-3 text-slate-700">
                          <span className="inline-flex items-center">
                            {name}
                            {note ? (
                              <MetricNoteIndicator metric={name} note={note} />
                            ) : null}
                          </span>
                        </td>
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
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-2">
            <p className="m-0 text-xs text-slate-500">
              SD is the sample standard deviation; it is blank for n = 1.
            </p>
          </div>
        </>
      )}
    </Panel>
  );
}
