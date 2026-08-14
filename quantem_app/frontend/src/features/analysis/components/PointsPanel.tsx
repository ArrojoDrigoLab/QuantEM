/**
 * Where the points landed, and whether that is more than chance.
 *
 * Two honesty rules live here. `n_off_tissue` is shown the moment it is
 * non-zero (rule 5) because those points are missing from every denominator
 * below it. And the aggregation rule is stated next to the numbers (rule 4):
 * these are one image's values, and combining images means an unweighted mean
 * over units, never a pooled count.
 */

import { Badge, Panel } from "@/shared/ui/design";
import { formatInteger, formatNumber, formatPercent } from "@/shared/ui/format";
import type { AnalysisComposition, AnalysisPoints } from "@/shared/types/analysis";

export interface PointsPanelProps {
  points: AnalysisPoints;
  composition: AnalysisComposition | null;
  /** "centroids" | "csv" — what the points are, so enrichment can be read. */
  pointsSource: string | null;
}

export function PointsPanel({ points, composition, pointsSource }: PointsPanelProps) {
  const names = Array.from(
    new Set([
      ...Object.keys(points.counts),
      ...Object.keys(points.enrichment),
    ])
  ).sort();

  return (
    <Panel className="p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="m-0 text-sm font-semibold text-slate-900">
          Point distribution
        </h3>
        <Badge tone="info">
          {pointsSource === "centroids"
            ? "Object centroids"
            : pointsSource === "csv"
              ? "Imported CSV"
              : "Points"}
        </Badge>
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">Total</dt>
          <dd className="m-0 text-sm font-semibold tabular-nums text-slate-900">
            {formatInteger(points.n_total)}
          </dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">
            On tissue
          </dt>
          <dd className="m-0 text-sm font-semibold tabular-nums text-slate-900">
            {formatInteger(points.n_on_tissue)}
          </dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">
            Off tissue
          </dt>
          <dd
            className={
              points.n_off_tissue > 0
                ? "m-0 text-sm font-semibold tabular-nums text-amber-700"
                : "m-0 text-sm font-semibold tabular-nums text-slate-900"
            }
          >
            {formatInteger(points.n_off_tissue)}
          </dd>
        </div>
        {/* Beside off-tissue, because the three of them are how `n_total` is
            spent and only one of them was on screen. `n_total ==
            n_on_tissue + n_off_tissue + n_unreadable`; an unreadable row is
            deliberately *not* counted as off-tissue, since a point that cannot
            be read is nowhere rather than outside something. Rendered only
            when the server sent the field: a run stored before it existed must
            not be shown a fabricated 0. */}
        {typeof points.n_unreadable === "number" ? (
          <div>
            <dt className="text-xs uppercase tracking-wide text-slate-500">
              Unreadable
            </dt>
            <dd
              className={
                points.n_unreadable > 0
                  ? "m-0 text-sm font-semibold tabular-nums text-amber-700"
                  : "m-0 text-sm font-semibold tabular-nums text-slate-900"
              }
            >
              {formatInteger(points.n_unreadable)}
            </dd>
          </div>
        ) : null}
        {typeof points.n_out_of_bounds === "number" ? (
          <div>
            <dt className="text-xs uppercase tracking-wide text-slate-500">
              Outside the image
            </dt>
            <dd
              className={
                points.n_out_of_bounds > 0
                  ? "m-0 text-sm font-semibold tabular-nums text-amber-700"
                  : "m-0 text-sm font-semibold tabular-nums text-slate-900"
              }
            >
              {formatInteger(points.n_out_of_bounds)}
            </dd>
          </div>
        ) : null}
      </dl>

      {points.n_off_tissue > 0 ? (
        <p className="mt-2 mb-0 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {formatInteger(points.n_off_tissue)} of{" "}
          {formatInteger(points.n_total)} points fell outside the analysis mask and
          were excluded. Every fraction and enrichment below is out of the{" "}
          {formatInteger(points.n_on_tissue)} points inside the analysis mask.
        </p>
      ) : null}

      {points.n_unreadable ? (
        <p className="mt-2 mb-0 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {formatInteger(points.n_unreadable)} of{" "}
          {formatInteger(points.n_total)}{" "}
          {points.n_unreadable === 1 ? "point has" : "points have"} a coordinate
          that is not a position (missing, or infinite) and{" "}
          {points.n_unreadable === 1 ? "was" : "were"} dropped before any
          measurement. {points.n_unreadable === 1 ? "It is" : "They are"} in no
          count, fraction or enrichment here, and not in the off-tissue total
          either — a point that cannot be read is nowhere, not outside
          something.
        </p>
      ) : null}

      {points.n_out_of_bounds ? (
        <p className="mt-2 mb-0 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {formatInteger(points.n_out_of_bounds)} of{" "}
          {formatInteger(points.n_total)}{" "}
          {points.n_out_of_bounds === 1 ? "point lies" : "points lie"} outside
          the image and {points.n_out_of_bounds === 1 ? "was" : "were"} clipped
          onto its border, which is where{" "}
          {points.n_out_of_bounds === 1 ? "it is" : "they are"} counted below.
          Coordinates are expected in image pixels; a whole point set landing on
          one edge is what a CSV in nanometres, or an export from a differently
          cropped copy of this image, looks like. Check the units before quoting
          any enrichment.
        </p>
      ) : null}

      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[520px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="py-1 pr-3 font-semibold">Compartment</th>
              <th className="py-1 pr-3 font-semibold">Points</th>
              <th className="py-1 pr-3 font-semibold">Point fraction</th>
              <th className="py-1 pr-3 font-semibold">Area fraction</th>
              <th className="py-1 font-semibold">Enrichment</th>
            </tr>
          </thead>
          <tbody>
            {names.map((name) => {
              const enrichment = points.enrichment[name] ?? null;
              return (
                <tr key={name} className="border-b border-slate-100">
                  <td className="py-1 pr-3 text-slate-700">{name}</td>
                  <td className="py-1 pr-3 tabular-nums text-slate-900">
                    {formatInteger(points.counts[name] ?? 0)}
                  </td>
                  <td className="py-1 pr-3 tabular-nums text-slate-900">
                    {formatPercent(points.fractions[name] ?? null)}
                  </td>
                  <td className="py-1 pr-3 tabular-nums text-slate-900">
                    {formatPercent(
                      composition?.area_fractions?.[name] ?? null
                    )}
                  </td>
                  <td className="py-1 tabular-nums text-slate-900">
                    {formatNumber(enrichment, 2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="mt-3 mb-0 text-xs text-slate-500">
        Enrichment is the point fraction divided by the area fraction: 1.0 is
        chance, and an em dash means the compartment has no area, so the ratio is
        undefined rather than infinite. These are one image&apos;s values — a
        group value is the unweighted mean over experimental units, never a
        pooled count weighted by how many points each image contributed.
      </p>
    </Panel>
  );
}
