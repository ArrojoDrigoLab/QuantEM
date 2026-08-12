/**
 * Everything one analysis run measured.
 *
 * The caveats come first, above any number they qualify. They are produced at
 * the point of measurement (uncalibrated image, no tissue mask, points that
 * fell off the tissue, a compartment whose enrichment is circular by
 * construction), and a reader who scrolls straight to a table has still seen
 * them.
 */

import { Badge, Panel } from "@/shared/ui/design";
import { formatNumber, formatTimestamp } from "@/shared/ui/format";
import { getAnalysisExportUrl } from "@/shared/api/analysis";
import type { AnalysisRun } from "@/shared/types/analysis";
import { BandHistogram } from "@/features/analysis/components/BandHistogram";
import { CompositionPanel } from "@/features/analysis/components/CompositionPanel";
import { MonteCarloPanel } from "@/features/analysis/components/MonteCarloPanel";
import { ObjectsPanel } from "@/features/analysis/components/ObjectsPanel";

const EXPORT_DESCRIPTIONS: Record<string, string> = {
  "objects.csv": "One row per confirmed object, every metric.",
  "image_summary.csv": "One row per image: fractions, density, enrichment, z.",
  "manifest.json": "What produced these numbers, and the aggregation rule.",
};

function ExportLink({ runId, name }: { runId: string; name: string }) {
  return (
    <a
      className="inline-flex flex-col rounded-md border border-slate-300 bg-white px-3 py-2 text-left shadow-sm hover:bg-slate-50"
      href={getAnalysisExportUrl(runId, name)}
      download
    >
      <span className="text-sm font-medium text-slate-900">{name}</span>
      <span className="text-xs text-slate-500">
        {EXPORT_DESCRIPTIONS[name] ?? "Export bundle file."}
      </span>
    </a>
  );
}

export interface AnalysisResultsProps {
  run: AnalysisRun;
}

/**
 * How many points this section is actually about.
 *
 * The badge above it used to divide `n_inside` by `points.n_total`, which is a
 * different set: `distance_to_boundary` drops every row whose coordinate is not
 * a position, so the median, the bands and `n_inside` are all over `n_measured`
 * and the denominator on screen was one of the few numbers on this page that
 * could be quietly wrong. This line reconciles the two whenever they differ,
 * beside the numbers rather than only in the caveat block at the top.
 *
 * `n_out_of_image` is inside `n_measured`, not beside it: those points were
 * clipped onto the border and measured from there, which is a measurement of a
 * pixel nobody chose. Silent when everything agrees.
 */
function DistanceCoverage({
  distances,
  points,
}: {
  distances: NonNullable<AnalysisRun["distances"]>;
  points: AnalysisRun["points"];
}) {
  const measured = distances.n_measured;
  const total = points?.n_total ?? null;
  const dropped =
    typeof measured === "number" && total !== null && total > measured
      ? total - measured
      : 0;
  const clipped = distances.n_out_of_image ?? 0;
  if (dropped === 0 && clipped === 0) return null;

  return (
    <p className="m-0 mt-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
      {dropped > 0 ? (
        <>
          These bands, the median and the inside count cover{" "}
          {formatNumber(measured, 0)} of the run&apos;s {formatNumber(total, 0)}{" "}
          points. The other {dropped} {dropped === 1 ? "has" : "have"} a
          coordinate that is not a position, so {dropped === 1 ? "it has" : "they have"}{" "}
          no distance to anything.{" "}
        </>
      ) : null}
      {clipped > 0 ? (
        <>
          {clipped} of the measured points {clipped === 1 ? "lies" : "lie"}{" "}
          outside the image and {clipped === 1 ? "was" : "were"} clipped onto the
          border, so {clipped === 1 ? "its" : "their"} distance is measured from
          that border pixel and not from the coordinates given. Every band and
          the median include {clipped === 1 ? "it" : "them"}.
        </>
      ) : null}
    </p>
  );
}

/**
 * Results only. The caller owns the run's *state*.
 *
 * This component used to render its own "failed" and "pending" panels beside
 * the screen's job panel, and the two were resolved independently: the screen
 * printed the job's traceback while this printed `run.error`, so one failure
 * appeared twice, under a heading that said the run was still pending. There
 * is now one reconciler (`resolveAnalysisRunState`) and one panel, and a
 * non-SUCCESS run never reaches this component.
 */
export function AnalysisResults({ run }: AnalysisResultsProps) {
  const calibrated = Boolean(run.calibrated);
  const wholeImage = !run.params?.tissue_segmentation_id;

  // Defensive: rendering half-written numbers from a run that has not finished
  // would be worse than rendering nothing, and the caller already shows why.
  if (run.status !== "SUCCESS") return null;

  return (
    <div className="flex flex-col gap-4">
      <Panel className="p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="m-0 text-base font-semibold text-slate-950">
              Analysis run
            </h2>
            <p className="m-0 mt-1 text-xs text-slate-500">
              {formatTimestamp(run.created_at)} · id {run.id}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {run.group ? <Badge tone="info">group: {run.group}</Badge> : null}
            {calibrated ? (
              <Badge tone="good">{formatNumber(run.pixel_size_nm, 2)} nm/px</Badge>
            ) : (
              <Badge tone="warning">uncalibrated</Badge>
            )}
          </div>
        </div>

        {run.caveats.length > 0 ? (
          <div className="mt-3 rounded-md border border-amber-200 bg-amber-50 p-3">
            <p className="m-0 text-xs font-semibold uppercase tracking-wide text-amber-800">
              Read before quoting these numbers
            </p>
            <ul className="m-0 mt-2 list-disc space-y-1 pl-5 text-sm text-amber-900">
              {run.caveats.map((caveat) => (
                <li key={caveat}>{caveat}</li>
              ))}
            </ul>
          </div>
        ) : null}

        {run.exports.length > 0 ? (
          <div className="mt-3">
            <p className="m-0 text-xs font-semibold uppercase tracking-wide text-slate-500">
              Export bundle
            </p>
            <div className="mt-2 flex flex-wrap gap-2">
              {run.exports.map((name) => (
                <ExportLink key={name} runId={run.id} name={name} />
              ))}
            </div>
            <p className="m-0 mt-2 text-xs text-slate-500">
              Written to {run.export_dir}
            </p>
          </div>
        ) : null}
      </Panel>

      {run.composition ? (
        <CompositionPanel
          composition={run.composition}
          calibrated={calibrated}
          pixelSizeNm={run.pixel_size_nm}
          wholeImageDenominator={wholeImage}
        />
      ) : null}

      {run.objects ? (
        <ObjectsPanel
          objects={run.objects}
          calibrated={calibrated}
          objectsCsvUrl={
            run.exports.includes("objects.csv")
              ? getAnalysisExportUrl(run.id, "objects.csv")
              : null
          }
          downloadStem={run.id}
        />
      ) : null}

      {run.distances ? (
        <Panel className="p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="m-0 text-sm font-semibold text-slate-900">
              Distance to {run.distances.target}
            </h3>
            <div className="flex items-center gap-2">
              <Badge>
                median {formatNumber(run.distances.median_nm, 1)} nm
              </Badge>
              {/* Out of the points this section *measured*, not out of the
                  run's point total. `distance_to_boundary` drops the rows that
                  are not positions, so with one unreadable row the screen read
                  "41 of 60 inside" over a median taken across 59. */}
              <Badge tone="info">
                {typeof run.distances.n_measured === "number"
                  ? `${run.distances.n_inside} of ${run.distances.n_measured} measured, inside`
                  : `${run.distances.n_inside} inside`}
              </Badge>
            </div>
          </div>
          <DistanceCoverage distances={run.distances} points={run.points} />
          <div className="mt-3">
            <BandHistogram
              labels={run.distances.band_labels}
              counts={run.distances.band_counts}
              fractions={run.distances.band_fractions}
              target={run.distances.target}
              downloadStem={run.id}
            />
          </div>
        </Panel>
      ) : run.points ? (
        <Panel className="p-4">
          <h3 className="m-0 text-sm font-semibold text-slate-900">
            Distance bands
          </h3>
          <p className="m-0 mt-1 text-sm text-slate-600">
            No distance target was set, or the image has no pixel size — the
            band edges are in nanometres, so this analysis needs a calibrated
            image.
          </p>
        </Panel>
      ) : null}

      {run.monte_carlo ? (
        <MonteCarloPanel
          monteCarlo={run.monte_carlo}
          selfCheck={run.monte_carlo_self_check}
          downloadStem={run.id}
        />
      ) : null}
    </div>
  );
}
