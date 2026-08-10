/**
 * Observed enrichment against complete spatial randomness.
 *
 * The p-value is empirical with the +1 correction, so its floor is
 * 1/(replicates+1). That floor is stated next to the column: at 20 replicates
 * nothing can be smaller than 0.048, and reporting "p < 0.001" from 20 draws
 * would be a fiction.
 *
 * A null with *no spread* is the case that floor makes dangerous. When every
 * replicate returns the same number the formula still produces something --
 * exactly 1/(R+1) if the observed value differs at all -- so the smallest,
 * most-significant-looking value the method can emit arrives whatever the data
 * are, including from a single point. `_p_two_sided` returns `None` there, as
 * `_z` already did. The footnote below covers both blanks, because it is the
 * only place on this screen that explains one.
 *
 * The self-check row is the manuscript's internal control run on *this* user's
 * masks: uniform points over their own geometry must recover enrichment ~1.0,
 * or the normalisation is biased for this image.
 */

import { Badge, Button, Panel } from "@/shared/ui/design";
import { formatNumber, formatPValue, NOT_MEASURED } from "@/shared/ui/format";
import { downloadCsv } from "@/utils/downloadText";
import type {
  AnalysisMonteCarlo,
  AnalysisMonteCarloSelfCheck,
} from "@/shared/types/analysis";

export interface MonteCarloPanelProps {
  monteCarlo: AnalysisMonteCarlo;
  selfCheck: AnalysisMonteCarloSelfCheck | null;
  downloadStem: string;
}

export function MonteCarloPanel({
  monteCarlo,
  selfCheck,
  downloadStem,
}: MonteCarloPanelProps) {
  const keys = Object.keys(monteCarlo.observed).sort();
  const pFloor = 1 / (monteCarlo.replicates + 1);

  const handleDownload = () => {
    downloadCsv(
      `monte-carlo-${downloadStem}.csv`,
      [
        "key",
        "observed",
        "null_mean",
        "null_sd",
        "z",
        "p_two_sided",
        "replicates",
        "seed",
      ],
      keys.map((key) => [
        key,
        monteCarlo.observed[key] ?? "",
        monteCarlo.null_mean[key] ?? "",
        monteCarlo.null_sd[key] ?? "",
        monteCarlo.z[key] ?? "",
        monteCarlo.p_two_sided[key] ?? "",
        monteCarlo.replicates,
        monteCarlo.seed,
      ])
    );
  };

  return (
    <Panel className="p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="m-0 text-sm font-semibold text-slate-900">
          Monte-Carlo null
        </h3>
        <div className="flex items-center gap-2">
          <Badge>{monteCarlo.replicates} replicates</Badge>
          <Badge>seed {monteCarlo.seed}</Badge>
        </div>
      </div>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[560px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="py-1 pr-3 font-semibold">Statistic</th>
              <th className="py-1 pr-3 font-semibold">Observed</th>
              <th className="py-1 pr-3 font-semibold">Null mean ± sd</th>
              <th className="py-1 pr-3 font-semibold">z</th>
              <th className="py-1 font-semibold">p (two-sided)</th>
            </tr>
          </thead>
          <tbody>
            {keys.map((key) => {
              const mean = monteCarlo.null_mean[key] ?? null;
              const sd = monteCarlo.null_sd[key] ?? null;
              return (
                <tr key={key} className="border-b border-slate-100">
                  <td className="py-1 pr-3 text-slate-700">{key}</td>
                  <td className="py-1 pr-3 tabular-nums text-slate-900">
                    {formatNumber(monteCarlo.observed[key] ?? null, 3)}
                  </td>
                  <td className="py-1 pr-3 tabular-nums text-slate-900">
                    {mean === null
                      ? NOT_MEASURED
                      : `${formatNumber(mean, 3)} ± ${formatNumber(sd, 3)}`}
                  </td>
                  <td className="py-1 pr-3 tabular-nums text-slate-900">
                    {formatNumber(monteCarlo.z[key] ?? null, 2)}
                  </td>
                  <td className="py-1 tabular-nums text-slate-900">
                    {formatPValue(monteCarlo.p_two_sided[key] ?? null)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* The blank cells are explained here because this is the only place on
          the screen that explains one, and there are now two of them. `p` used
          to read 0.048 wherever `z` was blank: the smallest value twenty
          replicates can produce, arriving from a null with no distribution to
          be extreme against, and the first number anyone compares to 0.05.
          `_p_two_sided` returns None for that case now, so the sentence that
          covered z has to cover p as well -- a blank with no explanation is
          read as a rendering fault, and a wrong 0.048 was read as a finding. */}
      <p className="mt-3 mb-0 text-xs text-slate-500">
        The null is complete spatial randomness inside the tissue mask, seeded
        per (image, replicate) so it does not depend on processing order. The
        p-value is empirical with the +1 correction, so with{" "}
        {monteCarlo.replicates} replicates it cannot go below{" "}
        {formatNumber(pFloor, 3)}. A blank z <em>or p</em> means the null had
        zero spread: every replicate returned the same value, which supports
        neither statistic. An empirical p against a flat null is not a test — it
        would be {formatNumber(pFloor, 3)} whenever the observed value differed
        from the null at all, and 1.000 when it did not, whatever the data are.
      </p>

      {selfCheck ? (
        <div className="mt-3 rounded-md border border-slate-200 bg-slate-50 p-3">
          <p className="m-0 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Self-check on your masks
          </p>
          <p className="m-0 mt-1 text-sm text-slate-700">
            {selfCheck.n_points.toLocaleString()} uniform points over this
            image&apos;s own geometry recovered enrichment within{" "}
            <span className="font-semibold tabular-nums">
              {formatNumber(selfCheck.max_abs_deviation, 3)}
            </span>{" "}
            of 1.0 in every compartment. A large deviation here means the
            normalisation is biased for this geometry and the enrichments above
            should not be quoted.
          </p>
        </div>
      ) : null}

      <div className="mt-3">
        <Button size="sm" onClick={handleDownload}>
          Download null table
        </Button>
      </div>
    </Panel>
  );
}
