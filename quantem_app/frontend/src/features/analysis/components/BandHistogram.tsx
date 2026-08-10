/**
 * Distance-band histogram, drawn as inline SVG.
 *
 * Hand-rolled on purpose: a plotting library for six rectangles would be more
 * bytes than the rest of this screen put together. The bars and the table below
 * them read the same three arrays, and the download button writes those arrays
 * out, so nothing is plotted that cannot be taken away.
 */

import { Button } from "@/shared/ui/design";
import { formatPercent } from "@/shared/ui/format";
import { downloadCsv } from "@/utils/downloadText";

const VIEW_WIDTH = 640;
const VIEW_HEIGHT = 260;
const PLOT_LEFT = 52;
const PLOT_RIGHT = 12;
const PLOT_TOP = 16;
const PLOT_BOTTOM = 46;

export interface BandHistogramProps {
  labels: string[];
  counts: number[];
  fractions: number[];
  /** Compartment the distances were measured to. */
  target: string;
  /** Filename stem for the CSV, usually the run id. */
  downloadStem: string;
}

export function BandHistogram({
  labels,
  counts,
  fractions,
  target,
  downloadStem,
}: BandHistogramProps) {
  const plotWidth = VIEW_WIDTH - PLOT_LEFT - PLOT_RIGHT;
  const plotHeight = VIEW_HEIGHT - PLOT_TOP - PLOT_BOTTOM;
  const maxCount = Math.max(1, ...counts);
  const slot = labels.length > 0 ? plotWidth / labels.length : plotWidth;
  const barWidth = Math.max(4, slot * 0.62);
  const ticks = [0, 0.5, 1].map((fraction) => Math.round(maxCount * fraction));

  const handleDownload = () => {
    downloadCsv(
      `distance-bands-${downloadStem}.csv`,
      ["band", "count", "fraction", "target"],
      labels.map((label, index) => [
        label,
        counts[index] ?? 0,
        fractions[index] ?? 0,
        target,
      ])
    );
  };

  return (
    <div className="flex flex-col gap-3">
      <svg
        viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
        className="h-auto w-full"
        role="img"
        aria-label={`Point counts by distance band to ${target}`}
      >
        <title>{`Point counts by distance band to ${target}`}</title>
        {ticks.map((tick) => {
          const y = PLOT_TOP + plotHeight - (tick / maxCount) * plotHeight;
          return (
            <g key={tick}>
              <line
                x1={PLOT_LEFT}
                x2={VIEW_WIDTH - PLOT_RIGHT}
                y1={y}
                y2={y}
                stroke="#e2e8f0"
                strokeWidth={1}
              />
              <text
                x={PLOT_LEFT - 8}
                y={y + 4}
                textAnchor="end"
                fontSize={11}
                fill="#64748b"
              >
                {tick}
              </text>
            </g>
          );
        })}
        <line
          x1={PLOT_LEFT}
          x2={PLOT_LEFT}
          y1={PLOT_TOP}
          y2={PLOT_TOP + plotHeight}
          stroke="#94a3b8"
          strokeWidth={1}
        />
        {labels.map((label, index) => {
          const count = counts[index] ?? 0;
          const height = (count / maxCount) * plotHeight;
          const x = PLOT_LEFT + slot * index + (slot - barWidth) / 2;
          const y = PLOT_TOP + plotHeight - height;
          return (
            <g key={label}>
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={Math.max(height, count > 0 ? 1 : 0)}
                fill="#0891b2"
                rx={2}
              />
              <text
                x={x + barWidth / 2}
                y={y - 5}
                textAnchor="middle"
                fontSize={11}
                fill="#0f172a"
              >
                {count}
              </text>
              <text
                x={PLOT_LEFT + slot * index + slot / 2}
                y={PLOT_TOP + plotHeight + 18}
                textAnchor="middle"
                fontSize={11}
                fill="#475569"
              >
                {label}
              </text>
            </g>
          );
        })}
        <text
          x={PLOT_LEFT + plotWidth / 2}
          y={VIEW_HEIGHT - 8}
          textAnchor="middle"
          fontSize={11}
          fill="#64748b"
        >
          Distance to {target} boundary
        </text>
      </svg>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[360px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="py-1 pr-3 font-semibold">Band</th>
              <th className="py-1 pr-3 font-semibold">Points</th>
              <th className="py-1 font-semibold">Fraction</th>
            </tr>
          </thead>
          <tbody>
            {labels.map((label, index) => (
              <tr key={label} className="border-b border-slate-100">
                <td className="py-1 pr-3 text-slate-700">{label}</td>
                <td className="py-1 pr-3 tabular-nums text-slate-900">
                  {counts[index] ?? 0}
                </td>
                <td className="py-1 tabular-nums text-slate-900">
                  {formatPercent(fractions[index] ?? null)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between gap-3">
        <p className="m-0 text-xs text-slate-500">
          Bands are distances to the {target} boundary, measured with an exact
          KD-tree; the same measure is used for the observed points and the null.
        </p>
        <Button size="sm" onClick={handleDownload}>
          Download band table
        </Button>
      </div>
    </div>
  );
}
