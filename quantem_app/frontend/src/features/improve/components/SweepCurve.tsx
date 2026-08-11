/**
 * The threshold sweep, drawn as inline SVG.
 *
 * What the chart has to make obvious, because it is the whole point of the
 * screen: the curve is the *training* Dice, the chosen point sits on it, and
 * the held-out marks are single scores at that point — not a curve, because a
 * held-out curve would be a threshold chosen on held-out data.
 *
 * The oracle is drawn as a dashed ceiling, above which nothing on this chart
 * can go, and labelled as such.
 */

import { Button } from "@/shared/ui/design";
import { formatNumber } from "@/shared/ui/format";
import { downloadCsv } from "@/utils/downloadText";
import type { AdapterSweep, SplitMode } from "@/shared/types/finetune";
import { SPLIT_MODE_LABELS } from "@/features/improve/splitMode";
import {
  PLOT,
  VIEW_HEIGHT,
  VIEW_WIDTH,
  polylinePoints,
  scaleX,
  scaleY,
  thresholdDomain,
  toCurvePoints,
} from "@/features/improve/sweepCurve";

const Y_TICKS = [0, 0.25, 0.5, 0.75, 1];

export interface SweepCurveProps {
  sweep: AdapterSweep;
  /** The same sweep for the un-adapted model, when the run produced one. */
  baseSweep?: AdapterSweep | null;
  splitMode: SplitMode;
  downloadStem: string;
}

export function SweepCurve({
  sweep,
  baseSweep,
  splitMode,
  downloadStem,
}: SweepCurveProps) {
  const domain = thresholdDomain(sweep.thresholds);
  const adapted = toCurvePoints(sweep.thresholds, sweep.train_dice);
  const base = baseSweep
    ? toCurvePoints(baseSweep.thresholds, baseSweep.train_dice)
    : [];
  const hasHeldout = splitMode !== "no-heldout";
  const defaultThreshold = 0.5;

  const handleDownload = () => {
    const rows = sweep.thresholds.map((threshold, index) => [
      threshold,
      sweep.train_dice[index] ?? "",
      baseSweep?.train_dice?.[index] ?? "",
    ]);
    downloadCsv(
      `threshold-sweep-${downloadStem}.csv`,
      ["threshold", "train_dice_adapted", "train_dice_base"],
      rows
    );
  };

  return (
    <div className="flex flex-col gap-3">
      <svg
        viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
        className="h-auto w-full"
        role="img"
        aria-label="Mean Dice on the fitted crops against threshold"
      >
        <title>Mean Dice on the fitted crops against threshold</title>

        {Y_TICKS.map((tick) => {
          const y = scaleY(tick);
          return (
            <g key={tick}>
              <line
                x1={PLOT.left}
                x2={PLOT.left + PLOT.width}
                y1={y}
                y2={y}
                stroke="#e2e8f0"
                strokeWidth={1}
              />
              <text
                x={PLOT.left - 8}
                y={y + 4}
                textAnchor="end"
                fontSize={11}
                fill="#64748b"
              >
                {tick.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* The default 0.5 the released model ships with. */}
        <line
          x1={scaleX(defaultThreshold, domain)}
          x2={scaleX(defaultThreshold, domain)}
          y1={PLOT.top}
          y2={PLOT.top + PLOT.height}
          stroke="#cbd5e1"
          strokeWidth={1}
          strokeDasharray="3 3"
        />
        <text
          x={scaleX(defaultThreshold, domain)}
          y={PLOT.top + PLOT.height + 30}
          textAnchor="middle"
          fontSize={10}
          fill="#94a3b8"
        >
          default 0.50
        </text>

        {sweep.heldout_oracle !== null && sweep.heldout_oracle !== undefined ? (
          <>
            <line
              x1={PLOT.left}
              x2={PLOT.left + PLOT.width}
              y1={scaleY(sweep.heldout_oracle)}
              y2={scaleY(sweep.heldout_oracle)}
              stroke="#f59e0b"
              strokeWidth={1.5}
              strokeDasharray="6 4"
            />
            <text
              x={PLOT.left + PLOT.width}
              y={scaleY(sweep.heldout_oracle) - 5}
              textAnchor="end"
              fontSize={10}
              fill="#b45309"
            >
              oracle ceiling {formatNumber(sweep.heldout_oracle, 3)}
            </text>
          </>
        ) : null}

        {base.length > 0 ? (
          <polyline
            points={polylinePoints(base, domain)}
            fill="none"
            stroke="#94a3b8"
            strokeWidth={1.5}
            strokeDasharray="5 4"
          />
        ) : null}

        <polyline
          points={polylinePoints(adapted, domain)}
          fill="none"
          stroke="#0891b2"
          strokeWidth={2}
        />

        {/* The chosen point: the maximum of the training curve. */}
        <line
          x1={scaleX(sweep.calibrated_threshold, domain)}
          x2={scaleX(sweep.calibrated_threshold, domain)}
          y1={PLOT.top}
          y2={PLOT.top + PLOT.height}
          stroke="#0f766e"
          strokeWidth={1.5}
        />
        <text
          x={scaleX(sweep.calibrated_threshold, domain)}
          y={PLOT.top - 5}
          textAnchor="middle"
          fontSize={11}
          fill="#0f766e"
        >
          chosen {formatNumber(sweep.calibrated_threshold, 2)}
        </text>

        {hasHeldout && sweep.heldout_dice_at_calibrated !== null ? (
          <circle
            cx={scaleX(sweep.calibrated_threshold, domain)}
            cy={scaleY(sweep.heldout_dice_at_calibrated)}
            r={5}
            fill="#7c3aed"
          />
        ) : null}
        {hasHeldout && sweep.heldout_dice_at_default !== null ? (
          <circle
            cx={scaleX(defaultThreshold, domain)}
            cy={scaleY(sweep.heldout_dice_at_default)}
            r={5}
            fill="#7c3aed"
            fillOpacity={0.45}
          />
        ) : null}

        <line
          x1={PLOT.left}
          x2={PLOT.left}
          y1={PLOT.top}
          y2={PLOT.top + PLOT.height}
          stroke="#94a3b8"
        />
        <line
          x1={PLOT.left}
          x2={PLOT.left + PLOT.width}
          y1={PLOT.top + PLOT.height}
          y2={PLOT.top + PLOT.height}
          stroke="#94a3b8"
        />
        <text
          x={PLOT.left + PLOT.width / 2}
          y={VIEW_HEIGHT - 8}
          textAnchor="middle"
          fontSize={11}
          fill="#64748b"
        >
          Threshold
        </text>
        <text
          x={14}
          y={PLOT.top + PLOT.height / 2}
          fontSize={11}
          fill="#64748b"
          transform={`rotate(-90 14 ${PLOT.top + PLOT.height / 2})`}
          textAnchor="middle"
        >
          Mean Dice
        </text>
      </svg>

      <ul className="m-0 flex list-none flex-wrap gap-4 p-0 text-xs text-slate-600">
        <li className="flex items-center gap-2">
          <span className="inline-block h-0.5 w-6 bg-cyan-600" />
          Adapted, fitted crops
        </li>
        {base.length > 0 ? (
          <li className="flex items-center gap-2">
            <span className="inline-block h-0.5 w-6 border-t border-dashed border-slate-400" />
            Base model, same crops
          </li>
        ) : null}
        {hasHeldout ? (
          <li className="flex items-center gap-2">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-violet-600" />
            Held-out score ({SPLIT_MODE_LABELS[splitMode]}) at the default and at
            the chosen threshold
          </li>
        ) : null}
      </ul>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="m-0 text-xs text-slate-500">
          The curve is the mean Dice over the crops the threshold was fitted on.
          The held-out marks are single scores at two thresholds, never a curve —
          picking the best point of a held-out curve would be fitting on it.
        </p>
        <Button size="sm" onClick={handleDownload}>
          Download sweep
        </Button>
      </div>
    </div>
  );
}
