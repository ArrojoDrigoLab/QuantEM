/**
 * The three indicators, drawn.
 *
 * One component for both places a run is watched — the Tasks drawer and the
 * labeling screen — because a user who reads "32 of 56 tiles" on one and a
 * bare percentage on the other has to work out for themselves whether they are
 * looking at the same run. Colours come from CSS custom properties so the dark
 * drawer and the light banner are two skins of one thing rather than two
 * components that drift.
 */

import type { ProgressRow } from "@/shared/progress/runProgress";
import "./RunProgressList.css";

interface RunProgressListProps {
  rows: ProgressRow[];
  /** Extra class on the wrapper; hosts use it to set the colour variables. */
  className?: string;
  "data-testid"?: string;
}

export function RunProgressList({
  rows,
  className,
  "data-testid": testId = "run-progress-list",
}: RunProgressListProps) {
  if (rows.length === 0) return null;
  return (
    <div className={`run-progress-list ${className ?? ""}`} data-testid={testId}>
      {rows.map((row) => (
        <div
          key={row.key}
          className={`run-progress-row run-progress-${row.kind} run-progress-tone-${row.tone}`}
          data-testid={`run-progress-row-${row.kind}`}
          role="group"
          aria-label={row.ariaLabel}
        >
          <span className="run-progress-glyph" aria-hidden="true">
            {row.glyph}
          </span>
          <span className="run-progress-name" title={row.name}>
            {row.name}
          </span>
          {/* No track when there is no fraction to draw. A waiting run and a
              run loading its model both have an honest answer -- "not yet" --
              and an empty rail reads as zero rather than as unknown. */}
          <span
            className={`run-progress-bar${row.percent === null ? " run-progress-bar-empty" : ""}`}
            aria-hidden="true"
          >
            {row.percent !== null && (
              <span
                className="run-progress-fill"
                style={{ width: `${Math.max(0, Math.min(100, row.percent))}%` }}
              />
            )}
          </span>
          {row.showPercentText && row.percent !== null && (
            <span className="run-progress-percent">{Math.round(row.percent)}%</span>
          )}
          <span className="run-progress-detail">{row.detail}</span>
        </div>
      ))}
    </div>
  );
}
