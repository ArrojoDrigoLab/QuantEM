/**
 * The distance section measures `n_measured`, not `n_total`.
 *
 * `distance_to_boundary` drops every row whose coordinate is not a position
 * before it measures anything, so the median, the bands and `n_inside` are all
 * over `dist.n`. The badge divided by `points.n_total` -- the run's whole point
 * set -- so with a single unreadable row it reported a denominator larger than
 * the set it had measured, and nothing on the panel reconciled the two. The
 * payload has carried `n_measured` since the analysis owner's change; this is
 * the screen catching up.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AnalysisResults } from "@/features/analysis/components/AnalysisResults";
import type {
  AnalysisDistances,
  AnalysisPoints,
  AnalysisRun,
} from "@/shared/types/analysis";

const POINTS: AnalysisPoints = {
  n_total: 60,
  n_on_tissue: 60,
  n_off_tissue: 0,
  counts: { mito: 20 },
  fractions: { mito: 0.333 },
  enrichment: { mito: 2.1 },
};

const DISTANCES: AnalysisDistances = {
  target: "mito",
  band_labels: ["0–50", "50–100", ">100"],
  band_counts: [20, 20, 15],
  band_fractions: [0.36, 0.36, 0.28],
  median_nm: 62.5,
  n_inside: 12,
  n_measured: 55,
  n_unreadable: 5,
  n_out_of_image: 0,
};

function makeRun(overrides: Partial<AnalysisRun> = {}): AnalysisRun {
  return {
    id: "run-1",
    segmentation_id: "seg-1",
    status: "SUCCESS",
    group: "",
    created_at: "2026-01-01T00:00:00Z",
    started_at: "2026-01-01T00:00:00Z",
    finished_at: "2026-01-01T00:01:00Z",
    params: { points_source: "csv", distance_target: "mito" },
    pixel_size_nm: 5,
    calibrated: true,
    composition: null,
    objects: null,
    points: POINTS,
    distances: DISTANCES,
    monte_carlo: null,
    monte_carlo_self_check: null,
    caveats: [],
    export_dir: "D:/runs/run-1",
    exports: [],
    error: "",
    ...overrides,
  };
}

describe("AnalysisResults distance section", () => {
  it("divides the inside count by the points it measured", () => {
    render(<AnalysisResults run={makeRun()} />);

    expect(screen.getByText("12 of 55 measured, inside")).toBeInTheDocument();
    // The number it used to divide by: the run's whole point set, which this
    // section never measured.
    expect(screen.queryByText("12 of 60 inside")).not.toBeInTheDocument();
  });

  it("reconciles the two totals next to the numbers", () => {
    render(<AnalysisResults run={makeRun()} />);

    expect(
      screen.getByText(/cover 55 of the run's 60 points/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/no distance to anything/)
    ).toBeInTheDocument();
  });

  it("says nothing extra when every point was measured", () => {
    render(
      <AnalysisResults
        run={makeRun({
          distances: { ...DISTANCES, n_measured: 60, n_unreadable: 0 },
        })}
      />
    );

    expect(screen.getByText("12 of 60 measured, inside")).toBeInTheDocument();
    expect(screen.queryByText(/cover 60 of the run's/)).not.toBeInTheDocument();
  });

  it("flags measured points that were clipped onto the border", () => {
    render(
      <AnalysisResults
        run={makeRun({
          distances: {
            ...DISTANCES,
            n_measured: 60,
            n_unreadable: 0,
            n_out_of_image: 4,
          },
        })}
      />
    );

    // Inside `n_measured`, not beside it: their distance is real arithmetic on
    // a pixel nobody chose.
    expect(
      screen.getByText(/clipped onto the\s+border/)
    ).toBeInTheDocument();
  });

  it("quotes no denominator at all for a run stored before n_measured existed", () => {
    // Falling back to `n_total` here would be the original defect, restored for
    // exactly the runs that cannot contradict it.
    const older: AnalysisDistances = { ...DISTANCES };
    delete older.n_measured;
    render(<AnalysisResults run={makeRun({ distances: older })} />);

    expect(screen.getByText("12 inside")).toBeInTheDocument();
    expect(screen.queryByText(/of 60/)).not.toBeInTheDocument();
  });
});
