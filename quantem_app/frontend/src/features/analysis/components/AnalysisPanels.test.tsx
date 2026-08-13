import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CompositionPanel } from "@/features/analysis/components/CompositionPanel";
import { PointsPanel } from "@/features/analysis/components/PointsPanel";
import { MonteCarloPanel } from "@/features/analysis/components/MonteCarloPanel";
import type {
  AnalysisComposition,
  AnalysisMonteCarlo,
  AnalysisPoints,
} from "@/shared/types/analysis";

const CALIBRATED: AnalysisComposition = {
  tissue_px: 1_000_000,
  tissue_um2: 25.0,
  area_fractions: { mito: 0.12, nucleus: 0.2, cytoplasm: 0.8 },
  areas_px: { mito: 120_000, nucleus: 200_000, cytoplasm: 800_000 },
  areas_um2: { mito: 3.0, nucleus: 5.0, cytoplasm: 20.0 },
};

const UNCALIBRATED: AnalysisComposition = {
  tissue_px: 1_000_000,
  tissue_um2: null,
  area_fractions: { mito: 0.12 },
  areas_px: { mito: 120_000 },
  areas_um2: null,
};

describe("CompositionPanel (honesty rule 6)", () => {
  it("renders no µm unit at all when the pixel size is unset", () => {
    const { container } = render(
      <CompositionPanel
        composition={UNCALIBRATED}
        calibrated={false}
        pixelSizeNm={null}
        wholeImageDenominator={false}
      />
    );
    expect(screen.getByText("Pixel size not set")).toBeInTheDocument();
    expect(container.textContent).not.toContain("µm");
  });

  it("renders µm² once the run reported a pixel size", () => {
    const { container } = render(
      <CompositionPanel
        composition={CALIBRATED}
        calibrated
        pixelSizeNm={5}
        wholeImageDenominator={false}
      />
    );
    expect(container.textContent).toContain("µm²");
    expect(screen.getByText("5 nm/px")).toBeInTheDocument();
  });

  it("warns when the whole image was used as the denominator", () => {
    render(
      <CompositionPanel
        composition={CALIBRATED}
        calibrated
        pixelSizeNm={5}
        wholeImageDenominator
      />
    );
    expect(screen.getByText("Whole image as denominator")).toBeInTheDocument();
  });

  it("does not claim tissue-area fractions when the denominator is the whole image", () => {
    // With no tissue mask the header badge says "Whole image as denominator";
    // a footer still reading "Fractions are of tissue area, not of the image"
    // contradicted it on the same panel.
    const { container } = render(
      <CompositionPanel
        composition={CALIBRATED}
        calibrated
        pixelSizeNm={5}
        wholeImageDenominator
      />
    );
    expect(container.textContent).not.toContain(
      "Fractions are of tissue area, not of the image."
    );
    expect(
      screen.getByText(/Fractions are of the whole image — this run had no tissue mask/)
    ).toBeInTheDocument();
  });

  it("keeps the tissue-area footer when the mask restricted the denominator", () => {
    const { container } = render(
      <CompositionPanel
        composition={CALIBRATED}
        calibrated
        pixelSizeNm={5}
        wholeImageDenominator={false}
      />
    );
    expect(container.textContent).toContain(
      "Fractions are of tissue area, not of the image."
    );
    expect(container.textContent).not.toContain("Fractions are of the whole image");
  });

  it("explains a derived cytoplasm rather than letting fractions look additive", () => {
    render(
      <CompositionPanel
        composition={CALIBRATED}
        calibrated
        pixelSizeNm={5}
        wholeImageDenominator={false}
      />
    );
    expect(screen.getByText(/do not sum to 1/)).toBeInTheDocument();
  });
});

describe("PointsPanel (honesty rules 4 and 5)", () => {
  const points: AnalysisPoints = {
    n_total: 120,
    n_on_tissue: 100,
    n_off_tissue: 20,
    counts: { mito: 40, nucleus: 10 },
    fractions: { mito: 0.4, nucleus: 0.1 },
    enrichment: { mito: 3.33, nucleus: null },
  };

  it("shows the excluded points and what they were excluded from", () => {
    render(
      <PointsPanel points={points} composition={CALIBRATED} pointsSource="csv" />
    );
    expect(
      screen.getByText(/fell outside the tissue mask and were excluded/)
    ).toBeInTheDocument();
  });

  it("states the aggregation rule next to the enrichment table", () => {
    render(
      <PointsPanel points={points} composition={CALIBRATED} pointsSource="csv" />
    );
    expect(
      screen.getByText(/unweighted mean over experimental units/)
    ).toBeInTheDocument();
  });

  it("renders an undefined enrichment as an em dash, not zero", () => {
    render(
      <PointsPanel points={points} composition={CALIBRATED} pointsSource="csv" />
    );
    const nucleusRow = screen.getByText("nucleus").closest("tr");
    expect(nucleusRow?.textContent).toContain("—");
  });

  it("says nothing about exclusions when every point is on tissue", () => {
    render(
      <PointsPanel
        points={{ ...points, n_total: 100, n_off_tissue: 0 }}
        composition={CALIBRATED}
        pointsSource="centroids"
      />
    );
    expect(
      screen.queryByText(/fell outside the tissue mask/)
    ).not.toBeInTheDocument();
  });

  /**
   * `assign_points` reports three ways a point can leave the denominator and
   * this panel showed one. `n_total == n_on_tissue + n_off_tissue +
   * n_unreadable`, so a reader who added the two visible numbers and got less
   * than the total had nothing on screen to explain the difference.
   */
  describe("the other two ways a point leaves the count", () => {
    it("counts unreadable rows separately from off-tissue ones", () => {
      render(
        <PointsPanel
          points={{ ...points, n_total: 123, n_unreadable: 3 }}
          composition={CALIBRATED}
          pointsSource="csv"
        />
      );

      expect(screen.getByText("Unreadable")).toBeInTheDocument();
      expect(
        screen.getByText(/not in the off-tissue total either/)
      ).toBeInTheDocument();
      // The distinction the backend is explicit about: nowhere, not outside.
      expect(
        screen.getByText(/a point that cannot be read is nowhere/)
      ).toBeInTheDocument();
    });

    it("flags points clipped onto the image border, and says why it matters", () => {
      render(
        <PointsPanel
          points={{ ...points, n_out_of_bounds: 118 }}
          composition={CALIBRATED}
          pointsSource="csv"
        />
      );

      expect(screen.getByText("Outside the image")).toBeInTheDocument();
      // The reading that saves an enrichment from being quoted: a whole set on
      // one edge is a units error, not a finding.
      expect(
        screen.getByText(/what a CSV in nanometres/)
      ).toBeInTheDocument();
    });

    it("shows a zero rather than hiding a field the server did send", () => {
      render(
        <PointsPanel
          points={{ ...points, n_unreadable: 0, n_out_of_bounds: 0 }}
          composition={CALIBRATED}
          pointsSource="csv"
        />
      );

      expect(screen.getByText("Unreadable")).toBeInTheDocument();
      expect(screen.getByText("Outside the image")).toBeInTheDocument();
      expect(
        screen.queryByText(/a point that cannot be read is nowhere/)
      ).not.toBeInTheDocument();
    });

    it("shows nothing for a run stored before the server reported them", () => {
      // Absent is "this build did not say", which is not the same claim as 0.
      render(
        <PointsPanel points={points} composition={CALIBRATED} pointsSource="csv" />
      );

      expect(screen.queryByText("Unreadable")).not.toBeInTheDocument();
      expect(screen.queryByText("Outside the image")).not.toBeInTheDocument();
    });
  });
});

describe("MonteCarloPanel", () => {
  const monteCarlo: AnalysisMonteCarlo = {
    replicates: 20,
    seed: 12345,
    observed: { enrichment_mito: 3.33 },
    null_mean: { enrichment_mito: 1.01 },
    null_sd: { enrichment_mito: 0.12 },
    z: { enrichment_mito: 19.3 },
    p_two_sided: { enrichment_mito: 0.048 },
  };

  it("states the p-value floor implied by the replicate count", () => {
    render(
      <MonteCarloPanel monteCarlo={monteCarlo} selfCheck={null} downloadStem="run-1" />
    );
    // 1 / (20 + 1) = 0.048, stated in the note under the table.
    expect(screen.getByText(/cannot go below/).textContent).toContain("0.048");
  });

  it("shows the null as mean ± sd rather than a bare mean", () => {
    render(
      <MonteCarloPanel monteCarlo={monteCarlo} selfCheck={null} downloadStem="run-1" />
    );
    expect(screen.getByText("1.010 ± 0.120")).toBeInTheDocument();
  });

  /**
   * The footnote is the only place this screen explains a blank cell, and there
   * are now two of them. `p` used to read 0.048 wherever `z` was blank -- the
   * smallest value twenty replicates can produce, from a null with no
   * distribution to be extreme against, and the first number anyone compares
   * to 0.05. Both are blank now; a blank nobody explains reads as a bug.
   */
  it("explains a blank p, not only a blank z", () => {
    render(
      <MonteCarloPanel
        monteCarlo={{
          ...monteCarlo,
          null_sd: { enrichment_mito: 0 },
          z: { enrichment_mito: null },
          p_two_sided: { enrichment_mito: null },
        }}
        selfCheck={null}
        downloadStem="run-1"
      />
    );

    const note = screen.getByText(/means the null had zero spread/);
    expect(note.textContent).toContain("A blank z or p");
    expect(note.textContent).toContain("every replicate returned the same value");
    // And why the number that would have appeared there was not a finding.
    expect(note.textContent).toContain("is not a test");
  });

  it("renders the blank p as an em dash rather than 0.048", () => {
    render(
      <MonteCarloPanel
        monteCarlo={{
          ...monteCarlo,
          null_sd: { enrichment_mito: 0 },
          z: { enrichment_mito: null },
          p_two_sided: { enrichment_mito: null },
        }}
        selfCheck={null}
        downloadStem="run-1"
      />
    );

    const row = screen.getByText("enrichment_mito").closest("tr");
    // Two em dashes: z and p. The observed value is still a real measurement.
    expect(row?.textContent).toContain("—");
    expect(row?.textContent).not.toContain("0.048");
  });

  it("reports the self-check deviation on the user's own masks", () => {
    render(
      <MonteCarloPanel
        monteCarlo={monteCarlo}
        selfCheck={{
          n_points: 5000,
          smallest_compartment_fraction: 0.12,
          enrichment: { mito: 1.001 },
          max_abs_deviation: 0.004,
        }}
        downloadStem="run-1"
      />
    );
    expect(screen.getByText("0.004")).toBeInTheDocument();
    expect(screen.getByText(/normalisation is biased/)).toBeInTheDocument();
  });
});
