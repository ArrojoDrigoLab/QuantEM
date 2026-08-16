import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ObjectsPanel } from "./ObjectsPanel";
import { metricNote } from "./objectsPanelUtils";
import type { AnalysisObjects } from "@/shared/types/analysis";

const CIRCULARITY_NOTE =
  "Measured on 2765 of 2766 confirmed objects; 1 was measured and could not be " +
  "reported. Their 4*pi*area/perimeter^2 came out above 1.015. The theoretical " +
  "ceiling is 1, and the estimator's measured envelope for a genuinely round object " +
  "reaches about 1.011 — so a value beyond 1.015 measures the estimator failing on " +
  "a small object, not the object, and is left blank rather than exported as a roundness.";

const MEAN_PROB_NOTE =
  "Measured on 2755 of 2766 confirmed objects; 11 carry no stored value for this " +
  "metric. User-drawn or defined objects have no model probability behind them.";

function objects(): AnalysisObjects {
  return {
    n: 2766,
    density: { count: 2766, tissue_um2: 25, per_um2: 110.64 },
    summary: {
      area_px: {
        n: 2766,
        n_objects: 2766,
        n_missing: 0,
        mean: 812.3,
        sd: 240.1,
        median: 790,
        iqr: 300,
        min: 40,
        max: 1310,
      },
      circularity: {
        n: 2765,
        n_objects: 2766,
        n_missing: 1,
        n_unreportable: 1,
        mean: 0.649,
        sd: 0.121,
        median: 0.66,
        iqr: 0.18,
        min: 0.401,
        max: 1.011,
      },
      mean_prob: {
        n: 2755,
        n_objects: 2766,
        n_missing: 11,
        mean: 0.82,
        sd: 0.06,
        median: 0.83,
        iqr: 0.08,
        min: 0.71,
        max: 0.9,
      },
    },
  };
}

function renderPanel(objectsCsvUrl: string | null = null) {
  return render(
    <ObjectsPanel
      objects={objects()}
      calibrated
      objectsCsvUrl={objectsCsvUrl}
      downloadStem="run-1"
    />
  );
}

describe("ObjectsPanel metric coverage indicators", () => {
  it("shows a yellow circularity indicator with the concise hover message", () => {
    const { container } = renderPanel();
    const indicator = screen.getByLabelText(`circularity measurement note: ${CIRCULARITY_NOTE}`);

    expect(indicator).toHaveAttribute("title", CIRCULARITY_NOTE);
    expect(indicator).toHaveClass("text-amber-700");
    expect(container.textContent).not.toContain("Quote it " + "as");
    expect(container.textContent).not.toContain("owner " + "ruling");
    expect(within(container).getAllByRole("row")).toHaveLength(4);
  });

  it("shows the mean-probability message in the normal text color", () => {
    renderPanel();
    const indicator = screen.getByLabelText(`mean_prob measurement note: ${MEAN_PROB_NOTE}`);

    expect(indicator).toHaveAttribute("title", MEAN_PROB_NOTE);
    expect(indicator).toHaveClass("text-slate-600");
    expect(indicator).not.toHaveClass("text-amber-700");
  });

  it("does not add an indicator when every object has the metric", () => {
    renderPanel();
    expect(screen.queryByLabelText(/area_px measurement note/)).not.toBeInTheDocument();
  });

  it("places the renamed downloads in the Objects header", () => {
    renderPanel("/objects.csv");
    expect(screen.getByRole("button", { name: "Download Summary Table" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Download All Objects" })).toBeInTheDocument();
    expect(screen.queryByText("Confirmed objects only")).not.toBeInTheDocument();
  });
});

describe("metricNote", () => {
  it("builds the requested circularity wording from structured counts", () => {
    expect(
      metricNote("circularity", {
        n: 2765,
        n_objects: 2766,
        n_missing: 1,
        n_unreportable: 1,
      })
    ).toBe(CIRCULARITY_NOTE);
  });

  it("builds the requested mean-probability wording", () => {
    expect(
      metricNote("mean_prob", { n: 2755, n_objects: 2766, n_missing: 11 })
    ).toBe(MEAN_PROB_NOTE);
  });

  it("ignores legacy estimator essays when nothing is missing", () => {
    expect(
      metricNote("circularity", {
        n: 5,
        n_objects: 5,
        n_missing: 0,
        note: "legacy long note",
        estimator_note: "legacy long note",
      })
    ).toBeNull();
  });
});
