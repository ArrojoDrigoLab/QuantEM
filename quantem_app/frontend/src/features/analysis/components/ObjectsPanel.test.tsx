/**
 * A metric note that exists and is correct must reach a human.
 *
 * The backend publishes `estimator_note` on every metric summary that has one,
 * whether or not anything was blanked, and the caveat list and the manifest
 * both carry it. The screen did not. On a run whose circularity column is fully
 * populated the table read `circularity  n=11  mean 0.649  max 0.889` and the
 * page contained the note nowhere — not in text, not in a `title` — while the
 * caveat block above it told the reader that a metric's reason lives "in the
 * summary table".
 *
 * What that cost, measured on real data under the estimator of the time
 * (regionprops.perimeter): eight mitochondrial outlines scaled to 0.6x — a
 * pure size change, identical shapes — moved mean circularity
 * 0.6186 -> 0.6409, paired t = 3.596, p = 0.0088, turning a correct
 * segmentation into "mitochondria became more circular after treatment". The
 * backend has since switched to perimeter_crofton (see the fixture below for
 * the current note), which shrinks that bias but does not remove the need for
 * the note — hence these tests.
 *
 * These tests are written against the generic path, not against circularity:
 * any metric that arrives carrying a note has to show it.
 */

import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ObjectsPanel } from "./ObjectsPanel";
import { metricNote } from "./objectsPanelUtils";
import type { AnalysisObjects } from "@/shared/types/analysis";

/**
 * Verbatim from `morphometrics.CIRCULARITY_ESTIMATOR_NOTE` as of the
 * perimeter_crofton ruling (2026-08-07). The panel renders whatever the
 * payload carries, so this fixture exists to keep the assertions below honest
 * about what the backend actually ships — when the note is rewritten
 * server-side, re-pin it here rather than letting the two drift.
 */
const ESTIMATOR_NOTE =
  "circularity is 4*pi*area/perimeter^2; 1.0 is its theoretical ceiling, " +
  "reached only by a perfect circle. The perimeter here is scikit-image's " +
  "perimeter_crofton on the pixel mask (owner ruling 2026-08-07; bundles " +
  "whose environment.perimeter_estimator field is absent or names " +
  "regionprops.perimeter used the earlier estimator, whose bias grew as " +
  "objects shrank and could turn a pure size change into a roundness " +
  "difference — perimeter and circularity are not comparable across that " +
  "boundary). Crofton is close to unbiased on round shapes: a disc measures " +
  "0.995 at r=10 px, 1.008 at r=20 and 1.001 at r=80, against a true 1.0. It " +
  "is still an estimator, not geometry. A genuinely round object scatters " +
  "within about 1.5% of 1.0, on both sides, so a value slightly above 1.0 " +
  "(up to 1.015) is reported as measured — it means as round as this " +
  "estimator can resolve, and withholding it would censor the roundest " +
  "objects and bias a round population's mean downward. Values beyond 1.015 " +
  "are estimator failures — discs below about r=10 px and tiny cornered " +
  "shapes produce them — and are blank. On cornered shapes crofton reads " +
  "high by a roughly constant factor (a square measures 0.900 at 20 px and " +
  "0.879 at 100 px against a true 0.785), which cancels between groups of " +
  "the same shape class. Below ~10 px radius the estimator is unreliable in " +
  "both directions; for populations dominated by such objects, report the " +
  "size distribution beside any circularity comparison.";

/** The other kind of note: coverage, not estimator. */
const COVERAGE_NOTE =
  "Measured on 4 of 11 confirmed objects; the other 7 are hand-drawn and " +
  "carry no stored value for it.";

/**
 * The run the reviewer described: eleven confirmed objects, every one of them
 * with a circularity, so nothing is missing and no coverage sentence exists.
 * `note` is what the backend actually ships in that state — the estimator
 * paragraph alone, because the coverage half of it is empty.
 */
function fullyMeasuredCircularity(): AnalysisObjects {
  return {
    n: 11,
    density: { count: 11, tissue_um2: 25.0, per_um2: 0.44 },
    summary: {
      circularity: {
        n: 11,
        n_objects: 11,
        n_missing: 0,
        mean: 0.649,
        sd: 0.121,
        median: 0.66,
        iqr: 0.18,
        min: 0.401,
        max: 0.889,
        note: ESTIMATOR_NOTE,
        estimator_note: ESTIMATOR_NOTE,
      },
      area_px: {
        n: 11,
        n_objects: 11,
        n_missing: 0,
        mean: 812.3,
        sd: 240.1,
        median: 790,
        iqr: 300,
        min: 402,
        max: 1310,
      },
    },
  };
}

function renderPanel(objects: AnalysisObjects) {
  return render(
    <ObjectsPanel
      objects={objects}
      calibrated
      objectsCsvUrl={null}
      downloadStem="run-1"
    />,
  );
}

describe("ObjectsPanel shows the note a metric carries", () => {
  it("shows the estimator note for a circularity column at full n", () => {
    const { container } = renderPanel(fullyMeasuredCircularity());

    // The numbers the reviewer saw are still there...
    expect(container.textContent).toContain("circularity");
    expect(container.textContent).toContain("0.649");
    expect(container.textContent).toContain("0.889");

    // ...and now so is what the estimator does to them. Asserted on the
    // substrings that carry the current note's claims (Crofton: near-unbiased
    // on round shapes, values in (1.0, 1.015] reported as measured, size
    // distribution beside any comparison of small objects), so a reworded
    // paragraph that still says these keeps passing.
    expect(container.textContent).toContain("estimator, not geometry");
    expect(container.textContent).toContain("reported as measured");
    expect(container.textContent).toContain(
      "report the size distribution beside any circularity comparison",
    );
  });

  it("puts the note in the document text, not only in a title attribute", () => {
    renderPanel(fullyMeasuredCircularity());

    // getByText walks rendered text nodes; a `title` would not satisfy it.
    // Hovering is not reading, and nobody hovers a table cell.
    const shown = screen.getByText(/estimator, not geometry/);
    expect(shown).toBeInTheDocument();
    expect(shown.getAttribute("title")).toBeNull();
  });

  it("shows a coverage note for a partly measured metric", () => {
    const objects = fullyMeasuredCircularity();
    objects.summary.mean_prob = {
      n: 4,
      n_objects: 11,
      n_missing: 7,
      mean: 0.82,
      sd: 0.06,
      median: 0.83,
      iqr: 0.08,
      min: 0.71,
      max: 0.9,
      note: COVERAGE_NOTE,
    };

    const { container } = renderPanel(objects);
    expect(container.textContent).toContain(
      "Measured on 4 of 11 confirmed objects",
    );
  });

  it("prints the estimator paragraph once when it is also inside note", () => {
    // The backend joins the coverage sentence and the estimator paragraph into
    // `note` and publishes the estimator paragraph again on its own key.
    // Rendering both verbatim would repeat several hundred words per row.
    const objects = fullyMeasuredCircularity();
    objects.summary.circularity.note = `${COVERAGE_NOTE} ${ESTIMATOR_NOTE}`;

    const { container } = renderPanel(objects);
    const text = container.textContent ?? "";
    const occurrences = text.split("estimator, not geometry").length - 1;
    expect(occurrences).toBe(1);
    expect(text).toContain("Measured on 4 of 11 confirmed objects");
  });

  it("leaves a metric with no note alone", () => {
    const objects: AnalysisObjects = {
      n: 3,
      density: { count: 3, tissue_um2: null, per_um2: null },
      summary: {
        area_px: {
          n: 3,
          mean: 100,
          sd: 10,
          median: 100,
          iqr: 12,
          min: 90,
          max: 110,
        },
      },
    };

    const { container } = renderPanel(objects);
    const rows = within(container).getAllByRole("row");
    // Header plus exactly one metric row: no empty note row is emitted.
    expect(rows).toHaveLength(2);
  });
});

describe("metricNote", () => {
  it("returns null when the metric says nothing", () => {
    expect(metricNote({ n: 5 })).toBeNull();
    expect(metricNote({ n: 5, note: "   ", estimator_note: null })).toBeNull();
  });

  it("returns the estimator note when there is no coverage sentence", () => {
    expect(metricNote({ n: 5, estimator_note: "biased upward" })).toBe(
      "biased upward",
    );
  });

  it("joins a coverage sentence and an estimator note that do not overlap", () => {
    expect(
      metricNote({ n: 5, note: "4 of 11.", estimator_note: "biased upward" }),
    ).toBe("4 of 11. biased upward");
  });

  it("does not duplicate an estimator note already folded into note", () => {
    expect(
      metricNote({
        n: 5,
        note: "4 of 11. biased upward",
        estimator_note: "biased upward",
      }),
    ).toBe("4 of 11. biased upward");
  });
});
