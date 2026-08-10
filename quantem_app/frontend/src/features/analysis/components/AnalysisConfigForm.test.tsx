/**
 * The numeric fields have to enforce their own bounds.
 *
 * `min={1} max={1000}` on a number input stops nothing in any browser: 20000
 * typed into Replicates is accepted, looks accepted, and the field says
 * nothing. The only objection was a message rendered next to "Run analysis" at
 * the bottom of a panel that is taller than a 720px window -- so the reported
 * experience was a button that did nothing at all. The bound is now stated at
 * the field, while it is still focused, exactly as the band-edge field has
 * always done.
 */

import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AnalysisConfigForm } from "@/features/analysis/components/AnalysisConfigForm";
import {
  defaultFormState,
  type AnalysisFormState,
} from "@/features/analysis/analysisOptions";
import type { ImageSegmentation } from "@/shared/types/images";

const MITO: ImageSegmentation = {
  id: "seg-mito",
  segmentation_type: {
    id: "type-mito",
    internal_name: "quantem_internal_mito",
    short_name: "Mitochondria",
    long_name: "Mitochondria",
    default_color: null,
    tags: [],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  },
  status_stage: "CANDIDATES_READY",
  status_progress: 100,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

/** Render the form as a controlled component so typing actually moves state. */
function renderForm(
  initial?: Partial<AnalysisFormState>,
  selectedSegmentation: ImageSegmentation | null = null
) {
  const onSubmit = vi.fn();
  function Harness() {
    const [state, setState] = useState<AnalysisFormState>({
      ...defaultFormState([MITO], MITO.id),
      ...initial,
    });
    return (
      <AnalysisConfigForm
        segmentations={[MITO]}
        selectedSegmentation={selectedSegmentation}
        state={state}
        onChange={setState}
        onSubmit={onSubmit}
        submitting={false}
        error={null}
      />
    );
  }
  render(<Harness />);
  return { onSubmit };
}

describe("AnalysisConfigForm", () => {
  it("says the replicate limit at the field, not only after a click", async () => {
    const user = userEvent.setup();
    renderForm();

    const replicates = screen.getByLabelText("Replicates");
    await user.clear(replicates);
    await user.type(replicates, "20000");

    expect(
      screen.getByText(/Replicates must be 1000 or fewer/)
    ).toBeInTheDocument();
    expect(replicates).toHaveAttribute("aria-invalid", "true");
  });

  it("objects to an empty replicate count too", async () => {
    const user = userEvent.setup();
    renderForm();

    await user.clear(screen.getByLabelText("Replicates"));

    expect(
      screen.getByText("Replicates must be a whole number of at least 1.")
    ).toBeInTheDocument();
  });

  it("says nothing about a value inside the range", async () => {
    const user = userEvent.setup();
    renderForm();

    const replicates = screen.getByLabelText("Replicates");
    await user.clear(replicates);
    await user.type(replicates, "500");

    expect(screen.queryByText(/Replicates must/)).not.toBeInTheDocument();
    expect(replicates).toHaveAttribute("aria-invalid", "false");
  });

  /**
   * Said before the run is spent. `run_analysis` blanks every physical unit
   * when the analysed segmentation's objects predate the image's calibration,
   * and until this notice the first place that said so was the finished
   * bundle — blank micron columns, minutes after the click.
   */
  describe("objects that predate the calibration", () => {
    it("warns above the Run button when the server says so", () => {
      renderForm(undefined, {
        ...MITO,
        objects_pixel_size: {
          produced_nm: [null],
          predates_calibration: true,
          unstamped_count: 0,
        },
      });

      const notice = screen.getByTestId("analysis-objects-pixel-size");
      expect(notice).toHaveTextContent("Objects predate the pixel size");
      expect(notice).toHaveTextContent(
        /produced while this image had no pixel size/
      );
      // The consequence for *this* screen: the run still works, in pixels.
      expect(notice).toHaveTextContent(/in pixels/);
      expect(notice).toHaveTextContent(/wrong-scale caveat/);
    });

    it("stays silent when the objects were made at the recorded scale", () => {
      renderForm(undefined, {
        ...MITO,
        objects_pixel_size: {
          produced_nm: [5],
          predates_calibration: false,
          unstamped_count: 0,
        },
      });

      expect(
        screen.queryByTestId("analysis-objects-pixel-size")
      ).not.toBeInTheDocument();
    });

    it("stays silent with no selected segmentation or an older backend", () => {
      renderForm();

      expect(
        screen.queryByTestId("analysis-objects-pixel-size")
      ).not.toBeInTheDocument();
    });
  });
});
