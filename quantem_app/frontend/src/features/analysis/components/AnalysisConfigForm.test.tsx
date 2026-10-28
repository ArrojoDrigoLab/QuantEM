import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import {
  defaultFormState,
  type AnalysisFormState,
} from "@/features/analysis/analysisOptions";
import { AnalysisConfigForm } from "@/features/analysis/components/AnalysisConfigForm";
import type { ImageSegmentation } from "@/shared/types/images";

function segmentation(
  id: string,
  internalName: string,
  longName: string
): ImageSegmentation {
  return {
    id,
    segmentation_type: {
      id: `type-${id}`,
      internal_name: internalName,
      short_name: longName,
      long_name: longName,
      category:
        internalName === "quantem_internal_tissue" ||
        internalName === "quantem_internal_cells" ||
        internalName === "quantem_internal_analysis_mask"
          ? "area"
          : "organelle",
      representation:
        internalName === "quantem_internal_cells" ? "instances" : "semantic",
      topology_policy: "non_overlapping",
      default_color: null,
      tags: [],
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    },
    status_stage: "CANDIDATES_READY",
    status_progress: 100,
    has_saved_area: true,
    is_complete: true,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}

const MITO = segmentation("seg-mito", "quantem_internal_mito", "Mitochondria");
const TISSUE = segmentation("seg-tissue", "quantem_internal_tissue", "Sample Area");
const CELLS = segmentation("seg-cells", "quantem_internal_cells", "Cells");
const ANALYSIS_MASK = segmentation(
  "seg-analysis-mask",
  "quantem_internal_analysis_mask",
  "Analysis Mask"
);
const SEGMENTATIONS = [MITO, TISSUE, CELLS, ANALYSIS_MASK];

function renderForm() {
  const onSubmit = vi.fn();

  function Harness() {
    const [state, setState] = useState<AnalysisFormState>(() =>
      defaultFormState(SEGMENTATIONS, MITO.id)
    );
    return (
      <AnalysisConfigForm
        segmentations={SEGMENTATIONS}
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
  it("offers Sample Area, Cells and independently toggleable outputs", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderForm();

    const selector = screen.getByLabelText("Measurement area") as HTMLSelectElement;
    expect(Array.from(selector.options).map((option) => option.text)).toEqual([
      "No area restriction",
      "Sample Area",
      "Cells",
      "Analysis Mask",
    ]);
    expect(selector.value).toBe(TISSUE.id);
    expect(screen.getByLabelText("Cells")).toHaveValue(CELLS.id);
    expect(screen.getByRole("checkbox", { name: "Whole image summary (always included)" })).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: "Per-cell breakdown" })).toBeChecked();
    expect(screen.getByText("Plain-language preview")).toBeInTheDocument();

    expect(screen.queryByLabelText("Point source")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Distance target")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Replicates")).not.toBeInTheDocument();
    expect(screen.queryByText(/Each one is rasterised/)).not.toBeInTheDocument();
    expect(
      screen.queryByText(
        "Runs in the background; the queue keeps going if you navigate away."
      )
    ).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Run Analysis" }));
    expect(onSubmit).toHaveBeenCalledOnce();
  });
});
