import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StepResults } from "@/features/finetune/components/StepResults";
import type { Adapter, AdapterSweep } from "@/shared/types/finetune";

function makeSweep(overrides: Partial<AdapterSweep> = {}): AdapterSweep {
  return {
    thresholds: [0.1, 0.3, 0.5, 0.7, 0.9],
    train_dice: [0.4, 0.72, 0.68, 0.55, 0.2],
    calibrated_threshold: 0.3,
    train_dice_at_calibrated: 0.72,
    train_dice_at_default: 0.68,
    heldout_dice_at_calibrated: 0.61,
    heldout_dice_at_default: 0.54,
    heldout_oracle: 0.79,
    improvement: 0.07,
    per_crop: { "roi-1": 0.74, "roi-2": 0.61 },
    train_crop_names: ["roi-1"],
    heldout_crop_names: ["roi-2"],
    ...overrides,
  };
}

function makeAdapter(overrides: Partial<Adapter> = {}): Adapter {
  const sweep = makeSweep();
  return {
    id: "ad-1",
    base_model: "quantem:mito",
    name: "mito @ liver",
    status: "SUCCESS",
    mode: "head",
    steps: 300,
    trainable_params: 5_775_000,
    segmentation_id: "seg-1",
    split_mode: "image-disjoint",
    train_crop_names: sweep.train_crop_names,
    heldout_crop_names: sweep.heldout_crop_names,
    sweep,
    calibrated_threshold: sweep.calibrated_threshold,
    heldout_dice: sweep.heldout_dice_at_calibrated,
    verified_reload: true,
    train_seconds: 91.4,
    applied_at: null,
    created_at: "2026-01-01T00:00:00Z",
    error: "",
    caveats: ["The threshold was fit on the training crops only."],
    ...overrides,
  };
}

/**
 * These cases are about the sweep numbers, not about where the annotations came
 * from, so they render with no provenance loaded. The composition panel has its
 * own tests.
 */
const NO_PROVENANCE = {
  provenance: null,
  provenanceLoading: false,
  provenanceError: null,
};

describe("StepResults", () => {
  it("badges the crops the threshold was fitted on (honesty rule 2)", () => {
    render(<StepResults adapter={makeAdapter()} baseSweep={null} {...NO_PROVENANCE} />);

    const fittedRow = screen.getByText("roi-1").closest("tr");
    const heldoutRow = screen.getByText("roi-2").closest("tr");
    expect(fittedRow).not.toBeNull();
    expect(heldoutRow).not.toBeNull();
    expect(
      within(fittedRow as HTMLElement).getByText("threshold fitted on this")
    ).toBeInTheDocument();
    expect(
      within(heldoutRow as HTMLElement).getByText("held out")
    ).toBeInTheDocument();
  });

  it("shows the oracle as a ceiling and the split mode next to the held-out score", () => {
    render(<StepResults adapter={makeAdapter()} baseSweep={null} {...NO_PROVENANCE} />);

    expect(screen.getByText(/Oracle ceiling — not a target/)).toBeInTheDocument();
    const heldoutCard = screen
      .getByText("Held-out, chosen threshold")
      .closest("div") as HTMLElement;
    expect(within(heldoutCard).getByText("0.610")).toBeInTheDocument();
    expect(within(heldoutCard).getByText("image-disjoint")).toBeInTheDocument();
    // Two held-out figures, each carrying the split mode; the sweep legend
    // names it a third time.
    expect(screen.getAllByText("image-disjoint").length).toBeGreaterThanOrEqual(2);
  });

  it("reports improvement as the held-out change, not the training change", () => {
    render(<StepResults adapter={makeAdapter()} baseSweep={null} {...NO_PROVENANCE} />);
    expect(screen.getByText("+0.070")).toBeInTheDocument();
  });

  it("hides both held-out numbers when nothing was held out", () => {
    const sweep = makeSweep({
      heldout_dice_at_calibrated: null,
      heldout_dice_at_default: null,
      heldout_oracle: null,
      improvement: null,
      heldout_crop_names: [],
      train_crop_names: ["roi-1", "roi-2"],
    });
    render(
      <StepResults
        adapter={makeAdapter({
          split_mode: "no-heldout",
          sweep,
          heldout_dice: null,
          train_crop_names: ["roi-1", "roi-2"],
          heldout_crop_names: [],
        })}
        baseSweep={null}
        {...NO_PROVENANCE}
      />
    );

    expect(screen.getAllByText("no held-out data").length).toBeGreaterThan(0);
    expect(screen.queryByText(/Oracle ceiling/)).not.toBeInTheDocument();
    // Both per-crop rows are marked as fitted, so neither can be read as held out.
    expect(screen.getAllByText("threshold fitted on this")).toHaveLength(2);
  });

  it("says so when the saved head was never re-scored", () => {
    render(
      <StepResults adapter={makeAdapter({ verified_reload: false })} baseSweep={null} {...NO_PROVENANCE} />
    );
    expect(screen.getByText("not re-scored after saving")).toBeInTheDocument();
  });
});
