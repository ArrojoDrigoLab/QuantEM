import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { GroundTruthProvenancePanel } from "@/features/improve/components/GroundTruthProvenance";

const REAL_SESSION = {
  confirmedFromModel: 86,
  drawnByHand: 4,
  rejected: 0,
  totalConfirmed: 90,
  regions: 1,
};

describe("GroundTruthProvenancePanel", () => {
  it("shows the confirmed-from-model vs drawn-by-hand split", () => {
    // The honesty gap: 86 of 90 "annotations" were the model's own candidates
    // the user confirmed, which makes a held-out Dice of 0.99 near-tautological.
    render(
      <GroundTruthProvenancePanel
        provenance={REAL_SESSION}
        loading={false}
        error={null}
      />
    );

    expect(screen.getByText("Confirmed from model")).toBeInTheDocument();
    expect(screen.getByText("86")).toBeInTheDocument();
    expect(screen.getByText("Drawn by hand")).toBeInTheDocument();
    expect(screen.getByText("4")).toBeInTheDocument();
    expect(screen.getByText("Rejected")).toBeInTheDocument();
    expect(screen.getByText("90")).toBeInTheDocument();
    expect(screen.getByText("96%")).toBeInTheDocument();
  });

  it("caveats a score measured mostly against the model's own output", () => {
    render(
      <GroundTruthProvenancePanel
        provenance={REAL_SESSION}
        loading={false}
        error={null}
      />
    );

    expect(screen.getByText(/measuring the model against itself/)).toBeInTheDocument();
    expect(screen.getByText(/draw a region from scratch/)).toBeInTheDocument();
  });

  it("does not caveat a reference with real independent annotations", () => {
    // Confirming model output is the intended workflow, not misconduct.
    render(
      <GroundTruthProvenancePanel
        provenance={{
          confirmedFromModel: 30,
          drawnByHand: 70,
          rejected: 5,
          totalConfirmed: 100,
          regions: 2,
        }}
        loading={false}
        error={null}
      />
    );

    expect(
      screen.queryByText(/measuring the model against itself/)
    ).not.toBeInTheDocument();
    expect(screen.getByText("30%")).toBeInTheDocument();
  });

  it("admits it could not count rather than implying a clean split", () => {
    render(
      <GroundTruthProvenancePanel
        provenance={null}
        loading={false}
        error="HTTP 500"
      />
    );

    expect(screen.getByText(/could not be counted/)).toBeInTheDocument();
    expect(screen.queryByText("Confirmed from model")).not.toBeInTheDocument();
  });

  it("says when there is no completed area to break down", () => {
    render(
      <GroundTruthProvenancePanel
        provenance={{ ...REAL_SESSION, regions: 0 }}
        loading={false}
        error={null}
      />
    );

    expect(screen.getByText(/No completed area was found/)).toBeInTheDocument();
  });
});
