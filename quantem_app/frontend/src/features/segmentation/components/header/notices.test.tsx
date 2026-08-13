/**
 * The two surfaces the failure-copy package added to the labeling header.
 *
 * Both are about the same thing: a screen that used to leave the user with a
 * red sentence and no way forward, or with an empty canvas and no explanation.
 * What is asserted here is the *rendered* text and the *rendered* control,
 * because that is what the user gets — a catalogue entry that exists and is
 * never rendered is worth nothing.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { HeaderNotices } from "./notices";
import type { ImageSegmentation } from "@/shared/types";

function makeSegmentation(
  overrides: Partial<ImageSegmentation> & Record<string, unknown> = {}
): ImageSegmentation {
  return {
    id: "seg-1",
    segmentation_type: {
      id: "type-1",
      internal_name: "quantem_internal_mito",
      short_name: "Mito",
      long_name: "Mitochondria",
      default_color: null,
      tags: [],
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    },
    status_stage: "CANDIDATES_READY",
    status_progress: 100,
    segment_counts: { CONFIRMED: 0, EXCLUDED: 0, INFERRED: 0, CANDIDATE: 0 },
    source_models: [],
    config: { supports_instance_params: true, instance_params: null },
    is_complete: false,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  } as ImageSegmentation;
}

function renderNotices(segmentation: ImageSegmentation | null) {
  return render(
    <HeaderNotices
      currentSegmentation={segmentation}
      isOrganelle
      appliedAdapter={null}
      runTargetLabel="QuantEM"
      isComplete={false}
    />
  );
}

describe("a failed run whose class the server named", () => {
  it("leads with the class, keeps the server's sentence, and offers the way out", () => {
    renderNotices(
      makeSegmentation({
        status_stage: "FAILED",
        status_error: "quantem:mito is not installed on this machine.",
        error_code: "model_not_installed",
      })
    );

    // The class's own headline, which is already the answer -- not "the run
    // failed", which is only the question.
    expect(
      screen.getByText(/This model is not installed on this computer/)
    ).toBeInTheDocument();
    // The blame, taken by name.
    expect(
      screen.getByText(/Nothing is wrong with your image or your work/)
    ).toBeInTheDocument();
    // The server's own sentence, which is the only text naming *which* pack.
    expect(screen.getByText(/quantem:mito is not installed/)).toBeInTheDocument();
    // And a control, in this application.
    const action = screen.getByRole("link", { name: "Open Models" });
    expect(action).toHaveAttribute("href", "#/models");
  });

  it("renders exactly as it did before when there is no code", () => {
    renderNotices(
      makeSegmentation({
        status_stage: "FAILED",
        status_error: "The worker stopped without saying why.",
      })
    );

    expect(
      screen.getByText(/The last run on this segmentation failed/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/The worker stopped without saying why/)
    ).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("ignores a code this build has never heard of", () => {
    // Forward compatibility: a newer backend must not blank the notice.
    renderNotices(
      makeSegmentation({
        status_stage: "FAILED",
        status_error: "Something specific went wrong.",
        error_code: "quantum_flux_inversion",
      })
    );

    expect(
      screen.getByText(/The last run on this segmentation failed/)
    ).toBeInTheDocument();
    expect(screen.getByText(/Something specific went wrong/)).toBeInTheDocument();
  });
});

describe("empty-run advice", () => {
  const emptyRun = {
    kind: "no_objects",
    message: "This run found no objects.",
    next_steps: ["Check the image's pixel size."],
  };

  it("does not render a separate notice box", () => {
    renderNotices(makeSegmentation({ run_notice: emptyRun }));
    expect(screen.queryByText("This run found no objects.")).not.toBeInTheDocument();
  });
});
