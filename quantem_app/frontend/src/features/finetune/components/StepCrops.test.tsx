/**
 * The other end of the "two things called complete" problem.
 *
 * The server's blocker is *"No completed ROI on this image. Mark the area you
 * have finished annotating as complete."* -- which, to somebody who has just
 * ticked a box labelled "Done (ER)" on the labeling screen, reads as an
 * instruction to do it again. The labeling screen no longer uses that word, and
 * this screen now names the control that actually produces a completed ROI.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { StepCrops } from "@/features/finetune/components/StepCrops";
import type { AdaptCropsResponse } from "@/shared/types/finetune";

const NO_COMPLETED_ROI =
  "No completed ROI on this image. Mark the area you have finished " +
  "annotating as complete: inside it every confirmed object is foreground and " +
  "everything else is background, and without that there is no valid " +
  "background to score against.";

function makeCrops(overrides: Partial<AdaptCropsResponse> = {}): AdaptCropsResponse {
  return {
    crops: [],
    split_mode: "within-image",
    n_images: 1,
    ready: false,
    blockers: [],
    warnings: [],
    train_crop_names: [],
    heldout_crop_names: [],
    modes: ["threshold_only"],
    ...overrides,
  };
}

function renderStep(crops: AdaptCropsResponse) {
  return render(
    <StepCrops crops={crops} loading={false} error={null} onRefresh={vi.fn()} />
  );
}

describe("StepCrops", () => {
  it("still shows the server's blocker verbatim", () => {
    renderStep(makeCrops({ blockers: [NO_COMPLETED_ROI] }));

    expect(screen.getByText(NO_COMPLETED_ROI)).toBeInTheDocument();
  });

  it("names the tool that makes a completed ROI", () => {
    renderStep(makeCrops({ blockers: [NO_COMPLETED_ROI] }));

    expect(screen.getByText("Confirmed area")).toBeInTheDocument();
    expect(
      screen.getByText(/Switch to Review, choose Correct, then "Confirmed area"/)
    ).toBeInTheDocument();
  });

  it("says which control does not count, by name", () => {
    renderStep(makeCrops({ blockers: [NO_COMPLETED_ROI] }));

    // The one the user had just used, and the reason they read the blocker as
    // advice to repeat themselves.
    expect(
      screen.getByText(/tick box beside the ER ROI list is a different thing/i)
    ).toBeInTheDocument();
  });

  it("says it for the other completed-area blocker too", () => {
    renderStep(
      makeCrops({
        blockers: [
          "2 completed area(s) found, but none of them could be used (too " +
            "small, or the image is unavailable).",
        ],
      })
    );

    expect(
      screen.getByText(/It is the polygon the/)
    ).toBeInTheDocument();
  });

  it("stays quiet about it when the blocker is about something else", () => {
    renderStep(
      makeCrops({
        blockers: [
          "No confirmed objects inside the completed area. Confirm the " +
            "objects you want the model to learn.",
        ],
        // That blocker does mention the completed area, so use one that does
        // not to pin the negative case.
      })
    );
    expect(screen.getByText(/It is the polygon the/)).toBeInTheDocument();

    renderStep(makeCrops({ blockers: ["This image has no probability map."] }));
    expect(screen.getAllByText(/It is the polygon the/)).toHaveLength(1);
  });

  it("adds nothing when there is nothing blocking", () => {
    renderStep(makeCrops({ ready: true }));

    expect(screen.queryByText(/Cannot proceed/)).not.toBeInTheDocument();
    expect(screen.queryByText(/It is the polygon the/)).not.toBeInTheDocument();
  });
});
