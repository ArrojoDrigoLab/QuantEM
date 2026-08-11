/**
 * The tick box called "Done", next to the thing the Adapt wizard calls "done".
 *
 * `Done (ER)` writes `RoiSegmentationStatus.is_complete`, which nothing outside
 * this list reads. What "Adapt a model" needs is a `CompletedROI` polygon, and
 * only the Confirmed area tool makes one. So a user ticked Done (ER), opened
 * the wizard, and was told *"No completed ROI on this image. Mark the area you
 * have finished annotating as complete."* -- an instruction to repeat what they
 * had just done, in the words they had just used.
 */

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import {
  ErRoiControls,
  type ErRoiSection,
} from "@/features/segmentation/components/ErRoiControls";
import type { SegmentationRoi } from "@/shared/types/segmentation";

function makeRoi(overrides: Partial<SegmentationRoi> = {}): SegmentationRoi {
  return {
    id: "roi-1",
    segmentation: "seg-1",
    x: 100,
    y: 200,
    width: 2048,
    height: 2048,
    source: "MANUAL",
    is_active: true,
    is_complete: false,
    completed_for_segmentation: false,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

function renderControls(overrides: Partial<ErRoiSection> = {}) {
  const props: ErRoiSection = {
    placementActive: false,
    pendingRoiActive: false,
    confirming: false,
    rois: [makeRoi()],
    activeRoiId: "roi-1",
    markingRoiId: null,
    deletingRoiId: null,
    onStartPlacement: vi.fn(),
    onCancelPlacement: vi.fn(),
    onConfirmRoi: vi.fn(),
    onMarkRoiDone: vi.fn(),
    onDeleteRoi: vi.fn(),
    onActivateRoi: vi.fn(),
    ...overrides,
  };
  return { props, ...render(<ErRoiControls {...props} />) };
}

describe("ErRoiControls", () => {
  it("does not call the per-ROI flag done or complete", () => {
    renderControls();

    // Either word sends the reader to the Adapt wizard's precondition, which
    // this flag has nothing to do with.
    expect(screen.queryByText(/done \(er\)/i)).not.toBeInTheDocument();
    expect(screen.getByLabelText("Reviewed")).toBeInTheDocument();
  });

  it("says what the flag is, and what it now counts for", () => {
    renderControls();

    expect(
      screen.getByText(/records that you have been through this ROI window/i)
    ).toBeInTheDocument();
    // This used to pin the opposite claim -- that ticking it "does not create a
    // confirmed area, which is what Adapt a model needs". A reviewed ROI is now
    // read as training data on the same terms a confirmed area is, so the copy
    // says so and this says the copy says so.
    expect(screen.getByText(/Fine-tuning trains on both/i)).toBeInTheDocument();
  });

  it("names the control that does produce one", () => {
    renderControls();

    // The wizard's advice was to do the thing you just did. This is the thing
    // you actually have to do.
    expect(
      screen.getByText(/Switch to Review, choose Correct, then "Confirmed area"/)
    ).toBeInTheDocument();
  });

  it("still toggles the flag it always toggled", async () => {
    const user = userEvent.setup();
    const onMarkRoiDone = vi.fn();
    renderControls({ onMarkRoiDone });

    await user.click(screen.getByLabelText("Reviewed"));

    expect(onMarkRoiDone).toHaveBeenCalledWith("roi-1", true);
  });

  it("reflects the stored per-organelle flag", () => {
    renderControls({ rois: [makeRoi({ completed_for_segmentation: true })] });

    expect(screen.getByLabelText("Reviewed")).toBeChecked();
  });

  it("says nothing about reviewing when there are no ROIs", () => {
    renderControls({ rois: [] });

    expect(
      screen.queryByText(/records that you have been through/i)
    ).not.toBeInTheDocument();
  });

  it("tells the truth about what deleting a ROI destroys", async () => {
    const user = userEvent.setup();
    renderControls();

    await user.click(
      screen.getByRole("button", { name: /Delete ROI 2048×2048 at \(100, 200\)/ })
    );

    // `CompletedROI` hangs off the segmentation, not off the ROI rectangle, so
    // deleting the window leaves any confirmed area standing. Saying so matters
    // here precisely because the two are being told apart.
    expect(
      await screen.findByText(/is a separate shape and is not affected/)
    ).toBeInTheDocument();
  });
});
