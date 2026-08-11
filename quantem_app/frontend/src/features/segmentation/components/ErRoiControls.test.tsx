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
    width: 512,
    height: 512,
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
    relocatingRoiId: null,
    confirming: false,
    rois: [makeRoi()],
    activeRoiId: "roi-1",
    markingRoiId: null,
    deletingRoiId: null,
    activatingRoiId: null,
    onStartPlacement: vi.fn(),
    onMoveRoi: vi.fn(),
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
  it("shows the active ROI, its source coordinates, and the 512px creation action", () => {
    renderControls();

    expect(screen.getByRole("heading", { name: "ROI" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "New 512×512 ROI" })).toBeInTheDocument();
    expect(screen.getByText("ROI 1: 512×512 px at (100, 200)")).toBeInTheDocument();
    expect(screen.getByText("active")).toBeInTheDocument();
  });

  it("offers explicit controls to open, move, delete, and mark an ROI done", async () => {
    const user = userEvent.setup();
    const onMoveRoi = vi.fn();
    const onMarkRoiDone = vi.fn();
    renderControls({ onMoveRoi, onMarkRoiDone });

    expect(screen.getByRole("button", { name: "Open" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Move" }));
    expect(onMoveRoi).toHaveBeenCalledWith("roi-1");
    await user.click(screen.getByRole("checkbox", { name: /mark roi 1.*done/i }));
    expect(onMarkRoiDone).toHaveBeenCalledWith("roi-1", true);
    expect(screen.getByRole("button", { name: /delete roi 1/i })).toBeInTheDocument();
  });

  it("shows creation and relocation as deliberate pending actions", () => {
    renderControls({ placementActive: true });
    expect(screen.getByText("Click to place a 512×512 ROI.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel placement" })).toBeInTheDocument();

    renderControls({
      placementActive: true,
      pendingRoiActive: true,
      relocatingRoiId: "roi-1",
    });
    expect(screen.getByRole("button", { name: "Move ROI" })).toBeInTheDocument();
  });

  it("explains the scope of a deletion before it happens", async () => {
    const user = userEvent.setup();
    renderControls();

    await user.click(screen.getByRole("button", { name: /delete roi 1/i }));

    expect(
      await screen.findByText(/removed from every segmentation on this image/i)
    ).toBeInTheDocument();
  });
});
