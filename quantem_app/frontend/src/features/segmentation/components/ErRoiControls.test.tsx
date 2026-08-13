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
    testingRoiId: null,
    onStartPlacement: vi.fn(),
    onMoveRoi: vi.fn(),
    onCancelPlacement: vi.fn(),
    onConfirmRoi: vi.fn(),
    onMarkRoiDone: vi.fn(),
    onDeleteRoi: vi.fn(),
    onActivateRoi: vi.fn(),
    onTestRoi: vi.fn(),
    ...overrides,
  };
  return { props, ...render(<ErRoiControls {...props} />) };
}

describe("ErRoiControls", () => {
  it("shows the active ROI, its source coordinates, and a size-neutral creation action", () => {
    renderControls();

    expect(screen.getByRole("heading", { name: "ROI" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add New ROI" })).toBeInTheDocument();
    expect(screen.getByText("ROI 1: 512×512 px at (100, 200)")).toBeInTheDocument();
    expect(screen.getByText("active")).toBeInTheDocument();
  });

  it("offers explicit controls to open, move, test, delete, and mark an ROI done", async () => {
    const user = userEvent.setup();
    const onMoveRoi = vi.fn();
    const onTestRoi = vi.fn();
    const onMarkRoiDone = vi.fn();
    renderControls({ onMoveRoi, onTestRoi, onMarkRoiDone });

    expect(screen.getByRole("button", { name: "Open" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "Move" }));
    expect(onMoveRoi).toHaveBeenCalledWith("roi-1");
    await user.click(screen.getByRole("button", { name: "Test" }));
    expect(onTestRoi).toHaveBeenCalledWith("roi-1");
    await user.click(screen.getByRole("checkbox", { name: /mark roi 1.*done/i }));
    expect(onMarkRoiDone).toHaveBeenCalledWith("roi-1", true);
    expect(screen.getByRole("button", { name: /delete roi 1/i })).toBeInTheDocument();
  });

  it("hides Move after the ROI is marked done", () => {
    renderControls({
      rois: [makeRoi({ completed_for_segmentation: true })],
    });

    expect(screen.queryByRole("button", { name: "Move" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open" })).toBeEnabled();
    expect(screen.getByRole("checkbox", { name: /mark roi 1.*done/i })).toBeChecked();
  });

  it("keeps ROI numbers in creation order when the active ROI changes", () => {
    const older = makeRoi({
      id: "roi-older",
      x: 10,
      y: 20,
      is_active: false,
      created_at: "2026-01-01T00:00:00Z",
    });
    const newer = makeRoi({
      id: "roi-newer",
      x: 30,
      y: 40,
      is_active: true,
      created_at: "2026-01-02T00:00:00Z",
    });
    const rendered = renderControls({
      rois: [newer, older],
      activeRoiId: newer.id,
    });

    expect(screen.getByText("ROI 1: 512×512 px at (10, 20)")).toBeInTheDocument();
    expect(screen.getByText("ROI 2: 512×512 px at (30, 40)")).toBeInTheDocument();

    rendered.rerender(
      <ErRoiControls
        {...rendered.props}
        rois={[
          { ...older, is_active: true },
          { ...newer, is_active: false },
        ]}
        activeRoiId={older.id}
      />
    );

    expect(screen.getByText("ROI 1: 512×512 px at (10, 20)")).toBeInTheDocument();
    expect(screen.getByText("ROI 2: 512×512 px at (30, 40)")).toBeInTheDocument();
  });

  it("shows creation and relocation as deliberate pending actions", () => {
    renderControls({ placementActive: true });
    expect(screen.getByText("Click to place the new ROI.")).toBeInTheDocument();
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
