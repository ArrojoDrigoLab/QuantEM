import { render, screen, within } from "@testing-library/react";
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
    onEditRoi: vi.fn(),
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
  it("puts the compact Add New action on the ROI heading and shows ROI dimensions", () => {
    renderControls();

    const heading = screen.getByRole("heading", { name: "ROI" }).parentElement;
    expect(heading).not.toBeNull();
    expect(within(heading!).getByRole("button", { name: "Add New" })).toBeInTheDocument();
    expect(screen.getByText("ROI 1: 512x512 px")).toBeInTheDocument();
    const active = screen.getByText("active");
    expect(active.parentElement).toHaveClass("er-roi-item-actions");
  });

  it("offers Edit Area on the title line with the remaining ROI actions below", async () => {
    const user = userEvent.setup();
    const roi = makeRoi();
    const onEditRoi = vi.fn();
    const onTestRoi = vi.fn();
    const onMarkRoiDone = vi.fn();
    renderControls({ rois: [roi], onEditRoi, onTestRoi, onMarkRoiDone });

    const editArea = screen.getByRole("button", { name: "Edit Area" });
    expect(editArea.parentElement).toHaveClass("er-roi-item-summary");
    await user.click(editArea);
    expect(onEditRoi).toHaveBeenCalledWith(roi);
    expect(screen.getByRole("button", { name: "Open" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "Test" }));
    expect(onTestRoi).toHaveBeenCalledWith("roi-1");
    await user.click(screen.getByRole("checkbox", { name: /mark roi 1.*done/i }));
    expect(onMarkRoiDone).toHaveBeenCalledWith("roi-1", true);
    expect(screen.getByRole("button", { name: /delete roi 1/i })).toBeInTheDocument();
  });

  it("keeps Edit Area visible but disabled after the ROI is marked done", () => {
    renderControls({
      rois: [makeRoi({ completed_for_segmentation: true })],
    });

    expect(screen.getByRole("button", { name: "Edit Area" })).toBeDisabled();
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
    const rendered = renderControls({ rois: [newer, older], activeRoiId: newer.id });

    expect(screen.getAllByText(/ROI [12]: 512x512 px/)).toHaveLength(2);

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

    expect(screen.getAllByText(/ROI [12]: 512x512 px/)).toHaveLength(2);
  });

  it("shows placement and area editing as deliberate pending actions", () => {
    const placing = renderControls({ placementActive: true });
    expect(screen.getByText("Click to place the new ROI.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
    placing.unmount();

    renderControls({
      placementActive: true,
      pendingRoiActive: true,
      relocatingRoiId: "roi-1",
    });
    expect(screen.getByRole("button", { name: "Save Area" })).toBeInTheDocument();
    expect(screen.getByText(/drag an edge or corner to resize/i)).toBeInTheDocument();
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
