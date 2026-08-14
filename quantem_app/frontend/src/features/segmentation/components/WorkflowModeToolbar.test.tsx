/**
 * The slot contract.
 *
 * Two other packages render controls into this toolbar without opening it, so
 * the slots are the interface between them and the file's owner. If a slot
 * silently stops rendering, the control it carries disappears from the app with
 * nothing failing anywhere near the package that owns it.
 */

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { WorkflowModeToolbar } from "@/features/segmentation/components/WorkflowModeToolbar";

function renderToolbar(extra: Partial<Parameters<typeof WorkflowModeToolbar>[0]> = {}) {
  return render(
    <WorkflowModeToolbar
      workflowMode="review"
      reviewPhase="correction"
      correctionTool="draw"
      hoverActionMode="confirm"
      drawBrushSize={24}
      draftOperation="include"
      hasDrawStrokes={false}
      onReviewPhaseChange={vi.fn()}
      onCorrectionToolChange={vi.fn()}
      onHoverActionModeChange={vi.fn()}
      onDrawBrushSizeChange={vi.fn()}
      onDraftOperationChange={vi.fn()}
      onConfirmShape={vi.fn()}
      onClearDrawing={vi.fn()}
      {...extra}
    />
  );
}

describe("WorkflowModeToolbar slots", () => {
  it("defaults to Include and can switch a draft to Exclude", async () => {
    const onDraftOperationChange = vi.fn();
    renderToolbar({
      reviewPhase: "correction",
      draftOperation: "include",
      onDraftOperationChange,
    });
    expect(screen.getByRole("button", { name: "Include" })).toHaveClass("active");
    await userEvent.click(screen.getByRole("button", { name: "Exclude" }));
    expect(onDraftOperationChange).toHaveBeenCalledWith("exclude");
  });

  it("renders an extra correction sub-tool without a phase toggle", () => {
    renderToolbar({
      reviewPhase: "correction",
      extraModes: <button type="button">Focus queue</button>,
    });

    expect(screen.queryByRole("button", { name: "Review" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Correct" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Focus queue" })).toBeInTheDocument();
  });

  it("renders extra tools at the end of the toolbar", () => {
    renderToolbar({ extraTools: <button type="button">Mark checked area</button> });

    expect(screen.getByRole("button", { name: "Mark checked area" })).toBeTruthy();
  });

  it("adds nothing to the toolbar when no package fills a slot", () => {
    const { container } = renderToolbar();

    expect(container.textContent).not.toContain("undefined");
    expect(screen.queryByRole("button", { name: "Review" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Correct" })).not.toBeInTheDocument();
  });

  it("shows brush and polygon icon tools for non-ER object segmentations", async () => {
    const onCorrectionToolChange = vi.fn();
    renderToolbar({
      reviewPhase: "correction",
      correctionTool: "draw",
      isErSegmentation: false,
      onCorrectionToolChange,
    });

    const brush = screen.getByRole("button", { name: "Brush" });
    const polygon = screen.getByRole("button", { name: "Polygon" });
    expect(brush.querySelector("svg")).not.toBeNull();
    expect(polygon.querySelector("svg")).not.toBeNull();
    expect(screen.queryByRole("button", { name: "Draw" })).toBeNull();

    await userEvent.click(polygon);
    expect(onCorrectionToolChange).toHaveBeenCalledWith("polygon");
  });

  it("offers a close action for a non-ER polygon draft", async () => {
    const onClosePolygon = vi.fn();
    renderToolbar({
      reviewPhase: "correction",
      correctionTool: "polygon",
      isErSegmentation: false,
      polygonHasDraft: true,
      polygonCanClose: true,
      onClosePolygon,
    });

    await userEvent.click(screen.getByRole("button", { name: "Close polygon (R)" }));
    expect(onClosePolygon).toHaveBeenCalledTimes(1);
  });
});
